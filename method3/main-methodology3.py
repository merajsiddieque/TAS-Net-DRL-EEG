from __future__ import print_function
import torch
import os.path as osp
import time
import argparse
import datetime
import os
import torch.nn as nn
import h5py
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli
from models import DSN
from utils import weights_init, save_checkpoint, inv_lr_scheduler, read_json
from rewards import compute_reward_det_coff, compute_reward_coff, compute_reward
from models import HighLevelActorCritic, LowLevelActorCritic, HierarchicalNetworkProxy
from rewards import compute_step_reward
from hierarchical_rl import TDActorCriticUpdater
import numpy as np
import random
from scipy.io import savemat
from Graph_Net import ClassifierGNN
from evaluate import evaluate
import math

parser = argparse.ArgumentParser()
parser.add_argument('--training', action='store_true', default=False, help='Training or Validate.')
parser.add_argument('--seed', type=int, default=27, help='Random seed')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--subject_id', type=int, default=0, help="subject index (default: 0)")
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--edge_features', type=int, default=32, help='graph edge features dimension.')
parser.add_argument('--n_feature', type=int, default=192, help='Number of hidden units.')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hid_dim', type=int, default=256, help='hidden unit dimension of DSN (default: 256).')
parser.add_argument('--deep_features', type=str, default=r'..\features\session_1\source_h5_file.h5', help='output directory and fragments')
parser.add_argument('--save_path', type=str, default=r'TAS-output\SEED\session1', help='output directory and fragments')
parser.add_argument('--fragment_length', type=int, default=8, help='Left or Right Maximum offset.')
parser.add_argument('--num_fragment', type=int, default=10, help='for eval emotion localization and unsupervised clustering.')
parser.add_argument('--reward_function', type=str, default='R1_R2', help='sim:R1, rep:R2 or mix:R1_R2.')
parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use.")

args = parser.parse_args()

torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    if args.training:
        datasets = h5py.File(args.deep_features, 'r')
        num_videos = len(datasets.keys())
        all_keys = list(datasets.keys())
        test_keys = all_keys[15*args.subject_id:15*(args.subject_id+1)]
        del all_keys[15*args.subject_id:15*(args.subject_id+1)]
        train_keys = all_keys
        print("# total videos {}. # train videos {}. # test videos {}.".format(num_videos, len(train_keys), len(test_keys)))
        # ------------------------------------------------------------------ #
        #  Networks                                                           #
        # ------------------------------------------------------------------ #
        Network1 = ClassifierGNN(
            in_features=args.n_feature,
            edge_features=args.edge_features,
            out_features=args.n_feature,
            device=DEVICE,
        )

        # High-Level Policy — selects coarse segments
        HighLevelPolicy = HighLevelActorCritic(
            in_dim=args.n_feature, hid_dim=args.hid_dim, num_layers=1, cell='gru'
        )
        # Low-Level Policy — selects frames within an active segment
        LowLevelPolicy = LowLevelActorCritic(
            in_dim=args.n_feature, hid_dim=args.hid_dim, num_layers=1, cell='gru'
        )

        HighLevelPolicy.apply(weights_init)
        LowLevelPolicy.apply(weights_init)

        print("Model size: {:.5f}M".format(
            sum(p.numel() for p in Network1.parameters()) / 1_000_000.0
        ))

        # ------------------------------------------------------------------ #
        #  Separate optimisers so high-level and low-level losses don't       #
        #  interfere with each other's gradients.                             #
        # ------------------------------------------------------------------ #
        LR = 0.001
        WD = 1e-4
        optimizer_gnn  = torch.optim.Adam(Network1.parameters(),       lr=LR, weight_decay=WD)
        optimizer_high = torch.optim.Adam(HighLevelPolicy.parameters(), lr=LR, weight_decay=WD)
        optimizer_low  = torch.optim.Adam(LowLevelPolicy.parameters(),  lr=LR, weight_decay=WD)

        scheduler_gnn  = lr_scheduler.StepLR(optimizer_gnn,  step_size=20, gamma=0.5)
        scheduler_high = lr_scheduler.StepLR(optimizer_high, step_size=20, gamma=0.5)
        scheduler_low  = lr_scheduler.StepLR(optimizer_low,  step_size=20, gamma=0.5)

        start_epoch = 0
        Network1       = Network1.to(DEVICE)
        HighLevelPolicy = HighLevelPolicy.to(DEVICE)
        LowLevelPolicy  = LowLevelPolicy.to(DEVICE)

        print("=====> Start Hierarchical RL + TD training <=====")
        start_time = time.time()

        # ------------------------------------------------------------------ #
        #  Logging                                                            #
        # ------------------------------------------------------------------ #
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, 'hierarchical_metrics.csv')
        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w') as f:
                f.write('epoch,high_level_loss,low_level_loss,TD_error,reward_per_step,recall\n')

        # TD-learning updater (R_t = r_t + gamma * V(s_{t+1}), gamma=0.99)
        td_updater = TDActorCriticUpdater(gamma=0.99)
        best_recall = 0.
        
        for epoch in range(start_epoch, args.epochs):
            Network1.train()
            HighLevelPolicy.train()
            LowLevelPolicy.train()

            idxs = np.arange(len(train_keys))
            np.random.shuffle(idxs)

            # Running accumulators for the epoch
            h_loss_total = 0.0
            l_loss_total = 0.0
            td_err_total = 0.0   # mean |TD-error| across all transitions
            r_step_total = 0.0   # mean per-step reward
            step_count   = 0

            for idx in idxs:
                key = train_keys[idx]
                seq = datasets[key]['features'][...]
                seq = torch.from_numpy(seq).to(DEVICE)

                # ---------------------------------------------------------- #
                #  Graph feature extraction (Network1 / GNN)                 #
                # ---------------------------------------------------------- #
                local_graphs0 = None
                seg_len = args.fragment_length * 2

                for n in range(math.ceil(seq.shape[0] / seg_len)):
                    if n != math.ceil(seq.shape[0] / seg_len) - 1:
                        data0 = seq[seg_len * n:seg_len * (n + 1), :]
                        sub_graph0, _ = Network1(data0)
                    else:
                        data0 = seq[seg_len * n:, :]
                        if data0.shape[0] == 1:
                            sub_graph0 = data0
                        else:
                            sub_graph0, _ = Network1(data0)

                    if local_graphs0 is not None:
                        local_graphs0 = torch.cat((local_graphs0, sub_graph0), dim=0)
                    else:
                        local_graphs0 = sub_graph0

                # Residual: combine raw sequence with graph-enriched features
                seq_graph0 = torch.add(seq, local_graphs0)

                num_segments = math.ceil(seq_graph0.shape[0] / seg_len)

                # ---------------------------------------------------------- #
                #  Hierarchical RL rollout                                   #
                #  Step 1: High-level policy selects coarse segments         #
                #  Step 2: Low-level policy selects frames inside segment    #
                # ---------------------------------------------------------- #
                high_transitions = []   # (reward, v_curr, log_prob)
                low_transitions  = []   # (reward, v_curr, log_prob)

                prev_h_action = None
                prev_l_action = None

                for s_i in range(num_segments):
                    start_idx  = s_i * seg_len
                    end_idx    = min(start_idx + seg_len, seq_graph0.shape[0])
                    seg_frames = seq_graph0[start_idx:end_idx]  # (T_seg, D)

                    # -- High-Level: should this segment be included? --
                    seg_feat     = seg_frames.mean(dim=0, keepdim=True).unsqueeze(0)  # (1,1,D)
                    h_prob, h_val = HighLevelPolicy(seg_feat)
                    h_m          = Bernoulli(h_prob)
                    h_action     = h_m.sample()
                    h_log_prob   = h_m.log_prob(h_action)

                    # Per-step reward: task + temporal consistency
                    h_reward     = compute_step_reward(
                        h_action.detach(), prev_h_action
                    ).to(DEVICE)
                    prev_h_action = h_action.detach()

                    high_transitions.append((
                        h_reward,
                        h_val.squeeze(),    # critic state value  V(s_t)
                        h_log_prob.squeeze()  # log π(a_t | s_t)
                    ))

                    # -- Low-Level: only runs inside a selected segment --
                    if h_action.item() == 1.0:
                        # Detach seg_frames from the GNN graph so that
                        # low_cost.backward() only touches LowLevelPolicy
                        # parameters and never tries to re-traverse Network1.
                        seg_frames_detached = seg_frames.detach()
                        l_probs, l_vals = LowLevelPolicy(seg_frames_detached.unsqueeze(0))  # (1,T,1)
                        l_probs = l_probs.squeeze(0)  # (T, 1)
                        l_vals  = l_vals.squeeze(0)   # (T, 1)

                        for f_i in range(seg_frames_detached.shape[0]):
                            l_m        = Bernoulli(l_probs[f_i])
                            l_action   = l_m.sample()
                            l_log_prob = l_m.log_prob(l_action)

                            l_reward   = compute_step_reward(
                                l_action.detach(), prev_l_action
                            ).to(DEVICE)
                            prev_l_action = l_action.detach()

                            low_transitions.append((
                                l_reward,
                                l_vals[f_i].squeeze(),
                                l_log_prob.squeeze()
                            ))

                # ---------------------------------------------------------- #
                #  TD Actor-Critic loss — High-Level Policy                  #
                #  R_t = r_t + gamma * V(s_{t+1}),  gamma = 0.99            #
                # ---------------------------------------------------------- #
                high_cost    = torch.tensor(0.0, device=DEVICE)
                epoch_h_cost = 0.0

                for t in range(len(high_transitions)):
                    r, v_curr, logp = high_transitions[t]
                    v_next = high_transitions[t + 1][1] if t + 1 < len(high_transitions) else None
                    actor_l, critic_l, td_err = td_updater.calculate_td_loss(
                        r, v_curr, v_next, logp
                    )
                    high_cost    = high_cost + actor_l + critic_l
                    epoch_h_cost += actor_l.item() + critic_l.item()
                    td_err_total += abs(td_err.item())   # track |δ|
                    r_step_total += r.item()
                    step_count   += 1

                # ---------------------------------------------------------- #
                #  TD Actor-Critic loss — Low-Level Policy                   #
                # ---------------------------------------------------------- #
                low_cost     = torch.tensor(0.0, device=DEVICE)
                epoch_l_cost = 0.0

                for t in range(len(low_transitions)):
                    r, v_curr, logp = low_transitions[t]
                    v_next = low_transitions[t + 1][1] if t + 1 < len(low_transitions) else None
                    actor_l, critic_l, td_err = td_updater.calculate_td_loss(
                        r, v_curr, v_next, logp
                    )
                    low_cost     = low_cost + actor_l + critic_l
                    epoch_l_cost += actor_l.item() + critic_l.item()
                    td_err_total += abs(td_err.item())   # track |δ|
                    r_step_total += r.item()
                    step_count   += 1

                h_loss_total += epoch_h_cost
                l_loss_total += epoch_l_cost

                # ---------------------------------------------------------- #
                #  Backward passes                                            #
                #  high_cost graph: Network1 + HighLevelPolicy               #
                #  low_cost graph:  LowLevelPolicy only (seg_frames detached) #
                # ---------------------------------------------------------- #

                # Step 1 — High-level + GNN (independent backward)
                optimizer_gnn.zero_grad()
                optimizer_high.zero_grad()
                if isinstance(high_cost, torch.Tensor) and high_cost.requires_grad:
                    high_cost.backward()   # retain_graph not needed; graphs are separate
                    torch.nn.utils.clip_grad_norm_(Network1.parameters(),        5.0)
                    torch.nn.utils.clip_grad_norm_(HighLevelPolicy.parameters(), 5.0)
                    optimizer_gnn.step()
                    optimizer_high.step()

                # Step 2 — Low-level only (fully detached from GNN graph)
                optimizer_low.zero_grad()
                if isinstance(low_cost, torch.Tensor) and low_cost.requires_grad:
                    low_cost.backward()
                    torch.nn.utils.clip_grad_norm_(LowLevelPolicy.parameters(), 5.0)
                    optimizer_low.step()

            # ---------------------------------------------------------- #
            #  Epoch-level evaluation & logging                          #
            # ---------------------------------------------------------- #
            avg_h  = h_loss_total / max(1, len(idxs))
            avg_l  = l_loss_total / max(1, len(idxs))
            avg_td = td_err_total / max(1, step_count)   # mean |TD-error|
            avg_rs = r_step_total / max(1, step_count)   # mean per-step reward

            # Use the hierarchical proxy to produce frame-level probabilities
            # that the shared evaluate() function expects.
            Network2 = HierarchicalNetworkProxy(
                HighLevelPolicy, LowLevelPolicy, args.fragment_length * 2
            )
            Recall = evaluate(args, Network1, Network2, datasets, test_keys)

            print(
                "Epoch [{}/{}]  "
                "H-Loss: {:.4f}  L-Loss: {:.4f}  "
                "|TD-err|: {:.4f}  R/step: {:.4f}  "
                "Recall: {:.4f}".format(
                    epoch + 1, args.epochs,
                    avg_h, avg_l,
                    avg_td, avg_rs,
                    Recall
                )
            )

            # Append one row to logs/hierarchical_metrics.csv
            with open(log_file_path, 'a') as f:
                f.write(
                    f"{epoch + 1},{avg_h:.6f},{avg_l:.6f},"
                    f"{avg_td:.6f},{avg_rs:.6f},{Recall:.6f}\n"
                )

            if Recall > best_recall:
                best_recall = Recall
                os.makedirs(args.save_path, exist_ok=True)
                save_checkpoint(
                    Network1.state_dict(),
                    osp.join(args.save_path,
                             f'pretrained_best_model1_seed_subject{args.subject_id}_{args.reward_function}.pth.tar')
                )
                save_checkpoint(
                    Network2.state_dict(),
                    osp.join(args.save_path,
                             f'pretrained_best_model2_seed_subject{args.subject_id}_{args.reward_function}.pth.tar')
                )

            scheduler_gnn.step()
            scheduler_high.step()
            scheduler_low.step()

        elapsed = round(time.time() - start_time)
        print("Finished. Total elapsed time (h:m:s): {}".format(str(datetime.timedelta(seconds=elapsed))))

    else:
        # Evaluate mode natively maps logic for methodologies via wrapper restoring
        datasets = h5py.File(args.deep_features, 'r')
        test_keys = list(datasets.keys())[15 * args.subject_id:15 * (args.subject_id + 1)]

        print("# test videos {}.".format(len(test_keys)))
        model1 = ClassifierGNN(in_features=args.n_feature, edge_features=args.edge_features, out_features=args.n_feature, device=DEVICE)
        
        HighLevelPolicy = HighLevelActorCritic(in_dim=args.n_feature, hid_dim=args.hid_dim, num_layers=1, cell='gru')
        LowLevelPolicy = LowLevelActorCritic(in_dim=args.n_feature, hid_dim=args.hid_dim, num_layers=1, cell='gru')
        model2 = HierarchicalNetworkProxy(HighLevelPolicy, LowLevelPolicy, args.fragment_length * 2)

        checkpoint_path1 = osp.join(args.save_path, 'pretrained_best_model1_seed_subject' + str(args.subject_id) + '_' + str(args.reward_function) + '.pth.tar')
        checkpoint1 = torch.load(checkpoint_path1)
        model1.load_state_dict(checkpoint1)
        model1 = model1.to(DEVICE)
        
        checkpoint_path2 = osp.join(args.save_path, 'pretrained_best_model2_seed_subject' + str(args.subject_id) + '_' + str(args.reward_function) + '.pth.tar')
        checkpoint2 = torch.load(checkpoint_path2)
        model2.load_state_dict(checkpoint2)
        model2 = model2.to(DEVICE)
        
        with torch.no_grad():
            model1.eval()
            model2.eval()
            out_path = os.path.join(args.save_path, 'result_output')
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            save_idx = open(os.path.join(out_path, 'log_subject' + str(args.subject_id) + '_' + str(args.reward_function) + '_' + str(args.num_fragment) + '.txt'), 'w')
            all_features = None
            all_labels = None
            num_segments_trial = []
            local_labels = [[[13, 22], [204, 218], [219, 235]],
                 [[50, 65], [132, 150], [164, 187], [206, 226]],
                 [[14, 27], [66, 80], [135, 149], [150, 165], [186, 206]],
                 [[4, 24], [26, 45], [95, 121], [131, 136], [166, 183], [202, 212]],
                 [[15, 35], [35, 50], [135, 150]],
                 [[10, 19], [40, 49], [63, 74], [91, 103], [120, 129], [165, 181]],
                 [[23, 40], [61, 75], [152, 165], [180, 195], [200, 212]],
                 [],
                 [[55, 70], [128, 143], [165, 180], [215, 235]],
                 [[14, 34], [58, 83], [98, 108], [141, 151]],
                 [],
                 [[45, 63], [76, 91], [148, 159], [188, 204], [209, 219], [229, 240]],
                 [[21, 31], [92, 103], [119, 129], [224, 240]],
                 [[49, 60], [138, 150], [162, 174], [195, 210]],
                 [[63, 80], [97, 113], [120, 134], [165, 180], [184, 205]]]
            for key_idx, key in enumerate(test_keys):
                seq = datasets[key]['features'][...]
                gt = datasets[key]['labels'][...]
                gt = torch.from_numpy(gt)
                label_idx = key_idx
                local_label = local_labels[label_idx]

                seq = torch.from_numpy(seq).to(DEVICE)
                sub_graphs = None
                for n in range(math.ceil(seq.shape[0] / (args.fragment_length * 2))):
                    if n != math.ceil(seq.shape[0] / (args.fragment_length * 2)) - 1:
                        data = seq[(args.fragment_length * 2) * n:(args.fragment_length * 2) * (n + 1), :]
                        sub_graph, _ = model1(data)
                    else:
                        data = seq[(args.fragment_length * 2) * n:, :]
                        if data.shape[0] == 1:
                            sub_graph = data
                        else:
                            sub_graph, _ = model1(data)

                    if sub_graphs is not None:
                        sub_graphs = torch.cat((sub_graphs, sub_graph), dim=0)
                    else:
                        sub_graphs = sub_graph
                seq_graph = torch.add(seq, sub_graphs)
                seq2seq = seq_graph.unsqueeze(dim=0)
                sig_probs = model2(seq2seq)

                probs_importance = sig_probs.data.cpu().squeeze().numpy()
                seq = seq.squeeze()
                limits = args.num_fragment
                order = np.argsort(probs_importance)[::-1]

                all_fragment = []
                n_t = 0
                if label_idx != 7 and label_idx != 10:
                    for j in range(len(local_label)):
                        for i in range(limits):
                            gt_left_idx = local_label[j][0]
                            gt_right_idx = local_label[j][1]
                            idx = order[i] + args.fragment_length
                            left_idx = idx - probs_importance[idx - args.fragment_length] * args.fragment_length
                            left_int_idx = int(np.ceil(left_idx))
                            right_idx = idx + probs_importance[idx - args.fragment_length] * args.fragment_length
                            right_int_idx = int(np.floor(right_idx))
                            if left_int_idx - args.fragment_length >= gt_right_idx or right_int_idx - args.fragment_length <= gt_left_idx:
                                tIOU = 0.
                            else:
                                idx_set = np.hstack((gt_left_idx, gt_right_idx))
                                idx_set = np.hstack((idx_set, left_int_idx - args.fragment_length))
                                idx_set = np.hstack((idx_set, right_int_idx - args.fragment_length))
                                idx_set = np.sort(idx_set)
                                tIOU = (idx_set[2] - idx_set[1]) / (idx_set[3] - idx_set[0])
                            if tIOU >= 0.5:
                                n_t += 1
                                break
                    local_recall = n_t / len(local_label)
                    log_str1 = 'i_th trial %.0f\tRecall %.02f' % (label_idx, local_recall)
                    save_idx.write(log_str1 + '\n')
                    save_idx.flush()

                for i in range(limits):
                    idx = order[i] + args.fragment_length
                    left_idx = idx - probs_importance[idx - args.fragment_length] * args.fragment_length
                    left_int_idx = int(np.ceil(left_idx)) - args.fragment_length
                    right_idx = idx + probs_importance[idx - args.fragment_length] * args.fragment_length
                    right_int_idx = int(np.floor(right_idx)) - args.fragment_length
                    one_fragment = seq[left_int_idx:right_int_idx, ]
                    all_fragment.append(one_fragment)
                    log_str0 = 'i_th fragment %.0f\tleft_idx %.0f\tright_idx %.0f' % (i, left_int_idx, right_int_idx)
                    save_idx.write(log_str0 + '\n')
                    save_idx.flush()

                all_fragment = torch.vstack(all_fragment)
                labels = torch.full((all_fragment.shape[0], 1), gt)
                num_segments_trial.append(all_fragment.shape[0])

                if all_features is not None:
                    all_features = torch.cat((all_features, all_fragment), dim=0)
                    all_labels = torch.cat((all_labels, labels), dim=0)
                else:
                    all_features = all_fragment
                    all_labels = labels
            all_features = all_features.cpu().data.numpy()
            all_labels = all_labels.cpu().data.numpy()
            mat_file = os.path.join(out_path, 'TAS_subject' + str(args.subject_id)  + '_' + str(args.reward_function) + '_' + str(args.num_fragment) + '.mat')
            savemat(mat_file, {'feature': all_features, 'label': all_labels})
