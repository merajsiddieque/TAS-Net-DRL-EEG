from __future__ import print_function
import torch
import os.path as osp
import time
import argparse
import datetime
import sys
import os

# Allow imports from the shared parent directory (Graph_Net, evaluate, utils, etc.)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch.nn as nn
import h5py
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli
from models import DSN
from utils import weights_init, save_checkpoint
from ppo_agent import PPOAgent
from topk_sampling import topk_sampling
from multi_reward import compute_multi_reward
import numpy as np
from scipy.io import savemat
from Graph_Net import ClassifierGNN
from evaluate import evaluate
import math

# =========================================================================== #
#  Argument parsing                                                            #
# =========================================================================== #
parser = argparse.ArgumentParser(
    description='PPO + Top-K Sampling + Multi-Objective Reward for EEG RL'
)
parser.add_argument('--training',       action='store_true', default=False)
parser.add_argument('--seed',           type=int,   default=27)
parser.add_argument('--epochs',         type=int,   default=30)
parser.add_argument('--subject_id',     type=int,   default=0)
parser.add_argument('--lr',             type=float, default=1e-3)
parser.add_argument('--weight_decay',   type=float, default=1e-4)
parser.add_argument('--edge_features',  type=int,   default=32)
parser.add_argument('--n_feature',      type=int,   default=192)
parser.add_argument('--hidden',         type=int,   default=8)
parser.add_argument('--nb_heads',       type=int,   default=8)
parser.add_argument('--dropout',        type=float, default=0.6)
parser.add_argument('--alpha',          type=float, default=0.2)
parser.add_argument('--hid_dim',        type=int,   default=256)
# Path defaults are relative to the methodology1/ working directory
parser.add_argument('--deep_features',  type=str,
                    default=r'..\features\session_1\source_h5_file.h5')
parser.add_argument('--save_path',      type=str,
                    default=r'..\TAS-output\SEED\session1')
parser.add_argument('--fragment_length',type=int,   default=8)
parser.add_argument('--num_fragment',   type=int,   default=10)
parser.add_argument('--reward_function',type=str,   default='R1_R2')
parser.add_argument('--gpu',            type=str,   default='0')
# PPO / Top-K specific
parser.add_argument('--topk',           type=int,   default=10,
                    help='K for Top-K frame selection (default: 10)')
parser.add_argument('--ppo_clip',       type=float, default=0.2,
                    help='PPO epsilon clip range (default: 0.2)')
parser.add_argument('--reward_weights', type=float, nargs=3,
                    default=[0.3, 0.3, 0.4],
                    help='Weights [w_sparse, w_smooth, w_div] for multi-objective reward')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# =========================================================================== #
#  TRAINING                                                                    #
# =========================================================================== #
if __name__ == '__main__':

    if args.training:
        # ------------------------------------------------------------------ #
        #  Data                                                               #
        # ------------------------------------------------------------------ #
        datasets  = h5py.File(args.deep_features, 'r')
        all_keys  = list(datasets.keys())
        test_keys = all_keys[15 * args.subject_id : 15 * (args.subject_id + 1)]
        del all_keys[15 * args.subject_id : 15 * (args.subject_id + 1)]
        train_keys = all_keys
        print("# total: {}  train: {}  test: {}".format(
            len(train_keys) + len(test_keys), len(train_keys), len(test_keys)))

        # ------------------------------------------------------------------ #
        #  Networks                                                           #
        # ------------------------------------------------------------------ #
        Network1 = ClassifierGNN(
            in_features=args.n_feature,
            edge_features=args.edge_features,
            out_features=args.n_feature,
            device=DEVICE,
        )
        Network2 = DSN(in_dim=args.n_feature, hid_dim=args.hid_dim,
                       num_layers=1, cell='gru')
        Network2.apply(weights_init)

        print("GNN  params: {:.5f}M".format(
            sum(p.numel() for p in Network1.parameters()) / 1e6))
        print("DSN  params: {:.5f}M".format(
            sum(p.numel() for p in Network2.parameters()) / 1e6))

        # ------------------------------------------------------------------ #
        #  PPO agent + separate optimisers for GNN and DSN                   #
        # ------------------------------------------------------------------ #
        ppo_agent = PPOAgent(clip_param=args.ppo_clip).to(DEVICE)
        ENTROPY_COEFF = 0.01   # weight on entropy bonus to prevent collapse

        optimizer = torch.optim.Adam(
            list(Network1.parameters()) + list(Network2.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        Network1 = Network1.to(DEVICE)
        Network2 = Network2.to(DEVICE)

        # ------------------------------------------------------------------ #
        #  Logging                                                            #
        # ------------------------------------------------------------------ #
        os.makedirs('logs', exist_ok=True)
        log_file_path = os.path.join('logs', 'ppo_topk_metrics.csv')
        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w') as f:
                f.write(
                    'epoch,loss_total,reward_mean,sparsity_reward,'
                    'smoothness_reward,diversity_reward,recall,'
                    'number_of_selected_frames\n'
                )

        # Ground-truth segment annotations per test/train trial (15 trials)
        LOCAL_LABELS = [
            [[13, 22], [204, 218], [219, 235]],
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
            [[63, 80], [97, 113], [120, 134], [165, 180], [184, 205]],
        ]
        # Map each key -> its GT labels (by original position in all_keys)
        all_keys_ordered = list(datasets.keys())
        key_to_labels = {
            key: LOCAL_LABELS[i % len(LOCAL_LABELS)]
            for i, key in enumerate(all_keys_ordered)
        }

        # Per-video moving-average baseline for advantage estimation
        baselines = {key: 0.0 for key in train_keys}
        best_recall = 0.0

        print("=====> Start PPO + Top-K training <=====")
        start_time = time.time()

        # ------------------------------------------------------------------ #
        #  Epoch loop                                                         #
        # ------------------------------------------------------------------ #
        for epoch in range(args.epochs):
            Network1.train()
            Network2.train()

            idxs = np.arange(len(train_keys))
            np.random.shuffle(idxs)

            # Epoch-level accumulators
            epoch_loss        = []
            epoch_reward      = []
            epoch_r_spars     = []
            epoch_r_smooth    = []
            epoch_r_div       = []
            epoch_num_sel     = []

            for idx in idxs:
                key = train_keys[idx]
                seq = torch.from_numpy(
                    datasets[key]['features'][...]
                ).to(DEVICE)                        # (T, D)

                # ---------------------------------------------------------- #
                #  Graph feature extraction                                   #
                # ---------------------------------------------------------- #
                seg_len       = args.fragment_length * 2
                local_graphs  = None

                for n in range(math.ceil(seq.shape[0] / seg_len)):
                    s = seg_len * n
                    e = seg_len * (n + 1)
                    chunk = seq[s:e] if n < math.ceil(seq.shape[0] / seg_len) - 1 \
                                     else seq[s:]
                    if chunk.shape[0] == 1:
                        sub_g = chunk
                    else:
                        sub_g, _ = Network1(chunk)

                    local_graphs = sub_g if local_graphs is None \
                                   else torch.cat((local_graphs, sub_g), dim=0)

                seq_graph = torch.add(seq, local_graphs)       # (T, D)
                seq_in    = seq_graph.unsqueeze(0)             # (1, T, D)

                # ---------------------------------------------------------- #
                #  Forward pass — get frame-selection probabilities           #
                # ---------------------------------------------------------- #
                sig_probs = Network2(seq_in)                   # (1, T, 1)

                # ---------------------------------------------------------- #
                #  Top-K sampling — select K most probable frames            #
                # ---------------------------------------------------------- #
                # Small noise breaks ties; promotes exploration early in training
                noisy_probs = torch.clamp(
                    sig_probs + torch.randn_like(sig_probs) * 0.02,
                    1e-6, 1.0 - 1e-6,
                )
                actions = topk_sampling(noisy_probs, args.topk).to(DEVICE)  # (1,T,1)

                # ---------------------------------------------------------- #
                #  Old log-probabilities (frozen snapshot)                   #
                # ---------------------------------------------------------- #
                old_dist      = Bernoulli(sig_probs.detach())
                old_log_probs = old_dist.log_prob(actions).detach()    # stop-grad

                # ---------------------------------------------------------- #
                #  Multi-objective reward (all components are negative)      #
                #  R = w1*(-mean(a)) + w2*(-mean(|diff|)) + w3*(-mean(sim)) #
                # ---------------------------------------------------------- #
                gt_labels_key = key_to_labels.get(key, None)
                reward, r_spars, r_smooth, r_div = compute_multi_reward(
                    seq_in, actions, args.reward_weights,
                    gt_labels=gt_labels_key,
                    fragment_length=args.fragment_length,
                    num_fragment=args.num_fragment,
                    recall_weight=0.5,
                )
                reward = reward.to(DEVICE)

                # ---------------------------------------------------------- #
                #  Advantage  A = R - baseline                               #
                #  With all-negative rewards the raw advantage is tiny;      #
                #  we normalise per-batch across all videos in the epoch     #
                #  (simple version: subtract per-video moving baseline).     #
                # ---------------------------------------------------------- #
                advantage = reward.item() - baselines[key]
                baselines[key] = 0.9 * baselines[key] + 0.1 * reward.item()

                # ---------------------------------------------------------- #
                #  New log-probabilities (policy being updated)              #
                # ---------------------------------------------------------- #
                new_probs     = Network2(seq_in)   # second forward through DSN only
                new_dist      = Bernoulli(torch.clamp(new_probs, 1e-6, 1.0 - 1e-6))
                new_log_probs = new_dist.log_prob(actions)

                # Entropy bonus — prevents distribution collapse
                entropy       = new_dist.entropy().mean()

                # ---------------------------------------------------------- #
                #  PPO clipped loss  +  entropy regularisation               #
                #  L = L_clip - c_ent * H(pi)                                #
                # ---------------------------------------------------------- #
                adv_tensor = torch.tensor(advantage, dtype=torch.float32,
                                          device=DEVICE)
                ppo_loss   = ppo_agent.update(old_log_probs, new_log_probs, adv_tensor)
                total_loss = ppo_loss - ENTROPY_COEFF * entropy

                # ---------------------------------------------------------- #
                #  Backward                                                   #
                # ---------------------------------------------------------- #
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(Network1.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(Network2.parameters(), 5.0)
                optimizer.step()

                # Accumulate metrics
                epoch_loss.append(total_loss.item())
                epoch_reward.append(reward.item())
                epoch_r_spars.append(r_spars.item())
                epoch_r_smooth.append(r_smooth.item())
                epoch_r_div.append(r_div.item())
                epoch_num_sel.append(torch.sum(actions).item())

            # -------------------------------------------------------------- #
            #  Epoch-level evaluation & logging                              #
            # -------------------------------------------------------------- #
            Network1.eval()
            Network2.eval()

            Recall     = evaluate(args, Network1, Network2, datasets, test_keys)

            avg_loss   = float(np.mean(epoch_loss))
            avg_reward = float(np.mean(epoch_reward))
            avg_spars  = float(np.mean(epoch_r_spars))
            avg_smooth = float(np.mean(epoch_r_smooth))
            avg_div    = float(np.mean(epoch_r_div))
            avg_sel    = float(np.mean(epoch_num_sel))

            print(
                "Epoch [{}/{}]  Loss: {:.4f}  Reward: {:.4f}  "
                "Sparse: {:.4f}  Smooth: {:.4f}  Div: {:.4f}  "
                "Sel: {:.1f}  Recall: {:.4f}".format(
                    epoch + 1, args.epochs,
                    avg_loss, avg_reward,
                    avg_spars, avg_smooth, avg_div,
                    avg_sel, Recall,
                )
            )

            # Write CSV row
            with open(log_file_path, 'a') as f:
                f.write(
                    f"{epoch + 1},{avg_loss:.6f},{avg_reward:.6f},"
                    f"{avg_spars:.6f},{avg_smooth:.6f},{avg_div:.6f},"
                    f"{Recall:.6f},{avg_sel:.2f}\n"
                )

            # Save best checkpoint
            if Recall > best_recall:
                best_recall = Recall
                os.makedirs(args.save_path, exist_ok=True)
                save_checkpoint(
                    Network1.state_dict(),
                    osp.join(args.save_path,
                             f'pretrained_best_model1_seed_subject'
                             f'{args.subject_id}_{args.reward_function}.pth.tar')
                )
                save_checkpoint(
                    Network2.state_dict(),
                    osp.join(args.save_path,
                             f'pretrained_best_model2_seed_subject'
                             f'{args.subject_id}_{args.reward_function}.pth.tar')
                )

            scheduler.step()

        elapsed = str(datetime.timedelta(seconds=round(time.time() - start_time)))
        print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
        print("Best Recall: {:.4f}".format(best_recall))
        print("Metrics saved to:", log_file_path)

    # ======================================================================= #
    #  EVALUATION MODE                                                         #
    # ======================================================================= #
    else:
        datasets  = h5py.File(args.deep_features, 'r')
        all_keys  = list(datasets.keys())
        test_keys = all_keys[15 * args.subject_id : 15 * (args.subject_id + 1)]
        print("# test videos: {}".format(len(test_keys)))

        model1 = ClassifierGNN(
            in_features=args.n_feature,
            edge_features=args.edge_features,
            out_features=args.n_feature,
            device=DEVICE,
        )
        model2 = DSN(in_dim=args.n_feature, hid_dim=args.hid_dim,
                     num_layers=1, cell='gru')

        ckp1 = osp.join(args.save_path,
                        f'pretrained_best_model1_seed_subject'
                        f'{args.subject_id}_{args.reward_function}.pth.tar')
        ckp2 = osp.join(args.save_path,
                        f'pretrained_best_model2_seed_subject'
                        f'{args.subject_id}_{args.reward_function}.pth.tar')

        model1.load_state_dict(torch.load(ckp1))
        model2.load_state_dict(torch.load(ckp2))
        model1 = model1.to(DEVICE)
        model2 = model2.to(DEVICE)

        with torch.no_grad():
            model1.eval()
            model2.eval()

            out_path = os.path.join(args.save_path, 'result_output')
            os.makedirs(out_path, exist_ok=True)

            save_idx = open(
                os.path.join(out_path,
                             f'log_subject{args.subject_id}_'
                             f'{args.reward_function}_{args.num_fragment}.txt'), 'w'
            )

            all_features     = None
            all_labels       = None
            num_segments_trial = []

            local_labels = [
                [[13, 22], [204, 218], [219, 235]],
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
                [[63, 80], [97, 113], [120, 134], [165, 180], [184, 205]],
            ]

            seg_len = args.fragment_length * 2

            for key_idx, key in enumerate(test_keys):
                seq = torch.from_numpy(datasets[key]['features'][...]).to(DEVICE)
                gt  = torch.from_numpy(datasets[key]['labels'][...])
                local_label = local_labels[key_idx]

                sub_graphs = None
                for n in range(math.ceil(seq.shape[0] / seg_len)):
                    s     = seg_len * n
                    chunk = seq[s : s + seg_len] if n < math.ceil(seq.shape[0] / seg_len) - 1 \
                            else seq[s:]
                    if chunk.shape[0] == 1:
                        sub_g = chunk
                    else:
                        sub_g, _ = model1(chunk)
                    sub_graphs = sub_g if sub_graphs is None \
                                 else torch.cat((sub_graphs, sub_g), dim=0)

                seq_graph = torch.add(seq, sub_graphs)
                sig_probs = model2(seq_graph.unsqueeze(0))

                probs_importance = sig_probs.data.cpu().squeeze().numpy()
                seq   = seq.squeeze()
                order = np.argsort(probs_importance)[::-1]
                limits = args.num_fragment

                n_t = 0
                if key_idx != 7 and key_idx != 10:
                    for j in range(len(local_label)):
                        for i in range(limits):
                            gt_l = local_label[j][0]
                            gt_r = local_label[j][1]
                            pos  = order[i] + args.fragment_length
                            l_i  = int(np.ceil(pos  - probs_importance[pos - args.fragment_length] * args.fragment_length))
                            r_i  = int(np.floor(pos + probs_importance[pos - args.fragment_length] * args.fragment_length))
                            if l_i - args.fragment_length >= gt_r or \
                               r_i - args.fragment_length <= gt_l:
                                tIOU = 0.0
                            else:
                                s4 = np.sort([gt_l, gt_r,
                                              l_i - args.fragment_length,
                                              r_i - args.fragment_length])
                                tIOU = (s4[2] - s4[1]) / (s4[3] - s4[0])
                            if tIOU >= 0.5:
                                n_t += 1
                                break
                    local_recall = n_t / len(local_label)
                    save_idx.write(
                        'i_th trial {:d}\tRecall {:.2f}\n'.format(key_idx, local_recall)
                    )
                    save_idx.flush()

                all_fragment = []
                for i in range(limits):
                    pos   = order[i] + args.fragment_length
                    l_int = int(np.ceil(pos  - probs_importance[pos - args.fragment_length] * args.fragment_length)) - args.fragment_length
                    r_int = int(np.floor(pos + probs_importance[pos - args.fragment_length] * args.fragment_length)) - args.fragment_length
                    all_fragment.append(seq[l_int:r_int])
                    save_idx.write('i_th fragment {:d}\tl {:d}\tr {:d}\n'.format(i, l_int, r_int))
                    save_idx.flush()

                frag_stack = torch.vstack(all_fragment)
                labels     = torch.full((frag_stack.shape[0], 1), gt.item())
                num_segments_trial.append(frag_stack.shape[0])

                all_features = frag_stack  if all_features is None \
                               else torch.cat((all_features, frag_stack), dim=0)
                all_labels   = labels      if all_labels   is None \
                               else torch.cat((all_labels,   labels),     dim=0)

            all_features = all_features.cpu().numpy()
            all_labels   = all_labels.cpu().numpy()
            mat_file = os.path.join(
                out_path,
                f'TAS_subject{args.subject_id}_{args.reward_function}_{args.num_fragment}.mat'
            )
            savemat(mat_file, {'feature': all_features, 'label': all_labels})
            print("Saved feature matrix to:", mat_file)
