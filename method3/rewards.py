import torch
import random
from utils import normalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def compute_reward(seq, actions, ignore_far_sim=False, temp_dist_thre=20):
    """
    Compute diversity reward and representativeness reward

    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU
    """
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze()
    pick_idxs = torch.nonzero(pick_idxs, as_tuple=False)
    pick_idxs = pick_idxs.squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1

    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        reward = reward.to(device)
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # compute diversity reward
    if num_picks == 1:
        reward_sim = torch.tensor(0.)
        reward_sim = reward_sim.to(device)
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        sim_mat = torch. matmul(normed_seq, normed_seq.t())  # similarity matrix [Eq.4]
        sim_submat = sim_mat[pick_idxs, :][:, pick_idxs]
        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            sim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_sim = sim_submat.sum() / (num_picks * (num_picks - 1.))  # diversity reward [Eq.3]

    # compute representativeness reward
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat = torch.addmm(input=dist_mat, beta=1, mat1=_seq, mat2=_seq.t(), alpha=-2)
    dist_mat = dist_mat[:, pick_idxs]
    if len(dist_mat.shape) == 1:
        dist_mat = dist_mat.unsqueeze(dim=1)
    dist_mat = torch.min(dist_mat, 1, keepdim=True)
    dist_mat = dist_mat[0]
    # reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]
    reward_rep = torch.exp(-dist_mat.mean())

    reward = (reward_sim + reward_rep) * 0.5
    return reward

def compute_reward_coff(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False,
                        div_coff=0.5, rep_coff=0.5):
    """
    Compute diversity reward and representativeness reward

    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU
    """
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1

    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # compute diversity reward
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        if use_gpu:
            reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1 - torch.matmul(normed_seq, normed_seq.t())  # dissimilarity matrix [Eq.4]
        dissim_submat = dissim_mat[pick_idxs, :][:, pick_idxs]
        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))  # diversity reward [Eq.3]

    # compute representativeness reward
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, _seq, _seq.t())
    dist_mat = dist_mat[:, pick_idxs]
    dist_mat = dist_mat.min(1, keepdim=True)[0]
    # reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]

    if _seq.size(1) == 2048:
        reward_rep = torch.exp(-dist_mat.mean()*0.1)
    else:
        reward_rep = torch.exp(-dist_mat.mean())

    # combine the two rewards
    reward = (reward_div * div_coff + reward_rep * rep_coff)
    # print("num_picks {} reward_div {}\t  reward_rep {}\t".format(num_picks, reward_div, reward_rep))
    return reward


def compute_reward_det_coff(seq, actions, det_scores, det_class,  episode,
                            ignore_far_sim=True, temp_dist_thre=20, use_gpu=False,
                            div_coff=50.0, rep_coff=50.0, det_coff =50.0):
    """
    Compute detection reward, diversity reward and representativeness reward

    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU

    """
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1

    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        if use_gpu:
            reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # compute diversity reward
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())  # dissimilarity matrix [Eq.4]
        dissim_submat = dissim_mat[pick_idxs, :][:, pick_idxs]
        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))  # diversity reward [Eq.3]

    # compute representativeness reward
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, _seq, _seq.t())
    dist_mat = dist_mat[:, pick_idxs]

    dist_mat = dist_mat.min(1, keepdim=True)[0]
    # reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]
    reward_rep = torch.exp(-dist_mat.mean())

    det_class = torch.from_numpy(det_class.astype(int))
    det_scores = torch.from_numpy(det_scores)
    sp_det_score = torch.zeros(n) # the score of standard plane detection

    for i in range(n):
        det_cls = det_class[i]
        if det_cls[0] != 3:
            sp_det_score[i] = det_scores[i][det_cls]
        else:
            sp_det_score[i] = -det_scores[i][3]

    pick_scores = sp_det_score[pick_idxs]
    reward_det = torch.sum(pick_scores)/num_picks # (norm_summ_scores)

    # combine the three rewards
    reward = reward_div * div_coff + reward_rep * rep_coff + reward_det  * det_coff
    # reward = (reward_div + reward_rep) * 0.25 + reward_det *0.5
    return reward


def compute_step_reward(action, prev_action, task_reward_scale=0.5, consistency_scale=0.5):
    """
    Per-step reward combining:
      1. Task reward   : reward for selecting (action=1) vs skipping (action=0).
                         Selecting is encouraged; sparse skipping is penalized to
                         prevent policy collapse (always-skip degeneracy).
      2. Temporal consistency reward : penalizes abrupt switches between consecutive
                         actions, encouraging smooth temporal decisions.

    Args:
        action      (Tensor): current binary action (scalar or 1-element tensor).
        prev_action (Tensor|None): previous step's action. None on the first step.
        task_reward_scale    (float): weight for the task reward component.
        consistency_scale    (float): weight for the temporal consistency component.

    Returns:
        Tensor: combined per-step scalar reward.
    """
    # --- Task Reward ---
    # Reward selecting informative frames/segments; lightly penalise always-skip.
    act_val = action.float().squeeze()
    task_reward = act_val * 1.0 + (1.0 - act_val) * (-0.2)

    # --- Temporal Consistency Reward ---
    if prev_action is None:
        # No previous action at the start of a sequence — no consistency signal.
        consistency_reward = torch.tensor(0.0, device=action.device)
    else:
        prev_val = prev_action.float().squeeze()
        # High consistency (same decision) → reward 1.0; switch → reward -1.0
        consistency_reward = 1.0 - 2.0 * torch.abs(act_val - prev_val)

    # --- Combined Per-step Reward ---
    reward = (task_reward_scale * task_reward + consistency_scale * consistency_reward)
    return reward
