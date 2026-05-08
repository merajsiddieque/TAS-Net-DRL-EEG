"""
multi_reward.py — Multi-Objective Reward for PPO + Top-K EEG selection.

    R_sparse = -mean(actions)
    R_smooth = -mean(|a_t - a_{t+1}|)
    R_div    = -mean(cosine_similarity between selected frames)
    R_recall =  mean tIOU overlap with ground-truth segments  [NEW]

    R = (1 - w_rec) * (w1*R_sparse + w2*R_smooth + w3*R_div)
        + w_rec * R_recall
"""

import torch
import torch.nn.functional as F
import numpy as np


# --------------------------------------------------------------------------- #
#  Individual reward components                                                #
# --------------------------------------------------------------------------- #

def sparsity_reward(actions: torch.Tensor) -> torch.Tensor:
    """R_sparse = -mean(actions).  In (-1, 0]."""
    return -torch.mean(actions.float())


def smoothness_reward(actions: torch.Tensor) -> torch.Tensor:
    """R_smooth = -mean(|a_t - a_{t+1}|).  In (-1, 0]."""
    a = actions.squeeze()
    if a.dim() == 0 or a.shape[0] <= 1:
        return torch.tensor(0.0, device=actions.device)
    diffs = torch.abs(a[1:].float() - a[:-1].float())
    return -torch.mean(diffs)


def diversity_reward(seq: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """R_div = -mean pairwise cosine-sim of selected frames.  In (-1, 0]."""
    seq_2d = seq.squeeze(0) if seq.dim() == 3 else seq   # (T, D)
    act_1d = actions.squeeze()                            # (T,)
    idx = torch.nonzero(act_1d, as_tuple=False).squeeze(1)
    if idx.numel() <= 1:
        return torch.tensor(0.0, device=seq.device)
    selected = seq_2d[idx]
    normed   = F.normalize(selected, p=2, dim=1)
    sim_mat  = torch.matmul(normed, normed.t())
    mask = ~torch.eye(sim_mat.shape[0], dtype=torch.bool, device=seq.device)
    return -torch.mean(sim_mat[mask])


def recall_reward(
    actions: torch.Tensor,
    gt_labels: list,
    fragment_length: int,
    num_fragment: int,
) -> torch.Tensor:
    """
    R_recall = fraction of GT segments hit by at least one top-K frame
               with tIOU >= 0.5.

    This is the KEY missing reward — it gives the policy direct incentive
    to move selected frames onto annotated ground-truth segments so that
    recall actually improves during training.

    Args:
        actions        : Top-K binary vector (1,T,1) or (T,).
        gt_labels      : list of [left, right] GT segment indices for this trial.
        fragment_length: half-window size (args.fragment_length).
        num_fragment   : how many top frames to evaluate (args.num_fragment).

    Returns:
        Scalar tensor in [0, 1].  Higher is better.
    """
    if not gt_labels:
        return torch.tensor(0.0, device=actions.device)

    # Work in numpy (no gradient needed — reward is a scalar constant for PPO)
    act_1d = actions.squeeze().float().detach().cpu().numpy()   # (T,)
    T      = act_1d.shape[0]
    order  = np.argsort(act_1d)[::-1]
    limits = min(num_fragment, max(0, T - fragment_length))

    n_t = 0
    for gt_l, gt_r in gt_labels:
        for i in range(limits):
            pos = int(order[i]) + fragment_length
            if pos >= T:
                continue
            prob_at_pos = float(act_1d[pos - fragment_length])
            left_int  = int(np.ceil( pos - prob_at_pos * fragment_length))
            right_int = int(np.floor(pos + prob_at_pos * fragment_length))
            adj_l = left_int  - fragment_length
            adj_r = right_int - fragment_length
            if adj_l >= gt_r or adj_r <= gt_l:
                tIOU = 0.0
            else:
                s4    = np.sort([gt_l, gt_r, adj_l, adj_r])
                denom = s4[3] - s4[0]
                tIOU  = (s4[2] - s4[1]) / denom if denom > 0 else 0.0
            if tIOU >= 0.5:
                n_t += 1
                break

    score = float(n_t) / len(gt_labels)
    return torch.tensor(score, dtype=torch.float32, device=actions.device)


# --------------------------------------------------------------------------- #
#  Combined multi-objective reward                                             #
# --------------------------------------------------------------------------- #

def compute_multi_reward(
    seq: torch.Tensor,
    actions: torch.Tensor,
    weights=(0.3, 0.3, 0.4),
    gt_labels=None,
    fragment_length: int = 8,
    num_fragment: int = 10,
    recall_weight: float = 0.5,
) -> tuple:
    """
    R = (1 - recall_weight) * (w1*R_sparse + w2*R_smooth + w3*R_div)
        + recall_weight * R_recall

    When gt_labels is None the recall term is dropped (structural only).

    Returns:
        (total_reward, r_sparse, r_smooth, r_div) — all scalar tensors.
    """
    w1, w2, w3 = weights

    r_sparse = sparsity_reward(actions)
    r_smooth = smoothness_reward(actions)
    r_div    = diversity_reward(seq, actions)

    structural = w1 * r_sparse + w2 * r_smooth + w3 * r_div

    if gt_labels is not None:
        r_rec  = recall_reward(actions, gt_labels, fragment_length, num_fragment)
        total  = (1.0 - recall_weight) * structural + recall_weight * r_rec
    else:
        total  = structural

    return total, r_sparse, r_smooth, r_div
