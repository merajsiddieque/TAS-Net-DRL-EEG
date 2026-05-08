import torch


def topk_sampling(probs, k):
    probs_flat = probs.squeeze()

    if probs_flat.dim() == 0:
        probs_flat = probs_flat.unsqueeze(0)

    k = min(k, probs_flat.shape[0])
    top_idx = torch.topk(probs_flat, k).indices

    actions = torch.zeros_like(probs_flat)
    actions[top_idx] = 1.0

    return actions.view(probs.shape)