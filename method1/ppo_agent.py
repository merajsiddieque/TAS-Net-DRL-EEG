import torch
import torch.nn as nn


class PPOAgent(nn.Module):
    """
    PPO clipped-surrogate objective.

    L = E[ min( r_t * A_t,  clip(r_t, 1-eps, 1+eps) * A_t ) ]
    where  r_t = exp(log_pi_new - log_pi_old)  and  eps = clip_param.
    """

    def __init__(self, clip_param: float = 0.2):
        super().__init__()
        self.clip_param = clip_param

    def compute_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantage: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            old_log_probs : log pi_old(a|s)  — detached from current graph
            new_log_probs : log pi_new(a|s)  — attached to current graph
            advantage     : A_t             — scalar or broadcastable tensor

        Returns:
            Scalar PPO loss (positive; gradient descent minimises it).
        """
        # Importance-sampling ratio  r_t = pi_new / pi_old
        ratio = torch.exp(new_log_probs - old_log_probs.detach())

        # Clipped surrogate
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

        # Pessimistic (min) objective — negate because we minimise loss
        loss = -torch.min(surr1, surr2).mean()
        return loss

    def update(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantage: torch.Tensor,
    ) -> torch.Tensor:
        """Alias called by the training loop — returns the clipped PPO loss."""
        return self.compute_loss(old_log_probs, new_log_probs, advantage)