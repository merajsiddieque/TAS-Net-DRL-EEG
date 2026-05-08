import torch

class TDActorCriticUpdater:
    def __init__(self, gamma=0.99):
        self.gamma = gamma

    def calculate_td_loss(self, reward, v_curr, v_next, log_prob):
        """
        Actor-Critic loss with TD Error bootstrapping.
        R_t = r_t + gamma * V(s_{t+1})
        td_error = R_t - V(s_t)
        Returns bounded actor_loss, critic_loss, and detached td_error mapping.
        """
        # Calculate TD Target
        if v_next is None:
            # Terminal state logically bootstraps to just the reward
            td_target = reward.clone()
        else:
            td_target = reward + self.gamma * v_next.detach()
            
        # Calculate Step Delta
        td_error = td_target - v_curr
        
        # Policy / Actor Loss (maximize expected reward -> minimize negative expected reward)
        # Detach error conceptually prevents rewarding environment state mutation
        actor_loss = -log_prob * td_error.detach()
        
        # Critic Loss (MSE) predicting absolute state value
        critic_loss = td_error ** 2
        
        return actor_loss, critic_loss, td_error.detach()
