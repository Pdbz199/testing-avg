import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        
        self.actor_body = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.actor_mu = nn.Linear(hidden_dim, action_dim)
        self.actor_log_sigma = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, rsample=False):
        features = self.actor_body(x)
        mu = self.actor_mu(features)
        log_sigma = self.actor_log_sigma(features)
        policy_distribution = torch.distributions.MultivariateNormal(mu, 0.01*torch.diag(torch.exp(log_sigma[0])))
        if rsample:
            action = policy_distribution.rsample()
        else:
            action = policy_distribution.sample()
        return action, policy_distribution.log_prob(action)
