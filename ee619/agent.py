"""Agent for a Walker2DBullet environment."""
from os.path import abspath, dirname, realpath, join

from gym.spaces.box import Box
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from ee619.model import GaussianPolicy, QNetwork


ROOT = dirname(abspath(realpath(__file__)))  # path to the ee619 directory

def soft_target_update(target, source, tau):
    """Partial update(for each gradient step) of model (inplace operation)"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_target_update(target, source):
    """Copy source model to target model (inplace operation)"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Agent:
    """Agent for a Walker2DBullet environment."""
    """Using Soft Actor Critic Agent"""
    def __init__(self, hidden_size=256, seed=0, lr=0.0003, gamma=0.99, tau=0.005, alpha=0.2):
        """All hyperparameter sizes are on Appendix D of SAC paper"""
        self.seed = seed
        torch.cuda.manual_seed(self.seed)
        self.action_space = Box(-1, 1, (6,))
        self.action_space.seed(self.seed)
        self.num_inputs = 22 # dimension of observation_space state

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic_q1 = QNetwork(self.seed, self.num_inputs, self.action_space.shape[0], hidden_size).to(self.device)
        self.critic_q1_optim = Adam(self.critic_q1.parameters(), lr=lr)
        self.critic_q2 = QNetwork(self.seed, self.num_inputs, self.action_space.shape[0], hidden_size).to(self.device)
        self.critic_q2_optim = Adam(self.critic_q2.parameters(), lr=lr)


        self.critic_q1_target = QNetwork(self.seed, self.num_inputs, self.action_space.shape[0], hidden_size).to(self.device)
        self.critic_q2_target = QNetwork(self.seed, self.num_inputs, self.action_space.shape[0], hidden_size).to(self.device)
        hard_target_update(self.critic_q1_target, self.critic_q1)
        hard_target_update(self.critic_q2_target, self.critic_q2)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        self.target_entropy = -torch.prod(torch.Tensor(self.action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=lr)
        self.policy = GaussianPolicy(self.seed, self.num_inputs, self.action_space.shape[0], \
                                         self.num_inputs, self.action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

    def act(self, observation: np.ndarray):
        """Decides which action to take for the given observation."""
        observation = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
        _, _, action = self.policy.sample(observation)
        return action.detach().cpu().numpy()[0]

    def select_action(self, observation: np.ndarray):
        """Return action selected by policy during training"""
        observation = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
        action, _, _ = self.policy.sample(observation)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, buffer, minibatch_size):
        # Sample a batch from buffer
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = buffer.sample(minibatch_size=minibatch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target = self.critic_q1_target(next_state_batch, next_state_action)
            qf2_next_target = self.critic_q2_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        # Two Q-functions to mitigate positive bias in the policy improvement step
        # Q1 update
        qf1 = self.critic_q1(state_batch, action_batch) 
        qf1_loss = F.mse_loss(qf1, next_q_value) 

        self.critic_q1_optim.zero_grad()
        qf1_loss.backward()
        self.critic_q1_optim.step()

        # Q2 update
        qf2 = self.critic_q2(state_batch, action_batch) 
        qf2_loss = F.mse_loss(qf2, next_q_value) 

        self.critic_q2_optim.zero_grad()
        qf2_loss.backward()
        self.critic_q2_optim.step()
        
        # Policy update
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi = self.critic_q1(state_batch, pi)
        qf2_pi = self.critic_q2(state_batch, pi)

        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = torch.mean((self.alpha * log_pi) - min_qf_pi) 

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # alpha update
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()
        
        soft_target_update(self.critic_q1_target, self.critic_q1, self.tau)
        soft_target_update(self.critic_q2_target, self.critic_q2, self.tau)


        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item()   


    def load(self):
        """Loads network parameters if there are any.

        Example:
            path = join(ROOT, 'model.pth')
            self.policy.load_state_dict(torch.load(path))
        """
        policy = join(ROOT, 'saved_model', 'weights_policy_final.pth')
        critic_q1 = join(ROOT, 'saved_model', 'weights_critic_q1_final.pth')
        critic_q2 = join(ROOT, 'saved_model', 'weights_critic_q2_final.pth')

        self.policy.load_state_dict(torch.load(policy, map_location=self.device))
        self.critic_q1.load_state_dict(torch.load(critic_q1, map_location=self.device))
        self.critic_q2.load_state_dict(torch.load(critic_q2, map_location=self.device))
