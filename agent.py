import torch
import torch.nn as nn
from torch.distributions import Categorical

import random
import numpy as np
from typing import Union, Optional

from env import TSP
from net import FC
from memory import PERBuffer, ReplayBuffer, RolloutBuffer, Experience
from utils import compute_gae, normalize, revert_norm_returns


class AgentBase(object):
    def __init__(self, *args, **kwargs):
        self._config = {}
        self._rng = random.Random()
        if "seed" in kwargs:
            self.seed(kwargs.get("seed"))

    def reset(self) -> None:
        """Resets data not associated with learning."""
        pass

    def seed(self, seed) -> None:
        """Sets a seed for all random number generators (RNG).

        Note that on top of local RNGs a global for the PyTorch is set.
        If this is undesiredable effect then:
        1) please additionally set `torch.manual_seed()` manually,
        2) let us know of your circumenstances.

        Parameters:
            seed: (int) Seed value for random number generators.

        """
        if not isinstance(seed, (int, float)):
            return

        self._rng.seed(seed)
        torch.manual_seed(seed)

        if hasattr(self, "buffer"):
            self.buffer.seed(seed)
    
    def update_coef(self):
        raise NotImplementedError
    
    def act(self):
        raise NotImplementedError
    
    def step(self):
        raise NotImplementedError
    
    def learn(self):
        raise NotImplementedError
    
    def _reset_mask(self):
        raise NotImplementedError



class DQAgent(AgentBase):
    def __init__(
        self, 
        env: TSP,
        net: Union[FC],
        sample_type: str = 'Uniform',
        double: bool = False,
        revert_discount: bool = False,
        device = 'cpu',
        epsilon = 1.0, epsilon_min = 0.01, 
        epsilon_decay = 0.999, 
        gamma = 0.99, lr = 1e-4,
        tau = 5e-3, batch_size = 128,
        max_grad_norm = 10,
        update_freq = 1,
        num_update = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        """
        Other objects
        """
        self.device = device
        self.env = env
        self.policy_net = net[0].to(self.device)
        self.target_net = net[1].to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr = lr, amsgrad = True)

        self.sample_type = sample_type
        if self.sample_type == 'PER':
            self.buffer = PERBuffer(batch_size = batch_size)
        elif self.sample_type == 'Uniform':
            self.buffer = ReplayBuffer(batch_size = batch_size)
        assert sample_type in ['PER', 'Uniform'], f"sample_type must be either `PER` or `Uniform`."

        """
        Acting mechanics
        """
        self.double = double
        self.revert_discount = revert_discount

        """
        For epsilon-greedy updates
        """
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        """
        Other parameters
        """
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.update_freq = update_freq
        self.num_update = num_update
        self.global_actions_taken = 0

        """
        Human intervention
        """
        self.base_mask = np.ones(self.env.data.shape[0], dtype = np.int8)

        """
        Log
        """
        self.exploitation = 0
        self.exploration = 0
        self._loss: float = float("nan")
    

    @property
    def loss(self) -> dict[str, float]:
        return {"loss": self._loss}
    @loss.setter
    def loss(self, value):
        if isinstance(value, dict):
            value = value["loss"]
        self._loss = value


    def update_coef(self, current_time_step: Optional[int] = None):
        if self.epsilon > self.epsilon_min:
            if current_time_step == None:
                self.epsilon = self.epsilon * self.epsilon_decay
            elif current_time_step != None:
                epsilon_bench = self.epsilon * self.epsilon_decay + 1 / current_time_step
                if epsilon_bench >= 1:
                    self.epsilon = self.epsilon * self.epsilon_decay
                elif epsilon_bench < 1:
                    self.epsilon = epsilon_bench

    @torch.no_grad()
    def act(self, exp: Experience) -> Experience:
        s = torch.tensor([exp.state], device = self.device)
        if len(self.env.stops) < self.env.n_stops:

            mask = self.base_mask
            mask[self.env.stops] = 0

            if random.random() > self.epsilon:
                a_ = self.policy_net(s)
                a_mask = np.expand_dims(mask, axis = 0)
                a_mask = torch.tensor(a_mask, dtype = torch.bool, device = self.device)
                a_[~a_mask] = -np.inf
                a = a_.max(1).indices.view(1, 1).item()
                self.exploitation += 1
            else:
                a = self.env.action_space.sample(mask = mask)
                self.exploration += 1
        
        elif len(self.env.stops) == self.env.n_stops:
            a = self.env.stops[0]
        
        else:
            print('WTF!')
        
        self.global_actions_taken += 1
        self.update_coef(self.global_actions_taken)
        
        return exp.update(action = a)
    

    def step(self, exp: Experience) -> None:
        assert isinstance(exp.action, int), f"DQN expects discrete actions (int), current type is {exp.action.__class__}"
        state_vec = torch.tensor([exp.state], dtype = torch.int32).to(self.device)
        action_vec = torch.tensor([[exp.action]], dtype = torch.int64).to(self.device)
        next_state_vec = torch.tensor([exp.next_state], dtype = torch.int32).to(self.device)
        reward_vec = torch.tensor([exp.reward], dtype = torch.float32).to(self.device)
        
        self.buffer.push(
            state = state_vec, 
            action = action_vec,
            next_state = next_state_vec,
            reward = reward_vec
        )

        if (len(self.buffer) >= self.batch_size) & (self.global_actions_taken % self.update_freq == 0):
            for _ in range(self.num_update):
                self.learn(experiences = self.buffer.sample())
            self.soft_update()


    def learn(self, experiences: dict[str, list]) -> None:

        state_batch = torch.cat(experiences['state'])
        action_batch = torch.cat(experiences['action'])
        next_state_batch = torch.cat(experiences['next_state'])
        reward_batch = torch.cat(experiences['reward'])
    
        with torch.no_grad():
            if self.double:
                max_action_from_policy = self.policy_net(next_state_batch).max(1).indices.unsqueeze(-1)
                max_Q_targets = self.target_net(next_state_batch).gather(1, max_action_from_policy).squeeze(-1)
            else:
                max_Q_targets = self.target_net(next_state_batch).max(1).values
        
        if self.revert_discount:
            Q_targets = ((max_Q_targets * (self.gamma**(self.env.n_stops - len(self.env.stops) - 1))) + reward_batch).unsqueeze(-1)
            # Q_targets = ((max_Q_targets * (1/self.gamma)) + reward_batch).unsqueeze(-1)
        else:
            Q_targets = ((max_Q_targets * self.gamma) + reward_batch).unsqueeze(-1)
        
        Q_expected = self.policy_net(state_batch).gather(1, action_batch)
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self._loss = float(loss.item())

    
    def soft_update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
    

    def inspect_qtable(self, nstops: int):
        net_ = self.policy_net
        net_.eval()
        x = torch.tensor([i for i in range(nstops)], dtype = torch.int32, device = self.device)

        with torch.no_grad():
            q_table = net_(x)
            q_table = q_table.cpu().detach().numpy()
        
        del net_, x
        return q_table
    

    def _reset_mask(self):
        self.base_mask = np.ones(self.env.data.shape[0], dtype = np.int8)



class PPOAgent(AgentBase):
    def __init__(
            self,
            env: TSP,
            net: Union[FC],
            revert_discount: bool = False,
            device: str = 'cpu',
            using_gae: bool = True,
            gae_lambda: float = 0.96,
            gamma: float = 0.99, 
            actor_lr: float = 3e-4,
            critic_lr: float = 1e-3,
            ppo_ratio_clip: float = 0.25,
            using_kl_div: bool = False,
            # rollout_length: int = 20,
            # batch_size: int = 20,
            actor_number_updates: int = 50,
            critic_number_updates: int = 50,
            entropy_coef: float = 1e-3,
            entropy_coef_decay: float = 0.999,
            max_grad_norm_actor: int = 100,
            max_grad_norm_critic: int = 100,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.revert_discount = revert_discount
        self.device = device
        self.env = env

        self.actor = net[0].to(self.device)
        self.actor.layer3.weight.data *= 1e-2 # https://arxiv.org/pdf/2006.05990
        self.critic = net[1].to(self.device)
        assert net[0].mode == 'actor', f"Reassign mode to 'actor', current assignment {net[0].mode}"
        assert net[1].mode == 'critic', f"Reassign mode to 'critic', current assignment {net[1].mode}"
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr, amsgrad = True)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr, amsgrad = True)
        self.max_grad_norm_actor = max_grad_norm_actor
        self.max_grad_norm_critic = max_grad_norm_critic
        self.actor_number_updates = actor_number_updates
        self.critic_number_updates = critic_number_updates

        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay

        self.batch_size = self.env.n_stops
        self.rollout_length = self.env.n_stops
        self.buffer = RolloutBuffer(batch_size = self.batch_size, buffer_size = self.batch_size)

        self.using_gae = using_gae
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.ppo_ratio_clip = ppo_ratio_clip

        self.using_kl_div = using_kl_div
        self.kl_beta = 0.1
        self.target_kl = 0.01
        self.kl_div = float("inf")

        self.global_actions_taken = 0
        self.base_mask = np.ones(self.env.data.shape[0], dtype = np.int8)

        self._loss_actor: float = float("nan")
        self._loss_critic: float = float("nan")
    

    @property
    def loss(self) -> dict[str, float]:
        return {"actor": self._loss_actor, "critic": self._loss_critic}
    @loss.setter
    def loss(self, value):
        if isinstance(value, dict):
            self._loss_actor = value["actor"]
            self._loss_critic = value["critic"]
        else:
            self._loss_actor = value
            self._loss_critic = value


    def update_coef(self, current_time_step: Optional[int] = None):
        if self.entropy_coef > 0.01:
            if current_time_step == None:
                self.entropy_coef = self.entropy_coef * self.entropy_coef_decay
            elif current_time_step != None:
                entropy_bench = self.entropy_coef * self.entropy_coef_decay + 1 / current_time_step
                if entropy_bench >= 1:
                    self.entropy_coef = self.entropy_coef * self.entropy_coef_decay
                elif entropy_bench < 1:
                    self.entropy_coef = entropy_bench


    @torch.no_grad()
    def act(self, exp: Experience) -> Experience:
        s = torch.tensor([exp.state], device = self.device)
        action_prob = self.actor(s)
        action_dist = Categorical(probs = action_prob)

        if len(self.env.stops) < self.env.n_stops:
            mask = self.base_mask
            mask[self.env.stops] = 0
            action_prob_clone = action_prob.clone()
            a_mask = np.expand_dims(mask, axis = 0)
            a_mask = torch.tensor(a_mask, dtype = torch.bool, device = self.device)
            action_prob_clone[~a_mask] = 0.0
            action_dist_clone = Categorical(probs = action_prob_clone)
            a = action_dist_clone.sample()
        elif len(self.env.stops) == self.env.n_stops:
            a = self.env.stops[0]
            a = torch.tensor([a], dtype = torch.int64).to(self.device)
        else:
            print('WTF!')
        
        logprob = action_dist.log_prob(a)
        state_value = self.critic.act(s)

        self.global_actions_taken += 1

        return exp.update(action = a.item(), logprob = logprob, state_value = state_value)


    def step(self, exp: Experience) -> None:
        state_vec = torch.tensor([exp.state], dtype = torch.int32).to(self.device)
        action_vec = torch.tensor([[exp.action]], dtype = torch.int64).to(self.device)
        next_state_vec = torch.tensor([exp.next_state], dtype = torch.int32).to(self.device)
        reward_vec = torch.tensor([exp.reward], dtype = torch.float32).to(self.device)
        logprob = exp.logprob
        state_value = exp.state_value

        self.buffer.push(
            state = state_vec, 
            action = action_vec,
            next_state = next_state_vec,
            reward = reward_vec,
            logprob = logprob,
            state_value = state_value
        )

        if self.global_actions_taken % self.rollout_length == 0:
            self.update_coef(self.global_actions_taken)
            self.learn(experiences = self.buffer.all_samples())
            self.buffer.clear()

    
    def learn(self, experiences: dict[str, list]) -> None:

        state_batch = torch.cat(experiences['state'])
        action_batch = torch.cat(experiences['action'])
        next_state_batch = torch.cat(experiences['next_state'])
        reward_batch = torch.cat(experiences['reward'])
        logprob_batch = torch.cat(experiences['logprob'])
        state_value_batch = torch.cat(experiences['state_value']).squeeze(-1)

        if self.revert_discount:
            gamma_ = self.gamma**(self.env.n_stops - len(self.env.stops) - 1)
        else:
            gamma_ = self.gamma

        with torch.no_grad():
            if self.using_gae:
                next_state_value_batch = self.critic.act(next_state_batch).squeeze(-1)
                advantages = compute_gae(
                    rewards = reward_batch, 
                    values = state_value_batch, 
                    next_values = next_state_value_batch,
                    gamma = gamma_, 
                    lamb = self.gae_lambda
                )
                advantages = normalize(advantages)
                returns = advantages + state_value_batch
            else:
                returns = revert_norm_returns(
                    rewards = reward_batch, 
                    gamma = gamma_
                )
                returns = returns.float()
                advantages = normalize(returns - state_value_batch)
        
        for i in range(1):
            idx = 0
            self.kl_div = 0
            while idx < self.rollout_length:
                _states = state_batch.detach()
                _actions = action_batch.detach()
                _logprobs = logprob_batch.detach()
                _returns = returns.detach()
                _advantages = advantages.detach()
                idx += self.batch_size
                samples = (_states, _actions, _logprobs, _returns, _advantages)

            self._loss_actor = 0.0

            for actor_iter in range(self.actor_number_updates):
                self.actor_optimizer.zero_grad()
                loss_actor, kl_div = self.compute_policy_loss(samples)
                self.kl_div += kl_div
                if kl_div > 1.5 * self.target_kl:
                    break
                loss_actor.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm_actor)
                self.actor_optimizer.step()
                self._loss_actor = loss_actor.item()

            for critic_iter in range(self.critic_number_updates):
                self.critic_optimizer.zero_grad()
                loss_critic = self.compute_value_loss(samples)
                loss_critic.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm_critic)
                self.critic_optimizer.step()
                self._loss_critic = float(loss_critic.item())

            self.kl_div = abs(self.kl_div) / (
                self.actor_number_updates  * self.rollout_length / self.batch_size
            )

            if self.using_kl_div:
                if self.kl_div > self.target_kl * 1.5:
                    self.kl_beta = min(1.5 * self.kl_beta, 1e2)  # Max 100
                elif self.kl_div < self.target_kl / 1.5:
                    self.kl_beta = max(0.75 * self.kl_beta, 1e-6)  # Min 0.000001

            if self.kl_div > self.target_kl * 1.5:
                break


    def compute_policy_loss(self, samples):
        states, actions, old_logprobs, _, advantages = samples
        action_probs = self.actor(states)
        action_dist = Categorical(probs = action_probs)
        entropy = action_dist.entropy()
        new_logprobs = action_dist.log_prob(actions)

        r_theta = (new_logprobs - old_logprobs).exp()
        r_theta_clip = torch.clamp(r_theta, 1.0 - self.ppo_ratio_clip, 1.0 + self.ppo_ratio_clip)

        # KL = E[log(P/Q)] = sum_{P}( P * log(P/Q) ) -- \approx --> avg_{P}( log(P) - log(Q) )
        approx_kl_div = (old_logprobs - new_logprobs).mean().item()
        if self.using_kl_div:
            # Ratio threshold for updates is 1.75 (although it should be configurable)
            policy_loss = -torch.mean(r_theta * advantages) + self.kl_beta * approx_kl_div
        else:
            joint_theta_adv = torch.stack((r_theta * advantages, r_theta_clip * advantages))
            assert joint_theta_adv.shape[0] == 2
            policy_loss = -torch.amin(joint_theta_adv, dim = 0).mean()
        entropy_loss = -self.entropy_coef * entropy.mean()

        loss = policy_loss + entropy_loss
        
        return loss, approx_kl_div


    def compute_value_loss(self, samples):
        states, _, _, returns, _ = samples
        values = self.critic(states).squeeze(-1)
        criterion = nn.SmoothL1Loss()
        loss = criterion(values, returns)
        return loss


    def _reset_mask(self):
        self.base_mask = np.ones(self.env.data.shape[0], dtype = np.int8)
