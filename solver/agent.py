import torch
import random
import numpy as np
from typing import Union

from solver.env import *
from solver.net import *
from solver.memory import *

class A2CAgent(object):
    pass



class DQAgent(object):
    def __init__(
        self, 
        env: TSP,
        net: Union[DQN],
        states_size: int, actions_size: int,
        device = 'cpu',
        epsilon = 1.0, epsilon_min = 0.01, 
        epsilon_decay = 0.999, 
        gamma = 0.99, lr = 1e-4,
        tau = 5e-3, batch_size = 128,
        max_grad_norm = 10,
        update_freq = 1
    ):
        self.device = device

        self.env = env
        self.policy_net = net[0].to(self.device)
        self.target_net = net[1].to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr = lr, amsgrad = True)

        self.states_size = states_size
        self.actions_size = actions_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.update_freq = update_freq

        self.base_mask = np.ones(actions_size, dtype = np.int8)

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


    def update_epsilon(self, current_time_step: Optional[int] = None):
        if self.epsilon > self.epsilon_min:
            if current_time_step == None:
                self.epsilon = self.epsilon * self.epsilon_decay
            elif current_time_step != None:
                epsilon_bench = self.epsilon * self.epsilon_decay + 1 / current_time_step
                if epsilon_bench >= 1:
                    self.epsilon = self.epsilon * self.epsilon_decay
                elif epsilon_bench < 1:
                    self.epsilon = epsilon_bench


    def act(
        self, 
        s: torch.tensor,
        inference: bool = False
    ):
        if len(self.env.stops) < self.env.n_stops:

            mask = self.base_mask
            mask[self.env.stops] = 0

            if not inference:
                if random.random() > self.epsilon:
                # if random.random() > 0.7:
                    with torch.no_grad():
                        a_ = self.policy_net(s)
                        a_mask = np.expand_dims(mask, axis = 0)
                        a_mask = torch.tensor(a_mask, dtype = torch.bool, device = self.device)
                        a_[~a_mask] = -np.inf
                        a = a_.max(1).indices.view(1, 1).item()
                    self.exploitation += 1
                else:
                    a = self.env.action_space.sample(mask = mask)
                    self.exploration += 1
            
            elif inference:
                self.policy_net.eval()
                with torch.no_grad():
                    a_ = self.policy_net(s)
                    a_mask = np.expand_dims(mask, axis = 0)
                    a_mask = torch.tensor(a_mask, dtype = torch.bool, device = self.device)
                    a_[~a_mask] = -np.inf
                    a = a_.max(1).indices.view(1, 1).item()
        
        elif len(self.env.stops) == self.env.n_stops:
            a = self.env.stops[0]
        
        else:
            print('WTF!')
        
        return a
    

    def learn(self, experience: ReplayMemory):
        if len(experience) < self.batch_size:
            return
        
        transitions = experience.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)
    
        with torch.no_grad():
            max_Q_targets = self.target_net(next_state_batch).max(1).values
        Q_targets = ((max_Q_targets * self.gamma) + reward_batch).unsqueeze(1)
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


    def inference(self):
        state, _ = self.env.reset()
        self._reset_mask()
        R = 0
        done = False

        while not done:
            action = self.act(
                s = torch.tensor([state], device = self.device),
                inference = True
            )
            next_state, reward, done, _ = self.env.step(action)
            reward = -1 * reward
            R += reward
            state = next_state

            if done:
                break
        
        return self.env.stops, R
    

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
        self.base_mask = np.ones(self.actions_size, dtype = np.int8)



class QAgent(object):
    def __init__(
        self, 
        env: TSP,
        states_size: int, actions_size: int, 
        epsilon = 1.0, epsilon_min = 0.01, 
        epsilon_decay = 0.999, 
        gamma = 0.95, lr = 0.8
    ):
        self.env = env

        self.states_size = states_size
        self.actions_size = actions_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.Q = np.zeros([states_size, actions_size])

        self.base_mask = np.ones(actions_size, dtype = np.int8)

        self.exploitation = 0
        self.exploration = 0


    def learn(self, s, a, r, s_next):
        TD_target_current = r + self.gamma*np.max(self.Q[s_next,a])
        TD_error_current = self.Q[s,a] - TD_target_current
        self.Q[s,a] = self.Q[s,a] - self.lr * TD_error_current


    def update_epsilon(self, current_time_step: Optional[int] = None):
        if self.epsilon > self.epsilon_min:
            if current_time_step == None:
                self.epsilon = self.epsilon * self.epsilon_decay
            elif current_time_step != None:
                epsilon_bench = self.epsilon * self.epsilon_decay + 1 / current_time_step
                if epsilon_bench >= 1:
                    self.epsilon = self.epsilon * self.epsilon_decay
                elif epsilon_bench < 1:
                    self.epsilon = epsilon_bench


    def act(self, s, inference: bool = False):
        if len(self.env.stops) < self.env.n_stops:

            # For exploitation
            q = np.copy(self.Q[s,:])
            q[self.env.stops] = -np.inf

            # For exploration
            mask = self.base_mask
            mask[self.env.stops] = 0

            if not inference:
                if random.random() > self.epsilon:
                # if random.random() > .5:
                    a = np.argmax(q)
                    self.exploitation += 1
                else:
                    a = self.env.action_space.sample(mask = mask)
                    self.exploration += 1
            
            elif inference:
                a = np.argmax(q)

        elif len(self.env.stops) == self.env.n_stops:
            a = self.env.stops[0]
        
        else:
            print('WTF!')
        
        return a
    

    def inference(self):
        state, _ = self.env.reset()
        self._reset_mask()
        R = 0
        done = False

        while not done:
            action = self.act(state, inference = True)
            next_state, reward, done, _ = self.env.step(action)
            reward = -1 * reward

            R += reward
            state = next_state

            if done:
                break
        
        return self.env.stops, R


    def inspect_qtable(self):
        return self.Q


    def _reset_mask(self):
        self.base_mask = np.ones(self.actions_size, dtype = np.int8)