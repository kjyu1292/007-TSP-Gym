from solver.env import TSP
from solver.agent import QAgent, DQAgent
# from solver.net import DQN
from solver.memory import *

from typing import Union, Optional
from collections import OrderedDict
from tqdm import tqdm
import os

import torch
import numpy as np

import imageio
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Runner():
    def __init__(
        self, 
        env: TSP, 
        agent: Union[DQAgent, QAgent],
        mem: ReplayMemory
    ):
        self.env = env
        self.agent = agent

        if agent.__class__ != QAgent:
            self.mem = mem
            self.device = self.agent.device
        elif agent.__class__ == QAgent:
            self.Q_memory = []
        
        self.R_memory = []
        self.loss_memory = []
        self.best_R = -np.inf
        self.best_route = []

        self.global_actions_taken = 0
        self.agent_str = str(self.agent.__class__).split(' ')[-1].split("'")[-2].split('.')[-1]
    

    def run(
        self, 
        num_episodes: int = 10_000,
        render_gif: bool = False, 
        render_each: Optional[int] = None, 
        duration: Optional[int] = None,
        save_every: int = 1_000
    ):
        self.num_episodes = num_episodes
        imgs = []
        final_img = None

        for e in tqdm(range(num_episodes)):
            self._run_ep_vec() if self.agent.__class__ != QAgent else self._run_ep()

            self.R_memory.append(self.R)
            if self.agent.__class__ == DQAgent:
                self.loss_memory.append(np.mean(self.L))
            elif self.agent.__class__ == QAgent:
                self.Q_memory.append(np.abs(self.agent.Q).sum().sum())

            if render_gif:
                if e % render_each == 0:
                    img = self.env.render(return_img = True)
                    imgs.append(img)

            '''
            BEST RESULT CHECKPOINT
            '''
            if self.R >= self.best_R:
                self.best_R = self.R
                self.best_route = self.env.stops
                final_img = self.env.render(return_img = True) if render_gif else None
                self.save_model(
                    episode = e,
                    epsilon = self.agent.epsilon,
                    rewards = self.R_memory,
                    losses = self.loss_memory,
                    policy_net = self.agent.policy_net.state_dict(),
                    target_net = self.agent.target_net.state_dict(),
                    optimizer = self.agent.optimizer.state_dict(),
                    best = True
                )
            
            '''
            NORMAL CHECKPOINT
            '''
            if (e % save_every == 0) & (self.agent.__class__ != QAgent):
                self.save_model(
                    episode = e,
                    epsilon = self.agent.epsilon,
                    rewards = self.R_memory,
                    losses = self.loss_memory,
                    policy_net = self.agent.policy_net.state_dict(),
                    target_net = self.agent.target_net.state_dict(),
                    optimizer = self.agent.optimizer.state_dict(),
                    best = False
                )
        
        if self.agent.__class__ == QAgent:
            for i in range(len(self.Q_memory)-1):
                self.loss_memory.append(
                    abs(self.Q_memory[i+1] - self.Q_memory[i])
                )

        self.training_chart()

        if render_gif:
            imgs.append(final_img)
            imageio.mimsave(
                f"training_{self.env.n_stops}_stops.gif",
                ims = imgs, duration = duration
            )

        return self.env, self.agent, self.best_R, self.best_route, self.fig


    def _run_ep_vec(self):
        state, _ = self.env.reset()
        self.agent._reset_mask()
        
        self.L = 0
        self.R = 0
        done = False

        while not done:
            action = self.agent.act(
                s = torch.tensor([state], device = self.device),
                inference = False
            )
            next_state, reward, done, _ = self.env.step(action)
            reward = -1 * reward

            state_vec, action_vec, next_state_vec, reward_vec = self._to_tensor(
                state = state, action = action,
                next_state = next_state, reward = reward
            )

            self.mem.push(state_vec, action_vec, next_state_vec, reward_vec)
            
            self.global_actions_taken += 1
            self.agent.update_epsilon(current_time_step = self.global_actions_taken)

            self.R += reward
            self.L += self.agent._loss
            state = next_state

            self.agent.learn(experience = self.mem)
            if (len(self.mem) >= self.agent.batch_size) and ((self.global_actions_taken % self.agent.update_freq) == 0):
                self.agent.soft_update()

            if done:
                break


    def _run_ep(self):
        state, _ = self.env.reset()
        self.agent._reset_mask()
        self.R = 0
        done = False

        while not done:
            action = self.agent.act(state, inference = False)
            next_state, reward, done, _ = self.env.step(action)
            reward = -1 * reward
            
            self.agent.learn(state, action, reward, next_state)
            self.global_actions_taken += 1
            self.agent.update_epsilon(current_time_step = self.global_actions_taken)

            if (self.env.cum_reward == True) & (done == True):
                self.R = reward
            else:
                self.R += reward

            state = next_state

            if done:
                break


    def save_model(
            self, episode: int,
            epsilon: float,
            rewards: list,
            losses: list,
            policy_net: OrderedDict,
            target_net: OrderedDict,
            optimizer: OrderedDict,
            best: bool
        ):

        path = os.getcwd()
        path_ = "\\".join(path.split("\\")[:-1])

        if best:
            checkfile = path_ + f'\\model\\checkpoint_{self.agent_str}_TSP{self.env.n_stops}_best.pt'
        elif not best:
            checkfile = path_ + f'\\model\\checkpoint_{self.agent_str}_TSP{self.env.n_stops}.pt'

        checkpoint = {
            'episode': episode,
            'epsilon': epsilon,
            'reward_list': rewards,
            'loss_list': losses,
            'policy_net': policy_net,
            'target_net': target_net,
            'optimizer': optimizer
        }
        torch.save(checkpoint, checkfile)
        del checkpoint
    

    def load_model(self, best: bool):
        path = os.getcwd()
        path_ = "\\".join(path.split("\\")[:-1])

        if best:
            checkfile = path_ + f'\\model\\checkpoint_{self.agent_str}_TSP{self.env.n_stops}_best.pt'
        elif not best:
            checkfile = path_ + f'\\model\\checkpoint_{self.agent_str}_TSP{self.env.n_stops}.pt'
        
        checkpoint = torch.load(checkfile)

        self.agent.epsilon = checkpoint['epsilon']
        self.R_memory = checkpoint['reward_list']
        self.loss_memory = checkpoint['loss_list']
        self.agent.policy_net.load_state_dict(checkpoint['policy_net'])
        self.agent.target_net.load_state_dict(checkpoint['target_net'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer'])

        del checkpoint


    def _to_tensor(
        self,
        state: int,
        action: int,
        next_state: int,
        reward: float
    ):
        state = torch.tensor([state], dtype = torch.int32).to(self.device)
        action = torch.tensor([[action]], dtype = torch.int64).to(self.device)
        next_state = torch.tensor([next_state], dtype = torch.int32).to(self.device)
        reward = torch.tensor([reward], dtype = torch.float32).to(self.device)

        return state, action, next_state, reward


    def training_chart(self, save_fig: bool = False):
        x_bar = [i for i in range(self.num_episodes)]
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x = x_bar, y = self.R_memory, name = "Reward"),
            secondary_y = False,
        )
        fig.add_trace(
            go.Scatter(x = x_bar, y = self.loss_memory, name = "Loss"),
            secondary_y = True,
        )

        fig.update_layout(
            title_text = f"Best Route: {self.best_route[:6] + ['...'] + self.best_route[-5:]} - Best R: {-self.best_R} - Exploit/Explore: {self.agent.exploitation, self.agent.exploration}"
        )

        fig.update_xaxes(title_text = "Episode")

        fig.update_yaxes(title_text = "<b>R</b>", secondary_y = False)
        fig.update_yaxes(title_text = "<b>L</b>", secondary_y = True)

        self.fig = fig
        if save_fig:
            fig.write_html(f"{self.agent_str}_TSP{self.env.data.shape[0]}.html")
    

    