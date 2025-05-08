import torch

from env import TSP
from agent import DQAgent, PPOAgent
from memory import Experience

from typing import Optional, Union
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import os

import imageio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Runner():
    def __init__(
        self, 
        env: TSP, 
        agent: Union[DQAgent, PPOAgent]
    ):
        self.env = env
        self.agent = agent

        self.device = self.agent.device

        self.loss_memory = []
        self.loss_actor_memory = []
        self.loss_critic_memory = []
        self.R_memory = []

        self.best_R = -np.inf
        self.best_Rs = []
        self.best_route = []
        self.best_routes = []
        self.since_last_best = 0

        self.agent_str = str(self.agent.__class__).split(' ')[-1].split("'")[-2].split('.')[-1]
        if hasattr(self.agent, "double"):
            if self.agent.double:
                self.agent_str = 'D' + self.agent_str
        if hasattr(self.agent, "sample_type"):
            self.agent_str = self.agent_str + self.agent.sample_type
                
    
    def run(
        self, 
        num_episodes: int = 10_000,
        render_gif: bool = False, 
        render_charts: list[bool, bool] = [False, False],
        render_each: Optional[int] = None, 
        duration: Optional[int] = None,
        save_every: int = 1_000
    ):
        self.num_episodes = num_episodes
        imgs = []
        final_img = None

        for e in tqdm(range(1, num_episodes+1, 1)):
        # for e in range(1, num_episodes+1, 1):
            self.e = e
            self._run_ep()

            self.R_memory.append(self.R)
            if self.agent.__class__ == DQAgent:
                self.loss_memory.append(sum(self.L)/len(self.L))
                # print(f"Episode {e} ; Reward {self.R} ; Loss {self.loss_memory[-1]}")
            elif self.agent.__class__ == PPOAgent:
                self.loss_actor_memory.append(sum(self.l_act)/len(self.l_act))
                self.loss_critic_memory.append(sum(self.l_cri)/len(self.l_cri))
                # print(f"Episode {e} ; Reward {self.R} ; ActorLoss {self.loss_actor_memory[-1]} ; CriticLoss {self.loss_critic_memory[-1]}")

            if render_gif:
                if e % render_each == 0:
                    img = self.env.render(return_img = True)
                    imgs.append(img)
            
            '''
            BEST RESULT CHECKPOINT
            '''
            if self.R >= self.best_R:
                self.best_R = self.R
                self.best_Rs.append(self.best_R)
                self.best_Rs = self.best_Rs[-20:]

                self.best_route = self.env.stops
                self.best_routes.append(self.best_route)
                self.best_routes = self.best_routes[-20:]

                self.since_last_best = 0

                final_img = self.env.render(return_img = True) if render_gif else None
            elif self.R < self.best_R:
                self.since_last_best += 1

            '''
            CHECKPOINT
            '''
            if (e % save_every == 0) & (e >= save_every):
                self.save_model()

        self.plot_training(save_fig = render_charts[0])
        self.plot_route(save_fig = render_charts[1])

        if render_gif:
            imgs.append(final_img)
            imageio.mimsave(
                f"training_{self.env.n_stops}_stops.gif",
                ims = imgs, duration = duration
            )

        return self.env, self.agent, self.best_R, self.best_route, self.training_chart, self.route_chart


    def _run_ep(self):
        state, _ = self.env.reset()
        self.agent._reset_mask()
        
        self.L = []
        self.l_act = []
        self.l_cri = []
        self.R = 0
        done = False

        while not done:
            experience = Experience(state = state)
            experience = self.agent.act(exp = experience)
            action = int(experience.action)
            next_state, reward, done, _ = self.env.step(action = action)
            reward = -1 * reward
            experience.update(action = action, next_state = next_state, reward = reward, done = done)
            
            self.agent.step(exp = experience)

            self.R += reward
            if self.agent.__class__ == DQAgent:
                self.L.append(self.agent._loss)
            elif self.agent.__class__ == PPOAgent:
                self.l_act.append(self.agent._loss_actor)
                self.l_cri.append(self.agent._loss_critic)
            state = next_state

            if done:
                break


    def save_model(self):
        
        path = os.getcwd()
        checkfile = path + f'\\model\\checkpoint_{self.agent_str}_TSP{self.env.n_stops}.pt'

        if self.agent.__class__ == DQAgent:
            checkpoint = {
                'episode': self.e,
                'best_R': self.best_R,
                'best_Rs': self.best_Rs,
                'best_route': self.best_route,
                'best_routes': self.best_routes,
                'epsilon': self.agent.epsilon,
                'rewards': self.R_memory,
                'losses': self.loss_memory,
                'policy_net': self.agent.policy_net.state_dict(),
                'target_net': self.agent.target_net.state_dict(),
                'optimizer': self.agent.optimizer.state_dict()
            }
        elif self.agent.__class__ == PPOAgent:
            checkpoint = {
                'episode': self.e,
                'best_R': self.best_R,
                'best_Rs': self.best_Rs,
                'best_route': self.best_route,
                'best_routes': self.best_routes,
                'rewards': self.R_memory,
                'actor_losses': self.loss_actor_memory,
                'critic_losses': self.loss_critic_memory,
                'actor': self.agent.actor.state_dict(),
                'critic': self.agent.critic.state_dict(),
                'actor_optimizer': self.agent.actor_optimizer.state_dict(),
                'critic_optimizer': self.agent.critic_optimizer.state_dict()
            }

        torch.save(checkpoint, checkfile)
        del checkpoint


    def load_model(self):

        path = os.getcwd()
        checkfile = path + f'\\model\\checkpoint_{self.agent_str}_TSP{self.env.n_stops}.pt'
        checkpoint = torch.load(checkfile)

        if self.agent.__class__ == DQAgent:
            self.e = checkpoint['episode']
            self.best_R = checkpoint['best_R']
            self.best_Rs = checkpoint['best_Rs']
            self.best_route = checkpoint['best_route']
            self.best_routes = checkpoint['best_routes']
            self.agent.epsilon = checkpoint['epsilon']
            self.R_memory = checkpoint['reward']
            self.loss_memory = checkpoint['losses']
            self.agent.policy_net.load_state_dict(checkpoint['policy_net'])
            self.agent.target_net.load_state_dict(checkpoint['target_net'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer'])
            self.num_episodes = len(self.R_memory)

        elif self.agent.__class__ == PPOAgent:
            self.e = checkpoint['episode']
            self.best_R = checkpoint['best_R']
            self.best_Rs = checkpoint['best_Rs']
            self.best_route = checkpoint['best_route']
            self.best_routes = checkpoint['best_routes']
            self.R_memory = checkpoint['rewards']
            self.loss_actor_memory = checkpoint['actor_losses']
            self.loss_critic_memory = checkpoint['critic_losses']
            self.agent.actor.load_state_dict(checkpoint['actor'])
            self.agent.critic.load_state_dict(checkpoint['critic'])
            self.agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.num_episodes = len(self.R_memory)

        del checkpoint


    def plot_training(self, save_fig: bool = True):
        avg_R = [np.nan]*99
        avg_l = [np.nan]*99
        avg_l_actor = [np.nan]*99
        avg_l_critic = [np.nan]*99

        if self.agent.__class__ == DQAgent:
            names = ['Reward', 'Avg_Reward', 'Loss', 'Avg_Loss']
            metrics = [self.R_memory, avg_R, self.loss_memory, avg_l]
        elif self.agent.__class__ == PPOAgent:
            names = ['Reward', 'Avg_Reward', 'ActorLoss', 'Avg_ActorLoss', 'CriticLoss', 'Avg_CriticLoss']
            metrics = [self.R_memory, avg_R, self.loss_actor_memory, avg_l_actor, self.loss_critic_memory, avg_l_critic]

        for ep in range(100, self.num_episodes+1, 1):
            for idx in range(0, len(metrics), 2):
                metrics[idx+1].append(np.mean(metrics[idx][ep-100:ep]))
        
        x_bar = [i for i in range(self.num_episodes)]
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for i, j in zip(range(len(metrics)), range(len(names))):
            if i <= 1:
                fig.add_trace(
                    go.Scatter(x = x_bar, y = metrics[i], name = names[j]),
                    secondary_y = False,
                )
            else:
                fig.add_trace(
                    go.Scatter(x = x_bar, y = metrics[i], name = names[j]),
                    secondary_y = True,
                )

        try:
            title = f"Best Route: {self.best_route[:6] + ['...'] + self.best_route[-5:]} - Best R: {-self.best_R} - Exploit/Explore: {self.agent.exploitation, self.agent.exploration}"
        except AttributeError:
            title = f"Best Route: {self.best_route[:6] + ['...'] + self.best_route[-5:]} - Best R: {-self.best_R}"

        fig.update_layout(title_text = title)
        fig.update_xaxes(title_text = "Episode")
        fig.update_yaxes(title_text = "<b>R</b>", secondary_y = False)
        fig.update_yaxes(title_text = "<b>L</b>", secondary_y = True)

        self.training_chart = fig
        if save_fig:
            fig.write_html(f"Training_{self.agent_str}_TSP{self.env.data.shape[0]}.html")
    

    def plot_route(self, save_fig: bool = False):
        fig = go.Figure()
        for route in self.best_routes:
            fig.add_trace(
                go.Scatter(
                    x = self.env.x[np.array(route)], 
                    y = self.env.y[np.array(route)],
                    mode = 'lines+markers',
                    text = route,
                    textposition = "top center",
                    textfont = dict(
                        color = ['red'] + ['black']*(len(route)-2) + ['red'], 
                        size = 12
                    ),
                    hoverinfo = 'text',
                    marker = dict(color = ['red']*len(route), size = 5)
                )
            )
        
        fig.data[-1].visible = True

        steps = []
        for i, r in zip(range(len(fig.data)), self.best_Rs):
            step = dict(
                method = "update",
                args= [
                    {"visible": [False] * len(fig.data)},
                    {"title": f"Tour Length: {r}"}
                ]
            )
            step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        sliders = [dict(
            active = 1,
            # currentvalue = {"prefix": "Frequency: "},
            pad = {"t": 50},
            steps = steps
        )]

        fig.update_layout(sliders = sliders)

        self.route_chart = fig
        if save_fig:
            fig.write_html(f"Route_{self.agent_str}_TSP{self.env.data.shape[0]}.html")
