from typing import Optional

import numpy as np
from math import radians
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import haversine_distances

import imageio
import matplotlib.pyplot as plt
plt.style.use("seaborn-dark")

import gymnasium as gym
from gymnasium import spaces


class TSP(gym.Env):
    def __init__(
        self, 
        data: np.ndarray,
        n_stops: int = None,
        distance_method: str = "haversine_distance",
        fixed_start: bool = False,
        cum_reward: Optional[bool] = True,
        render_mode: Optional[str] = "human"
    ):
        self.data = data
        self.n_stops = n_stops if self.data is None else data.shape[0]
        self.action_space = spaces.Discrete(self.n_stops)
        self.observation_space = spaces.Discrete(self.n_stops)
        self.fixed_start = fixed_start
        self.cum_reward = cum_reward

        if fixed_start:
            self.stops = [0]
        else:
            self.stops = []

        if cum_reward:
            self.EP_REWARD = 0

        self.render_mode = render_mode
        self.distance_method = distance_method

        self._generate_distance_matrix() # -> self.x, self.y, self.dist_matrix
        if self.render_mode == 'human':
            self.scaling() # Scaling for notation
            self.render()

    
    '''
    INITIALIZATION FUNCTIONS
    '''
    def _generate_distance_matrix(self):
        self.x = self.data[:,0]
        self.y = self.data[:,1]
        if self.distance_method in ["euclidean_distance"]:
            self.dist_matrix = cdist(self.data, self.data)
        elif self.distance_method == "haversine_distance":
            co_list = []
            for i in self.data:
                for j in i:
                    co_list.append(radians(j))
            self.dist_matrix = haversine_distances(np.array(co_list).reshape(self.n_stops, 2)) * 6371
        else:
            raise Exception("Method not recognized.")


    '''
    COMMON GYM FUNCTIONS
    '''
    def step(self, action: int, prev_state: Optional[np.ndarray] = None):
        obs = self.stops[-1]
        new_obs = action        
        self.stops.append(new_obs)

        reward = self._get_obs_reward(obs = obs, new_obs = new_obs)

        if self.stops[0] == self.stops[-1]:
            done = True
        else:
            done = False

        if self.cum_reward:
            self.EP_REWARD += reward
            return new_obs, self.EP_REWARD, done, {}
        else:
            return new_obs, reward, done, {}

    def _get_obs(self):
        pass

    def _get_truncated(self):
        pass

    def _get_terminated(self):
        pass

    def _get_obs_reward(self, obs: int, new_obs: int):
        return self.dist_matrix[obs, new_obs]

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[int] = None
    ):
        if self.cum_reward:
            self.EP_REWARD = 0

        if seed is not None:
            np.random.seed(seed)

        if self.fixed_start:
            self.stops = [0]
            first_stop = 0
        else:
            self.stops = []
            first_stop = np.random.randint(self.n_stops)
            self.stops.append(first_stop)

        return first_stop, {}


    '''
    RENDER FUNCTIONS
    '''
    def render(self, return_img = False):
        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(111)
        ax.set_title("Delivery Stops")

        # Show stops
        ax.scatter(self.x, self.y, c = "red", s = 50)
        
        # Show START
        if len(self.stops) > 0:
            xy = self.x[self.stops[0]], self.y[self.stops[0]]
            # xytext = xy[0] + .1, xy[1] - .05
            xytext = xy[0] + self.x_unit, xy[1] - self.y_unit
            ax.annotate(
                "START", 
                xy = xy, xytext = xytext,
                weight = "bold"
            )
        
        # Show route
        if len(self.stops) > 1:
            ax.plot(
                self.x[self.stops], self.y[self.stops], 
                c = "blue", linewidth = 1, linestyle = "--"
            )

            # # Show END
            # xy = self.x[self.stops[-1]], self.y[self.stops[-1]]
            # xytext = xy[0] + self.x_unit, xy[1] - self.y_unit
            # ax.annotate(
            #     "END", 
            #     xy = xy, 
            #     xytext = xytext, 
            #     weight = "bold"
            # )
        
        plt.xticks([])
        plt.yticks([])

        if return_img:
            fig.canvas.draw_idle()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype = 'uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image

    def scaling(self):
        x_min, x_max = self.x.min(), self.x.max()
        y_min, y_max = self.y.min(), self.y.max()
        self.x_unit = (x_max - x_min) / (self.n_stops * 10)
        self.y_unit = (y_max - y_min) / (self.n_stops * 10)