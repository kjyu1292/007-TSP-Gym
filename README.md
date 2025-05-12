# 007-TSP-Gym
Reworked environment backend, switch to Gymnasium module.

Definition of Traveling Salesman Problem: Given a set of nodes, find the shortest path that visit all but one node once, and said path must return to the starting node. 

| index  | latitude | longtitude | 
| ------------- | ------------- | ------------- |
| 0  | 0.9753  | 0.8564 |
| 1  | 0.5432  | 0.7935 |
| ...  | ...  | ... |
| ...  | ...  | ... |
| 8  | 0.4254  | 0.9723 |
| 9  | 0.1245  | 0.4752 |

With $\mathcal{L}(\pi)$ depicting the length $\mathcal{L}$ of a given path $\pi$. The exact nature of $\mathcal{L}$ in this particular application is a number, and $\pi$ is a list of indices, e.g: [**0**, 5, 6, ..., 9, **0**]. The first variable we can see here is $\pi$, and we have to find 1 that yields close-to-optimal within its number of permutations of $\frac{(N-1)!}{2}$, with N is the number of nodes of an instance of TSP that one would like to solve. Mathematically:

$$\mathcal{L}(\pi) = \sum_{i = 0}^{N-1} \mathcal{d}(\pi_{i},\pi_{i+1})$$

$\mathcal{d}$ is the function to calculate distance between 2 nodes, in this application, two provided functions are Euclidean and Haversine.

The implemented algorithms are built upon the principles of Reinforcement Learning. The Markov Decision Process (MDP) of the problem is as follow:
- State *s* depicting the node in which an agent finds itself in any time step; $s \in \mathcal{S}$, $|\mathcal{S}|=N$, with $\mathcal{S}$ called state space. $s_{i}$ can also be understood interchangeably with $\pi_{i}$.
- Action *a* depicting the node an agent chooses to move to in any time step; $a \in \mathcal{A}$, $|\mathcal{A}|=N$, with $\mathcal{A}$ called action space. Note that the number of states and the number of actions is the same.
- Reward *r* depicting the reward an agent receive from taking action *a* in node *s* to arrive at node *s'*. In this application, reward is the distance $\mathcal{d}$.  $r \in \mathcal{R}$, $|\mathcal{R}|=N \times N$, $\mathcal{R}$ called reward space. The reward space in the application of TSP is dense.
- $\gamma$ is the discount factor, commonly chosen to be a float ~0.9.
- $\pi$ is policy, this one is intrisically different from the one above depicting the tour. A policy $\pi$ can be understood as the probability distribution of states, given state *s*, the probability of choosing action *a* is $\pi(a|s), s \in \mathcal{S}, a \in \mathcal{A}$. $\sum_{i=0}^{N} \pi(a_{i}|s) = 1$ for any $s \in \mathcal{S}$.

Models implemented including Deep Q Network and its variants (Prioritized Experience Replay sampling method with Double Q), and Proximal Policy Optimization.
DQN with Uniform sampling yields the best result. PPO could not converge.

Discounted reward function is modified. The common practice places heavy portion of its focus on the near future steps. The modified version discounted rewards of a state place its focus on, instead, both ends of a trajectory, and with little to zero within the middle section.
e.g. State value of s_i at time step 39 has roughly 95% of its value constituted of rewards from step 40 and the final time step.
This modded version provide faster convergence and a more stable training.
