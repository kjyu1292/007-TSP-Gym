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
- State *s* depicting the node in which an agent finds itself in a any time step; $s \in \mathcal{S}$, $|\mathcal{S}|=N$, with $\mathcal{S}$ called state space.
