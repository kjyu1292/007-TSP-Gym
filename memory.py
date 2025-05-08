import copy
import math
import random
import torch
import numpy as np
from utils import to_list
from typing import Any, Sequence, Union, Deque, Iterator
from collections import deque, defaultdict


class Experience(object):
    checklist: list = [
        "state", "action", "next_state", "reward", "done", 
        "logprob", "state_value", "td_error"
    ]

    def __init__(self, **kwargs):
        self.data = {}
        self.update(**kwargs)
    

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Experience) and self.data == o.data


    def get(self, key: str):
        return self.data.get(key)


    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in Experience.checklist:
                self.data[key] = value
                self.__dict__[key] = value
        return self
    

    def get_dict(self, serialize = False) -> dict[str, Any]:
        if serialize:
            return {k: to_list(v) for (k, v) in self.data.items()}
        return self.data
    

class ReferenceBuffer(object):
    def __init__(self, buffer_size: int):
        self.buffer = dict()
        self.counter = defaultdict(int)
        self.buffer_size = buffer_size


    def __len__(self) -> int:
        return len(self.buffer)


    @staticmethod
    def _hash_element(el) -> Union[int,str]:
        if isinstance(el, np.ndarray):
            return hash(el.data.tobytes())
        elif isinstance(el, torch.tensor):
            return hash(str(el))
        else:
            return str(el)


    def push(self, el) -> Union[int,str]:
        idx = self._hash_element(el)
        self.counter[idx] += 1
        if self.counter[idx] < 2:
            self.buffer[idx] = el
        return idx


    def get(self, idx: Union[int,str]):
        return self.buffer[idx]


    def remove(self, idx: str):
        self.counter[idx] -= 1
        if self.counter[idx] < 1:
            self.buffer.pop(idx, None)
            del self.counter[idx]


class ReplayBuffer(object):
    type = "Replay"
    keylist = [
        "state", "action", "next_state", "reward"
        # , "done"
    ]

    def __init__(self, batch_size: int, buffer_size = int(1e5), **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.indices = range(batch_size)
        self.data: list[Experience] = []
        self._rng = random.Random(kwargs.get("seed"))


    def __len__(self) -> int:
        return len(self.data)


    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and isinstance(o, type(self))


    def seed(self, seed: int):
        self._rng = random.Random(seed)


    def clear(self):
        self.data = []


    def push(self, **kwargs):
        self.data.append(Experience(**kwargs))
        if len(self.data) > self.buffer_size:
            drop_exp = self.data.pop(0)


    def sample(self, keys: list[str] = keylist) -> dict[str, list]:
        sampled_exp: list[Experience] = self._rng.sample(self.data, self.batch_size)
        keys = keys if keys is not None else list(self.data[0].__dict__.keys())
        all_experiences = {k: [] for k in keys}
        for data in sampled_exp:
            for key in keys:
                value = getattr(data, key)
                all_experiences[key].append(value)
        return all_experiences


class PERBuffer(object):
    """
    https://danieltakeshi.github.io/2019/07/14/per/
    """
    type = "PER"

    def __init__(self, batch_size: int, buffer_size: int = int(1e5), alpha = 0.5, device = None, **kwargs):
        super(PERBuffer, self).__init__()
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        self.tree = SumTree(buffer_size)
        self.alpha: float = alpha
        self.__default_weights = np.ones(self.batch_size) / self.buffer_size
        self._rng = random.Random(kwargs.get("seed"))

        self.tiny_offset: float = 0.05

        self._states_mng = kwargs.get("compress_state", False)
        self._states = ReferenceBuffer(buffer_size + 20)


    def __len__(self) -> int:
        return len(self.tree)


    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and isinstance(o, type(self))


    def seed(self, seed: int):
        self._rng = random.Random(seed)
        return seed


    @property
    def data(self):
        # TODO @dawid: update to SumTree so that it return proper data
        return list(filter(lambda x: x is not None, self.tree.data))

    def push(self, *, priority: float = 0, **kwargs):
        priority += self.tiny_offset
        if self._states_mng:
            kwargs["obs_idx"] = self._states.add(kwargs.pop("obs"))
            if "next_obs" in kwargs:
                kwargs["next_obs_idx"] = self._states.add(kwargs.pop("next_obs"))
        # old_data = self.tree.insert(kwargs, pow(priority, self.alpha))
        old_data = self.tree.insert(Experience(**kwargs), pow(priority, self.alpha))

        if len(self.tree) >= self.buffer_size and self._states_mng and old_data is not None:
            self._states.remove(old_data["obs_idx"])
            self._states.remove(old_data["next_obs_idx"])


    def _sample_list(self, beta: float = 1, **kwargs) -> list[Experience]:
        """The method return samples randomly without duplicates"""
        if len(self.tree) < self.batch_size:
            return []

        samples = []
        experiences = []
        indices = []
        weights = self.__default_weights.copy()
        priorities = []

        for k in range(self.batch_size):
            r = self._rng.random()
            data, priority, index = self.tree.find(r)
            priorities.append(priority)
            weights[k] = pow(weights[k] / priority, beta) if priority > 1e-16 else 0
            indices.append(index)
            self.priority_update([index], [0])  # To avoid duplicating
            samples.append(data)

        self.priority_update(indices, priorities)  # Revert priorities
        weights = weights / max(weights)

        for experience, weight, index in zip(samples, weights, indices):
            experience.weight = weight
            experience.index = index
            if self._states_mng:
                experience.obs = self._states.get(experience.obs_idx)
                experience.next_obs = self._states.get(experience.next_obs_idx)
            experiences.append(experience)

        return experiences


    def sample(self, beta: float = 0.5) -> Union[dict[str, list], None]:
        all_experiences = defaultdict(lambda: [])
        sampled_exp = self._sample_list(beta=beta)
        if len(sampled_exp) == 0:
            return None

        for exp in sampled_exp:
            for key in exp.__dict__.keys():
                if self._states_mng and (key == "obs" or key == "next_obs"):
                    value = self._states.get(getattr(exp, key + "_idx"))
                else:
                    value = getattr(exp, key)
                all_experiences[key].append(value)
        return all_experiences


    def priority_update(self, indices: Sequence[int], priorities: list) -> None:
        """Updates prioprities for elements on provided indices."""
        for i, p in zip(indices, priorities):
            self.tree.weight_update(i, math.pow(p, self.alpha))


    def reset_alpha(self, alpha: float):
        """Resets the alpha wegith (p^alpha)"""
        tree_len = len(self.tree)
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [pow(self.tree[i], -old_alpha) for i in range(tree_len)]
        self.priority_update(range(tree_len), priorities)


class SumTree(object):
    """SumTree

    A binary tree where each level contains sum of its children nodes.
    """

    def __init__(self, leafs_num: int):
        """
        Parameters:
            leafs_num (int): Number of leaf nodes.

        """
        self.leafs_num = int(leafs_num)
        self.tree_height = math.ceil(math.log(leafs_num, 2)) + 1
        self.leaf_offset = 2 ** (self.tree_height - 1) - 1
        self.tree_size = 2**self.tree_height - 1
        self.tree = np.zeros(self.tree_size)
        self.data: list[dict | None] = [None] * self.leafs_num
        self.size = 0
        self.cursor = 0


    def __len__(self) -> int:
        return self.size


    def __getitem__(self, index) -> float:
        if isinstance(index, slice):
            return self.tree[self.leaf_offset :][index]
        return self.tree[self.leaf_offset + index]


    def insert(self, data, weight) -> Any:
        """Returns `data` of element that was removed"""
        index = self.cursor
        self.cursor = (self.cursor + 1) % self.leafs_num
        self.size = min(self.size + 1, self.leafs_num)

        old_data = copy.copy(self.data[index])
        self.data[index] = data
        self.weight_update(index, weight)
        return old_data


    def weight_update(self, index, weight) -> None:
        "Updates weight of a leaf node (by index)"
        tree_index = self.leaf_offset + index
        diff = weight - self.tree[tree_index]
        self._tree_update(tree_index, diff)


    def _tree_update(self, tindex, diff):
        while tindex >= 0:
            self.tree[tindex] += diff
            tindex = (tindex - 1) // 2


    def find(self, weight) -> tuple[Any, float, int]:
        """Returns (data, weight, index)"""
        assert 0 <= weight <= 1, "Expecting weight to be sampling weight [0, 1]"
        return self._find(weight * self.tree[0], 0)


    def _find(self, weight, index) -> tuple[Any, float, int]:
        """Recursively finds a data by the weight.

        Returns:
            Tuple of (data, weight, index) where the index is in the leaf layer.

        """
        if self.leaf_offset <= index:  # Moved to the leaf layer
            return (
                self.data[min(index - self.leaf_offset, self.leafs_num - 1)],
                self.tree[index],
                index - self.leaf_offset,
            )

        left_idx = 2 * index + 1
        left_weight = self.tree[left_idx]

        if weight <= left_weight:
            return self._find(weight, left_idx)
        else:
            return self._find(weight - left_weight, 2 * (index + 1))


    def get_n_first_nodes(self, n):
        return self.data[:n]
    

class RolloutBuffer(object):
    type = "Rollout"

    def __init__(self, batch_size: int, buffer_size=int(1e6), **kwargs):
        """
        A buffer that keeps and returns data in order.
        Commonly used with on-policy methods such as PPO.

        Parameters:
            batch_size (int): Maximum number of samples to return in each batch.
            buffer_size (int): Number of samples to store in the buffer.

        Keyword Arguments:
            compress_state (bool): Default False. Whether to manage memory used by states.
                Useful when states are "large" and frequently visited. Typical use case is
                dealing with images.

        """
        super().__init__()
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.data: Deque = deque()

        self._states_mng = kwargs.get("compress_state", False)
        self._states = ReferenceBuffer(buffer_size + 20)


    def __len__(self) -> int:
        return len(self.data)


    def __eq__(self, o: object) -> bool:
        return super().__eq__(o)


    def clear(self):
        self.data.clear()


    def push(self, **kwargs):
        if self._states_mng:
            kwargs["state_idx"] = self._states.add(kwargs.pop("state"))
            if "next_state" in kwargs:
                kwargs["next_state_idx"] = self._states.add(kwargs.pop("next_state", "None"))
        self.data.append(Experience(**kwargs))

        if len(self.data) > self.buffer_size:
            drop_exp = self.data.popleft()
            if self._states_mng:
                self._states.remove(drop_exp.state_idx)
                self._states.remove(drop_exp.next_state_idx)


    def sample(self, batch_size: Union[int,None] = None) -> Iterator[dict[str, list]]:
        """
        Samples the whole buffer. Iterates all gathered data.
        Note that sampling doesn't clear the buffer.

        Returns:
            A generator that iterates over all rolled-out samples.
        """
        data = self.data.copy()
        batch_size = batch_size if batch_size is not None else self.batch_size

        while len(data):
            batch_size = min(batch_size, len(data))
            all_experiences = defaultdict(lambda: [])
            for _ in range(batch_size):
                sample = data.popleft()
                for key, value in sample.get_dict().items():
                    all_experiences[key].append(value)

            yield all_experiences
    

    def all_samples(self):
        all_experiences = defaultdict(lambda: [])
        for sample in self.data:
            for key, value in sample.get_dict().items():
                all_experiences[key].append(value)

        return all_experiences