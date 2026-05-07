import random

import numpy as np


class JointExperienceReplayBuffer:
		"""
		Prioritized Joint Experience Replay Buffer.

		Same main interface as your old buffer:
			self.replay_buffer.push(joint_transition)
			batch = self.replay_buffer.sample(batch_size)
			len(self.replay_buffer)

		Extra optional method:
			self.replay_buffer.update_priorities(td_errors)

		Main idea:
			- high TD-error transitions are sampled more often
			- some uniform sampling is kept for stability
			- some recent sampling is kept for adaptation
		"""

		def __init__(
			self,
			capacity,
			alpha=0.6,
			priority_eps=1e-6,
			priority_ratio=0.70,
			uniform_ratio=0.20,
			recent_ratio=0.10,
			recent_window=None,
			skip_empty=True,
		):
			self.capacity = capacity

			self.buffer = []
			self.priorities = np.zeros(capacity, dtype=np.float32)

			self.position = 0
			self.size = 0

			# priority strength
			# alpha = 0 means uniform replay
			# alpha = 1 means strong prioritization
			self.alpha = alpha
			self.priority_eps = priority_eps

			# sampling mixture
			total = priority_ratio + uniform_ratio + recent_ratio
			self.priority_ratio = priority_ratio / total
			self.uniform_ratio = uniform_ratio / total
			self.recent_ratio = recent_ratio / total

			if recent_window is None:
				recent_window = max(100, capacity // 10)

			self.recent_window = recent_window

			# This stores indices of the last sampled batch.
			# It allows update_priorities(td_errors) without changing sample output.
			self.last_sampled_indices = []

			# If True, do not store invalid transitions with no vehicles.
			self.skip_empty = skip_empty

		def __len__(self):
			return self.size

		def push(self, joint_transition):
			"""
			Store one joint transition.

			joint_transition should contain:
				- 'vehicles'
				- 'global_reward'
				- other optional fields
			"""

			if self.skip_empty:
				vehicles = joint_transition.get("vehicles", [])
				if vehicles is None or len(vehicles) == 0:
					return

			# New samples should be seen at least once.
			if self.size > 0:
				max_priority = float(np.max(self.priorities[:self.size]))
				if max_priority <= 0:
					max_priority = 1.0
			else:
				max_priority = 1.0

			if self.size < self.capacity:
				self.buffer.append(joint_transition)
				idx = self.size
				self.size += 1
			else:
				idx = self.position
				self.buffer[idx] = joint_transition

			self.priorities[idx] = max_priority
			self.position = (self.position + 1) % self.capacity

		def _priority_probs(self):
			"""
			Compute sampling probabilities from priorities.
			"""

			priorities = self.priorities[:self.size].astype(np.float64)
			priorities = np.maximum(priorities, self.priority_eps)

			scaled = priorities ** self.alpha
			total = np.sum(scaled)

			if total <= 0 or not np.isfinite(total):
				return np.ones(self.size, dtype=np.float64) / self.size

			return scaled / total

		def _recent_indices(self):
			"""
			Return indices of the most recent transitions.
			"""

			if self.size == 0:
				return []

			n_recent = min(self.size, self.recent_window)

			indices = []
			for i in range(n_recent):
				idx = (self.position - 1 - i) % self.capacity
				if idx < self.size:
					indices.append(idx)

			return indices

		def sample(self, batch_size):
			"""
			Sample a batch.

			Return format is exactly the same as your old buffer:
				batch = list of joint_transition
			"""

			if self.size == 0:
				self.last_sampled_indices = []
				return []

			batch_size = min(batch_size, self.size)

			n_priority = int(batch_size * self.priority_ratio)
			n_recent = int(batch_size * self.recent_ratio)
			n_uniform = batch_size - n_priority - n_recent

			sampled_indices = []

			# 1. Prioritized samples
			if n_priority > 0:
				probs = self._priority_probs()

				indices = np.random.choice(
					self.size,
					size=n_priority,
					replace=True,
					p=probs,
				)

				sampled_indices.extend([int(i) for i in indices])

			# 2. Uniform samples
			if n_uniform > 0:
				n = min(n_uniform, self.size)
				indices = random.sample(range(self.size), n)
				sampled_indices.extend(indices)

			# 3. Recent samples
			if n_recent > 0:
				recent_indices = self._recent_indices()

				if len(recent_indices) > 0:
					if len(recent_indices) >= n_recent:
						indices = random.sample(recent_indices, n_recent)
					else:
						indices = random.choices(recent_indices, k=n_recent)

					sampled_indices.extend(indices)

			# If something went wrong or batch is too small, fill with uniform samples.
			while len(sampled_indices) < batch_size:
				sampled_indices.append(random.randrange(self.size))

			# Shuffle to avoid ordered batch structure.
			random.shuffle(sampled_indices)

			self.last_sampled_indices = sampled_indices

			batch = [self.buffer[i] for i in sampled_indices]

			return batch

		def update_priorities(self, td_errors):
			"""
			Update priorities using TD errors from the last sampled batch.

			Args:
				td_errors:
					array-like of shape [batch_size]
					usually abs(q_total_current - target)
			"""

			if len(self.last_sampled_indices) == 0:
				return

			td_errors = np.asarray(td_errors, dtype=np.float32).reshape(-1)

			n = min(len(self.last_sampled_indices), len(td_errors))

			for idx, td_error in zip(self.last_sampled_indices[:n], td_errors[:n]):
				priority = abs(float(td_error)) + self.priority_eps

				if np.isfinite(priority):
					self.priorities[idx] = priority

		def stats(self):
			if self.size == 0:
				return {
					"size": 0,
					"capacity": self.capacity,
					"max_priority": 0.0,
					"mean_priority": 0.0,
				}

			return {
				"size": self.size,
				"capacity": self.capacity,
				"max_priority": float(np.max(self.priorities[:self.size])),
				"mean_priority": float(np.mean(self.priorities[:self.size])),
				"priority_ratio": self.priority_ratio,
				"uniform_ratio": self.uniform_ratio,
				"recent_ratio": self.recent_ratio,
			}