import random
from collections import deque

class JointExperienceReplayBuffer:
    """
    Stores joint transitions: all vehicles' experiences from the same timestep.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, joint_transition):
        """
        Store a joint transition.
        
        Args:
            joint_transition: dict with keys:
                - 'timestep': int
                - 'vehicles': list of (vehicle_id, obs, action, next_obs, done)
                - 'global_reward': float (shared reward for all vehicles)
        """
        self.buffer.append(joint_transition)
    
    def sample(self, batch_size):
        """Sample batch_size joint transitions"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)