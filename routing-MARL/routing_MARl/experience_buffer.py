import random 
import numpy as np
from logs.logger import logger  # import our custom logger

class ExperienceReplayBuffer:
    """  
    Experience Replay Buffer for storing and sampling experiences  
    """ 
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, experience,exp=False):
        """  
        Store experience tuple (o_i, a_i, o'_i, r_i, # fake mews mean_action)  
        """  
        if exp: 
            o_i, a_i, op_i, r_i = experience
            experience = o_i, a_i, op_i, r_i+300 ############ make reward >0 for compare. 750=max simulation time
            

        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            
        assert isinstance(experience[3], (int, float)), \
        f"Reward must be scalar, got {type(experience[3])}: {experience[3]}"
        
        # logger.error(f"Current experience {experience[3]}")
        self.buffer.append(experience)

    def sample(self, batch_size):
        """  
        Sample a batch of experiences  
        """  
        return random.sample(self.buffer, min(len(self.buffer), batch_size))