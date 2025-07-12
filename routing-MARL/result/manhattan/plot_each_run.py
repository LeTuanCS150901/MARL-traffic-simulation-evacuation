import re  
import numpy as np  
import matplotlib.pyplot as plt  

# Step 1: Read data from the file  
# with open('reward_risk_model_50eps/training_log_20250505_230819.txt', 'r') as file:  
# with open('reward_risk_model_50eps/training_log_20250506_180354.txt', 'r') as file: 
# with open('reward_risk_model_100eps/training_log_20250507_011328.txt', 'r') as file:
# with open('reward_risk_model_100eps/training_log_20250509_175839.txt', 'r') as file:
# with open('reward_risk_model_100eps/training_log_20250509_213426.txt', 'r') as file:
# with open('reward_risk_model_100eps/training_log_20250513_003716.txt', 'r') as file: # good model: epch 65,idex
# with open('reward_risk_model_150eps/training_log_20250513_183102.txt', 'r') as file: # good model
# with open('reward_risk_model_150eps/training_log_20250513_201136.txt', 'r') as file: 
# with open('reward_risk_model_150eps/training_log_20250514_012205.txt', 'r') as file: # good model episode 179 --index 2
# with open('reward_risk_model_150eps/training_log_20250514_022117.txt', 'r') as file: # good model episode 180 --index 3
# with open('reward_risk_model_150eps/training_log_20250514_142657.txt', 'r') as file:
# with open('reward_3_150eps/training_log_20250517_195756.txt', 'r') as file:# good model R1 episode 150 --index 2
# with open('reward_3_150eps/training_log_20250517_235753.txt', 'r') as file:# good model R1 episode 150 --index 2
with open('training_log_20250711_132424.txt', 'r') as file: # good model R1, all cluster agents, congestions epch 300 index 1
# with open('training_log_20250625_072525.txt', 'r') as file: # good model R1, all cluster agents, congestions epch 300 index 1

    data = file.read()  

# Step 2: Split by episodes first, then parse each episode  
episode_pattern = r"Episode (\d+):(.*?)(?=Episode \d+:|$)"  
agent_pattern = r"Agent ([\w#]+):\s+Loss: \[(.*?)\]\s+Reward: \[(.*?)\]"  

# Initialize data structures  
episodes = []  
agents = set()  
agent_losses = {}  
agent_rewards = {}  

def parse_float_list(s):  
    """Parse list of floats, handling special values"""  
    vals = []  
    for x in s.split(','):  
        x = x.strip()  
        if not x:  
            continue  
        if x == 'inf' or re.match(r'1e\+\d+', x):  
            vals.append(np.nan)  
        else:  
            try:  
                vals.append(float(x))  
            except:  
                vals.append(np.nan)  
    return vals  

# Find all episodes  
episode_matches = re.findall(episode_pattern, data, re.DOTALL)  
print(f"Found {len(episode_matches)} episodes")  

# Process each episode  
for ep_num_str, ep_content in episode_matches:  
    ep_num = int(ep_num_str)  
    episodes.append(ep_num)  
    
    # Find all agents in this episode  
    agent_matches = re.findall(agent_pattern, ep_content, re.DOTALL)  
    
    for agent, loss_str, reward_str in agent_matches:  
        if agent not in agents:  
            agents.add(agent)  
            agent_losses[agent] = {}  
            agent_rewards[agent] = {}  
        
        agent_losses[agent][ep_num] = parse_float_list(loss_str)  
        agent_rewards[agent][ep_num] = parse_float_list(reward_str)  

# Sort episodes and convert agents to list  
episodes.sort()  
agents = list(agents)  
print(f"Found {len(agents)} agents: {agents}")  

# Calculate mean values per episode for each agent  
mean_losses = {}  
mean_rewards = {}  

for agent in agents:  
    mean_losses[agent] = []  
    mean_rewards[agent] = []  
    
    for ep in episodes:  
        if ep in agent_losses[agent]:  
            mean_losses[agent].append(np.nanmean(agent_losses[agent][ep]))  
        else:  
            mean_losses[agent].append(np.nan)  
            
        if ep in agent_rewards[agent]:  
            mean_rewards[agent].append(np.nanmean(agent_rewards[agent][ep]))  
        else:  
            mean_rewards[agent].append(np.nan)  

def moving_average(data, window_size):
    """
    Calculate the moving average of a NumPy array.

    Parameters:
    data (np.ndarray): Input array of numerical data.
    window_size (int): Number of elements to include in the moving average.

    Returns:
    np.ndarray: Array of moving averages.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a NumPy array.")
    if window_size <= 0:
        raise ValueError("Window size must be a positive integer.")
    if window_size > len(data):
        raise ValueError("Window size must be less than or equal to the length of the data.")
    
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Plot  
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']  
plt.figure(figsize=(14, 6))  

print("--------------------------- mean loss", [mean_losses[ag][-1] for ag in agents])
# Loss plot  
plt.subplot(1, 2, 1)  
for i, agent in enumerate(agents):  
    color = ""
    # if agent == "gneE19":
    #     color = "blue"
    # elif agent == "gneE21":
    #     color = "red"
  
    color = colors[i % len(colors)]  
    plt.plot(episodes, mean_losses[agent], label=f'Agent {agent}',   
             marker='o', color=color)  
# plt.xlim((0,75))
# plt.ylim((0,10))
plt.xlabel('Episode')  
plt.ylabel('Average Loss')  
plt.title('Average Loss per Episode')  
plt.legend()  
plt.grid(True)
plt.tight_layout()  


#Reward plot  
plt.subplot(1, 2, 2)  
for i, agent in enumerate(agents):  
    color = colors[i % len(colors)]  
    data = moving_average(np.array(mean_rewards[agent]), 1)
    # print(len(data))
    # print(data)
    # plt.ylim([0,1])
    plt.plot([i for i in range(len(data))], data, label=f'Agent {agent}',   
             marker='o', color=color) 
    
plt.xlabel('Episode')  
plt.ylabel('Average Reward')  
plt.title('Average Reward per Episode')  
plt.legend()  
plt.grid(True)  
plt.tight_layout()  
# plt.savefig('training_progress.png', dpi=300)  
plt.show()

# plt.savefig("Manhattan_net.pdf", dpi=300)