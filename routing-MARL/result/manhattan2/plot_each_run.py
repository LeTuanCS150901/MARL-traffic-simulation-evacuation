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
with open('training_log_20250606_235805.txt', 'r') as file: # good model R1, all cluster agents, congestions epch 300 index 1
# with open('reward_3_150eps/training_log_20250519_183607.txt', 'r') as file:
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

# plt.subplot(1, 2, 2)
# avg = 0.0
# for i, agent in enumerate(agents):  
#     avg += np.array(mean_rewards[agent])

# avg = avg / len(agents)

# plt.plot(episodes, avg, label=f'All agents',   
#              marker='o', color=color)  

# print("---------------------------------")
# print(mean_rewards[agents[0]])
# a = np.array(mean_rewards[agents[0]])
# b = np.array(mean_rewards[agents[1]])
# plt.plot(episodes,0.5*(a+b - np.abs(a-b)), label=f'min (R(blue),R(red))',   
#              marker='o', color='green') 
# plt.plot(episodes,0.5*(a+b + np.abs(a-b)), label=f'max (R(blue),R(red))',   
#              marker='o', color='black') 
# plt.ylim((0,1))
plt.xlabel('Episode')  
plt.ylabel('Average Reward')  
plt.title('Average Reward per Episode')  
plt.legend()  
plt.grid(True)  

plt.tight_layout()  
plt.savefig('training_progress.png', dpi=300)  
plt.show()



# import re  
# import numpy as np  
# import matplotlib.pyplot as plt  

# # Step 1: Read data from the file  
# with open('reward_risk_model_50eps/training_log_20250505_154245.txt', 'r') as file:  
#     data = file.read()  

# # Step 2: Extract data  
# pattern = r"Episode (\d+):\s+Agent gneE19:\s+Loss: \[(.*?)\]\s+Reward: \[(.*?)\]\s+Agent gneE21:\s+Loss: \[(.*?)\]\s+Reward: \[(.*?)\]"  
# matches = re.findall(pattern, data, re.DOTALL)  

# episodes = []  
# loss_19 = []  
# reward_19 = []  
# loss_21 = []  
# reward_21 = []  

# def parse_float_list(s):  
#     # Convert string list to float list; also convert 'inf' to np.nan  
#     vals = []  
#     for x in s.split(','):  
#         x = x.strip()  
#         print(x)
#         if x == 'inf':  
#             vals.append(np.nan)  
#         elif re.match(r'1e\+\d+', x):  # handle large rewards as inf-like (e.g. 1e+26)  
#             vals.append(np.nan)  
#         else:  
#             try:  
#                 vals.append(float(x))  
#             except:  
#                 vals.append(np.nan)  
#     return vals  

# for ep, l19, r19, l21, r21 in matches:  
#     episodes.append(int(ep))  
#     loss_19.append(parse_float_list(l19))  
#     reward_19.append(parse_float_list(r19))  
#     loss_21.append(parse_float_list(l21))  
#     reward_21.append(parse_float_list(r21))  
# print(loss_19)
# print(loss_21)
# # Step 3: Align data length for plotting average per episode  
# mean_loss_19 =[np.nanmean(x) for x in loss_19]  
# mean_loss_21 = [np.nanmean(x) for x in loss_21]  
# mean_reward_19 =  [np.nanmean(x) for x in reward_19]  
# mean_reward_21 =  [np.nanmean(x) for x in reward_21]  
# print(reward_19)

# # Step 4: Plot  
# plt.figure(figsize=(14, 6))  

# plt.subplot(1, 2, 1)  
# plt.plot(episodes, mean_loss_19, label='Agent gneE19 Loss', marker='o')  
# plt.plot(episodes, mean_loss_21, label='Agent gneE21 Loss', marker='o')  
# plt.xlabel('Episode')  
# plt.ylabel('Average Loss')  
# plt.title('Average Loss per Episode')  
# plt.legend()  
# plt.grid(True)  

# print("Mean :", np.array(mean_reward_19))
# indices = np.where(np.array(mean_reward_19)  >= 0)[0] 
# print("Mean ind:", indices)
# values = np.array(mean_reward_19)[indices]
# t_val = (1/values) - 1
# print("Mean t:", t_val)
# plt.subplot(1, 2, 2)  
# # plt.plot(indices, 500*t_val * (0.1*np.sin(np.sqrt(2)*indices + t_val)+2*np.exp(-indices/10)), label='Agent gneE19 Travel time', linestyle='--')
# # plt.scatter(indices, 500*t_val* (0.1*np.sin(np.sqrt(2)*indices + t_val)+2*np.exp(-indices/10)), label='Agent gneE19 Travel time', marker='x', color='red')  
# plt.plot(episodes, mean_reward_21, label='Agent gneE21 Reward', marker='o')  
# plt.plot(episodes, mean_reward_19, label='Agent gneE19 Reward', marker='o')  

# plt.xlabel('Episode')  
# plt.ylabel('Average reward')  
# #plt.ylim([0, 1])
# plt.title('Average reward per Episode')  
# plt.legend()  
# plt.grid(True)  

# plt.tight_layout()  
# plt.show()