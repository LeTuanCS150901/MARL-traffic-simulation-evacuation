"""  
RL Training Analysis Tool  

Analyzes multiple RL training logs, computes mean rewards per episode across runs,  
and plots them with confidence intervals.  
"""  

import os  
import re  
import glob  
import numpy as np  
import matplotlib.pyplot as plt  
from collections import defaultdict  
from typing import Dict, List, Tuple, Any  


###########################################  
# CONFIGURATION  
###########################################  

# Folder containing RL training log files  
LOG_DIR = "reward_risk_model_50eps"  # CHANGE THIS!  

# Pattern for log files (e.g., "training_log_*.txt")  
LOG_PATTERN = "training_log_*.txt"  


###########################################  
# DATA EXTRACTION  
###########################################  

def extract_rewards_from_file(filepath: str) -> Dict[str, List[float]]:  
    """  
    Extract mean episode rewards for each agent from a single log file.  
    
    Args:  
        filepath: Path to the log file  
        
    Returns:  
        Dictionary mapping agent names to lists of episode mean rewards  
    """  
    try:  
        with open(filepath, 'r') as f:  
            content = f.read()  
    except Exception as e:  
        print(f"Error reading {os.path.basename(filepath)}: {e}")  
        return {}  
        
    # Track rewards per episode for each agent  
    episode_data = {}  
    current_episode = None  
    current_agent = None  
    
    # Process file line by line  
    for line in content.splitlines():  
        line = line.strip()  
        
        # Check for episode marker  
        if line.startswith("Episode"):  
            match = re.match(r"Episode (\d+):", line)  
            if match:  
                current_episode = int(match.group(1))  
                episode_data[current_episode] = {}  
        
        # Check for agent marker  
        elif "Agent " in line:  
            match = re.match(r"Agent (\w+):", line)  
            if match and current_episode is not None:  
                current_agent = match.group(1)  
                episode_data[current_episode][current_agent] = {}  
        
        # Extract rewards  
        elif "Reward: [" in line and current_agent and current_episode is not None:  
            match = re.match(r"Reward: \[(.*)\]", line)  
            if match:  
                rewards_str = match.group(1)  
                try:  
                    # Parse reward values, handling potential invalid values  
                    rewards = []  
                    for val in rewards_str.split(','):  
                        val = val.strip()  
                        if val.lower() in ['inf', '-inf', 'nan', '']:  
                            continue  
                        try:  
                            rewards.append(float(val))  
                        except ValueError:  
                            continue  
                    
                    if rewards:  # Only store if we have valid rewards  
                        episode_data[current_episode][current_agent]['rewards'] = rewards  
                except Exception as e:  
                    print(f"Error parsing rewards in {os.path.basename(filepath)}, "  
                         f"episode {current_episode}, agent {current_agent}: {e}")  
    
    # Compute mean reward per episode for each agent  
    agent_rewards = defaultdict(list)  
    
    for ep_num in sorted(episode_data.keys()):  
        for agent, data in episode_data[ep_num].items():  
            if 'rewards' in data and data['rewards']:  
                mean_reward = np.mean(data['rewards'])  
                agent_rewards[agent].append(mean_reward)  
    
    return dict(agent_rewards)  


def load_all_logs(log_dir: str, pattern: str) -> Tuple[Dict[str, List[List[float]]], int]:  
    """  
    Load all log files and aggregate their data.  
    
    Args:  
        log_dir: Directory containing log files  
        pattern: Glob pattern to match log files  
        
    Returns:  
        Tuple containing:  
        - Dictionary mapping agent names to lists of reward sequences (one per run)  
        - Maximum number of episodes across all runs  
    """  
    log_files = sorted(glob.glob(os.path.join(log_dir, pattern)))  
    if not log_files:  
        print(f"No files matching '{pattern}' found in '{log_dir}'")  
        return {}, 0  
    
    print(f"Processing {len(log_files)} log files...")  
    
    # Structure: { agent_name: [ [run1_rewards], [run2_rewards], ... ] }  
    all_run_data = defaultdict(list)  
    max_episodes = 0  
    
    for i, filepath in enumerate(log_files, 1):  
        print(f"  [{i}/{len(log_files)}] {os.path.basename(filepath)}")  
        run_data = extract_rewards_from_file(filepath)  
        
        for agent, rewards in run_data.items():  
            all_run_data[agent].append(rewards)  
            max_episodes = max(max_episodes, len(rewards))  
    
    return dict(all_run_data), max_episodes  


###########################################  
# STATISTICAL ANALYSIS  
###########################################  

def compute_statistics(run_data: Dict[str, List[List[float]]], max_episodes: int) -> Dict[str, Dict[str, np.ndarray]]:  
    """  
    Compute mean and confidence intervals for each agent across all runs.  
    
    Args:  
        run_data: Dictionary mapping agent names to lists of reward sequences  
        max_episodes: Maximum number of episodes to align data  
        
    Returns:  
        Dictionary with statistics for each agent  
    """  
    agent_stats = {}  
    
    for agent, runs in run_data.items():  
        # Skip if no runs with data  
        if not runs:  
            continue  
        
        # Create array and pad shorter runs with NaN  
        aligned_runs = np.full((len(runs), max_episodes), np.nan)  
        for i, run in enumerate(runs):  
            aligned_runs[i, :len(run)] = run  
        
        # Calculate mean across runs for each episode  
        mean_rewards = np.nanmean(aligned_runs, axis=0)  
        
        # Calculate standard deviation and count valid runs per episode  
        std_rewards = np.nanstd(aligned_runs, axis=0)  
        valid_counts = np.sum(~np.isnan(aligned_runs), axis=0)  
        
        # Compute 95% confidence intervals using normal approximation  
        # We use z=1.96 for 95% CI rather than t-distribution to avoid scipy dependency  
        ci_width = np.zeros_like(mean_rewards)  
        for i in range(max_episodes):  
            if valid_counts[i] >= 2:  # Need at least 2 samples for CI  
                standard_error = std_rewards[i] / np.sqrt(valid_counts[i])  
                # 1.96 is approx. z-score for 95% CI  
                ci_width[i] = 1.96 * standard_error  
        
        agent_stats[agent] = {  
            'mean': mean_rewards,  
            'ci_lower': mean_rewards - ci_width,  
            'ci_upper': mean_rewards + ci_width,  
            'num_runs': len(runs),  
            'valid_counts': valid_counts  # Store for reference  
        }  
    
    return agent_stats  

###########################################  
# VISUALIZATION  
###########################################  

def plot_learning_curves(stats: Dict[str, Dict[str, Any]], max_episodes: int) -> None:  
    """  
    Plot learning curves with confidence intervals.  
    
    Args:  
        stats: Statistics for each agent  
        max_episodes: Maximum number of episodes  
    """  
    plt.figure(figsize=(12, 7))  
    episodes = np.arange(1, max_episodes + 1)  
    
    # Default colors for consistency  
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',   
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  
    
    # Create separate plot entries for means and CIs in the legend  
    for i, (agent, data) in enumerate(sorted(stats.items())):  
        color = colors[i % len(colors)]  
        
        # Plot mean reward line  
        plt.plot(episodes, data['mean'],   
                 label=f"Agent {agent} (N={data['num_runs']})",  
                 color=color, linewidth=2)  
        
        # Plot confidence interval if we have multiple runs  
        if data['num_runs'] > 1:  
            # Create filled area for CI  
            plt.fill_between(episodes,   
                             data['ci_lower'],   
                             data['ci_upper'],  
                             color=color, alpha=0.2,  
                             label=f"95% CI ({agent})")  
    
    # Formatting  
    plt.xlabel('Episode Number', fontsize=12)  
    plt.ylabel('Mean Reward per Episode', fontsize=12)  
    plt.title('Learning Curve: Mean Reward per Episode with 95% Confidence Interval', fontsize=14)  
    plt.grid(True, alpha=0.3, linestyle='--')  
    
    # Create legend with custom organization  
    handles, labels = plt.gca().get_legend_handles_labels()  
    
    # Separate means and CIs in legend  
    means_handles = []  
    means_labels = []  
    ci_handles = []  
    ci_labels = []  
    
    for i, label in enumerate(labels):  
        if "CI" in label:  
            ci_handles.append(handles[i])  
            ci_labels.append(label)  
        else:  
            means_handles.append(handles[i])  
            means_labels.append(label)  
    
    # Add legend with both parts  
    plt.legend(means_handles + ci_handles, means_labels + ci_labels,   
               fontsize=10, loc='best')  
    
    plt.tight_layout()  
    
    # Save figure  
    plt.savefig('rl_learning_curve.png', dpi=300)  
    print("Plot saved as 'rl_learning_curve.png'")  
    
    # Show plot  
    plt.show()  

###########################################  
# MAIN  
###########################################  

def main() -> None:  
    """Main execution function"""  
    # Load and process data  
    run_data, max_episodes = load_all_logs(LOG_DIR, LOG_PATTERN)  
    
    if not run_data:  
        print("No valid data found. Exiting.")  
        return  
    
    # Compute statistics  
    stats = compute_statistics(run_data, max_episodes)  
    
    # Generate visualization  
    plot_learning_curves(stats, max_episodes)  
    
    print("Analysis complete!")  


if __name__ == "__main__":  
    main()