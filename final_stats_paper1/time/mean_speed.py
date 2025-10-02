import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

def compare_algorithms(file_dict):
    """
    Compare multiple SUMO summary files from different algorithms
    
    Args:
        file_dict: Dictionary with algorithm names as keys and file paths as values
                   Example: {'Algorithm A': 'summary_a.xml', 'Algorithm B': 'summary_b.xml'}
    """
    # Color palette for different algorithms
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', 
              '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#d35400']

      # Fallback colors if algorithm name not in dict
    # colors = ['#2E86DE', '#EE5A6F', '#26DE81', '#FD79A8', 
    #                  '#A29BFE', '#FD7272', '#54A0FF', '#48DBFB']
    
    # Store data for all algorithms
    all_data = {}
    
    # Parse all XML files
    for algo_name, xml_file in file_dict.items():
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        times = []
        mean_speeds = []
        running = []
        halting = []
        
        for step in root.findall('step'):
            times.append(float(step.get('time')))
            mean_speed = float(step.get('meanSpeed'))
            mean_speeds.append(max(0, mean_speed))
            running.append(int(step.get('running')))
            halting.append(int(step.get('halting')))
        
        all_data[algo_name] = {
            'times': times,
            'mean_speeds': mean_speeds,
            'running': running,
            'halting': halting
        }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Mean Speed Comparison
    for idx, (algo_name, data) in enumerate(all_data.items()):
        color = colors[idx % len(colors)]
        
        # Area chart with gradient effect
        ax1.fill_between(data['times'], data['mean_speeds'], 
                         alpha=0.3, color=color)
        ax1.plot(data['times'], data['mean_speeds'], 
                color=color, linewidth=2.5, label=algo_name, alpha=0.8)
        # ax1.scatter(data['times'], data['mean_speeds'], 
        #            color=color, s=20, alpha=0.5, zorder=5)
    
    ax1.set_ylabel('Mean Speed (m/s)', fontsize=13, fontweight='bold')
    ax1.set_title('Mean Speed Over Time - Algorithm Comparison', 
                  fontsize=15, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.set_ylim(bottom=0)
    
    # Plot 2: Running and Halting Vehicles
    x_offset = 0
    bar_width = 0.35
    
    for idx, (algo_name, data) in enumerate(all_data.items()):
        color = colors[idx % len(colors)]
        
        # Running vehicles (solid line with area)
        ax2.fill_between(data['times'], data['running'], 
                         alpha=0.25, color=color)
        ax2.plot(data['times'], data['running'], 
                color=color, linewidth=2.5, label=f'{algo_name} (Running)', 
                alpha=0.8, linestyle='-')
        
        # Halting vehicles (dashed line with area)
        # ax2.fill_between(data['times'], data['halting'], 
        #                  alpha=0.15, color=color)
        # ax2.plot(data['times'], data['halting'], 
        #         color=color, linewidth=2, label=f'{algo_name} (Halting)', 
        #         alpha=0.7, linestyle='--')
    
    ax2.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Number of Vehicles', fontsize=13, fontweight='bold')
    ax2.set_title('Running and Halting Vehicles Over Time - Algorithm Comparison', 
                  fontsize=15, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax2.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print_summary_statistics(all_data)

def print_summary_statistics(all_data):
    """Print comparison statistics for all algorithms"""
    print(f"\n{'='*80}")
    print(f"{'ALGORITHM COMPARISON STATISTICS':^80}")
    print(f"{'='*80}\n")
    
    for algo_name, data in all_data.items():
        mean_speeds = [s for s in data['mean_speeds'] if s > 0]
        running = data['running']
        halting = data['halting']
        
        if mean_speeds:
            avg_speed = np.mean(mean_speeds)
            max_speed = np.max(mean_speeds)
            min_speed = np.min(mean_speeds)
        else:
            avg_speed = max_speed = min_speed = 0
        
        avg_running = np.mean(running)
        max_running = np.max(running)
        avg_halting = np.mean(halting)
        max_halting = np.max(halting)
        halting_pct = (avg_halting / avg_running * 100) if avg_running > 0 else 0
        
        print(f"{algo_name}:")
        print(f"  Mean Speed:")
        print(f"    - Average: {avg_speed:.2f} m/s")
        print(f"    - Maximum: {max_speed:.2f} m/s")
        print(f"    - Minimum: {min_speed:.2f} m/s")
        print(f"  Vehicles:")
        print(f"    - Avg Running: {avg_running:.2f}")
        print(f"    - Max Running: {max_running}")
        print(f"    - Avg Halting: {avg_halting:.2f}")
        print(f"    - Max Halting: {max_halting}")
        print(f"    - Halting %: {halting_pct:.2f}%")
        print(f"{'-'*80}\n")
    
    print(f"{'='*80}\n")

# Usage Example
file_dict = {
    'Dynamic A*': 'dynamic_a/summary.xml',
    'Static A*': 'static_a/summary.xml',
    'SUMO DUE': 'sumo/summary_499.xml',
    'C-logit': 'hassan/summary_499.xml',
    "MARL": 'RL/summary.xml'
}

compare_algorithms(file_dict)