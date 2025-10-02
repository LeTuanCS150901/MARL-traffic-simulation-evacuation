import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

def compare_travel_times_boxplot(file_dict):
    """
    Compare mean travel time distributions across algorithms using box plot
    
    Args:
        file_dict: Dictionary with algorithm names as keys and file paths as values
    """
    all_stats = {}
    
    # Collect data from all files
    for algo_name, xml_file in file_dict.items():
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        times = []
        mean_travel_times = []
        arrived = []
        
        for step in root.findall('step'):
            time = float(step.get('time'))
            travel_time = float(step.get('meanTravelTime'))
            num_arrived = int(step.get('arrived'))
            
            times.append(time)
            arrived.append(num_arrived)
            
            if travel_time > 0:
                mean_travel_times.append(travel_time)
        
        # Calculate evacuation time
        evacuation_time = 0
        for i in range(len(arrived)-1, -1, -1):
            if arrived[i] > 0:
                evacuation_time = times[i]
                break
        
        all_stats[algo_name] = {
            'travel_times': mean_travel_times,
            'evacuation_time': evacuation_time,
            'total_arrived': max(arrived) if arrived else 0,
            'mean_tt': np.mean(mean_travel_times) if mean_travel_times else 0,
            'median_tt': np.median(mean_travel_times) if mean_travel_times else 0,
            'std_tt': np.std(mean_travel_times) if mean_travel_times else 0
        }
    
    # Create box plot - FIX: Changed from (2, 2) to single subplot
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot 1: Box plots comparison
    travel_time_data = [stats['travel_times'] for stats in all_stats.values() 
                        if stats['travel_times']]
    labels = [name for name, stats in all_stats.items() if stats['travel_times']]
    
    if travel_time_data:
        bp = ax1.boxplot(travel_time_data, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True,
                        boxprops=dict(linewidth=2),
                        whiskerprops=dict(linewidth=2),
                        capprops=dict(linewidth=2),
                        medianprops=dict(color='red', linewidth=2.5),
                        meanprops=dict(color='blue', linewidth=2.5, linestyle='--'))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax1.set_ylabel('Travel Time (s)', fontsize=12, fontweight='bold')
    ax1.set_title('Travel Time Distribution Comparison', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2.5, label='Median'),
        Line2D([0], [0], color='blue', linewidth=2.5, linestyle='--', label='Mean')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    

# Usage Example
file_dict = {
    'Dynamic A*': 'dynamic_a/summary.xml',
    'Static A*': 'static_a/summary.xml',
    'SUMO DUE': 'sumo/summary_499.xml',
    'C-logit': 'hassan/summary_499.xml',
    "MARL": 'RL/summary.xml'
}

compare_travel_times_boxplot(file_dict)