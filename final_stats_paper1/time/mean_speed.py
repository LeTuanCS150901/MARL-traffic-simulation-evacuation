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
        inserted = []
        
        for step in root.findall('step'):
        
            times.append(float(step.get('time')))
            mean_speed = float(step.get('meanSpeed'))
            mean_speeds.append(max(0, mean_speed))
            running.append(int(step.get('running')))
            halting.append(int(step.get('halting')))
            inserted.append(int(step.get('inserted')))

            if step.get('arrived') == arrival_max:
                break

        culmulated_arrivals = [ins - run for ins,run in zip(inserted,running )]
        
        all_data[algo_name] = {
            'times': times,
            'mean_speeds': mean_speeds,
            'running': running,
            'halting': halting,
            'inserted': inserted,
            'culmulated_arrivals': culmulated_arrivals
        }
    
    # Create figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
    
    # Plot 1: MACRO fundamental diagram
    for idx, (algo_name, data) in enumerate(all_data.items()):
        color = colors[idx % len(colors)]
        
        # Area chart with gradient effect
        # ax1.fill_between(data['times'], data['mean_speeds'], 
        #                  alpha=0.3, color=color)

     
            # ax1.scatter(data['running'], data['running'] * np.array(data['mean_speeds']), 
            #         color=color, linewidth=0.3, label=algo_name, alpha=0.8)

        ax1.scatter(
                data['running'], 
                data['running'] * np.array(data['mean_speeds']),
                color=color, 
                linewidth=0.3, 
                label=algo_name, 
                alpha=0.8,
                marker='x',      # change marker to "x"
                s=10           # smaller point size (default is 36)
            )
    
    ax1.set_xlabel('Density', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Flow', fontsize=13, fontweight='bold')
    ax1.set_title('Macro fundamental diagram of traffic', 
                  fontsize=15, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax1.legend(loc='best', fontsize=11,  markerscale=5)

    # ax1.set_ylim(bottom=0)
    
    # Plot 2: mean speed overtime
    x_offset = 0
    bar_width = 0.35
    
    for idx, (algo_name, data) in enumerate(all_data.items()):
        color = colors[idx % len(colors)]
        
        # Running vehicles (solid line with area)
        # ax2.fill_between(data['times'], data['mean_speeds'], 
                        #  alpha=0.25, color=color)
        # if algo_name == "C-logit":
        ax2.plot(data['times'], data['mean_speeds'], 
                    color=color, linewidth=2.5, label=f'{algo_name}', 
                    alpha=0.8, linestyle='-')
            
    
    ax2.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Mean Speed (m/s)', fontsize=13, fontweight='bold')
    ax2.set_title('Mean Speed of network over time', 
                  fontsize=15, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax2.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
    ax2.set_ylim(bottom=0)
    
    # plot 3: accumulated vehs
        # plot 3: accumulated vehs
    for idx, (algo_name, data) in enumerate(all_data.items()):
        color = colors[idx % len(colors)]
        times = np.array(data['times'])
        arrivals = np.array(data['culmulated_arrivals'])

        # Scatter plot
        ax3.plot(times, arrivals, color=color, linewidth=2, label=algo_name, alpha=0.8)

        # Get last point
        last_time = times[-1]
        last_value = arrivals[-1]

        # Draw vertical line from y=0 to y=last_value
        # ax3.vlines(x=last_time, ymin=0, ymax=last_value,
        #         color=color, linestyle='--', linewidth=0.2, alpha=0.7)

    ax3.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Number of Vehicles', fontsize=13, fontweight='bold')
    ax3.set_title('Cumulative vehicles arrived', fontsize=15, fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax3.legend(loc='best', fontsize=11, framealpha=0.9)
   
    # ax1.set_ylim(bottom=0)
    plt.tight_layout(w_pad=5)  # Increase horizontal space between subplots
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
# config the number of vehs need to simulate
arrival_max  = str(2000)

file_dict = {
    'Dynamic A*': 'dynamic_a/summary.xml',
    # 'Static A*': 'static_a/summary.xml',
    'SUMO DUE': 'due/summary_001.xml',
    'C-logit': 'hassan/summary_399.xml',
    # "MARL with disaster": 'RL/disaster/summary.xml',
    "MARL no disaster": 'RL/no_disaster/summary_fin.xml',
    'dynamic_a_new': 'dynamic_a_new/summary.xml'

}

compare_algorithms(file_dict)