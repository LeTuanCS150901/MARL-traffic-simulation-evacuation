"""
Vehicle Risk Analysis and Algorithm Comparison
This script analyzes vehicle risk in disaster scenarios and compares multiple routing algorithms
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to folder A
folder_a_path = os.path.join(current_dir, '..', 'routing_MARL')

# Add folder A to the system path
sys.path.append(folder_a_path)

from graph_manager import GraphManager
from read_config import read_config

# ============================================================================
# DATA EXTRACTION FUNCTIONS
# ============================================================================

def extract_trip_destinations(xml_file, print_output=False):
    """
    Extract trip destinations from XML file
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    trip_destinations = {}
    for trip in root.findall('trip'):
        trip_id = trip.attrib['id']
        trip_to = trip.attrib['to']
        trip_destinations[trip_id] = trip_to

    if print_output:
        print("trip_destinations = {")
        for k, v in trip_destinations.items():
            print(f'    "{k}": "{v}",')
        print("}")
    return trip_destinations


def compute_min_distance_edge(graph, current_edge):
    """
    Compute minimum distance from current edge to any disaster edge
    """
    min_distance = float('inf')
    min_disaster_edge = None

    # Loop through disaster edges
    disaster_zones = graph.CONFIG['disaster']['fire']['disaster_zones']
    selected_zone_id = None

    for zone_id in disaster_zones:
        for edge_disaster in disaster_zones[zone_id]:
            distance = graph.euclidean_distance_between_edges(current_edge, edge_disaster)

            if distance < min_distance:
                min_distance = distance
                min_disaster_edge = edge_disaster
                selected_zone_id = zone_id

    return min_distance, min_disaster_edge, selected_zone_id


def calculate_survival_probability(graph, current_edge, disaster_zone):
    """
    Calculate survival probability for a vehicle on a given edge
    """
    disaster_radius = graph.disaster_info[disaster_zone]['radius'] + 180
    # disaster_radius = 100
    
    if current_edge in graph.edges_dist_disaster:
        ds = graph.edges_dist_disaster[current_edge][disaster_zone]

        # print(f"^^^^^^^^^^^^^^^^^^^^^ Radius= {disaster_radius}")
        # Calculate not survival probability
        # not_survive_proba = 0 if ds > disaster_radius else 1 - (ds / disaster_radius)

        not_survive_proba = 0 if ds > disaster_radius else np.exp(-ds/disaster_radius)
        

        # not_survive_proba = 1/(ds/disaster_radius) if ds > disaster_radius else np.exp(-ds/disaster_radius)
        # not_survive_proba = 1/(ds/disaster_radius) if ds > disaster_radius else ds/disaster_radius
        return 1 - not_survive_proba  # Survival probability
    else:
        return 0.0


# ============================================================================
# TEMPORAL DATA COLLECTION
# ============================================================================

def log_disaster_metric_with_time(demand_file, network_file, output_routes, trips_to_control, 
                                   testcase, saved_file):
    """
    Enhanced version that captures temporal information along with risk metrics
    """
    # READ GRAPH
    CONFIG = read_config(config_filename=testcase)
    net_file = CONFIG['net_file']
    route_file = CONFIG['route_file']
    graph = GraphManager(route_file, net_file)

    # Calculate distances
    graph.calculate_distances(net_file)
    graph.calculate_distance_to_disaster_zones(net_file)
    graph.calculate_dist_OD(net_file)

    # Load the output routes XML file
    try:
        tree = ET.parse(output_routes)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except FileNotFoundError:
        print(f"Route file not found: {output_routes}")
        return None

    vehicle_risk_timeline = []  # Store temporal data

    # Log information on found vehicles
    vehicle_count = 0
    for vehicle in root.findall('vehicle'):
        vehicle_id = vehicle.get('id')
        depart = float(vehicle.get('depart', 0))  # Get departure time
        
        if vehicle_id in trips_to_control:
            vehicle_count += 1
            route = vehicle.find('route')
            
            if route is not None:
                edges = route.get('edges').split()
                
                # Calculate cumulative risk along the route
                total_length = 0
                product_survival = 1
                
                for edge in edges:
                    min_distance, min_disaster_edge, disaster_zone = compute_min_distance_edge(graph, edge)
                    survival_probability = calculate_survival_probability(graph, edge, disaster_zone)
                    
                    edge_length = graph.edges[edge]['length']
                    total_length += edge_length
                    survive_proba_power_length = survival_probability ** edge_length
                    product_survival *= survive_proba_power_length

                # Calculate final path survival probability
                if total_length > 0:
                    path_survival_probability = product_survival ** (1/total_length)
                    risk = 1 - path_survival_probability
                    
                    # Estimate arrival time (replace with actual if available in XML)
                    # Assuming average speed of 10 m/s (36 km/h)
                    travel_time = total_length / 10.0
                    
                    # Try to get actual arrival time from XML, otherwise use estimate
                    arrival_str = vehicle.get('arrival', None)
                    if arrival_str:
                        arrival = float(arrival_str)
                    else:
                        arrival = depart + travel_time
                    
                    vehicle_risk_timeline.append({
                        'vehicle_id': vehicle_id,
                        'depart': depart,
                        'arrival': arrival,
                        'survival_probability': path_survival_probability,
                        'risk': risk,
                        'route_length': total_length
                    })

    # Save enhanced results to CSV
    if vehicle_risk_timeline:
        save_temporal_to_csv(vehicle_risk_timeline, saved_file)
        print(f"✓ Total vehicles processed: {vehicle_count}")
    
    return vehicle_risk_timeline


def save_temporal_to_csv(data, saved_file):
    """
    Save temporal data to CSV file
    """
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'result/disaster_metric') 
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, saved_file)

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"  Data saved to: {filepath}")


# ============================================================================
# MULTI-ALGORITHM DATA COLLECTION
# ============================================================================

def collect_all_algorithm_data(base_path, network_file, demand_file, testcase):
    """
    Collect data from all algorithm runs for comparison
    """
    # Define all your algorithms/scenarios
    # MODIFY THIS DICTIONARY to match your folder structure
    algorithms = {
        'Dynamic A': ('dynamic_a', 'routes.xml'),
        'SUMO DUE': ('sumo', 'vehroute_499.xml'),
        'C-logit': ('hassan', 'vehroute_499.xml'),
        'Proposed MARL': ('RL', 'routes.xml')
       
    
        # 'Static': ('static', 'routes.xml'),
        # Add more algorithms as needed
        # 'Algorithm Name': ('subfolder_name', 'routes_filename.xml'),
    }
    
    all_data = {}
    
    # Extract controlled trips once
    trip_destinations = extract_trip_destinations(demand_file, print_output=False)
    trips_to_control = set(trip_destinations.keys())
    
    for algo_name, (subfolder, rou_file) in algorithms.items():
        print(f"\n{'='*60}")
        print(f"Processing algorithm: {algo_name}")
        print(f"{'='*60}")
        
        # Build the path to the route file
        if subfolder == "baseline" or subfolder == ".":
            output_routes = os.path.join(base_path, rou_file)
        else:
            output_routes = os.path.join(base_path, subfolder, rou_file)
        
        if not os.path.exists(output_routes):
            print(f"⚠ Warning: Route file not found at {output_routes}, skipping...")
            continue
        
        # Collect data for this algorithm
        vehicle_data = log_disaster_metric_with_time(
            demand_file, network_file, output_routes, 
            trips_to_control, testcase, f"{algo_name.replace(' ', '_')}_temporal.csv"
        )
        
        if vehicle_data:
            all_data[algo_name] = vehicle_data
            print(f"✓ Collected data for {algo_name}: {len(vehicle_data)} vehicles")
        else:
            print(f"⚠ No data collected for {algo_name}")
    
    return all_data


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_algorithms_comparison(all_data, save_path=None):
    """
    Compare network risk metrics across all algorithms over time
    """
    # fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # # Define colors for each algorithm
    # colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
    # color_map = dict(zip(all_data.keys(), colors))
    # colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', 
    #           '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#d35400']
    
    # # Find global time range
    # all_times = []
    # for data in all_data.values():
    #     df = pd.DataFrame(data)
    #     all_times.extend([df['depart'].min(), df['arrival'].max()])
    
    # min_time = min(all_times)
    # max_time = max(all_times)
    # time_points = np.linspace(min_time, max_time, 500)
    
    # # Process each algorithm
    # for idx, (algo_name, vehicle_data) in enumerate(all_data.items()):
    #     df = pd.DataFrame(vehicle_data)
        
    #     active_vehicles = []
    #     avg_risk = []
    #     vehicles_at_risk = []
        
    #     for t in time_points:
    #         active = df[(df['depart'] <= t) & (df['arrival'] >= t)]
            
    #         active_vehicles.append(len(active))
            
    #         if len(active) > 0:
    #             avg_risk.append(active['risk'].mean())
    #             at_risk = len(active[active['survival_probability'] < 0.99])
    #             vehicles_at_risk.append(at_risk)
    #         else:
    #             avg_risk.append(0)
    #             vehicles_at_risk.append(0)
        
    #     # # Plot 1: Active vehicles
    #     # axes[0].plot(time_points, active_vehicles, linewidth=2.5, 
    #     #             label=algo_name, color=color_map[algo_name], alpha=0.8)
        
    #     color = colors[idx % len(colors)]
    #     # Plot 2: Average risk
    #     # axes[1].plot(time_points, avg_risk, linewidth=2.5,
    #     #             label=algo_name, color=color_map[algo_name], alpha=0.8)

    #     axes.plot(time_points, avg_risk, linewidth=2.5,
    #                 label=algo_name, color=color, alpha=0.8)
        
    #     axes.fill_between(time_points,avg_risk, 
    #                      alpha=0.25, color=color)
    
        
        # # Plot 3: Vehicles at risk
        # axes[2].plot(time_points, vehicles_at_risk, linewidth=2.5,
        #             label=algo_name, color=color_map[algo_name], alpha=0.8)
    
    # Formatting

    # axes[0].set_ylabel('Active Vehicles', fontsize=13, fontweight='bold')
    # axes[0].set_title('Algorithm Comparison: Network Traffic and Risk Over Time', 
    #                  fontsize=15, fontweight='bold')
    # axes[0].legend(loc='best', fontsize=11, framealpha=0.9)
    # axes[0].grid(True, alpha=0.3)
    
    # axes.set_ylabel('Average Risk Level', fontsize=13, fontweight='bold')
    # axes.legend(loc='best', fontsize=11, framealpha=0.9)
    # axes.grid(True, alpha=0.3)
    # axes.ylim()
    
    # axes[2].set_ylabel('Vehicles at High Risk\n(survival < 0.99)', fontsize=13, fontweight='bold')
    # axes[2].set_xlabel('Simulation Time (seconds)', fontsize=13, fontweight='bold')
    # axes[2].legend(loc='best', fontsize=11, framealpha=0.9)
    # axes[2].grid(True, alpha=0.3)
    

    fig, ax1 = plt.subplots(figsize=(12, 10))

    # Define colors for each algorithm
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', 
            '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#d35400']

    # Find global time range
    all_times = []
    for data in all_data.values():
        df = pd.DataFrame(data)
        all_times.extend([df['depart'].min(), df['arrival'].max()])

    min_time = min(all_times)
    max_time = max(all_times)
    time_points = np.linspace(min_time, max_time, 500)

    # Process each algorithm
    for idx, (algo_name, vehicle_data) in enumerate(all_data.items()):
        df = pd.DataFrame(vehicle_data)
        
        avg_risk = []
        
        for t in time_points:
            active = df[(df['depart'] <= t) & (df['arrival'] >= t)]
            
            if len(active) > 0:
                avg_risk.append(active['risk'].mean())
            else:
                avg_risk.append(0)
        
        color = colors[idx % len(colors)]
        
        # Plot average risk
        ax1.plot(time_points, avg_risk, linewidth=2.5,
                label=algo_name, color=color, alpha=0.8)
        
        ax1.fill_between(time_points, avg_risk, 
                        alpha=0.25, color=color)

  
    # Formatting
    ax1.set_ylabel('Average Risk Exposure', fontsize=18)
    ax1.set_xlabel('Time (seconds)', fontsize=18)
    ax1.set_title('Average Risk Over Time', 
                fontsize=20)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True)
    ax1.set_ylim([0,0.5])
    ax1.set_xlim([0,2900])
    plt.tight_layout()
    plt.show()

    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to: {save_path}")
    


def plot_cumulative_risk_comparison(all_data, save_path=None):
    """
    Plot cumulative risk (total vehicles exposed to risk) over time
    """
    fig, ax1 = plt.subplots(2,1, figsize=(14, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
    color_map = dict(zip(all_data.keys(), colors))
    
    # Find global time range
    all_times = []
    for data in all_data.values():
        df = pd.DataFrame(data)
        all_times.extend([df['depart'].min(), df['arrival'].max()])
    
    min_time = min(all_times)
    max_time = max(all_times)
    time_points = np.linspace(min_time, max_time, 500)
    
    for algo_name, vehicle_data in all_data.items():
        df = pd.DataFrame(vehicle_data)
        
        cumulative_risk_exposure = []
        
        for t in time_points:
            # Count vehicles that have been exposed to risk by time t
            exposed = df[df['depart'] <= t]
            total_risk = exposed['risk'].sum()
            cumulative_risk_exposure.append(total_risk)
        
        ax1.plot(time_points, cumulative_risk_exposure, linewidth=3,
               label=algo_name, color=color_map[algo_name], alpha=0.8)
    
    ax.set_xlabel('Simulation Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Risk Exposure', fontsize=13, fontweight='bold')
    ax.set_title('Cumulative Risk Exposure Comparison', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Cumulative risk plot saved to: {save_path}")
    
    plt.show()


def plot_risk_distribution_comparison(all_data, save_path=None):
    """
    Compare risk distributions across algorithms using box plots and violins
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prepare data
    algo_names = list(all_data.keys())
    risk_data = [pd.DataFrame(all_data[algo])['risk'].values for algo in algo_names]
    survival_data = [pd.DataFrame(all_data[algo])['survival_probability'].values 
                     for algo in algo_names]
    
    # Box plot for risk
    bp1 = axes[0].boxplot(risk_data, labels=algo_names, patch_artist=True,
                          showmeans=True, meanline=True)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(algo_names)))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[0].set_ylabel('Risk Level', fontsize=13, fontweight='bold')
    axes[0].set_title('Risk Distribution Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Violin plot for survival probability
    parts = axes[1].violinplot(survival_data, positions=range(1, len(algo_names)+1),
                               showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    axes[1].set_xticks(range(1, len(algo_names)+1))
    axes[1].set_xticklabels(algo_names, rotation=45)
    axes[1].set_ylabel('Survival Probability', fontsize=13, fontweight='bold')
    axes[1].set_title('Survival Probability Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Distribution plot saved to: {save_path}")
    
    plt.show()


def plot_algorithm_heatmap_comparison(all_data, save_path=None):
    """
    Create side-by-side heatmaps for each algorithm
    """
    n_algos = len(all_data)
    fig, axes = plt.subplots(1, n_algos, figsize=(6*n_algos, 6))
    
    if n_algos == 1:
        axes = [axes]
    
    # Find global ranges for consistent color scale
    all_times = []
    all_risks = []
    for data in all_data.values():
        df = pd.DataFrame(data)
        all_times.extend([df['depart'].min(), df['arrival'].max()])
        all_risks.extend(df['risk'].values)
    
    min_time = min(all_times)
    max_time = max(all_times)
    max_risk = max(all_risks) if all_risks else 1.0
    
    time_bins = np.linspace(min_time, max_time, 50)
    risk_bins = np.linspace(0, max_risk, 10)
    
    for idx, (algo_name, vehicle_data) in enumerate(all_data.items()):
        df = pd.DataFrame(vehicle_data)
        
        # Create heatmap data
        heatmap_data = np.zeros((len(risk_bins)-1, len(time_bins)-1))
        
        for t_idx in range(len(time_bins)-1):
            t_start, t_end = time_bins[t_idx], time_bins[t_idx+1]
            active = df[(df['depart'] <= t_end) & (df['arrival'] >= t_start)]
            
            for r_idx in range(len(risk_bins)-1):
                r_start, r_end = risk_bins[r_idx], risk_bins[r_idx+1]
                count = len(active[(active['risk'] >= r_start) & (active['risk'] < r_end)])
                heatmap_data[r_idx, t_idx] = count
        
        im = axes[idx].imshow(heatmap_data, aspect='auto', origin='lower',
                             cmap='YlOrRd', interpolation='bilinear',
                             vmin=0, vmax=np.max([h.max() for h in [heatmap_data]]))
        
        axes[idx].set_title(f'{algo_name}', fontsize=13, fontweight='bold')
        axes[idx].set_xlabel('Time (s)', fontsize=11)
        
        if idx == 0:
            axes[idx].set_ylabel('Risk Level', fontsize=11)
        
        # Set ticks
        axes[idx].set_xticks(np.linspace(0, len(time_bins)-1, 5))
        axes[idx].set_xticklabels([f'{int(t)}' for t in np.linspace(min_time, max_time, 5)])
        
        axes[idx].set_yticks(np.arange(0, len(risk_bins)-1, 2))
        axes[idx].set_yticklabels([f'{r:.3f}' for r in risk_bins[::2]])
        
        plt.colorbar(im, ax=axes[idx], label='Vehicles', fraction=0.046, pad=0.04)
    
    plt.suptitle('Risk Distribution Heatmap Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Heatmap comparison saved to: {save_path}")
    
    plt.show()


def plot_individual_vehicle_timeline(vehicle_data, algo_name='Algorithm', save_path=None):
    """
    Plot individual vehicle risk as horizontal bars (Gantt-style)
    """
    df = pd.DataFrame(vehicle_data)
    df = df.sort_values('risk', ascending=False)  # Sort by risk level
    
    # Take top 20 riskiest vehicles for clarity
    df_plot = df.head(20)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create color map based on risk
    colors = plt.cm.RdYlGn_r(df_plot['risk'] / df_plot['risk'].max())
    
    for idx, (i, row) in enumerate(df_plot.iterrows()):
        duration = row['arrival'] - row['depart']
        ax.barh(idx, duration, left=row['depart'], 
                color=colors[idx], edgecolor='black', linewidth=0.5)
        
        # Add vehicle ID text
        ax.text(row['depart'] + duration/2, idx, row['vehicle_id'], 
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels([f"{row['vehicle_id']}\n(risk: {row['risk']:.3f})" 
                        for _, row in df_plot.iterrows()], fontsize=9)
    ax.set_xlabel('Simulation Time (seconds)', fontsize=12)
    ax.set_ylabel('Vehicle ID (sorted by risk)', fontsize=12)
    ax.set_title(f'{algo_name}: Individual Vehicle Risk Timeline (Top 20 Highest Risk)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, 
                               norm=plt.Normalize(vmin=0, vmax=df['risk'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Risk Level', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Individual timeline plot saved to: {save_path}")
    
    plt.show()


# ============================================================================
# STATISTICS FUNCTIONS
# ============================================================================

def print_algorithm_statistics(all_data):
    """
    Print comprehensive statistics comparing all algorithms
    """
    print("\n" + "="*80)
    print("ALGORITHM PERFORMANCE COMPARISON")
    print("="*80)
    
    stats_data = []
    
    for algo_name, vehicle_data in all_data.items():
        df = pd.DataFrame(vehicle_data)
        
        stats = {
            'Algorithm': algo_name,
            'Total Vehicles': len(df),
            'Avg Risk': df['risk'].mean(),
            'Max Risk': df['risk'].max(),
            'Std Risk': df['risk'].std(),
            'Vehicles at High Risk (>0.01)': len(df[df['risk'] > 0.01]),
            'Avg Survival Prob': df['survival_probability'].mean(),
            'Min Survival Prob': df['survival_probability'].min(),
            'Avg Route Length (m)': df['route_length'].mean(),
            'Total Simulation Time (s)': df['arrival'].max() - df['depart'].min()
        }
        stats_data.append(stats)
    
    stats_df = pd.DataFrame(stats_data)
    
    # Print formatted table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print("\n" + stats_df.to_string(index=False))
    
    # Find best algorithm for each metric
    print("\n" + "-"*80)
    print("BEST PERFORMING ALGORITHMS:")
    print("-"*80)
    
    best_avg_risk = stats_df.loc[stats_df['Avg Risk'].idxmin(), 'Algorithm']
    best_survival = stats_df.loc[stats_df['Avg Survival Prob'].idxmax(), 'Algorithm']
    least_at_risk = stats_df.loc[stats_df['Vehicles at High Risk (>0.01)'].idxmin(), 'Algorithm']
    
    print(f"✓ Lowest Average Risk: {best_avg_risk}")
    print(f"✓ Highest Survival Probability: {best_survival}")
    print(f"✓ Fewest Vehicles at High Risk: {least_at_risk}")
    print("="*80 + "\n")
    
    return stats_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION - MODIFY THESE PATHS TO MATCH YOUR SETUP
    # ========================================================================
    
    base_path = r'D:\PhD_Life\Work\Code\marl-traffic-simulation-evacuation\final_stats_paper1\time'
    network_file = r'D:\PhD_Life\Work\Code\marl-traffic-simulation-evacuation\network\manhattan2\manhattan2.net.xml'
    demand_file = r'D:\PhD_Life\Work\Code\marl-traffic-simulation-evacuation\routing-MARL\network\manhattan2\manhattan2.rou.xml'
    testcase = 'manhattan2'
    
    # ========================================================================
    # STEP 1: Collect data from all algorithms
    # ========================================================================
    
    print("\n" + "="*80)
    print("STARTING MULTI-ALGORITHM RISK ANALYSIS")
    print("="*80)
    print("Collecting data from all algorithms...")
    
    all_data = collect_all_algorithm_data(base_path, network_file, demand_file, testcase)
    
    if not all_data:
        print("\n⚠ ERROR: No data collected. Please check:")
        print("  1. File paths are correct")
        print("  2. Algorithm names and folders in collect_all_algorithm_data()")
        print("  3. XML files exist and are properly formatted")
        exit(1)
    
    print(f"\n{'='*80}")
    print(f"✓ Successfully collected data from {len(all_data)} algorithms")
    print(f"{'='*80}")
    
    # ========================================================================
    # STEP 2: Create output directory for comparison plots
    # ========================================================================
    
    comparison_dir = os.path.join(base_path, 'comparison_plots')
    os.makedirs(comparison_dir, exist_ok=True)
    print(f"\nOutput directory: {comparison_dir}")
    
    # ========================================================================
    # STEP 3: Generate all comparison visualizations
    # ========================================================================
    
    print("\n" + "="*80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*80)
    
    print("\n[1/5] Creating algorithm timeline comparison...")
    plot_algorithms_comparison(
        all_data,
        save_path=os.path.join(comparison_dir, 'algorithms_timeline_comparison.png')
    )
    
    # print("\n[2/5] Creating cumulative risk comparison...")
    # plot_cumulative_risk_comparison(
    #     all_data,
    #     save_path=os.path.join(comparison_dir, 'cumulative_risk_comparison.png')
    # )
    
    # print("\n[3/5] Creating risk distribution comparison...")
    # plot_risk_distribution_comparison(
    #     all_data,
    #     save_path=os.path.join(comparison_dir, 'risk_distribution_comparison.png')
    # )
    
    # print("\n[4/5] Creating heatmap comparison...")
    # plot_algorithm_heatmap_comparison(
    #     all_data,
    #     save_path=os.path.join(comparison_dir, 'heatmap_comparison.png')
    # )
    
    # print("\n[5/5] Creating individual vehicle timelines for each algorithm...")
    # for algo_name, vehicle_data in all_data.items():
    #     algo_filename = algo_name.replace(' ', '_').lower()
    #     plot_individual_vehicle_timeline(
    #         vehicle_data,
    #         algo_name=algo_name,
    #         save_path=os.path.join(comparison_dir, f'{algo_filename}_vehicle_timeline.png')
    #     )
    
    # ========================================================================
    # STEP 4: Generate and save statistics
    # ========================================================================
    
    print("\n" + "="*80)
    print("GENERATING STATISTICS")
    print("="*80)
    
    stats_df = print_algorithm_statistics(all_data)
    
    # Save statistics to CSV
    stats_path = os.path.join(comparison_dir, 'algorithm_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"\n✓ Statistics saved to: {stats_path}")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    
    print("\n" + "="*80)
    print("✓ ALL ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {comparison_dir}")
    print("\nGenerated files:")
    print("  - algorithms_timeline_comparison.png")
    print("  - cumulative_risk_comparison.png")
    print("  - risk_distribution_comparison.png")
    print("  - heatmap_comparison.png")
    print("  - [algorithm]_vehicle_timeline.png (for each algorithm)")
    print("  - algorithm_statistics.csv")
    print("  - [algorithm]_temporal.csv (for each algorithm)")
    print("\n" + "="*80 + "\n")