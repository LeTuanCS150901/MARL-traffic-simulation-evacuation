import time
import random
import numpy as np
import argparse  
import csv
from pathlib import Path
import xml.etree.ElementTree as ET
from read_config import read_config

from group_agents import MultiAgentManager
from graph_manager import GraphManager
from logs.logger import logger, setLoggerEnable  # import our custom logger

from simulation import run_simulation, reset_simulation, run_simulation_inference
from disaster_infor.disaster_move import DisasterMove

import torch
torch.set_num_threads(64)

#  CUDA/GPU Setup
if torch.cuda.is_available():
    # Print to confirm Pytorch sees the GPU
    print(f"✅ CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Optional: Optimizes performance if your input sizes (obs_dim) don't change
    torch.backends.cudnn.benchmark = True 
    
    # 3. Reproducibility (Important for RL)
    # Set seeds for CPU, NumPy, and GPU
    # SEED = 42  # You can pick any number
    # random.seed(SEED)
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
else:
    print("⚠️ CUDA is NOT available. Running on CPU.")

SCRIPT_DIR = Path(__file__).resolve().parent


def set_seed(seed):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_config_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def resolve_sumo_file_value(value, base_dir):
    paths = [item.strip() for item in value.split(",")]
    resolved_paths = []
    for path_text in paths:
        path = Path(path_text)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        resolved_paths.append(str(path))
    return ",".join(resolved_paths)


def ensure_child(parent, tag):
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def set_child_value(parent, tag, value):
    child = ensure_child(parent, tag)
    child.set("value", str(value))
    return child


def prepare_run_sumo_config(CONFIG, output_dir, seed=None):
    if output_dir is None:
        return

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    CONFIG.setdefault("_base_sumo_config", CONFIG["sumo_config"])
    source_config = resolve_config_path(CONFIG["_base_sumo_config"])
    source_dir = source_config.parent

    tree = ET.parse(source_config)
    root = tree.getroot()

    input_node = ensure_child(root, "input")
    for tag in ("net-file", "route-files"):
        node = input_node.find(tag)
        if node is not None and node.get("value"):
            node.set("value", resolve_sumo_file_value(node.get("value"), source_dir))

    output_node = ensure_child(root, "output")
    output_files = {
        "summary-output": output_path / "summary.xml",
        "tripinfo-output": output_path / "tripinfo.xml",
        "vehroute-output": output_path / "routes.xml",
    }
    for tag, file_path in output_files.items():
        set_child_value(output_node, tag, file_path)

    report_node = ensure_child(root, "report")
    set_child_value(report_node, "log", output_path / "sumo.log")

    if seed is not None:
        random_number_node = ensure_child(root, "random_number")
        set_child_value(random_number_node, "seed", seed)

    generated_config = output_path / "run.sumocfg"
    ET.indent(tree, space="    ")
    tree.write(generated_config, encoding="utf-8", xml_declaration=True)

    CONFIG["sumo_config"] = str(generated_config)
    CONFIG["saved_route"] = str(output_path / "routes.xml")


def parse_seed_list(value):
    if value is None:
        return []
    seeds = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    return seeds


def parse_best_seed_value(raw_value):
    value = raw_value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    return value


def load_complete_seed_pool(best_seeds_file, top_k=5):
    path = Path(best_seeds_file).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Best-seeds file not found: {path}")

    entries = []
    current = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "-":
            if current:
                entries.append(current)
            current = {}
            continue
        if current is None or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = parse_best_seed_value(raw_value)
        current[key] = value
    if current:
        entries.append(current)

    complete_seeds = []
    for entry in entries:
        if entry.get("status") != "complete":
            continue
        seed = entry.get("seed")
        if seed is None:
            continue
        demand = entry.get("demand_vehicles")
        arrived = entry.get("arrived_vehicles")
        summary_arrived = entry.get("summary_arrived")
        if demand not in (None, "") and arrived not in (None, ""):
            if int(float(arrived)) != int(float(demand)):
                continue
        if summary_arrived not in (None, "") and arrived not in (None, ""):
            if int(float(summary_arrived)) != int(float(arrived)):
                continue
        complete_seeds.append(int(float(seed)))

    if len(complete_seeds) < top_k:
        raise ValueError(
            f"Need at least {top_k} complete seeds in {path}, found {len(complete_seeds)}. "
            "Run the seed search until the candidate gate passes."
        )
    return complete_seeds[:top_k]


def default_best_seeds_path(testcase):
    return SCRIPT_DIR.parent / "exp_runs" / "proposed" / str(testcase) / "best_seeds.yaml"


def write_finetune_seed_history_header(history_path):
    if history_path.exists():
        return
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["episode", "seed", "output_dir", "duration_seconds"],
        )
        writer.writeheader()


def append_finetune_seed_history(history_path, episode, seed, output_dir, duration_seconds):
    with history_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["episode", "seed", "output_dir", "duration_seconds"],
        )
        writer.writerow(
            {
                "episode": episode,
                "seed": seed,
                "output_dir": str(output_dir),
                "duration_seconds": f"{duration_seconds:.6f}",
            }
        )


def choose_finetune_seed(seed_pool, sampler, recent_seeds, diversity_window=20, minimum_distinct=3):
    selected_seed = sampler.choice(seed_pool)
    if len(seed_pool) < minimum_distinct or len(recent_seeds) < diversity_window - 1:
        return selected_seed

    recent_window = recent_seeds[-(diversity_window - 1):]
    if len(set(recent_window + [selected_seed])) >= minimum_distinct:
        return selected_seed

    alternatives = [
        seed
        for seed in seed_pool
        if len(set(recent_window + [seed])) >= minimum_distinct
    ]
    if alternatives:
        return sampler.choice(alternatives)
    return selected_seed


def initialize(CONFIG):
    net_file = CONFIG['net_file']
    route_file = CONFIG['route_file']
    graph =  GraphManager(route_file, net_file)
    # calculate distance of edges to destination edges
    graph.calculate_distances(net_file)
    # calculate distance of edges to disaster zones
    graph.calculate_distance_to_disaster_zones(net_file)
    # calculate distance from O->d
    graph.calculate_dist_OD(net_file)

    logger.info(f"Graph loaded. Number of nodes: {graph.network.number_of_nodes()}")
    logger.info(f"Number of edges: {graph.network.number_of_edges()}")

    # Global dictionary to track each vehicle's last processed edge.
    # visited_edge = {vehicle_id: [] for vehicle_id in graph.trips.keys()}
    visited_edge = {}
    multi_agents = MultiAgentManager(graph)
    multi_agents.divide_trips_to_agent_groups()

    # Init disaster object
    poi_file = None
    try:
        poi_file = CONFIG['disaster']['POI_path'] # path to file contains disaster coordinates
    except Exception:
        logger.warning("No POI file of disaster found => disater object = None")
    disaster_move = None
    if poi_file:
        logger.success("ACtivate offline training for data collection")
        disaster_move = DisasterMove(poi_file, graph)
        disaster_move.init_disaster_params()

    init = {
        "edges": graph.edges,
        "trips": graph.trips,
        "visited_edge": visited_edge,
        "multi_agents": multi_agents,
        "disaster_move": disaster_move,
    }
    return init

def train(num_episodes, simulation_steps, save_freq, CONFIG=None):  
    """Training function with model saving"""  
    # Initialize environment and agents  
    flag_save_freq = True
    while flag_save_freq:
        try:
            save_freq = int(save_freq)
            if save_freq <= 0:
                int(None)
            flag_save_freq = False
            print("Thank you very much !")
        except:
            save_freq = input("Please input a positive value for the save frequency !\n")

    result = initialize(CONFIG)
    edges = result["edges"]
    trips = result["trips"]
    last_edge = result["visited_edge"]
    multi_agents = result["multi_agents"]
    disaster_move = result["disaster_move"]

    # Initialize plotter  
    multi_agents.initialize_plotter(max_episodes=num_episodes)  
    
    for episode in range(num_episodes):  
        start_time = time.time()
        logger.info(f"Starting episode {episode + 1}/{num_episodes}")  
        
        # Reset the simulation  
        reset_simulation(episode, CONFIG)  
        last_edge.clear()  
        
        # random activate/deactiate disaster for a new episode
        # if no disaster provided. The function do nothing
        multi_agents.random_active_disaster()  
        
        # Run one episode  
        run_simulation(edges=edges, trips=trips, last_edge=last_edge,  
                      multi_agents=multi_agents, CONFIG=CONFIG, simulation_steps=simulation_steps,
                      disaster_move=disaster_move)  
        
        logger.info(f"Episode {episode + 1} completed.")  
        multi_agents.plotter.increase_episode()  

        # track the time taken for each epoch
        end_time = time.time()
        epoch_duration = end_time - start_time
        logger.trace(f"Epoch {episode}/{num_episodes} took {epoch_duration:.2f} seconds")
        print(f"Epoch {episode}/{num_episodes} took {epoch_duration} seconds")
        
        # Save models periodically after 10 eps, but not on the final episode (we'll do that outside the loop)  
        if episode % save_freq == 0 and episode < num_episodes - 1:  
            multi_agents.save_all_models(episode=episode+1)  
        
        time.sleep(1)  

    # Final save at the end of training (only once)  
    timestamp = multi_agents.save_all_models(episode=num_episodes)  
    # multi_agents.plotter.write_to_file() 
    
    logger.info(f"Training completed. Models saved with timestamp: {timestamp}")  
    return timestamp  

def run_inference(index=None, episode=None, simulation_steps=None, CONFIG=None, offline_infer=None):  
    """Run inference using saved models"""  
    if simulation_steps is None:  
        simulation_steps = CONFIG['simulation_steps']  
    
    # Initialize environment and agents  
    result = initialize(CONFIG)
    edges = result["edges"]
    trips = result["trips"]
    last_edge = result["visited_edge"]
    multi_agents = result["multi_agents"]
    disaster_move = result["disaster_move"]
        
    # initialize plotter
    multi_agents.initialize_plotter(max_episodes=1)  
     
    # set mode as inference
    multi_agents.set_training(mode=False)

    # load models, either from offline training or online training
    if offline_infer:
        # set mode for selecting action in offline way
        multi_agents.set_offline_infer(mode=True)
        success = multi_agents.load_offline_models()
    else:
        success = multi_agents.load_models(index=index, episode=episode, eval_mode=True)  
    
    if not success:  
        logger.error("Failed to load models. Cannot run inference.")  
        return False
    
    logger.info("Models loaded successfully. Running inference...")  

    # Reset simulation  
    reset_simulation(episode=0, CONFIG=CONFIG)  
    last_edge.clear()  
    
    # random activate/deactiate disaster for a new episode
    # if no disaster provided. The function do nothing
    multi_agents.random_active_disaster()  
        
    # Run simulation with loaded models (no learning) 
    # For now no disaster is added into the model
    run_simulation_inference(edges=edges, trips=trips, last_edge=last_edge,  
                  multi_agents=multi_agents, CONFIG=CONFIG, simulation_steps=simulation_steps, disaster_move=disaster_move)  
    
    logger.info("Inference completed")  
    return True

def fine_tune(
    index,
    checkpoint_episode,
    num_finetune_episodes,
    simulation_steps,
    save_freq,
    CONFIG=None,
    seed_pool=None,
    output_dir=None,
    sampler_seed=None,
):
    """
    Fine-tuning function:
    - Loads pretrained models from checkpoints.
    - Freezes encoders, trains MLP/output layers with updated reward weighting.
    - Updates optimizer (small lr, fresh).
    """
    logger.info("===== Starting Fine-Tuning Mode =====")
    if not seed_pool:
        raise ValueError("Fine-tuning requires a non-empty seed pool from --finetune-seeds or --best-seeds-file")
    if output_dir is None:
        raise ValueError("Fine-tuning with a seed pool requires --output-dir for per-episode SUMO configs")

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    history_path = output_path / "finetune_seed_history.csv"
    write_finetune_seed_history_header(history_path)
    sampler = random.Random(sampler_seed)

    flag_save_freq = True
    while flag_save_freq:
        try:
            save_freq = int(save_freq)
            if save_freq <= 0:
                int(None)
            flag_save_freq = False
            print("Thank you very much !")
        except:
            save_freq = input("Please input a positive value for the save frequency !\n")

    # Step 1: Initialize environment and agents
    result = initialize(CONFIG)
    edges = result["edges"]
    trips = result["trips"]
    last_edge = result["visited_edge"]
    multi_agents = result["multi_agents"]
    disaster_move = result["disaster_move"]

    # Step 2: Load pretrained models, and freeze encoder, unfreeze MLP, create fine-tune optimizer
    success = multi_agents.load_models(index=index, episode=checkpoint_episode, eval_mode=False, fine_tune_mode=True)  
    if not success:
        logger.error("Failed to load pretrained models. Cannot start fine-tuning.")
        return None

    # Step 4: Initialize plotter
    multi_agents.initialize_plotter(max_episodes=num_finetune_episodes)

    # Step 5: Run fine-tuning episodes
    timestamp = None
    recent_episode_seeds = []
    for episode in range(1, num_finetune_episodes + 1):
        start_time = time.time()
        episode_seed = choose_finetune_seed(seed_pool, sampler, recent_episode_seeds)
        recent_episode_seeds.append(episode_seed)
        episode_output_dir = output_path / f"episode_{episode:04d}_seed_{episode_seed}"
        set_seed(episode_seed)
        prepare_run_sumo_config(CONFIG, episode_output_dir, seed=episode_seed)
        logger.info(
            f"[Fine-tune] Episode {episode}/{num_finetune_episodes} uses seed {episode_seed} "
            f"from pool {seed_pool}"
        )

        reset_simulation(episode, CONFIG)
        last_edge.clear()

        # random activate/deactiate disaster for fine-tuning
        multi_agents.random_active_disaster()

        # Use the same simulation logic, but with learning ON and new reward weights
        run_simulation(edges=edges, trips=trips, last_edge=last_edge,
                    multi_agents=multi_agents, CONFIG=CONFIG, 
                    simulation_steps=simulation_steps, disaster_move=disaster_move)

        multi_agents.plotter.increase_episode()

        end_time = time.time()
        episode_duration = end_time - start_time
        append_finetune_seed_history(
            history_path=history_path,
            episode=episode,
            seed=episode_seed,
            output_dir=episode_output_dir,
            duration_seconds=episode_duration,
        )
        logger.trace(f"Fine-tune episode took {episode_duration:.2f}s")

        # Save fine-tuned models periodically
        if episode % save_freq == 0:
            timestamp = multi_agents.save_all_models(episode=f"ft_{episode}")

    if timestamp is None:
        timestamp = multi_agents.save_all_models(episode=f"ft_{num_finetune_episodes}")

    logger.info(f"Fine-tuning completed. Models saved with timestamp: {timestamp}")
    logger.info(f"Fine-tune seed history saved to {history_path}")

    return timestamp


def main():  
    parser = argparse.ArgumentParser(description='RL for Traffic Routing')  
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference', 'finetune'],  
                       help='Run mode: train or inference')  
    parser.add_argument('--episodes', type=int, default=None,  
                       help='Number of episodes for training')  
    parser.add_argument('--testcase', type=str, default=None,  
                       help='The testcase config yaml file in config folder')  
    parser.add_argument('--steps', type=int, default=None,  
                       help='Number of simulation steps per episode')  
    parser.add_argument('--save_freq', type=int, default=None,  
                       help='Save frequency (episodes)')  
    parser.add_argument('--index', type=int, default=None,  
                       help='index is the run number of training. E.g, traning number 1,2,3,...')  
    parser.add_argument('--episode', type=int, default=None,  
                       help='Episode number of saved models to load for inference')  
    parser.add_argument('--finetune-episodes', type=int, default=None,
                       help='Number of episodes to run in finetune mode. Defaults to --episode for backward compatibility')
    parser.add_argument('--gui',       action='store_true',
                        help='Launch SUMO with GUI (sumo-gui) instead of sumo') # boolean flag for gui
    parser.add_argument('--offline_infer',  action='store_true',
                        help='Run offline model for inference') # boolean flag for offline_inference
    parser.add_argument('--log', action='store_true',
                        help='Enable Loguru console output')  # <-- added option
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for Python, NumPy, PyTorch, and SUMO')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory for per-run SUMO outputs and generated run.sumocfg')
    parser.add_argument('--finetune-seeds', type=str, default=None,
                       help='Comma-separated complete seed pool for anti-overfit fine-tuning, e.g. 44,48,52,60,61')
    parser.add_argument('--best-seeds-file', type=str, default=None,
                       help='Path to best_seeds.yaml. Used for finetune seed pool when --finetune-seeds is not provided')
    parser.add_argument('--seed-pool-size', type=int, default=5,
                       help='Number of complete seeds to load from best_seeds.yaml for fine-tuning')
    
    args = parser.parse_args()  

    # Set the config file
    CONFIG = read_config(config_filename=args.testcase)
    CONFIG["_base_sumo_config"] = CONFIG["sumo_config"]
    set_seed(args.seed)
    if args.mode != "finetune":
        prepare_run_sumo_config(CONFIG, args.output_dir, seed=args.seed)
    
    # Override CONFIG values if specified  
    num_episodes = args.episodes if args.episodes is not None else CONFIG['num_episodes']  
    simulation_steps = args.steps if args.steps is not None else CONFIG['simulation_steps']  
    # Set which SUMO binary to call
    CONFIG['sumo_binary'] = 'sumo-gui' if args.gui else 'sumo'

    offline_infer = False
    if args.offline_infer:
        offline_infer = True

    # enable loguru console output
    if args.log:
        setLoggerEnable(True)

    if args.mode == 'train':  
        logger.info(f"Starting training for {num_episodes} episodes, {simulation_steps} steps per episode")  
        train(num_episodes, simulation_steps, save_freq=args.save_freq, CONFIG=CONFIG)  

    if args.mode == 'inference':  # inference mode  
        logger.info("Starting inference with saved models")  
        success = run_inference(index=args.index, episode=args.episode, simulation_steps=simulation_steps, CONFIG=CONFIG,
                      offline_infer=offline_infer)  
        if not success:
            raise RuntimeError("Inference failed")
        
    if args.mode == 'finetune':
        logger.info("Starting fine tune with saved models")  
        seed_pool = parse_seed_list(args.finetune_seeds)
        if not seed_pool:
            best_seeds_file = args.best_seeds_file or default_best_seeds_path(args.testcase)
            seed_pool = load_complete_seed_pool(best_seeds_file, top_k=args.seed_pool_size)
        if len(seed_pool) < args.seed_pool_size:
            raise ValueError(
                f"Fine-tuning requires at least {args.seed_pool_size} complete seeds, got {len(seed_pool)}"
            )
        finetune_episodes = args.finetune_episodes if args.finetune_episodes is not None else args.episode
        fine_tune(
            index=args.index,
            checkpoint_episode=args.episode,
            num_finetune_episodes=finetune_episodes,
            simulation_steps=simulation_steps,
            save_freq=args.save_freq,
            CONFIG=CONFIG,
            seed_pool=seed_pool[:args.seed_pool_size],
            output_dir=args.output_dir,
            sampler_seed=args.seed,
        )

if __name__ == "__main__":  
    main()  
