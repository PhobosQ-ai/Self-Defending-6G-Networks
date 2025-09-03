import torch
import logging
from gan_model import Generator, Discriminator
from rl_agent import PPOAgent
from edge_node import EdgeNode
from federated_learning import FederatedServer
from data_loader import get_dataloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NUM_EDGE_NODES = 10
LATENT_DIM = 100
SIMULATION_STEPS = 100
FL_SYNC_INTERVAL = 20

def main():
    logger.info("Initializing Self-Defending 6G Network Simulation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        dataloader, NUM_FEATURES, NUM_CLASSES = get_dataloader(batch_size=64)
        logger.info(f"Data dimensions: Features={NUM_FEATURES}, Classes={NUM_CLASSES}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    global_generator = Generator(LATENT_DIM, NUM_CLASSES, NUM_FEATURES, device)
    global_discriminator = Discriminator(NUM_FEATURES, NUM_CLASSES, device)
    
    edge_nodes = [EdgeNode(node_id=i, generator_model=global_generator, device=device) for i in range(NUM_EDGE_NODES)]
    
    fl_server = FederatedServer(edge_nodes, global_generator, device)
    
    state_dim = 3 
    action_dim = 3 
    ppo_agent = PPOAgent(state_dim, action_dim, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=4, eps_clip=0.2, device=device)
    
    logger.info("\n--- Starting Simulation ---")
    current_decoy_type = 0
    
    for step in range(SIMULATION_STEPS):
        logger.info(f"\n[Step {step+1}/{SIMULATION_STEPS}]")
        total_alerts_this_step = 0
        
        for node in edge_nodes:
            interaction_data = node.monitor_traffic()
            if interaction_data is not None:
                total_alerts_this_step += 1
                
        avg_network_load = torch.rand(1).item() 
        current_state = [total_alerts_this_step, avg_network_load, current_decoy_type]
        
        action = ppo_agent.select_action(current_state)
        
        if action == 1:
            logger.info("RL Action: Increasing decoy density.")
            for node in edge_nodes:
                node.deploy_decoys(num_decoys=10, decoy_type=current_decoy_type)
        elif action == 2:
            current_decoy_type = (current_decoy_type + 1) % NUM_CLASSES
            logger.info(f"RL Action: Changing decoy type to {current_decoy_type}.")
            for node in edge_nodes:
                node.deploy_decoys(num_decoys=5, decoy_type=current_decoy_type)
        else:
            logger.info("RL Action: Maintaining current strategy.")

        if (step + 1) % FL_SYNC_INTERVAL == 0:
            fl_server.run_round()
            
    logger.info("\n--- Simulation Finished ---")

if __name__ == '__main__':
    main()
