def main():
    import torch
    import logging
    import random
    from gan_model import Generator, Discriminator
    from rl_agent import PPOAgent
    from edge_node import EdgeNode
    from federated_learning import FederatedServer
    from data_loader import get_dataloader

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    NUM_EDGE_NODES = 5
    SIMULATION_STEPS = 100
    FL_SYNC_INTERVAL = 10
    PPO_UPDATE_TIMESTEP = 20
    LATENT_DIM = 100
    LR_GAN = 0.0002
    B1_GAN = 0.5
    B2_GAN = 0.999
    GP_WEIGHT = 10
    STATE_DIM = 3
    ACTION_DIM = 3
    LR_ACTOR = 0.0003
    LR_CRITIC = 0.001
    GAMMA = 0.99
    K_EPOCHS = 4
    EPS_CLIP = 0.2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        _, NUM_FEATURES, NUM_CLASSES = get_dataloader(batch_size=64)
    except Exception as e:
        return
    global_generator = Generator(LATENT_DIM, NUM_CLASSES, NUM_FEATURES, device)
    edge_nodes = []
    for i in range(NUM_EDGE_NODES):
        gen = Generator(LATENT_DIM, NUM_CLASSES, NUM_FEATURES, device)
        gen.load_state_dict(global_generator.state_dict())
        disc = Discriminator(NUM_FEATURES, NUM_CLASSES, device)
        node_dataloader, _, _ = get_dataloader(batch_size=64)
        node = EdgeNode(
            node_id=i, 
            generator_model=gen, 
            discriminator_model=disc, 
            dataloader=node_dataloader, 
            device=device, 
            latent_dim=LATENT_DIM, 
            num_classes=NUM_CLASSES,
            lr=LR_GAN,
            b1=B1_GAN,
            b2=B2_GAN,
            gp_weight=GP_WEIGHT
        )
        edge_nodes.append(node)
    fl_server = FederatedServer(edge_nodes, global_generator, device)
    ppo_agent = PPOAgent(STATE_DIM, ACTION_DIM, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS, EPS_CLIP, device)
    current_decoy_type = 0
    time_step = 0
    for node in edge_nodes:
        node.deploy_decoys(num_decoys=5, decoy_type=current_decoy_type)
    for step in range(SIMULATION_STEPS):
        total_alerts_this_step = 0
        for node in edge_nodes:
            if node.monitor_traffic() is not None:
                total_alerts_this_step += 1
        avg_network_load = random.random()
        current_state = [total_alerts_this_step, avg_network_load, current_decoy_type]
        action = ppo_agent.select_action(current_state)
        reward = 0
        if total_alerts_this_step > 0:
            reward += total_alerts_this_step
        if action in [1, 2]:
            reward -= 0.1
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(False)
        time_step += 1
        if action == 1:
            for node in edge_nodes:
                node.deploy_decoys(num_decoys=10, decoy_type=current_decoy_type)
        elif action == 2:
            current_decoy_type = (current_decoy_type + 1) % NUM_CLASSES
            for node in edge_nodes:
                node.deploy_decoys(num_decoys=5, decoy_type=current_decoy_type)
        if time_step % PPO_UPDATE_TIMESTEP == 0:
            ppo_agent.update()
        if (step + 1) % FL_SYNC_INTERVAL == 0:
            fl_server.run_round()

if __name__ == '__main__':
    main()
