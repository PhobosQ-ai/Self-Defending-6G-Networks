import random
import torch
import logging

logger = logging.getLogger(__name__)

class EdgeNode:
    """
    Simulates a 6G Edge Node responsible for deploying and monitoring decoys.
    Corresponds to Sections 3.2.1 and 3.3.1.
    """
    def __init__(self, node_id: int, generator_model: torch.nn.Module, device: torch.device):
        self.node_id = node_id
        self.device = device
        self.local_generator = generator_model.to(device)
        self.active_decoys = []
        self.attack_log = []
        logger.info(f"Edge Node {self.node_id} initialized.")

    def deploy_decoys(self, num_decoys: int, decoy_type: int):
        """Generates and 'deploys' new decoys."""
        # This is a simulation of decoy deployment.
        self.active_decoys = []
        for _ in range(num_decoys):
            # Generate a decoy based on a random latent vector and the specified type
            z = torch.randn(1, 100, device=self.device)  # latent_dim = 100
            y = torch.zeros(1, 10, device=self.device)  # num_classes = 10
            y[0, decoy_type] = 1
            
            with torch.no_grad():
                decoy_features = self.local_generator(z, y)
                self.active_decoys.append(decoy_features)
        
        logger.info(f"Node {self.node_id}: Deployed {len(self.active_decoys)} decoys of type {decoy_type}.")

    def monitor_traffic(self):
        """
        Simulates traffic monitoring. Any interaction is flagged.
        Returns data for the adversarial feedback loop.
        """
        # Simulate a random chance of an attacker interacting with a decoy
        if random.random() < 0.1: # 10% chance of interaction per time step
            attacker_ip = f"192.168.10.{random.randint(100, 200)}"
            interaction_data = self.active_decoys[0] # Simulate interaction with the first decoy
            
            logger.warning(f"ALERT! Node {self.node_id}: Detected interaction from {attacker_ip} with a decoy.")
            self.attack_log.append(interaction_data)
            return interaction_data
        return None

    def get_local_update(self):
        """
        Simulates local model training for Federated Learning.
        Corresponds to Section 3.4.1.
        """
        # In a real implementation, this would involve training the local_generator
        # on local benign data and the collected self.attack_log.
        logger.info(f"Node {self.node_id}: Performing local model update for federated learning.")
        # Return the model's state dictionary (weights)
        return self.local_generator.state_dict()
