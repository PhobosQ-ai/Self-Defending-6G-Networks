import torch
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

class FederatedServer:
    def __init__(self, edge_nodes: list, global_model: torch.nn.Module, device: torch.device):
        self.edge_nodes = edge_nodes
        self.global_model = global_model.to(device)
        self.device = device
        logger.info("Federated Learning Server initialized.")

    def run_round(self):
        """Executes one round of federated learning."""
        logger.info("\n--- Starting Federated Learning Round ---")
        
        # 1. Select a subset of nodes (clients)
        # For simplicity, we use all nodes here
        selected_nodes = self.edge_nodes
        
        # 2. Collect updates from selected nodes
        local_updates = [node.get_local_update() for node in selected_nodes]
        
        # 3. Aggregate updates (Federated Averaging)
        global_weights = self.aggregate_updates(local_updates)
        self.global_model.load_state_dict(global_weights)
        
        # 4. Distribute the new global model back to all nodes
        self.distribute_model()
        
        logger.info("--- Federated Learning Round Complete ---\n")

    def aggregate_updates(self, local_updates: list) -> OrderedDict:
        """Averages the weights from local models as per FedAvg algorithm."""
        agg_weights = OrderedDict()
        
        for key in local_updates[0].keys():
            agg_weights[key] = torch.stack([updates[key].to(self.device) for updates in local_updates], 0).sum(0)
        
        num_updates = len(local_updates)
        for key in agg_weights:
            agg_weights[key] = torch.div(agg_weights[key], num_updates)
            
        logger.info("Server: Aggregated updates from all nodes.")
        return agg_weights

    def distribute_model(self):
        """Sends the updated global model to all edge nodes."""
        for node in self.edge_nodes:
            node.local_generator.load_state_dict(self.global_model.state_dict())
        logger.info("Server: Distributed updated global model to all nodes.")
