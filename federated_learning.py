import torch
from collections import OrderedDict

class FederatedServer:
    def __init__(self, edge_nodes: list, global_model: torch.nn.Module, device: torch.device):
        self.edge_nodes = edge_nodes
        self.global_model = global_model.to(device)
        self.device = device

    def run_round(self):
        selected_nodes = self.edge_nodes
        local_updates = [node.get_local_update() for node in selected_nodes]
        global_weights = self.aggregate_updates(local_updates)
        self.global_model.load_state_dict(global_weights)
        self.distribute_model()

    def aggregate_updates(self, local_updates: list[tuple[OrderedDict, int]]) -> OrderedDict:
        if not local_updates:
            return self.global_model.state_dict()
        total_data_points = sum(num_points for _, num_points in local_updates)
        if total_data_points == 0:
            return self.global_model.state_dict()
        agg_weights = OrderedDict()
        for key in local_updates[0][0].keys():
            agg_weights[key] = torch.zeros_like(local_updates[0][0][key], device=self.device)
        for weights, num_points in local_updates:
            if num_points == 0:
                continue
            weight_contribution = num_points / total_data_points
            for key in weights.keys():
                agg_weights[key] += weights[key].to(self.device) * weight_contribution
        return agg_weights

    def distribute_model(self):
        for node in self.edge_nodes:
            node.local_generator.load_state_dict(self.global_model.state_dict())
