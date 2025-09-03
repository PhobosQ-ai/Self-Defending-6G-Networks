import random
import torch
from gan_model import compute_gradient_penalty

class EdgeNode:
    def __init__(self, node_id: int, generator_model: torch.nn.Module, discriminator_model: torch.nn.Module, 
                 dataloader: torch.utils.data.DataLoader, device: torch.device, latent_dim: int, 
                 num_classes: int, lr: float, b1: float, b2: float, gp_weight: float):
        self.node_id = node_id
        self.device = device
        self.local_generator = generator_model.to(device)
        self.local_discriminator = discriminator_model.to(device)
        self.optimizer_G = torch.optim.Adam(self.local_generator.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_D = torch.optim.Adam(self.local_discriminator.parameters(), lr=lr, betas=(b1, b2))
        self.dataloader = dataloader
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.gp_weight = gp_weight
        self.active_decoys = []
        self.attack_log = []

    def deploy_decoys(self, num_decoys: int, decoy_type: int):
        self.local_generator.eval()
        self.active_decoys = []
        labels = torch.zeros(num_decoys, self.num_classes, device=self.device)
        labels[:, decoy_type] = 1
        z = torch.randn(num_decoys, self.latent_dim, device=self.device)
        with torch.no_grad():
            decoy_features = self.local_generator(z, labels)
            self.active_decoys = list(torch.split(decoy_features, 1))

    def monitor_traffic(self) -> torch.Tensor | None:
        if self.active_decoys and random.random() < 0.1:
            attacker_ip = f"192.168.10.{random.randint(100, 200)}"
            interaction_data = random.choice(self.active_decoys)
            self.attack_log.append(interaction_data)
            return interaction_data
        return None

    def get_local_update(self) -> tuple[dict, int]:
        self.local_generator.train()
        self.local_discriminator.train()
        data_points_trained = 0
        try:
            real_samples, real_labels_idx = next(iter(self.dataloader))
            real_samples = real_samples.to(self.device)
            real_labels_idx = real_labels_idx.to(self.device)
            real_labels = torch.nn.functional.one_hot(real_labels_idx, num_classes=self.num_classes).float()
            self.optimizer_D.zero_grad()
            real_validity = self.local_discriminator(real_samples, real_labels)
            z = torch.randn(real_samples.size(0), self.latent_dim, device=self.device)
            gen_labels_idx = torch.randint(0, self.num_classes, (real_samples.size(0),), device=self.device)
            gen_labels = torch.nn.functional.one_hot(gen_labels_idx, num_classes=self.num_classes).float()
            fake_samples = self.local_generator(z, gen_labels)
            fake_validity = self.local_discriminator(fake_samples.detach(), gen_labels)
            gradient_penalty = compute_gradient_penalty(self.local_discriminator, real_samples.data, fake_samples.data, real_labels.data, self.device, self.gp_weight)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty
            d_loss.backward()
            self.optimizer_D.step()
            if self.node_id % 5 == 0:
                self.optimizer_G.zero_grad()
                z = torch.randn(real_samples.size(0), self.latent_dim, device=self.device)
                gen_labels_idx = torch.randint(0, self.num_classes, (real_samples.size(0),), device=self.device)
                gen_labels = torch.nn.functional.one_hot(gen_labels_idx, num_classes=self.num_classes).float()
                gen_samples = self.local_generator(z, gen_labels)
                fake_validity = self.local_discriminator(gen_samples, gen_labels)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                self.optimizer_G.step()
            data_points_trained = real_samples.size(0)
        except StopIteration:
            pass
        except Exception as e:
            pass
        return self.local_generator.state_dict(), data_points_trained
