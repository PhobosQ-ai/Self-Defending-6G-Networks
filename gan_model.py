import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class Generator(nn.Module):
    """
    Conditional GAN Generator.
    Generates a decoy (network flow features) based on a noise vector and a condition.
    """
    def __init__(self, latent_dim: int, condition_dim: int, output_dim: int, device: torch.device):
        super(Generator, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, output_dim),
            nn.Tanh()
        ).to(device)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        combined_input = torch.cat([z, y], dim=1)
        return self.model(combined_input)

class Discriminator(nn.Module):
    """
    Conditional GAN Discriminator (Critic in WGAN-GP).
    Evaluates the realism of a given network flow conditioned on its type.
    """
    def __init__(self, input_dim: int, condition_dim: int, device: torch.device):
        super(Discriminator, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        ).to(device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        combined_input = torch.cat([x, y], dim=1)
        return self.model(combined_input)

def compute_gradient_penalty(discriminator: Discriminator, real_samples: torch.Tensor, fake_samples: torch.Tensor, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Calculates the gradient penalty loss for WGAN GP as per the paper's formulation."""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.randn(real_samples.size(0), 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates, labels)
    
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
