# Self-Defending 6G Networks Through AI-Driven Adaptive Decoy Generation

This repository contains the simulation code for the research paper titled "Self-Defending 6G Networks Through AI-Driven Adaptive Decoy Generation at the Edge".

## Overview

The framework provides a proactive security mechanism for 6G networks using three core AI components:
1. **Conditional GAN (cGAN)**: To generate realistic, context-aware network decoys.
2. **Reinforcement Learning (PPO)**: To dynamically manage decoy deployment strategies.
3. **Federated Learning**: To synchronize models across distributed edge nodes in a privacy-preserving manner.

## Features

- Simulation of edge nodes in a 6G network environment.
- Generation of adaptive decoys using conditional GANs.
- Dynamic decision-making with Proximal Policy Optimization (PPO).
- Privacy-preserving model synchronization via Federated Learning.
- Data preprocessing and visualization for the CIC-IoT-2023 dataset.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd py-files
   ```

# Self-Defending 6G Networks — Adaptive Decoy Generation (Math-enabled README)

This folder contains the simulation code that implements the Adaptive Decoy Generation framework described in the paper "Self-Defending 6G Networks Through AI-Driven Adaptive Decoy Generation at the Edge".

This README includes the core mathematical expressions used by the implementation so they render in Markdown viewers that support TeX (MathJax/KaTeX).

## Quick start

1. Clone the repository and change into the python folder:

```bash
git clone <repository-url>
cd "c:\PhobosQ - docs\latex docs\6g files\py files"
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the simulation:

```bash
python main.py
```

## What this code implements

- Conditional Generative Adversarial Network (cGAN) for context-aware decoy generation.
- Reinforcement Learning (PPO) agent for dynamic decoy deployment and resource management.
- Federated Learning (FedAvg-like) synchronization across distributed edge nodes.
- Data loading, preprocessing, plotting utilities and a simulation harness.

## Mathematical details (rendered in TeX-aware Markdown)

Below are the key equations and loss functions referenced in the implementation and paper. Put simply, these are the formal contracts that guide training and synchronization.

### 1) cGAN objective

The canonical conditional GAN objective used to train generator G and discriminator D (conditioned on y) is:

$$
\min_G \max_D \; \mathbb{E}_{x\sim p_{\text{data}},\,y\sim p_y}[\log D(x,y)] \\
\quad\; + \, \mathbb{E}_{z\sim p_z,\,y\sim p_y}[\log(1 - D(G(z,y),y))].
$$

In practice we use a stabilized variant (WGAN-GP) for improved training stability.

### 2) WGAN-GP discriminator loss (used for stability)

Using the Wasserstein loss with gradient penalty (coefficient $\lambda$):

$$
L_D = \mathbb{E}_{\tilde{x} \sim p_g}[D(\tilde{x},y)] - \mathbb{E}_{x\sim p_{\text{data}}}[D(x,y)] 
\; + \; \lambda\,\mathbb{E}_{\hat{x}\sim p_{\hat{x}}}\big[ (\|\nabla_{\hat{x}} D(\hat{x},y)\|_2 - 1)^2 \big]
$$

where $\hat{x}$ is sampled uniformly on straight lines between real and generated samples.

Generator loss (Wasserstein style) typically minimizes the negative critic score:

$$
L_G = -\mathbb{E}_{z\sim p_z}[D(G(z,y),y)].
$$

### 3) Adversarial feedback (attack-driven fine-tuning)

When attack interaction data $\mathcal{D}_{\text{attack}}$ is available, the generator fine-tuning loss term used in the paper is:

$$
\mathcal{L}_{\text{attack}} = \mathbb{E}_{x_{\text{attack}}\sim\mathcal{D}_{\text{attack}}} \Big[ \min_{z} \| G(z, y_{\text{context}}) - x_{\text{attack}} \|_2^2 \Big].
$$

This expresses a GAN inversion (finding latent $z$ that reconstructs attacker-observed samples) or training an encoder to map attacks into the latent space for targeted fine-tuning.

### 4) Reinforcement learning objective (policy optimization)

The agent seeks a policy $\pi$ that maximizes the expected discounted return:

$$
J(\pi) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t,a_t)\right]
$$

For PPO (Proximal Policy Optimization) the clipped surrogate objective is commonly used:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \big[ \min\big( r_t(\theta)\hat{A}_t, \; \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t \big) \big]
$$

where

$$
r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}, \quad \hat{A}_t = \text{advantage estimate}, \quad \epsilon\text{ is clip parameter.}
$$

### 5) Federated averaging (synchronization)

The federated aggregation step used by the server in the paper is expressed as (a FedAvg-like weighted update):

$$
	heta_{\text{global}}^{(t+1)} \leftarrow \theta_{\text{global}}^{(t)} - \eta \sum_{k\in S_t} \frac{n_k}{n} \Delta\theta_k^{(t)}
$$

where $n_k$ is the size of node $k$'s local dataset, $n=\sum_k n_k$, and $\Delta\theta_k^{(t)}$ denotes the local model update returned by node $k$.

### 6) Evaluation metrics (detection rates)

Detection Rate (DR):

$$
\mathrm{DR} = \frac{\text{Number of detected attacks}}{\text{Total number of attacks}} \times 100\%.
$$

False Positive Rate (FPR):

$$
\mathrm{FPR} = \frac{\text{Number of false positives}}{\text{Total legitimate interactions}} \times 100\%.
$$

Latency (interaction-to-alert) is measured as the elapsed time between the first packet interacting with a decoy and the generated alert at the edge node.

## Project structure

- `main.py` — simulation harness and orchestrator.
- `gan_model.py` — cGAN (Generator / Discriminator) implementations.
- `rl_agent.py` — PPO agent and utilities.
- `edge_node.py` — simulated edge node behaviors and decoy deployment.
- `data_loader.py` — dataset loading and preprocessing (CIC-IoT-2023 adapter).
- `federated_learning.py` — simple FedAvg orchestration used in the simulation.
- `plots.py` — helper functions to render figures used in the paper.
- `requirements.txt` — Python dependencies.