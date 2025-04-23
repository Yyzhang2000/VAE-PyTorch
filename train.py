from train_engine import train
import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from dataset import PokemonDataset

from model.vae import VAE
from model.config import ModelConfig


#### Config
from dataclasses import dataclass

from utils import *


@dataclass
class TRAIN_CONFIG:
    # Config for training
    seed = 42
    log_dir: str = "./logs"
    task_name: str = "pokemon2000-vae"
    model_dir: str = "models"
    num_epochs: int = 10
    batch_size: int = 64

    # Optimizer
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 1e-5

    def __post_init__(self):
        self.model_dir = os.path.join(self.log_dir, self.task_name)


if __name__ == "__main__":
    training_config = TRAIN_CONFIG()

    os.makedirs(training_config.model_dir, exist_ok=True)
    model_config = ModelConfig()
    save_config(training_config, model_config)
    set_logger(os.path.join(training_config.model_dir, "train.log"))

    logging.info(
        f"Configuration saved to {os.path.join(training_config.model_dir, 'config.yaml')}"
    )

    # Common setup
    set_seed(training_config.seed)
    writer = set_tensorboard_writer(training_config.model_dir)
    set_logger(os.path.join(training_config.model_dir, "train.log"))
    device = get_device()
    # End of common setup

    # Dataset
    train_dataset = PokemonDataset(root_dir="data/train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dataset = PokemonDataset(root_dir="data/val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    # End of dataset

    # Model
    model = VAE(model_config)
    model.to(device)
    # End of model

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG.learning_rate,
        betas=TRAIN_CONFIG.betas,
        weight_decay=TRAIN_CONFIG.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_config.num_epochs, eta_min=training_config.min_lr
    )
    # End of optimizer

    logging.info("Starting training...")

    # Loss function for VAE
    def criterion(recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss, kl_loss

    train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        writer=writer,
        train_config=training_config,
        scheduler=scheduler,
        device=device,
    )
