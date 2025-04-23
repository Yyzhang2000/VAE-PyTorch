import torch

from tqdm import tqdm
import numpy as np
import logging
import os
import shutil
import random
from torchvision.utils import make_grid
import torchvision


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    writer,
    epoch,
    scheduler=None,
    device=torch.device("cpu"),
):
    model.train()
    losses = []

    progress_bar = tqdm(
        enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}"
    )
    for batch_idx, image in progress_bar:

        image = image.float().to(device)

        (x_recon, z_mean, z_logvar) = model(image)
        recon_loss, kl_loss = criterion(x_recon, image, z_mean, z_logvar)
        loss = recon_loss + kl_loss

        if torch.isnan(loss):
            print("NaN detected in loss!")
            print("mu:", recon_loss.abs().mean().item())
            print("logvar:", z_logvar.min().item(), z_logvar.max().item())
            raise RuntimeError("Loss went NaN")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        recon_loss = recon_loss.item()
        losses.append(loss.item())

        if writer is not None:
            writer.add_scalar(
                "train/recon_loss", recon_loss, epoch * len(train_loader) + batch_idx
            )
            writer.add_scalar(
                "train/kl_loss", kl_loss.item(), epoch * len(train_loader) + batch_idx
            )
            writer.add_scalar(
                "train/loss", loss.item(), epoch * len(train_loader) + batch_idx
            )

        progress_bar.set_postfix(
            recon_loss=recon_loss,
            loss=loss.item(),
        )

    return np.mean(losses)


def train(model, optimizer, train_loader, criterion, writer, train_config, device):
    model.to(device)

    # Get random sample from the dataset
    image = next(iter(train_loader))
    random_indices = torch.randint(low=0, high=len(image), size=(10,))
    random_images = image[random_indices]
    image_dir = os.path.join(train_config.model_dir, "sample_images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)

    images_grid = make_grid(random_images, nrow=5, normalize=True)
    writer.add_image("train/sample_images", images_grid, 0)
    # Save the random images
    torchvision.utils.save_image(
        images_grid,
        os.path.join(image_dir, "sample_images.png"),
    )
    random_images = random_images.to(device)
    # Save images

    best_loss = float("inf")
    for epoch in range(train_config.num_epochs):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            writer=writer,
            epoch=epoch,
            device=device,
        )
        logging.info(
            f"Epoch [{epoch + 1}/{train_config.num_epochs}], "
            f"Train Loss: {train_loss:.4f}"
        )
        torch.save(
            model.state_dict(),
            os.path.join(train_config.model_dir, f"latest_vae_model.pth"),
        )
        if train_loss < best_loss:
            best_loss = train_loss
            shutil.copyfile(
                os.path.join(train_config.model_dir, "latest_vae_model.pth"),
                os.path.join(train_config.model_dir, "best_vae_model.pth"),
            )
            logging.info(f"Best model saved with loss: {best_loss:.4f}")

        if (epoch + 1) % 20 == 0:
            # Save a sample of the model's output
            with torch.no_grad():
                model.eval()
                sample_images = model.sample(images=random_images)
                sample_images_grid = make_grid(sample_images, nrow=5, normalize=True)
                writer.add_image("train/sample_images", sample_images_grid, (epoch + 1))
                # Save the sample images
                sample_images_path = os.path.join(
                    image_dir,
                    f"sample_images_epoch_{epoch + 1}.png",
                )
                torchvision.utils.save_image(sample_images_grid, sample_images_path)
                logging.info(
                    f"Sample images at {epoch + 1} saved at {os.path.join(image_dir, f'sample_images_epoch_{epoch + 1}.png')}"
                )

    logging.info(f"Training complete. Best train loss: {best_loss:.2f}%")
