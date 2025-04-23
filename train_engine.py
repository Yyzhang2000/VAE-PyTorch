import torch

from tqdm import tqdm
import numpy as np
import logging
import os
import shutil
import random
from torchvision.utils import make_grid
import torchvision


def kl_annealing_scheduler(epoch, total_epochs, start=0.0, end=1.0):
    """Anneal the KL divergence term in the loss function."""
    # Sigmoid function to control the KL annealing
    return float(1 / (1 + np.exp(-(epoch - total_epochs / 2) / (total_epochs / 10))))


def evaluate_model(
    model,
    test_loader,
    criterion,
    writer,
    epoch,
    device=torch.device("cpu"),
):
    model.eval()
    losses = []
    recon_losses = []
    kl_losses = []

    real_images = []
    generated_images = []
    progress_bar = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc=f"Evaluating epoch {epoch+1}",
    )
    with torch.no_grad():
        for batch_idx, image in progress_bar:
            image = image.float().to(device)
            (x_recon, z_mean, z_logvar) = model(image)
            recon_loss, kl_loss = criterion(x_recon, image, z_mean, z_logvar)
            loss = recon_loss + kl_loss

            loss = loss.item()
            recon_loss = recon_loss.item()
            kl_loss = kl_loss.item()
            losses.append(loss)
            recon_losses.append(recon_loss)
            kl_losses.append(kl_loss)

            real_images.append(image.cpu())
            generated_images.append(x_recon.cpu())

            if writer is not None:
                writer.add_scalar(
                    "recon_loss/test", recon_loss, epoch * len(test_loader) + batch_idx
                )
                writer.add_scalar(
                    "kl_loss/test", kl_loss, epoch * len(test_loader) + batch_idx
                )
                writer.add_scalar(
                    "loss/test", loss, epoch * len(test_loader) + batch_idx
                )

    return np.mean(losses)


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    writer,
    total_epochs,
    epoch,
    scheduler=None,
    device=torch.device("cpu"),
):
    model.train()
    losses = []
    recon_losses = []

    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Training epoch {epoch+1}",
    )
    for batch_idx, image in progress_bar:
        beta = kl_annealing_scheduler(epoch, total_epochs=total_epochs)
        image = image.float().to(device)

        (x_recon, z_mean, z_logvar) = model(image)
        recon_loss, kl_loss = criterion(x_recon, image, z_mean, z_logvar)
        kl_loss = beta * kl_loss
        loss = recon_loss + beta * kl_loss

        if torch.isnan(loss):
            print("NaN detected in loss!")
            print("mu:", recon_loss.abs().mean().item())
            print("logvar:", z_logvar.min().item(), z_logvar.max().item())
            raise RuntimeError("Loss went NaN")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recon_loss = recon_loss.item()
        losses.append(loss.item())
        recon_losses.append(recon_loss)

        if writer is not None:
            writer.add_scalar(
                "recon_loss/train", recon_loss, epoch * len(train_loader) + batch_idx
            )
            writer.add_scalar(
                "kl_loss/train", kl_loss.item(), epoch * len(train_loader) + batch_idx
            )
            writer.add_scalar(
                "loss/train", loss.item(), epoch * len(train_loader) + batch_idx
            )
            writer.add_scalar("beta/train", beta, epoch * len(train_loader) + batch_idx)

        progress_bar.set_postfix(
            recon_loss=np.mean(recon_losses),
            loss=np.mean(losses),
        )

    if writer is not None:
        writer.add_scalar("learning_rate/train", optimizer.param_groups[0]["lr"], epoch)
    if scheduler is not None:
        scheduler.step()

    return np.mean(losses)


def train(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    criterion,
    writer,
    train_config,
    device,
):
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
            scheduler=scheduler,
            writer=writer,
            total_epochs=train_config.num_epochs,
            epoch=epoch,
            device=device,
        )

        val_loss = evaluate_model(
            model=model,
            test_loader=val_loader,
            criterion=criterion,
            writer=writer,
            epoch=epoch,
            device=device,
        )

        logging.info(
            f"Epoch [{epoch + 1}/{train_config.num_epochs}], "
            f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}"
        )

        torch.save(
            model.state_dict(),
            os.path.join(train_config.model_dir, f"latest_vae_model.pth"),
        )
        if val_loss < best_loss:
            best_loss = val_loss
            shutil.copyfile(
                os.path.join(train_config.model_dir, "latest_vae_model.pth"),
                os.path.join(train_config.model_dir, "best_vae_model.pth"),
            )
            logging.info(f"Best model saved with loss: {best_loss:.4f}")

        if (epoch + 1) % 2 == 0:
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
