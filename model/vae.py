import torch
import torch.nn as nn
import torch.nn.functional as F


from .config import ModelConfig


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.in_channels = config.in_channels
        self.image_size = config.image_size

        layers = list()

        layers.append(
            nn.Conv2d(
                self.in_channels,
                config.encoder_channels[0],
                kernel_size=config.encoder_kernel_sizes[0],
                stride=config.encoder_strides[0],
                padding=config.encoder_padding[0],
            )
        )
        layers.append(nn.LeakyReLU(0.2))

        for i in range(1, len(config.encoder_channels)):
            layers.append(
                nn.Conv2d(
                    config.encoder_channels[i - 1],
                    config.encoder_channels[i],
                    kernel_size=config.encoder_kernel_sizes[i],
                    stride=config.encoder_strides[i],
                    padding=config.encoder_padding[i],
                )
            )
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm2d(config.encoder_channels[i]))

        self.encoder = nn.Sequential(*layers)

        self.output_size = self._compute_output_size()
        self.fc_mu = nn.Linear(self.output_size, config.latent_dim * 2)

    def _compute_output_size(self):
        pseudo_input = torch.randn(
            1, self.in_channels, self.image_size, self.image_size
        )
        output = self.encoder(pseudo_input)
        return output.shape[1] * output.shape[2] * output.shape[3]

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        z_mean, z_logvar = torch.chunk(self.fc_mu(x), 2, dim=-1)

        return z_mean, z_logvar


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.latent_dim = config.latent_dim
        self.output_size = config.image_size
        self.config = config

        self.spatial_h, self.spatial_w = self._compute_spatial_dim(config)

        self.fc = nn.Sequential(
            nn.Linear(
                config.latent_dim,
                config.decoder_channels[0] * self.spatial_h * self.spatial_w,
            ),
            nn.ReLU(),
        )

        layers = list()

        for i in range(1, len(config.decoder_channels)):
            layers.append(
                nn.ConvTranspose2d(
                    config.decoder_channels[i - 1],
                    config.decoder_channels[i],
                    kernel_size=config.decoder_kernel_sizes[i - 1],
                    stride=config.decoder_strides[i - 1],
                    padding=config.decoder_padding[i - 1],
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(config.decoder_channels[i]))

        layers.append(
            nn.ConvTranspose2d(
                config.decoder_channels[-1],
                config.in_channels,
                kernel_size=config.decoder_kernel_sizes[-1],
                stride=config.decoder_strides[-1],
                padding=config.decoder_padding[-1],
            )
        )

        self.decoder = nn.Sequential(*layers)

    def _compute_spatial_dim(self, config):
        dummy_input = torch.randn(
            1, config.in_channels, config.image_size, config.image_size
        )
        encoder = Encoder(config)
        output = encoder.encoder(dummy_input)
        return output.shape[2], output.shape[3]

    def forward(self, z: torch.Tensor):
        z = self.fc(z)
        z = z.view(
            z.size(0),
            self.config.decoder_channels[0],
            self.spatial_h,
            self.spatial_w,
        )
        x_recon = self.decoder(z)
        return x_recon


class VAE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        z_mean, z_logvar = self.encoder(x)
        z_logvar = torch.clamp(z_logvar, min=-10, max=10)  # Prevent numerical issues
        z = self.reparameterize(z_mean, z_logvar)
        x_recon = self.decoder(z)

        return x_recon, z_mean, z_logvar

    @torch.no_grad()
    def sample(self, num_samples: int = 8, images=None):
        if images is None:
            z = torch.randn(num_samples, self.encoder.fc_mu.out_features // 2)
        else:
            z_mean, z_logvar = self.encoder(images)
            z = self.reparameterize(z_mean, z_logvar)
        x_recon = self.decoder(z)
        return x_recon


if __name__ == "__main__":
    # Example usage
    config = ModelConfig()
    vae = VAE(config)

    # Create a random input tensor
    x = torch.randn(8, config.in_channels, config.image_size, config.image_size)

    # Forward pass
    x_recon, z_mean, z_logvar = vae(x)

    print()
    print("Input shape:", x.shape)
    print("Reconstructed shape:", x_recon.shape)
    print("Latent mean shape:", z_mean.shape)
    print("Latent log variance shape:", z_logvar.shape)
