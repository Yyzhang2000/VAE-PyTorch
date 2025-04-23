import torch
import torch.nn as nn
import torch.nn.functional as F


from .config import ModelConfig


class _ResidualBlock(nn.Module):
    def __init__(self, in_dim: int = 64, out_dim: int = 64, groups: int = 1, scale=1.0):
        super().__init__()

        self.scale = scale
        self.hidden_dim = int(out_dim * scale)
        self.groups = groups

        if in_dim != out_dim:
            self.expand = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.expand = nn.Identity()

        self.conv1 = nn.Conv2d(
            in_dim,
            self.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.hidden_dim)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(
            self.hidden_dim,
            out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.relu2 = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor):
        identity = self.expand(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out += identity

        out = self.relu2(self.bn2(out))
        return out


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        assert (
            2 ** len(config.encoder_channels)
        ) * 4 == config.image_size, "Image size must be divisible by 2^N, where N is the number of encoder channels."

        self.in_channels = config.in_channels
        self.image_size = config.image_size

        layers = list()

        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    config.encoder_channels[0],
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    bias=False,
                ),
                nn.BatchNorm2d(config.encoder_channels[0]),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2),
            )
        )

        for i in range(1, len(config.encoder_channels)):
            layers.append(
                nn.Sequential(
                    _ResidualBlock(
                        config.encoder_channels[i - 1],
                        config.encoder_channels[i],
                        groups=1,
                        scale=1,
                    ),
                    nn.AvgPool2d(2),
                )
            )

        layers.append(
            _ResidualBlock(
                config.encoder_channels[-1],
                config.encoder_channels[-1],
                groups=1,
                scale=1,
            )
        )

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

        assert (
            2 ** len(config.encoder_channels)
        ) * 4 == config.image_size, "Image size must be divisible by 2^N, where N is the number of encoder channels."

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

        layers.append(
            _ResidualBlock(
                config.decoder_channels[0],
                config.decoder_channels[0],
                groups=1,
                scale=1,
            ),
        )
        layers.append(
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        layers.append(nn.BatchNorm2d(config.decoder_channels[0]))

        for i in range(1, len(config.decoder_channels)):
            layers.append(
                _ResidualBlock(
                    config.decoder_channels[i - 1],
                    config.decoder_channels[i],
                    groups=1,
                    scale=1,
                ),
            )
            layers.append(
                nn.Upsample(scale_factor=2, mode="nearest"),
            )
            layers.append(nn.BatchNorm2d(config.decoder_channels[i]))

        layers.append(
            _ResidualBlock(
                config.decoder_channels[-1],
                config.decoder_channels[-1],
                groups=1,
                scale=1,
            )
        )

        layers.append(
            nn.Conv2d(
                config.decoder_channels[-1],
                config.in_channels,
                kernel_size=5,
                stride=1,
                padding=2,
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
