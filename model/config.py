from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Image Information
    image_size: int = 256
    in_channels: int = 3

    # Encoder
    encoder_channels: list = field(
        default_factory=lambda: [64, 128, 256, 512, 512, 512]
    )

    # Decoder
    decoder_channels: list = field(
        default_factory=lambda: [512, 512, 512, 256, 128, 64]
    )

    # Latent Space
    latent_dim: int = 512
