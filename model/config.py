from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Image Information
    image_size: int = 256
    in_channels: int = 3

    # Encoder
    encoder_channels: list = field(default_factory=lambda: [32, 64, 128, 256])
    encoder_kernel_sizes: list = field(default_factory=lambda: [3, 3, 3, 3])
    encoder_strides: list = field(default_factory=lambda: [2, 2, 2, 2])
    encoder_padding: list = field(default_factory=lambda: [1, 1, 1, 1])

    # Decoder
    decoder_channels: list = field(default_factory=lambda: [256, 128, 64, 32])
    decoder_kernel_sizes: list = field(default_factory=lambda: [4, 4, 4, 4])
    decoder_strides: list = field(default_factory=lambda: [2, 2, 2, 2])
    decoder_padding: list = field(default_factory=lambda: [1, 1, 1, 1])

    # Latent Space
    latent_dim: int = 128
