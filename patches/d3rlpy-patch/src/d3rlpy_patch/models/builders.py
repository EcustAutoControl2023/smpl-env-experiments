from typing import Sequence
from d3rlpy.models.encoders import EncoderFactory
from .torch.imitators import CustomConditionalVAE, TemporalConditionalVAE

def create_temporal_conditional_vae(
    observation_shape: Sequence[int],
    action_size: int,
    latent_size: int,
    beta: float,
    encoder_factory: EncoderFactory,
    min_logstd: float = -20.0,
    max_logstd: float = 2.0,
    tl: int = 20,
    k: int = 128,
    net_type: str = "GRU",
    num_layers: int = 1,
) -> TemporalConditionalVAE:
    encoder_encoder = encoder_factory.create_with_action(
        observation_shape, action_size
    )
    decoder_encoder = encoder_factory.create_with_action(
        observation_shape, latent_size
    )
    return TemporalConditionalVAE(
        encoder_encoder,
        decoder_encoder,
        beta,
        min_logstd=min_logstd,
        max_logstd=max_logstd,
        tl=tl,
        k=k,
        net_type=net_type,
        num_layers=num_layers,
    )

def create_custom_conditional_vae(
    observation_shape: Sequence[int],
    action_size: int,
    latent_size: int,
    beta: float,
    encoder_factory: EncoderFactory,
    min_logstd: float = -20.0,
    max_logstd: float = 2.0,
) -> CustomConditionalVAE:
    encoder_encoder = encoder_factory.create_with_action(
        observation_shape, action_size
    )
    decoder_encoder = encoder_factory.create_with_action(
        observation_shape, latent_size
    )
    return CustomConditionalVAE(
        encoder_encoder,
        decoder_encoder,
        beta,
        min_logstd=min_logstd,
        max_logstd=max_logstd,
    )

