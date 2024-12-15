from typing import Any, Dict, List, Optional, Tuple, Union

import flax.linen as nn
import jax.numpy as jnp

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.attention_flax import FlaxFeedForward, FlaxJointTransformerBlock, FlaxAttention
from ...models.attention_processor_flax import FlaxJointAttnProcessor2_0
from ...models.modeling_flax_utils import FlaxModelMixin
from ...models.normalization_flax_utils import FlaxAdaLayerNormContinuous, FlaxAdaLayerNormZero
from ...utils import logging
from ..embeddings_flax import FlaxCombinedTimestepTextProjEmbeddings, FlaxPatchEmbed


logger = logging.get_logger(__name__)


class FlaxSD3SingleTransformerBlock(nn.Module):
    """
    A Single Transformer block as part of the MMDiT architecture, used in Stable Diffusion 3 ControlNet.

    Reference: https://arxiv.org/abs/2403.03206

    Args:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dtype (`jnp.dtype`, defaults to jnp.float32): The dtype of the computation.
    """

    dim: int 
    num_attention_heads: int
    attention_head_dim: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.norm1 = FlaxAdaLayerNormZero(self.dim, dtype=self.dtype)
        self.attn = FlaxAttention(
            query_dim=self.dim,
            dim_head=self.attention_head_dim,
            heads=self.num_attention_heads,  
            out_dim=self.dim,
            bias=True,
            processor=FlaxJointAttnProcessor2_0(),
            eps=1e-6,
            dtype=self.dtype
        )
        self.norm2 = nn.LayerNorm(self.dim, epsilon=1e-6, dtype=self.dtype, use_bias=False)
        self.ff = FlaxFeedForward(dim=self.dim, dim_out=self.dim, activation_fn="gelu", dtype=self.dtype)

    def __call__(self, hidden_states: jnp.ndarray, temb: jnp.ndarray) -> jnp.ndarray:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        
        # Attention
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None
        )

        # Process attention outputs
        attn_output = gate_msa.reshape(-1, 1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.reshape(-1, 1) * ff_output

        hidden_states = hidden_states + ff_output

        return hidden_states


class FlaxSD3Transformer2DModel(FlaxModelMixin, ConfigMixin):
    """
    The Transformer model introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        sample_size (`int`): The width of the latent images.
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to 16): Number of channels in the input.
        num_layers (`int`, defaults to 18): Number of transformer blocks.
        attention_head_dim (`int`, defaults to 64): Number of channels in each head.
        num_attention_heads (`int`, defaults to 18): Number of heads for attention.
        cross_attention_dim (`int`): Number of encoder_hidden_states dimensions.
        caption_projection_dim (`int`): Dimensions for projecting encoder_hidden_states.
        pooled_projection_dim (`int`): Dimensions for projecting pooled_projections.
        out_channels (`int`, defaults to 16): Number of output channels.
        dtype (`jnp.dtype`, defaults to jnp.float32): The dtype of the computation.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        dual_attention_layers: Tuple[int, ...] = (),
        qk_norm: Optional[str] = None,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.dtype = dtype

    def setup(self):
        self.pos_embed = FlaxPatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=self.config.pos_embed_max_size,
            dtype=self.dtype
        )

        self.time_text_embed = FlaxCombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=self.config.pooled_projection_dim,
            dtype=self.dtype
        )

        self.context_embedder = nn.Dense(
            features=self.config.caption_projection_dim,
            dtype=self.dtype,
            use_bias=True
        )

        self.transformer_blocks = [
            FlaxJointTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=self.config.num_attention_heads,
                attention_head_dim=self.config.attention_head_dim,
                context_pre_only=i == self.config.num_layers - 1,
                qk_norm=self.config.qk_norm,
                use_dual_attention=True if i in self.config.dual_attention_layers else False,
                dtype=self.dtype
            )
            for i in range(self.config.num_layers)
        ]

        self.norm_out = FlaxAdaLayerNormContinuous(
            self.inner_dim,
            self.inner_dim,
            elementwise_affine=False,
            eps=1e-6,
            dtype=self.dtype
        )
        
        self.proj_out = nn.Dense(
            features=self.config.patch_size * self.config.patch_size * self.out_channels,
            dtype=self.dtype,
            use_bias=True
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        pooled_projections: Optional[jnp.ndarray] = None,
        timestep: Optional[jnp.ndarray] = None,
        block_controlnet_hidden_states: Optional[List] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
        deterministic: bool = True,
    ) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
        height, width = hidden_states.shape[-2:]

        # Embed and add positions
        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # Process blocks
        for index_block, block in enumerate(self.transformer_blocks):
            # Skip specified layers
            if skip_layers is not None and index_block in skip_layers:
                continue

            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                deterministic=deterministic
            )

            # Add controlnet residuals if provided
            if block_controlnet_hidden_states is not None and not block.context_pre_only:
                interval_control = len(self.transformer_blocks) / len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[int(index_block / interval_control)]

        # Final norm and projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # Unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            (-1, height, width, patch_size, patch_size, self.out_channels)
        )
        # Equivalent to torch.einsum("nhwpqc->nchpwq", hidden_states)
        hidden_states = jnp.transpose(hidden_states, (0, 5, 1, 3, 2, 4))
        output = hidden_states.reshape(
            (-1, self.out_channels, height * patch_size, width * patch_size)
        )

        if not return_dict:
            return (output,)

        return {"sample": output}