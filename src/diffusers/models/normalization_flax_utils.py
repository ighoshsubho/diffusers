from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..utils import logging


logger = logging.get_logger(__name__)

class FlaxSD35AdaLayerNormZeroX(nn.Module):
    """
    Norm layer adaptive layer norm zero (AdaLN-Zero) for SD 3.5.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        norm_type (`str`): Type of normalization to use.
        bias (`bool`): Whether to include bias in linear layer.
        dtype (`jnp.dtype`): The dtype of the computation.
    """
    embedding_dim: int
    norm_type: str = "layer_norm"
    bias: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.silu = nn.silu
        self.linear = nn.Dense(
            features=9 * self.embedding_dim,
            use_bias=self.bias,
            dtype=self.dtype
        )

        if self.norm_type == "layer_norm":
            self.norm = nn.LayerNorm(
                epsilon=1e-6,
                use_bias=False,
                dtype=self.dtype
            )
        else:
            raise ValueError(f"Unsupported `norm_type` ({self.norm_type}) provided. Only 'layer_norm' is currently supported.")

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        emb: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, ...]:
        emb = self.linear(self.silu(emb))

        # Split into 9 chunks for shifts, scales and gates
        chunks = jnp.split(emb, indices_or_sections=9, axis=1)
        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        shift_mlp, scale_mlp, gate_mlp = chunks[3], chunks[4], chunks[5]
        shift_msa2, scale_msa2, gate_msa2 = chunks[6], chunks[7], chunks[8]

        norm_hidden_states = self.norm(hidden_states)
        hidden_states = norm_hidden_states * (1 + scale_msa[:, None]) + shift_msa[:, None]
        norm_hidden_states2 = norm_hidden_states * (1 + scale_msa2[:, None]) + shift_msa2[:, None]

        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2


class FlaxAdaLayerNormZero(nn.Module):
    """
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        norm_type (`str`): Type of normalization to use.
        bias (`bool`): Whether to include bias in linear layer.
        dtype (`jnp.dtype`): The dtype of the computation.
    """
    embedding_dim: int
    norm_type: str = "layer_norm"
    bias: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.silu = nn.silu
        self.linear = nn.Dense(
            features=6 * self.embedding_dim,
            use_bias=self.bias,
            dtype=self.dtype
        )

        if self.norm_type == "layer_norm":
            self.norm = nn.LayerNorm(
                epsilon=1e-6,
                use_bias=False,
                dtype=self.dtype
            )
        elif self.norm_type == "fp32_layer_norm":
            # In Flax, we handle dtype conversions differently
            self.norm = nn.LayerNorm(
                epsilon=1e-6,
                use_bias=False,
                dtype=jnp.float32
            )
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({self.norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def __call__(
        self,
        x: jnp.ndarray,
        emb: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        emb = self.linear(self.silu(emb))

        # Split into 6 chunks for shifts, scales and gates
        chunks = jnp.split(emb, indices_or_sections=6, axis=1)
        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        shift_mlp, scale_mlp, gate_mlp = chunks[3], chunks[4], chunks[5]

        # Handle fp32 normalization if needed
        if self.norm_type == "fp32_layer_norm":
            x = self.norm(x.astype(jnp.float32)).astype(self.dtype)
        else:
            x = self.norm(x)

        x = x * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class FlaxAdaLayerNormContinuous(nn.Module):
    """
    Continuous adaptive layer normalization for SD3.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        conditioning_embedding_dim (`int`): Dimension of the conditioning embeddings.
        elementwise_affine (`bool`): Whether to use elementwise affine parameters.
        eps (`float`): Small value for numerical stability.
        bias (`bool`): Whether to include bias in linear layer.
        norm_type (`str`): Type of normalization to use.
        dtype (`jnp.dtype`): The dtype of the computation.
    """
    embedding_dim: int
    conditioning_embedding_dim: int
    elementwise_affine: bool = True
    eps: float = 1e-5
    bias: bool = True
    norm_type: str = "layer_norm"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.silu = nn.silu
        self.linear = nn.Dense(
            features=self.embedding_dim * 2,
            use_bias=self.bias,
            dtype=self.dtype
        )

        if self.norm_type == "layer_norm":
            self.norm = nn.LayerNorm(
                epsilon=self.eps,
                use_bias=self.bias,
                dtype=self.dtype,
                param_dtype=self.dtype if self.elementwise_affine else None
            )
        elif self.norm_type == "rms_norm":
            # Note: RMSNorm in Flax is slightly different from PyTorch
            self.norm = RMSNorm(
                dim=self.embedding_dim,
                eps=self.eps,
                dtype=self.dtype,
                param_dtype=self.dtype if self.elementwise_affine else None
            )
        else:
            raise ValueError(f"unknown norm_type {self.norm_type}")

    def __call__(self, x: jnp.ndarray, conditioning_embedding: jnp.ndarray) -> jnp.ndarray:
        # Ensure conditioning_embedding is in the correct dtype
        conditioning_embedding = conditioning_embedding.astype(x.dtype)

        emb = self.linear(self.silu(conditioning_embedding))
        scale, shift = jnp.split(emb, indices_or_sections=2, axis=1)

        x = self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]
        return x


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Parameters:
        dim (`int`): The dimension to normalize over.
        eps (`float`): Small value for numerical stability.
        dtype (`jnp.dtype`): The dtype of the computation.
        param_dtype (`Optional[jnp.dtype]`): The dtype for parameters.
    """
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: Optional[jnp.dtype] = None

    def setup(self):
        if self.param_dtype is not None:
            self.weight = self.param(
                'weight',
                nn.initializers.ones,
                (self.dim,),
                self.param_dtype
            )
        else:
            self.weight = None

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        # Cast to float32 for better numerical stability
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)

        if self.weight is not None:
            hidden_states = hidden_states * self.weight

        return hidden_states.astype(self.dtype)
