# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def get_sinusoidal_embeddings(
    timesteps: jnp.ndarray,
    embedding_dim: int,
    freq_shift: float = 1,
    min_timescale: float = 1,
    max_timescale: float = 1.0e4,
    flip_sin_to_cos: bool = False,
    scale: float = 1.0,
) -> jnp.ndarray:
    """Returns the positional encoding (same as Tensor2Tensor).

    Args:
        timesteps (`jnp.ndarray` of shape `(N,)`):
            A 1-D array of N indices, one per batch element. These may be fractional.
        embedding_dim (`int`):
            The number of output channels.
        freq_shift (`float`, *optional*, defaults to `1`):
            Shift applied to the frequency scaling of the embeddings.
        min_timescale (`float`, *optional*, defaults to `1`):
            The smallest time unit used in the sinusoidal calculation (should probably be 0.0).
        max_timescale (`float`, *optional*, defaults to `1.0e4`):
            The largest time unit used in the sinusoidal calculation.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip the order of sinusoidal components to cosine first.
        scale (`float`, *optional*, defaults to `1.0`):
            A scaling factor applied to the positional embeddings.

    Returns:
        a Tensor of timing signals [N, num_channels]
    """
    assert timesteps.ndim == 1, "Timesteps should be a 1d-array"
    assert embedding_dim % 2 == 0, f"Embedding dimension {embedding_dim} should be even"
    num_timescales = float(embedding_dim // 2)
    log_timescale_increment = math.log(max_timescale / min_timescale) / (num_timescales - freq_shift)
    inv_timescales = min_timescale * jnp.exp(jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)
    emb = jnp.expand_dims(timesteps, 1) * jnp.expand_dims(inv_timescales, 0)

    # scale embeddings
    scaled_time = scale * emb

    if flip_sin_to_cos:
        signal = jnp.concatenate([jnp.cos(scaled_time), jnp.sin(scaled_time)], axis=1)
    else:
        signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)
    signal = jnp.reshape(signal, [jnp.shape(timesteps)[0], embedding_dim])
    return signal

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False, extra_tokens: int = 0, base_size: int = 16) -> np.ndarray:
    """
    Creates 2D sinusoidal positional embeddings.

    Args:
        embed_dim (`int`): The embedding dimension.
        grid_size (`int`): Size of the grid (height and width).
        cls_token (`bool`, defaults to `False`): Whether to add classification token.
        extra_tokens (`int`, defaults to `0`): Number of extra tokens.
        base_size (`int`, defaults to `16`): Base size for positional embeddings.

    Returns:
        `np.ndarray`: The positional embeddings with shape `[grid_size * grid_size, embed_dim]`
        or `[1 + grid_size * grid_size, embed_dim]` if using cls_token.
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size)
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """
    Generate 2D sinusoidal positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension.
        grid (`np.ndarray`): Grid of positions.

    Returns:
        `np.ndarray`: Shape `[H * W, embed_dim]`
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # Use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Generate 1D positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension.
        pos (`np.ndarray`): 1D tensor of positions.

    Returns:
        `np.ndarray`: Shape `[M, D]`
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class FlaxTimestepEmbedding(nn.Module):
    r"""
    Time step Embedding Module. Learns embeddings for input time steps.

    Args:
        time_embed_dim (`int`, *optional*, defaults to `32`):
            Time step embedding dimension.
        dtype (`jnp.dtype`, *optional*, defaults to `jnp.float32`):
            The data type for the embedding parameters.
    """

    time_embed_dim: int = 32
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, temb):
        temb = nn.Dense(self.time_embed_dim, dtype=self.dtype, name="linear_1")(temb)
        temb = nn.silu(temb)
        temb = nn.Dense(self.time_embed_dim, dtype=self.dtype, name="linear_2")(temb)
        return temb


class FlaxTimesteps(nn.Module):
    r"""
    Wrapper Module for sinusoidal Time step Embeddings as described in https://arxiv.org/abs/2006.11239

    Args:
        dim (`int`, *optional*, defaults to `32`):
            Time step embedding dimension.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip the sinusoidal function from sine to cosine.
        freq_shift (`float`, *optional*, defaults to `1`):
            Frequency shift applied to the sinusoidal embeddings.
    """

    dim: int = 32
    flip_sin_to_cos: bool = False
    freq_shift: float = 1

    @nn.compact
    def __call__(self, timesteps):
        return get_sinusoidal_embeddings(
            timesteps, embedding_dim=self.dim, flip_sin_to_cos=self.flip_sin_to_cos, freq_shift=self.freq_shift
        )

class FlaxPatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding with support for SD3 cropping.

    Args:
        height (`int`): The height of the image.
        width (`int`): The width of the image.
        patch_size (`int`): The size of the patches.
        in_channels (`int`): The number of input channels.
        embed_dim (`int`): The output dimension of the embedding.
        layer_norm (`bool`): Whether to use layer normalization.
        flatten (`bool`): Whether to flatten the output.
        bias (`bool`): Whether to use bias.
        pos_embed_type (`str`): Type of positional embeddings to use.
        pos_embed_max_size (`int`): Maximum size of positional embeddings.
        dtype (`jnp.dtype`): The dtype of the computation.
    """
    height: int = 224
    width: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 768
    layer_norm: bool = False
    flatten: bool = True
    bias: bool = True
    pos_embed_type: str = "sincos"
    pos_embed_max_size: Optional[int] = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.num_patches = (self.height // self.patch_size) * (self.width // self.patch_size)
        kernel_shape = (self.patch_size, self.patch_size)

        # Setup projection
        self.proj = nn.Conv(
            features=self.embed_dim,
            kernel_size=kernel_shape,
            strides=self.patch_size,
            padding="VALID",
            use_bias=self.bias,
            dtype=self.dtype,
        )

        # Setup normalization if needed
        if self.layer_norm:
            self.norm = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype)
        else:
            self.norm = None

        # Calculate dimensions
        self.patch_height = self.height // self.patch_size
        self.patch_width = self.width // self.patch_size
        self.base_size = self.height // self.patch_size

        # Initialize positional embeddings
        grid_size = self.pos_embed_max_size if self.pos_embed_max_size else int(self.num_patches ** 0.5)

        if self.pos_embed_type == "sincos":
            pos_embed = get_2d_sincos_pos_embed(self.embed_dim, grid_size, base_size=self.base_size)
            self.pos_embed = self.param(
                'pos_embed',
                lambda _: jnp.array(pos_embed)[None, ...],
                jnp.zeros((1, pos_embed.shape[0], self.embed_dim))
            )
        elif self.pos_embed_type is not None:
            raise ValueError(f"Unsupported pos_embed_type: {self.pos_embed_type}")

    def cropped_pos_embed(self, height: int, width: int) -> jnp.ndarray:
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size

        if height > self.pos_embed_max_size or width > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) and width ({width}) must be less than pos_embed_max_size: {self.pos_embed_max_size}."
            )

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2

        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = jax.lax.dynamic_slice(
            spatial_pos_embed,
            (0, top, left, 0),
            (1, height, width, spatial_pos_embed.shape[-1])
        )
        return spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])

    def __call__(self, latent: jnp.ndarray) -> jnp.ndarray:
        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            height = latent.shape[-2] // self.patch_size
            width = latent.shape[-1] // self.patch_size

        latent = self.proj(latent)

        if self.flatten:
            latent = jnp.reshape(latent, (latent.shape[0], -1, self.embed_dim))

        if self.layer_norm:
            latent = self.norm(latent)

        if hasattr(self, "pos_embed"):
            if self.pos_embed_max_size:
                pos_embed = self.cropped_pos_embed(height, width)
            else:
                if self.patch_height != height or self.patch_width != width:
                    pos_embed = get_2d_sincos_pos_embed(
                        embed_dim=self.pos_embed.shape[-1],
                        grid_size=(height, width),
                        base_size=self.base_size
                    )
                    pos_embed = jnp.array(pos_embed)[None, ...]
                else:
                    pos_embed = self.pos_embed

            latent = latent + pos_embed

        return latent.astype(self.dtype)

class FlaxCombinedTimestepTextProjEmbeddings(nn.Module):
    """
    Combined embeddings for timesteps and text projections.

    Args:
        embedding_dim (`int`): The embedding dimension.
        pooled_projection_dim (`int`): The dimension of pooled projections.
        dtype (`jnp.dtype`): The dtype of the computation.
    """
    embedding_dim: int
    pooled_projection_dim: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.time_proj = FlaxTimesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            dtype=self.dtype
        )

        # Setup timestep embedder
        self.timestep_embedder = nn.Sequential([
            nn.Dense(features=self.embedding_dim, dtype=self.dtype),
            nn.SiLU(),
            nn.Dense(features=self.embedding_dim, dtype=self.dtype),
        ])

        # Setup text projection
        self.text_proj = nn.Sequential([
            nn.LayerNorm(dtype=self.dtype),
            nn.Dense(features=self.embedding_dim, dtype=self.dtype),
        ])

    def __call__(self, timestep: jnp.ndarray, pooled_projection: jnp.ndarray) -> jnp.ndarray:
        # Project timesteps
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj)

        # Project text embeddings
        pooled_projections = self.text_proj(pooled_projection)

        # Combine embeddings
        conditioning = timesteps_emb + pooled_projections

        return conditioning.astype(self.dtype)
