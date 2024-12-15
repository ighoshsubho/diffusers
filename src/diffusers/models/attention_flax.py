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

import functools
import math
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from diffusers.models.attention_processor_flax import FlaxJointAttnProcessor2_0
from diffusers.models.normalization_flax_utils import (
    FlaxAdaLayerNormContinuous,
    FlaxAdaLayerNormZero,
    FlaxSD35AdaLayerNormZeroX,
)


def _query_chunk_attention(query, key, value, precision, key_chunk_size: int = 4096):
    """Multi-head dot product attention with a limited number of queries."""
    num_kv, num_heads, k_features = key.shape[-3:]
    v_features = value.shape[-1]
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / jnp.sqrt(k_features)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(query, key, value):
        attn_weights = jnp.einsum("...qhd,...khd->...qhk", query, key, precision=precision)

        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = jax.lax.stop_gradient(max_score)
        exp_weights = jnp.exp(attn_weights - max_score)

        exp_values = jnp.einsum("...vhf,...qhv->...qhf", value, exp_weights, precision=precision)
        max_score = jnp.einsum("...qhk->...qh", max_score)

        return (exp_values, exp_weights.sum(axis=-1), max_score)

    def chunk_scanner(chunk_idx):
        # julienne key array
        key_chunk = jax.lax.dynamic_slice(
            operand=key,
            start_indices=[0] * (key.ndim - 3) + [chunk_idx, 0, 0],  # [...,k,h,d]
            slice_sizes=list(key.shape[:-3]) + [key_chunk_size, num_heads, k_features],  # [...,k,h,d]
        )

        # julienne value array
        value_chunk = jax.lax.dynamic_slice(
            operand=value,
            start_indices=[0] * (value.ndim - 3) + [chunk_idx, 0, 0],  # [...,v,h,d]
            slice_sizes=list(value.shape[:-3]) + [key_chunk_size, num_heads, v_features],  # [...,v,h,d]
        )

        return summarize_chunk(query, key_chunk, value_chunk)

    chunk_values, chunk_weights, chunk_max = jax.lax.map(f=chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size))

    global_max = jnp.max(chunk_max, axis=0, keepdims=True)
    max_diffs = jnp.exp(chunk_max - global_max)

    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(axis=0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)

    return all_values / all_weights


def jax_memory_efficient_attention(
    query, key, value, precision=jax.lax.Precision.HIGHEST, query_chunk_size: int = 1024, key_chunk_size: int = 4096
):
    r"""
    Flax Memory-efficient multi-head dot product attention. https://arxiv.org/abs/2112.05682v2
    https://github.com/AminRezaei0x443/memory-efficient-attention

    Args:
        query (`jnp.ndarray`): (batch..., query_length, head, query_key_depth_per_head)
        key (`jnp.ndarray`): (batch..., key_value_length, head, query_key_depth_per_head)
        value (`jnp.ndarray`): (batch..., key_value_length, head, value_depth_per_head)
        precision (`jax.lax.Precision`, *optional*, defaults to `jax.lax.Precision.HIGHEST`):
            numerical precision for computation
        query_chunk_size (`int`, *optional*, defaults to 1024):
            chunk size to divide query array value must divide query_length equally without remainder
        key_chunk_size (`int`, *optional*, defaults to 4096):
            chunk size to divide key and value array value must divide key_value_length equally without remainder

    Returns:
        (`jnp.ndarray`) with shape of (batch..., query_length, head, value_depth_per_head)
    """
    num_q, num_heads, q_features = query.shape[-3:]

    def chunk_scanner(chunk_idx, _):
        # julienne query array
        query_chunk = jax.lax.dynamic_slice(
            operand=query,
            start_indices=([0] * (query.ndim - 3)) + [chunk_idx, 0, 0],  # [...,q,h,d]
            slice_sizes=list(query.shape[:-3]) + [min(query_chunk_size, num_q), num_heads, q_features],  # [...,q,h,d]
        )

        return (
            chunk_idx + query_chunk_size,  # unused ignore it
            _query_chunk_attention(
                query=query_chunk, key=key, value=value, precision=precision, key_chunk_size=key_chunk_size
            ),
        )

    _, res = jax.lax.scan(
        f=chunk_scanner,
        init=0,
        xs=None,
        length=math.ceil(num_q / query_chunk_size),  # start counter  # stop counter
    )

    return jnp.concatenate(res, axis=-3)  # fuse the chunked result back


class FlaxAttention(nn.Module):
    r"""
    A Flax multi-head attention module as described in: https://arxiv.org/abs/1706.03762

    Parameters:
        query_dim (:obj:`int`):
            Input hidden states dimension
        heads (:obj:`int`, *optional*, defaults to 8):
            Number of heads
        dim_head (:obj:`int`, *optional*, defaults to 64):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        split_head_dim (`bool`, *optional*, defaults to `False`):
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`

    """

    query_dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0
    use_memory_efficient_attention: bool = False
    split_head_dim: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim_head * self.heads
        self.scale = self.dim_head**-0.5

        # Weights were exported with old names {to_q, to_k, to_v, to_out}
        self.query = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_q")
        self.key = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_k")
        self.value = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_v")

        self.proj_attn = nn.Dense(self.query_dim, dtype=self.dtype, name="to_out_0")
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def __call__(self, hidden_states, context=None, deterministic=True):
        context = hidden_states if context is None else context

        query_proj = self.query(hidden_states)
        key_proj = self.key(context)
        value_proj = self.value(context)

        if self.split_head_dim:
            b = hidden_states.shape[0]
            query_states = jnp.reshape(query_proj, (b, -1, self.heads, self.dim_head))
            key_states = jnp.reshape(key_proj, (b, -1, self.heads, self.dim_head))
            value_states = jnp.reshape(value_proj, (b, -1, self.heads, self.dim_head))
        else:
            query_states = self.reshape_heads_to_batch_dim(query_proj)
            key_states = self.reshape_heads_to_batch_dim(key_proj)
            value_states = self.reshape_heads_to_batch_dim(value_proj)

        if self.use_memory_efficient_attention:
            query_states = query_states.transpose(1, 0, 2)
            key_states = key_states.transpose(1, 0, 2)
            value_states = value_states.transpose(1, 0, 2)

            # this if statement create a chunk size for each layer of the unet
            # the chunk size is equal to the query_length dimension of the deepest layer of the unet

            flatten_latent_dim = query_states.shape[-3]
            if flatten_latent_dim % 64 == 0:
                query_chunk_size = int(flatten_latent_dim / 64)
            elif flatten_latent_dim % 16 == 0:
                query_chunk_size = int(flatten_latent_dim / 16)
            elif flatten_latent_dim % 4 == 0:
                query_chunk_size = int(flatten_latent_dim / 4)
            else:
                query_chunk_size = int(flatten_latent_dim)

            hidden_states = jax_memory_efficient_attention(
                query_states, key_states, value_states, query_chunk_size=query_chunk_size, key_chunk_size=4096 * 4
            )
            hidden_states = hidden_states.transpose(1, 0, 2)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        else:
            # compute attentions
            if self.split_head_dim:
                attention_scores = jnp.einsum("b t n h, b f n h -> b n f t", key_states, query_states)
            else:
                attention_scores = jnp.einsum("b i d, b j d->b i j", query_states, key_states)

            attention_scores = attention_scores * self.scale
            attention_probs = nn.softmax(attention_scores, axis=-1 if self.split_head_dim else 2)

            # attend to values
            if self.split_head_dim:
                hidden_states = jnp.einsum("b n f t, b t n h -> b f n h", attention_probs, value_states)
                b = hidden_states.shape[0]
                hidden_states = jnp.reshape(hidden_states, (b, -1, self.heads * self.dim_head))
            else:
                hidden_states = jnp.einsum("b i j, b j d -> b i d", attention_probs, value_states)
                hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

        hidden_states = self.proj_attn(hidden_states)
        return self.dropout_layer(hidden_states, deterministic=deterministic)


class FlaxBasicTransformerBlock(nn.Module):
    r"""
    A Flax transformer block layer with `GLU` (Gated Linear Unit) activation function as described in:
    https://arxiv.org/abs/1706.03762


    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        only_cross_attention (`bool`, defaults to `False`):
            Whether to only apply cross attention.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        split_head_dim (`bool`, *optional*, defaults to `False`):
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
    """

    dim: int
    n_heads: int
    d_head: int
    dropout: float = 0.0
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32
    use_memory_efficient_attention: bool = False
    split_head_dim: bool = False

    def setup(self):
        # self attention (or cross_attention if only_cross_attention is True)
        self.attn1 = FlaxAttention(
            self.dim,
            self.n_heads,
            self.d_head,
            self.dropout,
            self.use_memory_efficient_attention,
            self.split_head_dim,
            dtype=self.dtype,
        )
        # cross attention
        self.attn2 = FlaxAttention(
            self.dim,
            self.n_heads,
            self.d_head,
            self.dropout,
            self.use_memory_efficient_attention,
            self.split_head_dim,
            dtype=self.dtype,
        )
        self.ff = FlaxFeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype)
        self.norm1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self, hidden_states, context, deterministic=True):
        # self attention
        residual = hidden_states
        if self.only_cross_attention:
            hidden_states = self.attn1(self.norm1(hidden_states), context, deterministic=deterministic)
        else:
            hidden_states = self.attn1(self.norm1(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        # cross attention
        residual = hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states), context, deterministic=deterministic)
        hidden_states = hidden_states + residual

        # feed forward
        residual = hidden_states
        hidden_states = self.ff(self.norm3(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        return self.dropout_layer(hidden_states, deterministic=deterministic)

class FlaxJointTransformerBlock(nn.Module):
    """
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
        qk_norm (`str`, optional): The type of normalization to use for query and key values.
        use_dual_attention (`bool`): Whether to use dual attention mechanism.
        dtype (`jnp.dtype`): The dtype of the computation (default: jnp.float32)
    """
    dim: int
    num_attention_heads: int
    attention_head_dim: int
    context_pre_only: bool = False
    qk_norm: Optional[str] = None
    use_dual_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.use_dual_attention = self.use_dual_attention
        self.context_pre_only = self.context_pre_only
        context_norm_type = "ada_norm_continous" if self.context_pre_only else "ada_norm_zero"

        # Setup main normalization layers
        if self.use_dual_attention:
            self.norm1 = FlaxSD35AdaLayerNormZeroX(self.dim, dtype=self.dtype)
        else:
            self.norm1 = FlaxAdaLayerNormZero(self.dim, dtype=self.dtype)

        # Setup context normalization layers
        if context_norm_type == "ada_norm_continous":
            self.norm1_context = FlaxAdaLayerNormContinuous(
                self.dim,
                self.dim,
                elementwise_affine=False,
                eps=1e-6,
                bias=True,
                norm_type="layer_norm",
                dtype=self.dtype
            )
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context = FlaxAdaLayerNormZero(self.dim, dtype=self.dtype)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )

        # Setup attention layers
        self.attn = FlaxAttention(
            query_dim=self.dim,
            cross_attention_dim=None,
            added_kv_proj_dim=self.dim,
            heads=self.num_attention_heads,
            dim_head=self.attention_head_dim,
            out_dim=self.dim,
            context_pre_only=self.context_pre_only,
            bias=True,
            processor=FlaxJointAttnProcessor2_0(),
            qk_norm=self.qk_norm,
            eps=1e-6,
            dtype=self.dtype
        )

        if self.use_dual_attention:
            self.attn2 = FlaxAttention(
                query_dim=self.dim,
                cross_attention_dim=None,
                heads=self.num_attention_heads,
                dim_head=self.attention_head_dim,
                out_dim=self.dim,
                bias=True,
                processor=FlaxJointAttnProcessor2_0(),
                qk_norm=self.qk_norm,
                eps=1e-6,
                dtype=self.dtype
            )
        else:
            self.attn2 = None

        # Setup feed-forward layers
        self.norm2 = nn.LayerNorm(self.dim, epsilon=1e-6, use_bias=False, dtype=self.dtype)
        self.ff = FlaxFeedForward(dim=self.dim, dim_out=self.dim, activation_fn="gelu", dtype=self.dtype)

        if not self.context_pre_only:
            self.norm2_context = nn.LayerNorm(self.dim, epsilon=1e-6, use_bias=False, dtype=self.dtype)
            self.ff_context = FlaxFeedForward(dim=self.dim, dim_out=self.dim, activation_fn="gelu", dtype=self.dtype)
        else:
            self.norm2_context = None
            self.ff_context = None

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray,
        temb: jnp.ndarray,
        deterministic: bool = True,
    ) -> Tuple[Optional[jnp.ndarray], jnp.ndarray]:
        # Apply normalization and get gates
        if self.use_dual_attention:
            norm_out = self.norm1(hidden_states, emb=temb)
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = norm_out
        else:
            norm_out = self.norm1(hidden_states, emb=temb)
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_out

        # Process context
        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_out_context = self.norm1_context(encoder_hidden_states, emb=temb)
            (norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp) = norm_out_context

        # Apply attention
        attn_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            deterministic=deterministic
        )
        attn_output, context_attn_output = attn_outputs

        # Process attention outputs for hidden states
        attn_output = gate_msa.reshape(-1, 1) * attn_output
        hidden_states = hidden_states + attn_output

        # Apply dual attention if enabled
        if self.use_dual_attention:
            attn_output2 = self.attn2(hidden_states=norm_hidden_states2, deterministic=deterministic)
            attn_output2 = gate_msa2.reshape(-1, 1) * attn_output2
            hidden_states = hidden_states + attn_output2

        # Apply feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states, deterministic=deterministic)
        ff_output = gate_mlp.reshape(-1, 1) * ff_output
        hidden_states = hidden_states + ff_output

        # Process context states if needed
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            # Process attention outputs for encoder hidden states
            context_attn_output = c_gate_msa.reshape(-1, 1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            # Apply feed-forward to context
            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            context_ff_output = self.ff_context(norm_encoder_hidden_states, deterministic=deterministic)
            context_ff_output = c_gate_mlp.reshape(-1, 1) * context_ff_output
            encoder_hidden_states = encoder_hidden_states + context_ff_output

        return encoder_hidden_states, hidden_states

class FlaxTransformer2DModel(nn.Module):
    r"""
    A Spatial Transformer layer with Gated Linear Unit (GLU) activation function as described in:
    https://arxiv.org/pdf/1506.02025.pdf


    Parameters:
        in_channels (:obj:`int`):
            Input number of channels
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        depth (:obj:`int`, *optional*, defaults to 1):
            Number of transformers block
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        use_linear_projection (`bool`, defaults to `False`): tbd
        only_cross_attention (`bool`, defaults to `False`): tbd
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        split_head_dim (`bool`, *optional*, defaults to `False`):
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
    """

    in_channels: int
    n_heads: int
    d_head: int
    depth: int = 1
    dropout: float = 0.0
    use_linear_projection: bool = False
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32
    use_memory_efficient_attention: bool = False
    split_head_dim: bool = False

    def setup(self):
        self.norm = nn.GroupNorm(num_groups=32, epsilon=1e-5)

        inner_dim = self.n_heads * self.d_head
        if self.use_linear_projection:
            self.proj_in = nn.Dense(inner_dim, dtype=self.dtype)
        else:
            self.proj_in = nn.Conv(
                inner_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )

        self.transformer_blocks = [
            FlaxBasicTransformerBlock(
                inner_dim,
                self.n_heads,
                self.d_head,
                dropout=self.dropout,
                only_cross_attention=self.only_cross_attention,
                dtype=self.dtype,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                split_head_dim=self.split_head_dim,
            )
            for _ in range(self.depth)
        ]

        if self.use_linear_projection:
            self.proj_out = nn.Dense(inner_dim, dtype=self.dtype)
        else:
            self.proj_out = nn.Conv(
                inner_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )

        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self, hidden_states, context, deterministic=True):
        batch, height, width, channels = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height * width, channels)
            hidden_states = self.proj_in(hidden_states)
        else:
            hidden_states = self.proj_in(hidden_states)
            hidden_states = hidden_states.reshape(batch, height * width, channels)

        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, context, deterministic=deterministic)

        if self.use_linear_projection:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, channels)
        else:
            hidden_states = hidden_states.reshape(batch, height, width, channels)
            hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states + residual
        return self.dropout_layer(hidden_states, deterministic=deterministic)


class FlaxFeedForward(nn.Module):
    r"""
    Flax module that encapsulates two Linear layers separated by a non-linearity. It is the counterpart of PyTorch's
    [`FeedForward`] class, with the following simplifications:
    - The activation function is currently hardcoded to a gated linear unit from:
    https://arxiv.org/abs/2002.05202
    - `dim_out` is equal to `dim`.
    - The number of hidden dimensions is hardcoded to `dim * 4` in [`FlaxGELU`].

    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # The second linear layer needs to be called
        # net_2 for now to match the index of the Sequential layer
        self.net_0 = FlaxGEGLU(self.dim, self.dropout, self.dtype)
        self.net_2 = nn.Dense(self.dim, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.net_0(hidden_states, deterministic=deterministic)
        hidden_states = self.net_2(hidden_states)
        return hidden_states


class FlaxGEGLU(nn.Module):
    r"""
    Flax implementation of a Linear layer followed by the variant of the gated linear unit activation function from
    https://arxiv.org/abs/2002.05202.

    Parameters:
        dim (:obj:`int`):
            Input hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim * 4
        self.proj = nn.Dense(inner_dim * 2, dtype=self.dtype)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.proj(hidden_states)
        hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=2)
        return self.dropout_layer(hidden_linear * nn.gelu(hidden_gelu), deterministic=deterministic)
