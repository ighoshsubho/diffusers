from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

class FlaxJointAttnProcessor2_0:
    """
    Attention processor used typically in processing the SD3-like self-attention projections.
    Flax implementation of the joint attention processor.
    """
    
    def __call__(
        self,
        attn: nn.Module,
        hidden_states: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape for multi-head attention
        query = jnp.reshape(query, (batch_size, -1, attn.heads, head_dim))
        query = jnp.transpose(query, (0, 2, 1, 3))
        key = jnp.reshape(key, (batch_size, -1, attn.heads, head_dim))
        key = jnp.transpose(key, (0, 2, 1, 3))
        value = jnp.reshape(value, (batch_size, -1, attn.heads, head_dim))
        value = jnp.transpose(value, (0, 2, 1, 3))

        # Apply normalization if available
        if hasattr(attn, 'norm_q') and attn.norm_q is not None:
            query = attn.norm_q(query)
        if hasattr(attn, 'norm_k') and attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections
        if encoder_hidden_states is not None:
            # Project context embeddings
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            # Reshape context projections
            encoder_hidden_states_query_proj = jnp.reshape(
                encoder_hidden_states_query_proj, 
                (batch_size, -1, attn.heads, head_dim)
            )
            encoder_hidden_states_query_proj = jnp.transpose(encoder_hidden_states_query_proj, (0, 2, 1, 3))
            
            encoder_hidden_states_key_proj = jnp.reshape(
                encoder_hidden_states_key_proj, 
                (batch_size, -1, attn.heads, head_dim)
            )
            encoder_hidden_states_key_proj = jnp.transpose(encoder_hidden_states_key_proj, (0, 2, 1, 3))
            
            encoder_hidden_states_value_proj = jnp.reshape(
                encoder_hidden_states_value_proj, 
                (batch_size, -1, attn.heads, head_dim)
            )
            encoder_hidden_states_value_proj = jnp.transpose(encoder_hidden_states_value_proj, (0, 2, 1, 3))

            # Apply normalization to context if available
            if hasattr(attn, 'norm_added_q') and attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if hasattr(attn, 'norm_added_k') and attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # Concatenate sample and context projections
            query = jnp.concatenate([query, encoder_hidden_states_query_proj], axis=2)
            key = jnp.concatenate([key, encoder_hidden_states_key_proj], axis=2)
            value = jnp.concatenate([value, encoder_hidden_states_value_proj], axis=2)

        # Compute scaled dot-product attention
        scale = jnp.sqrt(head_dim).astype(query.dtype)
        attention_scores = jnp.matmul(query, jnp.transpose(key, (0, 1, 3, 2))) / scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        hidden_states = jnp.matmul(attention_weights, value)

        # Reshape attention output
        hidden_states = jnp.transpose(hidden_states, (0, 2, 1, 3))
        hidden_states = jnp.reshape(hidden_states, (batch_size, -1, attn.heads * head_dim))

        if encoder_hidden_states is not None:
            # Split attention outputs
            hidden_states, encoder_hidden_states = (
                hidden_states[:, :residual.shape[1]],
                hidden_states[:, residual.shape[1]:]
            )
            if not attn.context_pre_only and hasattr(attn, 'to_add_out'):
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # Apply output projections
        hidden_states = attn.to_out[0](hidden_states)
        # Note: Flax handles dropout differently during training/inference via deterministic flag
        hidden_states = attn.to_out[1](hidden_states, deterministic=deterministic)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states