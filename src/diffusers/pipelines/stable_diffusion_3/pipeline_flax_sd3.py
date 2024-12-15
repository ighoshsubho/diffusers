from functools import partial
from typing import Dict, List, Optional, Union

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from transformers import CLIPTokenizer, FlaxCLIPTextModel, FlaxCLIPTextModelWithProjection, FlaxT5EncoderModel

from ...models import FlaxAutoencoderKL
from ...models.transformers.transformer_sd3_flax import FlaxSD3Transformer2DModel
from ...schedulers import FlaxDDIMScheduler
from ...utils import logging, replace_example_docstring
from ..pipeline_flax_utils import FlaxDiffusionPipeline


logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import jax
        >>> import numpy as np
        >>> from flax.jax_utils import replicate
        >>> from flax.training.common_utils import shard
        >>> from diffusers import FlaxStableDiffusion3Pipeline

        >>> pipeline, params = FlaxStableDiffusion3Pipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-base", variant="bf16", dtype=jax.numpy.bfloat16
        ... )

        >>> prompt = "a photo of an astronaut riding a horse on mars"

        >>> prng_seed = jax.random.PRNGKey(0)
        >>> num_inference_steps = 50

        >>> num_samples = jax.device_count()
        >>> prompt = num_samples * [prompt]
        >>> prompt_ids = pipeline.prepare_inputs(prompt)

        >>> params = replicate(params)
        >>> prng_seed = jax.random.split(prng_seed, jax.device_count())
        >>> prompt_ids = shard(prompt_ids)

        >>> images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
        >>> images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
        ```
"""

class FlaxStableDiffusion3Pipeline(FlaxDiffusionPipeline):
    def __init__(
        self,
        vae: FlaxAutoencoderKL,
        text_encoder: FlaxCLIPTextModel,
        text_encoder_2: FlaxCLIPTextModelWithProjection,
        text_encoder_3: FlaxT5EncoderModel,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        tokenizer_3: CLIPTokenizer,
        transformer: FlaxSD3Transformer2DModel,
        scheduler: FlaxDDIMScheduler,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        self.dtype = dtype

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_sample_size = self.transformer.config.sample_size
        self.patch_size = self.transformer.config.patch_size

    def prepare_inputs(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        max_length: Optional[int] = None,
    ):
        if not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # Use prompt if prompt_2/3 not provided
        prompt_2 = prompt_2 or prompt
        prompt_3 = prompt_3 or prompt

        # Handle string inputs
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(prompt_2, str):
            prompt_2 = [prompt_2]
        if isinstance(prompt_3, str):
            prompt_3 = [prompt_3]

        max_length = max_length or self.tokenizer.model_max_length

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="np",
        )

        text_inputs_2 = self.tokenizer_2(
            prompt_2,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="np",
        )

        text_inputs_3 = self.tokenizer_3(
            prompt_3,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="np",
        )

        return text_inputs.input_ids, text_inputs_2.input_ids, text_inputs_3.input_ids

    def _encode_prompt(
        self,
        prompt_ids: jnp.array,
        prompt_2_ids: jnp.array,
        prompt_3_ids: jnp.array,
        params: Union[Dict, FrozenDict],
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
    ):
        batch_size = prompt_ids.shape[0]

        # Get text embeddings from all encoders
        prompt_embeds = self.text_encoder(prompt_ids, params=params["text_encoder"])[0]
        prompt_2_embeds = self.text_encoder_2(prompt_2_ids, params=params["text_encoder_2"])[0]
        prompt_3_embeds = self.text_encoder_3(prompt_3_ids, params=params["text_encoder_3"])[0]

        # Get pooled embeddings from CLIP models
        pooled_prompt_embeds = self.text_encoder(prompt_ids, params=params["text_encoder"])[1]
        pooled_prompt_2_embeds = self.text_encoder_2(prompt_2_ids, params=params["text_encoder_2"])[1]

        # Concatenate text embeddings and pooled embeddings
        prompt_embeds = jnp.concatenate([prompt_embeds, prompt_2_embeds, prompt_3_embeds], axis=-1)
        pooled_prompt_embeds = jnp.concatenate([pooled_prompt_embeds, pooled_prompt_2_embeds], axis=-1)

        # Duplicate for each image
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, axis=0)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(num_images_per_prompt, axis=0)

        # Get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size

            max_length = prompt_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            ).input_ids

            uncond_embeddings = self.text_encoder(uncond_input, params=params["text_encoder"])[0]
            uncond_embeddings_2 = self.text_encoder_2(uncond_input, params=params["text_encoder_2"])[0]
            uncond_embeddings_3 = self.text_encoder_3(uncond_input, params=params["text_encoder_3"])[0]

            uncond_embeddings = jnp.concatenate(
                [uncond_embeddings, uncond_embeddings_2, uncond_embeddings_3], axis=-1
            )

            # Duplicate unconditional embeddings for each generation per prompt
            uncond_embeddings = uncond_embeddings.repeat(num_images_per_prompt, axis=0)

            # Concatenate unconditional and text embeddings
            prompt_embeds = jnp.concatenate([uncond_embeddings, prompt_embeds])
            pooled_prompt_embeds = jnp.concatenate([pooled_prompt_embeds, pooled_prompt_embeds])

        return prompt_embeds, pooled_prompt_embeds

    def _generate(
        self,
        prompt_ids: jnp.array,
        prompt_2_ids: jnp.array,
        prompt_3_ids: jnp.array,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.random.PRNGKey,
        num_inference_steps: int,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        latents: Optional[jnp.ndarray] = None,
    ):
        # 0. Default height and width to transformer config
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        if height % (self.vae_scale_factor * self.patch_size) != 0 or width % (self.vae_scale_factor * self.patch_size) != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * self.patch_size} but are {height} and {width}."
            )

        # 1. Encode input prompt
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(
            prompt_ids,
            prompt_2_ids,
            prompt_3_ids,
            params,
            num_images_per_prompt,
            guidance_scale > 1.0,
        )

        # 2. Prepare scheduler state
        scheduler_state = self.scheduler.set_timesteps(
            params["scheduler"], num_inference_steps=num_inference_steps
        )

        # 3. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents_shape = (
            prompt_ids.shape[0] * num_images_per_prompt,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if latents is None:
            latents = jax.random.normal(prng_seed, latents_shape, dtype=jnp.float32)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler_state.init_noise_sigma

        # 4. Denoising loop
        def loop_body(step, args):
            latents, scheduler_state = args

            # Expand latents for classifier free guidance
            latents_input = jnp.concatenate([latents] * 2) if guidance_scale > 1.0 else latents

            # Get current timestep
            t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
            timestep = jnp.broadcast_to(t, latents_input.shape[0])

            # Predict noise
            noise_pred = self.transformer.apply(
                {"params": params["transformer"]},
                latents_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
            ).sample

            # Perform guidance if scale > 1
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = jnp.split(noise_pred, 2, axis=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute previous noisy sample x_t -> x_t-1
            latents, scheduler_state = self.scheduler.step(
                scheduler_state,
                noise_pred,
                t,
                latents,
            ).to_tuple()

            return latents, scheduler_state

        # Run denoising loop
        latents, _ = jax.lax.fori_loop(0, num_inference_steps, loop_body, (latents, scheduler_state))

        # 5. Scale and decode the image latents with vae
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.apply({"params": params["vae"]}, latents, method=self.vae.decode).sample

        image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
        return image

    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt_ids: jnp.array,
        prompt_2_ids: Optional[jnp.array] = None,
        prompt_3_ids: Optional[jnp.array] = None,
        params: Union[Dict, FrozenDict] = None,
        prng_seed: Optional[jax.random.PRNGKey] = None,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        latents: Optional[jnp.ndarray] = None,
        return_dict: bool = True,
        jit: bool = False,
    ):
        # Prepare inputs
        if prompt_2_ids is None:
            prompt_2_ids = prompt_ids
        if prompt_3_ids is None:
            prompt_3_ids = prompt_ids

        # Generate with pmapped function if jit=True
        if jit:
            images = _p_generate(
                self,
                prompt_ids,
                prompt_2_ids,
                prompt_3_ids,
                params,
                prng_seed,
                num_inference_steps,
                height,
                width,
                guidance_scale,
                num_images_per_prompt,
                latents,
            )
        else:
            images = self._generate(
                prompt_ids,
                prompt_2_ids,
                prompt_3_ids,
                params,
                prng_seed,
                num_inference_steps,
                height,
                width,
                guidance_scale,
                num_images_per_prompt,
                latents,
            )

        if not return_dict:
            return (images,)

        return {"images": images}

@partial(
    jax.pmap,
    in_axes=(None, 0, 0, 0, 0, 0, None, None, None, None, None, 0),
    static_broadcasted_argnums=(0, 6, 7, 8),
)
def _p_generate(
    pipe,
    prompt_ids,
    prompt_2_ids,
    prompt_3_ids,
    params,
    prng_seed,
    num_inference_steps,
    height,
    width,
    guidance_scale,
    num_images_per_prompt,
    latents,
):
    return pipe._generate(
        prompt_ids,
        prompt_2_ids,
        prompt_3_ids,
        params,
        prng_seed,
        num_inference_steps,
        height,
        width,
        guidance_scale,
        num_images_per_prompt,
        latents,
    )
