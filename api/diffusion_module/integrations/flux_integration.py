
import os
from typing import Any, List, override

from PIL import Image
from loguru import logger
from api.utils.helpers import HTTPLoggedException
from api.diffusion_module.integrations.base_integration import ModelIntegration
from api.diffusion_module.schemas import GenerateImageRequest
import torch
from diffusers import Flux2KleinPipeline, FluxPipeline, DiffusionPipeline

from api.utils.statics import default_flux_model, quantization_types

class Flux_Integration(ModelIntegration):
    _available_models: List[str] = ["black-forest-labs/FLUX.2-klein-4B", "black-forest-labs/FLUX.1-schnell"]
    def __init__(self, type:str = "flux", cache_default_model:bool = False, enable_cpu_offload:bool=False, quant_type:str = None):
        super().__init__(type=type, cache_default_model=cache_default_model, enable_cpu_offload=enable_cpu_offload, quant_type=quant_type)
        self.default_integration_model = default_flux_model
        if self._with_cached_model:
            self._cache_default_pipe()
            logger.info(f"-- FLUX integration >> current cached model: {self.cached_model} | device: {self._cached_pipe_device}--")
        if self._hf_token is not None:
            logger.info(f"-- FLUX INIT >> HF TOKEN {self._hf_token} --")

    @override
    def clear_cache(self, free_resources:bool = True):
        del self._cached_pipe
        self.cached_model = None
        self._cached_pipe_device = None
        if free_resources:
            self.free_resources()

    @override
    def generate_image(self, req: GenerateImageRequest, device:str = "cuda") -> Any:
        try:
            target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
            pipe: DiffusionPipeline = self._get_pipe_for_request(req)
            if pipe is None:
                raise HTTPLoggedException(status_code=500, detail=f"-- Unable to load pipe for request. Model: {req.diffuser_name} --")
            #self._cached_pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
            if req.seed is not None and req.seed < 0:
                req.seed = None
            prompt = req.prompt
            image = pipe(
                prompt=prompt,
                height=req.height,
                width=req.width,
                guidance_scale=req.guidance,
                num_inference_steps=req.inference_steps,
                generator=torch.Generator(device=target_device).manual_seed(req.seed) if req.seed is not None else None
            ).images[0]
            return image
        except Exception as e:
            logger.error(f"-- an error has occured while trying to generate an image with Flux integration: {e} --")
            raise HTTPLoggedException(status_code=500, detail=f"-- Flux Integration >> An error has occured while trying to generate the image --")
            
    # @abstract & override
    def _cache_default_pipe(self):
        target_device:str = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = self.get_pipe_for_model(default_flux_model, device=target_device, with_auto_offload=self._enable_auto_offload)
        """
        Offloading automatically takes care of moving the individual components vae, text_encoder, text_encoder_2, 
        tokenizer, tokenizer_2, transformer, scheduler, image_encoder, feature_extractor to GPU when needed. 
        To make sure offloading works as expected, you should consider moving the pipeline back to CPU: `pipeline.to('cpu')` or removing the move altogether if you use offloading.

        """
        self._cached_pipe = pipe
        self._cached_pipe_device = target_device
        self.cached_model = default_flux_model

    @override
    def get_pipe_for_model(self, model:str, device:str = "cuda", with_auto_offload:bool = False) -> DiffusionPipeline:
        try:
            target_device:str = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if target_device == "cuda" else torch.float32
            is_v1: bool = "FLUX.1" in model
            if is_v1:
                if self._quantization_cfg is None:
                    cfg = quantization_types[0] if len(quantization_types) > 0 else "4bit"
                    logger.warning(f"-- Attempting to load Flux.1 model {model} without quantization >> using default {cfg} quantization --")
                    self._quantization_cfg = self.get_quantization_cfg(cfg)
                pipe = FluxPipeline.from_pretrained(model, torch_dtype=torch.bfloat16, quantization_config=self._quantization_cfg, token=self._hf_token)
            else:
                pipe = Flux2KleinPipeline.from_pretrained(model, torch_dtype=dtype, quantization_config=self._quantization_cfg, token=self._hf_token)
            if with_auto_offload:
                logger.info("-- FLUX INTEGRATION >> AUTO OFFLOAD ENABLED --")
                pipe.enable_model_cpu_offload()
            else:
                logger.info(f"-- FLUX INTEGRATION >> PIPE TO {target_device.upper()} --")
                pipe.to(target_device)
                self._apply_pipe_optimizations(pipe=pipe, target_device=target_device)
            logger.info(f"-- Flux Integration: {model} >> Pipeline memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB --")
            return pipe
        except:
            logger.error(f"-- an error has occured while trying to load the diffusion pipe for model: {model} --")
            return None
        