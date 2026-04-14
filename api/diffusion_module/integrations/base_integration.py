
from abc import ABC, abstractmethod
import gc
import os
from typing import Any, List
from api.diffusion_module.schemas import GenerateImageRequest, IntegrationSettings
from api.utils.helpers import HTTPLoggedException
from api.utils.statics import quantization_types, default_quant_type
from diffusers import DiffusionPipeline
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from diffusers.quantizers import PipelineQuantizationConfig

from loguru import logger
import torch


class ModelIntegration(ABC):
    model_type:str
    _cached_pipe: DiffusionPipeline = None
    _cached_pipe_device:str = "cuda"
    cached_model:str = None
    _hf_token: str = None
    _with_cached_model: bool = False
    _enable_auto_offload:bool = False
    _quantization_cfg: PipelineQuantizationConfig = None
    _current_quant_cfg:str = None
    default_integration_model:str = None
    _available_models: List[str] = []

    def __init__(self, type:str, cache_default_model: bool = False, enable_cpu_offload:bool = False, quant_type:str = None):
        self._hf_token = os.environ["HF_KEY"]
        self.model_type = type
        self._with_cached_model = cache_default_model
        self._enable_auto_offload = enable_cpu_offload
        self._quantization_cfg = self.get_quantization_cfg(quant_type) if quant_type is not None else None
        self._cached_pipe_device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_current_settings(self) -> IntegrationSettings:
        return IntegrationSettings(
            name=self.model_type,
            default_model=self.default_integration_model,
            current_model=self.cached_model,
            current_device=self._cached_pipe_device,
            current_quantization=self._current_quant_cfg,
            auto_offload_enabled=self._enable_auto_offload,
            current_memory_usage=self.get_current_gpu_memory(),
            available_models=self._available_models
        )
    
    def get_current_gpu_memory(self) -> str:
        return f"{torch.cuda.max_memory_reserved() / 1024**3:.3f} GB"
    
    def free_resources(self, device: str = None):
        logger.info("-- freeing hardware resources --")
        match device:
            case "cuda":
                logger.info("-- clearing CUDA cache --")
                torch.cuda.empty_cache()
            case "gpu":
                logger.info("-- clearing GC --")
                gc.collect()
            case None:
                logger.info("-- clearing GPU & CPU cache --")
                torch.cuda.empty_cache()
                gc.collect()
    
    @abstractmethod
    def clear_cache(self, free_resources:bool = True):
        pass

    @abstractmethod
    def get_pipe_for_model(self, model:str, device:str = "cuda", with_auto_offload:bool = False) -> DiffusionPipeline:
        pass

    @abstractmethod
    def generate_image(self, req: GenerateImageRequest, device:str = "cuda") -> Any:
        pass
    
    def load_and_cache_model(self, model: str, device:str = "cuda", with_auto_offload:bool = False, quantization: str = None) -> str | None:
        if model is None:
            raise HTTPLoggedException(status=400, status_code="-- No model name has been specified --")
        logger.info(f"-- Loading model: {model} to device {device} --")
        target_device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        if target_device != device:
            logger.warning(f"-- requested device {device} not available. Falling back to {target_device} --")
        if model not in self._available_models:
            logger.warning(f"-- Failed to load model: {model}. Model not available for the current integration --")
            return None
        if self._cached_pipe is not None:
            logger.info(f"-- clearing current cached model: {self.cached_model} --")
            self.clear_cache()
        self._quantization_cfg = self.get_quantization_cfg(requested_cfg=quantization)
        if self._quantization_cfg is None:
            logger.warning(f"-- No quantization config is configured for model: {model} --")
        self.cached_model = None
        self._cached_pipe_device = None
        pipe = self.get_pipe_for_model(model=model, device=target_device, with_auto_offload=with_auto_offload)
        if pipe is None:
            logger.error(f"-- Failed to load pipe for model: {model} --")
            return None
        self._cached_pipe = pipe
        self.cached_model = model
        self._enable_auto_offload = with_auto_offload
        self._cached_pipe_device = target_device
        logger.info(f"-- Model loaded and cached successfully: {self.cached_model} >> device {self._cached_pipe_device} --")
        return self.cached_model
    
    def update_quantization_config(self, config_name:str, reload_cache: bool = False) -> IntegrationSettings:
        if reload_cache:
            logger.info(f"-- reloading current model {self.cached_model} with new quantization config: {config_name} --")
            self.load_and_cache_model(self.cached_model, self._cached_pipe_device, self._enable_auto_offload, config_name)
        else:
            logger.info(f"-- updating quantization config without reload >>  new quantization config: {config_name} --")
            self._quantization_cfg = self.get_quantization_cfg(config_name)
        return self.get_current_settings()
    
    def _cache_default_pipe(self):
        target_device:str = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = self.get_pipe_for_model(self.default_integration_model, device=target_device, with_auto_offload=self._enable_auto_offload)
        """
        Offloading automatically takes care of moving the individual components vae, text_encoder, text_encoder_2, 
        tokenizer, tokenizer_2, transformer, scheduler, image_encoder, feature_extractor to GPU when needed. 
        To make sure offloading works as expected, you should consider moving the pipeline back to CPU: `pipeline.to('cpu')` or removing the move altogether if you use offloading.

        """
        self._cached_pipe = pipe
        self._cached_pipe_device = target_device
        self.cached_model = self.default_integration_model
    
    def _get_pipe_for_request(self, req: GenerateImageRequest) -> DiffusionPipeline:
        pipe: DiffusionPipeline = None
        target_device: str = "cuda" if torch.cuda.is_available() else "cpu"
        if self._cached_pipe is None:
            cached_model = self.default_integration_model if self.cached_model is None else self.cached_model
            logger.warning(f"-- No cached pipe found >> creating pipe for model: {cached_model} --")
            if req.diffuser_name == self.cached_model:
                logger.info(f"-- caching default model: {cached_model} --")
                self._cache_default_pipe()
                pipe = self._cached_pipe
            else:
                logger.info(f"-- caching new pipe for model: {cached_model} --")
                self.load_and_cache_model(req.diffuser_name, device=target_device, quantization=self._current_quant_cfg, with_auto_offload=self._enable_auto_offload)
                pipe = self._cached_pipe
        else:
            logger.info(f"-- cached pipe found for model: {self.cached_model} --")
            # WG: free resources & update cached pipe?
            if req.diffuser_name != self.cached_model:
                logger.warning(f"-- cached model: {self.cached_model} is not the requested model {req.diffuser_name} >> UPDATING CACHED PIPE --")
                if req.cache_diffusion_pipe is False:
                    logger.warning(f"-- requested model {req.diffuser_name} is not the default cached model. Removing cached references to free up memory --")
                    self.clear_cache()
                    pipe = self.get_pipe_for_model(req.diffuser_name, device=target_device, with_auto_offload=self._enable_auto_offload)
                    if pipe is None:
                        raise HTTPLoggedException(status_code=400, detail=f"-- No Flux model has been found for request diffuser: {req.diffuser_name} --")
                    else:
                        logger.info(f"-- using uncached pipe for model: {req.diffuser_name} --")
                else:
                    self.load_and_cache_model(req.diffuser_name, device=target_device, quantization=self._current_quant_cfg, with_auto_offload=self._enable_auto_offload)
                    logger.warning(f"-- caching new pipe for model: {req.diffuser_name} --")
                    pipe = self._cached_pipe
            else:
                logger.info(f"-- requested model {req.diffuser_name} is already cached >> USING CACHED PIPE --")
                pipe = self._cached_pipe
        return pipe
    """
    ['feature_extractor', 'image_encoder', 'scheduler', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'transformer', 'vae'],
    """
    def get_quantization_cfg(self, requested_cfg:str) -> PipelineQuantizationConfig:
        cfg: PipelineQuantizationConfig = None
        if requested_cfg is not None and requested_cfg not in quantization_types:
            logger.warning(f"-- no quantization config type found for: {requested_cfg} >> using default quantization type --")
            requested_cfg = quantization_types[0] if len(quantization_types) > 0 else default_quant_type
        match requested_cfg:
            case "4bit":
                # Pipeline memory usage: 10.389 GB FLUX.1
                # Pipeline memory usage: 9.766  GB FLUX.2
                cfg = PipelineQuantizationConfig(
                    quant_mapping={
                        "transformer": DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
                        "text_encoder_2": TransformersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
                    }
            )
            case "4bit-full":
                # Pipeline memory usage: 10.461 GB FLUX.1
                # Pipeline memory usage: 4.887  GB FLUX.2
                cfg = PipelineQuantizationConfig(
                    quant_mapping={
                        "transformer": DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
                        "text_encoder": TransformersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
                        "text_encoder_2": TransformersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
                    }
            )
            case "4bit+":
                # Pipeline memory usage: 12.133 GB FLUX.1
                # Pipeline memory usage: 6.365  GB FLUX.2
                cfg = PipelineQuantizationConfig(
                    quant_mapping={
                        "transformer": DiffusersBitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16
                        ),
                        "text_encoder": TransformersBitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16
                        ),
                        "text_encoder_2": TransformersBitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16
                        ),
                    })
            case "4bit++":
                # Pipeline memory usage: 11.982 GB FLUX.1
                # Pipeline memory usage: 9.766  GB FLUX.2
                cfg = PipelineQuantizationConfig(
                    quant_mapping={
                        "transformer": DiffusersBitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16
                        ),
                        "text_encoder_2": TransformersBitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16
                        ),
                    })
            case "8b":
                #KO: no carga
                cfg = PipelineQuantizationConfig(
                    quant_mapping={
                    "transformer": DiffusersBitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        llm_int8_enable_fp32_cpu_offload=True,
                        llm_int8_threshold=4.0
                    ),
                    "text_encoder": TransformersBitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                    ),
                    "text_encoder_2": TransformersBitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                    ),
                })
        if cfg is not None:
            logger.warning(f"-- using quantization config: {requested_cfg} --")
            self._current_quant_cfg = requested_cfg
        else:
            logger.warning(f"-- No quantization config in use --")
            self._current_quant_cfg = None
        return cfg
    
    def _apply_pipe_optimizations(self, pipe: DiffusionPipeline, target_device: str):
        """Apply memory optimization techniques to a diffusion pipeline"""
        if target_device == "cuda":
            logger.info("-- Applying CUDA optimizations --")
            
            # Enable xformers memory-efficient attention (if available)
            try:
                if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                    pipe.enable_xformers_memory_efficient_attention()
                    logger.info("✓ xformers memory-efficient attention enabled")
            except Exception as e:
                logger.warning(f"xformers not available, falling back to attention slicing: {e}")
                # Fallback: attention slicing
                try:
                    if hasattr(pipe, "enable_attention_slicing"):
                        pipe.enable_attention_slicing("auto")
                        logger.info("✓ Attention slicing enabled (fallback)")
                except Exception as e:
                    logger.warning(f"Attention slicing also failed: {e}")
            
            # Enable VAE tiling for large images to reduce memory
            if hasattr(pipe, "vae"):
                if hasattr(pipe.vae, "enable_tiling"):
                    pipe.vae.enable_tiling()
                    logger.info("✓ VAE tiling enabled for large images")
            
            # Enable sequential CPU offload for memory-constrained scenarios
            # This trades speed for memory: only enabled if needed
            # pipe.enable_sequential_cpu_offload()  # Uncomment for extreme memory constraints
            
            # Use gradient checkpointing if available
            if hasattr(pipe, "unet") and hasattr(pipe.unet, "enable_gradient_checkpointing"):
                try:
                    pipe.unet.enable_gradient_checkpointing()
                    logger.info("✓ Gradient checkpointing enabled on UNet")
                except:
                    logger.debug("Gradient checkpointing not available")
            
            # TF32 allows faster matmul operations with slightly lower precision  
            # Recommended for RTX 40-series and newer
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("✓ TF32 precision enabled (faster matmul)")
            except:
                pass
