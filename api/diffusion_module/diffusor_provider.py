import os
from typing import List
from fastapi import FastAPI
from api.diffusion_module.integrations.base_integration import ModelIntegration
from api.diffusion_module.integrations.flux_integration import Flux_Integration
from api.diffusion_module.integrations.sd_integration import SD_Integration
from api.diffusion_module.schemas import IntegrationSettings, ProviderSettings, SetProviderRequest, UpdateQuantConfigRequest

#custom pipeline

from loguru import logger

from api.utils.helpers import HTTPLoggedException

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
            return cls._instances[cls]
        
"""
 VAE Options:
      1. "stabilityai/sd-vae-ft-mse" (more stable, fewer artifacts)
      2. "stabilityai/sd-vae-ft-ema" (slightly smoother but more blurry)
      3. "madebyollin/sdxl-vae-fp16-fix" (SDXL vae, sharper but needs SDXL unet)
    
    UNet Options:
      1. "runwayml/stable-diffusion-v1-5" (best general quality)
      2. "runwayml/stable-diffusion-inpainting" (for inpainting tasks)
      3. "stabilityai/stable-diffusion-2" (higher quality but slower)
    
    Text Encoder Options:
      1. "openai/clip-vit-large-patch14" (standard, 77 tokens, 768-dim)
      2. "openai/clip-vit-base-patch32" (faster, lower quality, 49 tokens)
"""
class ModelProvider(metaclass=Singleton):
    current_integration_name:str = None
    integration: ModelIntegration = None
    init = False
    available_integrations: List[str]=["stablediffusion", "flux"]

    def __init__(self, app:FastAPI, integration_name:str, with_cached_diffusor:bool = True, with_auto_offload:bool = False, quantization: str = None):
        self.app = app
        if len(os.environ["HF_KEY"]) == 0:
            raise HTTPLoggedException(status_code=500, detail="No Huggingface Token has been found in environment")
        self.current_integration_name = integration_name
        match self.current_integration_name:
            case "stablediffusion":
                self.integration = SD_Integration(cache_default_model=with_cached_diffusor, enable_cpu_offload=with_auto_offload, quant_type=quantization)
            case "flux":
                self.integration = Flux_Integration(cache_default_model=with_cached_diffusor, enable_cpu_offload=with_auto_offload, quant_type=quantization)
        #self.init = self.validate_init() <- current_integration_name = integration class & integration.cached_model if with_cached_diffusor & HF_TOKEN 
        self.init = False if self.integration is None else True
        if self.init:
            logger.info(f"-- MODEL PROVIDER INIT >> INTEGRATION: {integration_name} --")
        else:
            logger.error(f"-- MODEL PROVIDER INIT >> FAILED TO INITIALIZE --")

    def get_current_settings(self) -> ProviderSettings:
        return ProviderSettings(
            init=self.init,
            available=self.is_available(),
            available_integrations=self.available_integrations, 
            current_integration_settings=self.get_current_integration_settings()
        )
    
    def get_current_integration_settings(self) -> IntegrationSettings:
        return self.integration.get_current_settings() if self.integration is not None else None
    
    def set_integration_model(self, request: SetProviderRequest) -> IntegrationSettings | None:
        logger.info(f"-- MODEL PROVIDER >> setting {request.name} integration --")
        self.init = False
        match request.name:
            case "stablediffusion":
                if isinstance(self.integration, SD_Integration) is False:
                    logger.warning("-- setting up SD integration >> clearing cache --")
                    self.clear()
                    if not request.with_cached_pipe:
                        logger.info(f"-- setting {request.name} integration without default cached model --")
                        self.integration = SD_Integration(enable_cpu_offload=request.with_auto_offload, quant_type=request.quantization)
                        if request.cache_model is not None:
                            logger.info(f"-- loading {request.cache_model} model --")
                            self.integration.load_and_cache_model(model=request.cache_model, device=request.target_device, with_auto_offload=request.with_auto_offload, quantization=request.quantization)
                    else:
                        logger.info(f"-- setting {request.name} integration with default cached model --")
                        self.integration = SD_Integration(cache_default_model=True, enable_cpu_offload=request.with_auto_offload, quant_type=request.quantization)
                    self.current_integration_name = request.name
                else:
                    if request.cache_model is not None:
                        logger.info(f"-- updating {request.name} integration cached model --")
                        self.integration.load_and_cache_model(model=request.cache_model, device=request.target_device, with_auto_offload=request.with_auto_offload, quantization=request.quantization)
            case "flux":
                if isinstance(self.integration, Flux_Integration) is False:
                    logger.warning("-- setting up FLUX integration >> clearing cache --")
                    self.clear()
                    if not request.with_cached_pipe:
                        logger.info(f"-- setting {request.name} integration without default cached model --")
                        self.integration = Flux_Integration(enable_cpu_offload=request.with_auto_offload, quant_type=request.quantization)
                        if request.cache_model is not None:
                            logger.info(f"-- loading {request.cache_model} model --")
                            self.integration.load_and_cache_model(model=request.cache_model, device=request.target_device, with_auto_offload=request.with_auto_offload, quantization=request.quantization)
                    else:
                        logger.info(f"-- setting {request.name} integration with default cached model --")
                        self.integration = Flux_Integration(cache_default_model=True, enable_cpu_offload=request.with_auto_offload, quant_type=request.quantization)
                    self.current_integration_name = request.name
                else:
                    if request.cache_model is not None:
                        logger.info(f"-- updating {request.name} integration cached model --")
                        self.integration.load_and_cache_model(model=request.cache_model, device=request.target_device, with_auto_offload=request.with_auto_offload, quantization=request.quantization)
        self.init = True if self.integration is not None else False
        if self.init:
            logger.info(f"-- MODEL PROVIDER >> INTEGRATION: {self.current_integration_name} --")
        else:
            logger.error(f"-- MODEL PROVIDER >> FAILED TO UPDATE --")
        return self.integration.get_current_settings()
    
    def update_quant_config(self, request: UpdateQuantConfigRequest) -> IntegrationSettings:
        if self.integration is None:
            raise HTTPLoggedException(status_code=500, detail="-- No diffusor integration has been loaded --")
        return self.integration.update_quantization_config(config_name=request.config_name, reload_cache=request.with_cache_reload)
    
    def is_available(self) -> bool:
        return True if self.init and self.integration is not None and len(self.integration.get_current_settings().available_models) > 0 else False
    
    def clear(self, device: str = None):
        if self.integration is not None:
            self.integration.free_resources()
            del self.integration
        self.current_integration_name = None
        self.init = False

      
