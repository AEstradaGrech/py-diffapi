

from typing import List, Optional
from pydantic import BaseModel, Field

class GenerateImageRequest(BaseModel):
    prompt: str = Field(description="The text that will be used to generate the image")
    num_gen: int = Field(description="The number of images to generate with this prompt", default=1)
    seed: Optional[int] = Field(description="Using the same seed will generate the same image for the same prompt. Pass a negative value to ignore the parameter", default=None)
    height: float = Field(description="Height of the images that will be generated", default=1024)
    width: float = Field(description="Width of the images that will be generated", default=1024)
    name: Optional[str] = Field(description="Name of the image file. A guid will be used instead if no name is provided", default=None)
    guidance: float = Field(description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.",default=7.5)
    inference_steps: Optional[int] = Field(description="Adjusts the number of steps involved in the generations. The more steps, the better the final result", default=None)
    diffuser_name: str = Field(description="The name of the diffuser or inference endpoint to use", default="runwayml/stable-diffusion-v1-5")
    tag: Optional[str] = Field(None, description="Tags a chat session with something meaningful for the user to provide more context about the image")
    file_save:bool = Field(description="Flag to indicate wether to save the file to a local folder or not", default=True)
    db_save: bool = Field(description="Flag to indicate wether to save a document with the image in base64 format or not", default=False)
    cache_diffusion_pipe: Optional[bool] = Field(description="Flag to indicate whether to cache the used diffusion pipe or not in case it is not the current one", default=False) 

class StableDiffusionImageRequest(GenerateImageRequest):
    use_refiner: Optional[bool] = Field(description="Use the SDXL refiner pipe or the default 'generate_image' / single diffusor method", default=False)

class GenerateRefinedImageRequest(GenerateImageRequest):
    negative_prompt: Optional[str] = Field(description="Negative prompt to guide the diffuser away from certain concepts", default=None)
    denoising_split_fraction: Optional[float] = Field(description="Ratio of denoise action between base and refiner models. This will split 'inference_steps' according to this ratio", default=0.7)
    refiner_guidance: Optional[float] = Field(description="guidance scale for the refiner. Pipe works better when refiner's guidance is LOWER than the base model guidance", default=6.5)
    refiner_negative_prompt: Optional[str] = Field(description="Negative prompt to pass to the refiner or None", default=None)
    copy_base_negative_prompt: Optional[bool] = Field(description="If TRUE and a negative_prompt is passed for the base model, it will REUSE it for the refiner", default=False)
    with_random_refiner_seed: Optional[bool] = Field(description="If TRUE will pass None (random) as seed for the refiner. Otherwise will use the same torch generator & seed than for the base model", default=False)

class SetProviderRequest(BaseModel):
    name:str = Field(description="Name of the integration to use. stablediffusion | flux")
    with_cached_pipe: bool = Field(description="True to cache a diffusor pipe for subsequent requests. 'None' to initialize the integration without cached pipe", default=None)
    cache_model:Optional[str] = Field(description="Full model name to load and cache in the integration. 'None' to use the default", default=None)
    with_auto_offload: bool = Field(description="Flag to indicate whether to let the pipeline handle the memory or use the selected device applying the pipe optimization", default=False)
    target_device: Optional[str] = Field(description="Selected device to load the model. 'cuda' | 'cpu'", default="cuda")
    quantization: Optional[str] = Field(description="Quantization default config name to use. 'None' to load full model (except FLUX.1 that must be always quantized)", default=None)
  
class IntegrationSettings(BaseModel):
    name:str = Field(description="Name of the integration to use. 'stablediffusion | flux'")
    current_model: Optional[str] = Field(description="Full name of the current cached model")
    current_device: Optional[str] = Field(description="Current target device (cuda | cpu)")
    current_quantization: Optional[str] = Field(description="current quantization config (if any)", default=None)
    auto_offload_enabled: bool = Field(description="Pipe sequential_offload enabled | device + optimizations")
    current_memory_usage: str = Field(description="Current GPU memory weight for the current cached model")
    available_models: List[str] = Field(description="Current available diffusion models")
    default_model: str = Field(description="Current default model for the integration to cache if non specified")

class ProviderSettings(BaseModel):
    init: bool = Field(description="Indicates whether the model provider has been initialized with an integration and available to use")
    available: bool = Field(description="Indicates whether the current integration is busy and cannot accept another request or not")
    available_integrations: List[str] = Field(description="Avaliable diffusor integrations. 'stablediffusion' | 'flux'", default=[])
    current_integration_settings: Optional[IntegrationSettings] = Field(description="Settings for the current integration", default=None)

class UpdateQuantConfigRequest(BaseModel):
    config_name:str = Field(description="Name of the default / templated configuration to set in the current integration")
    with_cache_reload: bool = Field(description="If true, the integration will reload the current cached model using the new quantization config. Otherwise will be used for newly instantiated pipes")

