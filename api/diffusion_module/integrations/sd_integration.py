
import os
from typing import List, override
from diffusers import StableDiffusion3Pipeline, AutoencoderKL, DiffusionPipeline, SD3Transformer2DModel
from loguru import logger
import torch
import random

from api.diffusion_module.integrations.base_integration import ModelIntegration
from api.diffusion_module.schemas import GenerateImageRequest, GenerateRefinedImageRequest
from api.utils.helpers import HTTPLoggedException
from api.utils.statics import default_sdxl_vae, sdxl_base_model, sdxl_refiner_model, default_quant_type, sd3_default_model, default_sd_model

class SD_Integration(ModelIntegration):
    _cached_refiner: DiffusionPipeline =  None
    _available_models: List[str] = ["stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/stable-diffusion-3.5-medium"]
    def __init__(self, type:str = "stablediffusion", cache_default_model:bool = False, enable_cpu_offload:bool = False, quant_type:str = None):
        super().__init__(type=type, cache_default_model=cache_default_model, enable_cpu_offload=enable_cpu_offload, quant_type=quant_type)
        self.default_integration_model = default_sd_model
        if cache_default_model:
            try:
                self._cache_default_pipe()    
            except:
                logger.warning(f"-- Unable to cache the default SD pipe --")
        if self._hf_token is not None:
            logger.info(f"-- SD INIT >> HF TOKEN {self._hf_token} --")

    @override
    def clear_cache(self, free_resources:bool = True):
        if self._cached_pipe:
            del self._cached_pipe
        if self._cached_refiner:
            del self._cached_refiner
        self.cached_model = None
        self._cached_pipe_device = None
        if free_resources:
            self.free_resources()

    @override
    def get_pipe_for_model(self, model:str, device:str = "cuda", with_auto_offload:bool = False) -> DiffusionPipeline:
        device_name = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        target_device = torch.device(device_name)
        #adapt torch bfloat type to stablediffusion3 models
        torch_dtype = torch.bfloat16 if target_device.type == "cuda" else torch.float32
        logger.info(f"-- stable diffusion pipeline for {model} on {device_name} using {torch_dtype} --")
        is_v3: bool = True if "stable-diffusion-3" in model else False
        pipe: DiffusionPipeline = None
        if not is_v3:
            logger.info(f"-- Loading SDXL Base model >> device: {device_name.upper()} --")
            vae = self.load_vae(target_device=target_device, model=default_sdxl_vae)
            if vae is None:
                raise HTTPLoggedException(status_code=500, detail="-- failed to load VAE for SDXL pipe --")
            pipe = self.load_sdxl_pipe(vae=vae, target_device=device_name)
        else:
            pipe = self.load_sd3_pipe(model=model)
        if not with_auto_offload:
            logger.info(f"-- SD INTEGRATION >> PIPE TO {device_name.upper()} --")
            pipe.to(target_device)
            self._apply_pipe_optimizations(pipe=pipe, target_device=device_name)
        else:
            logger.info("-- SD INTEGRATION >> AUTO OFFLOAD ENABLED --")
            pipe.enable_model_cpu_offload()
        
        self._enable_auto_offload = with_auto_offload
        return pipe
    
    @override
    def generate_image(self, req:GenerateImageRequest, device:str="cuda"):
        logger.info("-- ON GENERATE IMAGE - SD INTEGRATION --")
        if req.diffuser_name not in self._available_models:
            raise HTTPLoggedException(status_code=400, detail="Requested diffuser is not available")
        try:
            params = {
                "file_name": req.name,
                "guidance": req.guidance,
                "inference_steps": 20 if req.inference_steps <= 0 else req.inference_steps,
                "height": req.height,
                "width": req.width,
                "img_num": req.num_gen,
                "seed": req.seed if req.seed is not None else int(random.Random().random() * 1000000000)
            }

            device_name = device if device == "cuda" and torch.cuda.is_available() else "cpu"
            target_device = torch.device(device_name)
            #adapt torch bfloat type to stablediffusion3 models
            torch_dtype = torch.bfloat16 if target_device.type == "cuda" else torch.float32
            
            if "stable-diffusion-3" in req.diffuser_name and req.diffuser_name != self.cached_model:
                logger.warning(f"-- Loading 'stable-diffusion-3' model >> setting AUTO_OFFLOAD ON --")
                self._enable_auto_offload = True
            pipe: DiffusionPipeline = self._get_pipe_for_request(req)
            
            if pipe is None:
                raise HTTPLoggedException(status_code=500, detail=f"-- Unable to load pipe for request. Model: {req.diffuser_name} --")
            
            if params["img_num"] <= 0:
                params["img_num"] = 1

            prompt_text = req.prompt
            generator = None
            if params["seed"] >= 0:
                generator = torch.Generator(device=target_device).manual_seed(params["seed"])

            # With 16GB VRAM, avoid sequential CPU offload (it's slower)
            # Only use it if generating very large batches or high resolutions
            if target_device.type == "cuda" and params["img_num"] > 4 and (params["height"] * params["width"]) > 786432:  # >786432 = >768x1024
                pipe.enable_sequential_cpu_offload()
                logger.info("Sequential CPU offload enabled for large batch/resolution")

            with torch.autocast(device_type="cuda", dtype=torch_dtype, cache_enabled=True) if target_device.type == "cuda" else torch.no_grad():
                output = pipe(
                    prompt_text,
                    guidance_scale=params["guidance"],
                    num_inference_steps=params["inference_steps"],
                    height=params["height"],
                    width=params["width"],
                    num_images_per_prompt=params["img_num"],
                    generator=generator,
                )
            return output.images
        except Exception as e:
            logger.error(f"An error has occured while generating SD image >> Exception: {e}")
            raise HTTPLoggedException(status_code=500, detail="An error has occured while generating image")
        
    """
    SDXL consists of an ensemble of experts pipeline for latent diffusion: 
    In a first step, the base model (stabilityai/stable-diffusion-xl-base-1.0) is used to generate (noisy) latents,
    which are then further processed with a refinement model specialized for the final denoising steps. 
    Note that the base model can be used as a standalone module.
    """
    def refined_pipe(self, req: GenerateRefinedImageRequest, device:str = "cuda", free_resources:bool = True):
        try:
            if req.seed == -1:
                req.seed = None
            if req.inference_steps <= 0:
                req.inference_steps = 1
            if req.num_gen <= 0:
                req.num_gen = 1
            target_device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
            n_steps = req.inference_steps
            guidance = req.guidance
            refiner_guidance = req.refiner_guidance if req.refiner_guidance is not None else None
            
            width = req.width
            height = req.height
            # Create torch.Generator for reproducible results (diffusers expects 'generator' param)
            generator = torch.Generator(target_device).manual_seed(req.seed) if req.seed is not None else None
            negative_prompt = None if req.negative_prompt == "" else req.negative_prompt
            refiner_neg_prompt = req.refiner_negative_prompt if req.refiner_negative_prompt is not None else None
            if negative_prompt is not None and req.copy_base_negative_prompt:
                refiner_neg_prompt = negative_prompt
            # ============================================================
            # CRITICAL PARAMETER: Controls base/refiner split
            # ============================================================
            # high_noise_frac = 0.7 means:
            #   - Base model    processes noise from t=1000 to t=300 (70% of denoising)
            #   - Refiner model processes noise from t=300 to t=0   (30% of denoising)
            # This creates a "handoff" where specialized models handle their optimal range
            # Tune this to balance speed vs quality:
            #   - Lower (0.5-0.6): More refinement, slower, higher quality
            #   - Higher (0.8-0.9): Less refinement, faster, base model dominant
            # ============================================================
            high_noise_frac = req.denoising_split_fraction
            
            pipe = None
            refiner = None
            if req.cache_diffusion_pipe:
                if self._cached_pipe is None or self._cached_refiner is None or self.cached_model != "stabilityai/stable-diffusion-xl-base-1.0":
                    logger.info("-- SDXL refiner pipe not available >> caching SDXL refiner pipe components --")
                    self.clear_cache()
                    self._cache_sdxl_refiner_pipe()
                    if self._cached_pipe is None or self._cached_refiner is None:
                        raise HTTPLoggedException(status_code=500, detail="-- an error has occured while caching the SDXL refiner pipe --")
                pipe = self._cached_pipe
                refiner = self._cached_refiner
            else:
                logger.info("-- loading uncached SDXL pipe --")
                try:
                    logger.warning("-- loading SDXL pipe >> clearing cached resources --")
                    self.clear_cache()
                    vae = self.load_vae(model=default_sdxl_vae,target_device=target_device)
                    pipe = self.load_sdxl_pipe(vae=vae, target_device=target_device)
                    refiner = self.load_refiner(vae=vae, target_device=target_device)
                    self._apply_pipe_optimizations(pipe=pipe, target_device=target_device)
                    self._apply_pipe_optimizations(pipe=refiner, target_device=target_device)    
                except Exception as e:
                    logger.error(f"-- An error has occured while loading the refiner pipe -- Exception: {e}")
                    raise HTTPLoggedException(status_code=500, detail="An error has occured while loading the SDXL refiner")
            prompt = req.prompt
            
            # ============================================================
            # STAGE 1: Base Model - Rough Composition
            # ============================================================
            # denoising_end=high_noise_frac stops the base model at 70% completion
            # Input:  Pure random noise (t=1000)
            # Output: Structured latent (t=300, still noisy but structured)
            # Role:   Creates overall composition and layout
            # 
            # ⚠️ PARAMETER NOTES:
            # ============================================================
            # guidance_scale: Controls prompt adherence (0.0-20.0)
            #   - QUALITY RISK: Values >10 often produce artifacts, oversaturation
            #   - DEFAULT (7.5) works best for SDXL - don't override unless needed
            #   - Too high (>12): Causes unnaturalness, color bleeding, artifacts
            #   - If degraded quality: ensure guidance is between 5.0-8.5
            #
            # width/height: Image dimensions in pixels
            #   - Must be multiples of 64 for SDXL
            #   - RTX 4060 Ti safe: up to 1024x1024
            #   - Higher resolution = more VRAM, slightly slower
            #   - SDXL works best at 1024x1024 or similar aspect ratios
            #
            # seed: Reproducibility generator (torch.Generator object)
            #   - Use: torch.Generator(device).manual_seed(int_value)
            #   - None = random seed each generation
            #   - Same seed + same prompt = identical output (for debugging)
            #
            # negative_prompt: Concepts to AVOID in the image
            #   - Example: "blurry, low quality, deformed, ugly"
            #   - ONLY effective when guidance_scale >7.5 [!!!]
            #   - If empty string "": Must convert to None to avoid errors
            #   - Excessive negative prompts (>50 tokens) can hurt quality
            # ============================================================
            logger.info(f"-- STAGE 1: Base model denoising (0% -> {high_noise_frac*100}%) --")
            image = pipe(
                prompt=prompt, 
                num_inference_steps=n_steps,
                num_images_per_prompt=req.num_gen,
                guidance_scale=guidance,  # ⚠️ Keep between 5.0-8.5 for best quality
                width=width,
                height=height,
                generator=generator,  # torch.Generator object for reproducibility
                negative_prompt=negative_prompt, 
                denoising_end=high_noise_frac,  # Stop at 70% - pass to refiner
                output_type="latent"  # Return latent format (efficient for stage 2)
            ).images

            # ============================================================
            # STAGE 2: Refiner Model - Quality Enhancement
            # ============================================================
            # denoising_start=high_noise_frac starts refiner from 70% completion
            # Input:  Base model's latent output (t=300, partially denoised)
            # Output: Final PIL Image (t=0, fully denoised and decoded)
            # Role:   Refines details, textures, consistency
            #
            # ⚠️ REDUNDANT PARAMETERS NOTE:
            # ============================================================
            # The refiner ALSO RECEIVES GUIDANCE_SCALE, but SDXL refiners work BEST
            # with guidance values LOWER THAN the BASE model:
            #   - Base model: 7.5 (default)
            #   - Refiner: 5.0-7.0 (recommended)
            #   - Reason: Refiner specializes in detail, doesn't need high guidance
            #   - Over-guiding the refiner causes artifacts & detail degradation
            #
            # height/width: Refiner maintains same resolution from base model
            #   - Should match base model output dimensions
            #   - DiffusionPipeline handles automatic dimension conversion
            #
            # seed: CRITICAL for consistency between base and refiner
            #   - MUST be same Generator object (or None for both) [!!!]
            #   - Different seeds = temporal artifacts between stages
            #
            # negative_prompt: Refiner applies same negative guidance
            #   - For refinement stage, less effective than in base
            #   - Still useful for "clean up" specific unwanted elements
            # ============================================================
            logger.info(f"-- STAGE 2: Refiner model ({high_noise_frac*100}% -> 100%) --")
            images = refiner(
                prompt=prompt, 
                num_inference_steps=n_steps, 
                guidance_scale=refiner_guidance,  # ⚠️ Consider reducing to 5.0-6.5 for refiner
                width=width,
                height=height,
                generator=None if req.with_random_refiner_seed else generator,  # Must match base model generator for consistency
                negative_prompt=refiner_neg_prompt,
                num_images_per_prompt=req.num_gen, 
                denoising_start=high_noise_frac,  # Start from where base ended (70%)
                image=image  # Takes latent from base, refines it
            ).images
            #logger.warning(f"-- ON REFINER RESULT {type(image)} --")
            
            if free_resources:
                self.free_resources(device=target_device)
            
            logger.info(f"-- REFINED PIPE COMPLETE >> REQ IMAGES: {req.num_gen} >> NUM IMAGES: {len(images)} --")
            return images
        except Exception as e:
            logger.error(f"-- An error has occured while infering the SDXL refiner pipe >> Exception: {e}")
            raise HTTPLoggedException(status_code=500, detail="An error has occured while infering the refined SDXL pipe")
    
    @override
    def _cache_default_pipe(self):
        if self._cached_pipe is not None:
            self.clear_cache()
        logger.info(f"-- SD Integration >> default pipe: {self.default_integration_model}")
        if "sdxl" in self.default_integration_model:
            logger.info("-- SD Integration >> caching sdxl refiner pipe --")
            self._cache_sdxl_refiner_pipe()
        else:
            logger.info(f"-- SD Integration >> caching sd3 pipe")
            self._cache_sd3_pipe()

    def _cache_sdxl_refiner_pipe(self):
        """
        Optimized SDXL pipeline caching for RTX 4060 Ti (16GB VRAM, 48GB RAM)
        
        Major Optimizations Applied:
            1 - Precision: float16 over bfloat16

            RTX 4060 Ti has better native float16 support
            Saves ~2-3GB VRAM with minimal quality loss
            
            2 - Memory-Efficient Attention (New!)

            Added _apply_pipe_optimizations() helper method
            Primary: xformers memory-efficient attention
            Fallback: attention slicing for broad compatibility
            Saves ~30-40% memory during inference
            
            3 - VAE Tiling

            Handles large images without OOM
            Processes in chunks instead of all-at-once
            Essential for 1024x1024+ resolutions
            
            4 - TensorFloat32 (TF32) Acceleration

            Faster matrix multiplication operations
            ~10-30% speed improvement on RTX 40-series
            
            5 -Model Loading Optimizations

            low_cpu_mem_usage=True: Streams layers to avoid loading spikes
            variant="fp16": Uses smaller model variant
            safety_check=None: Removes safety classifier (~1.5GB saved)
            Shared VAE between base+refiner (eliminates duplication)
            
            6 - Enhanced Logging & Error Handling

            Better debugging with detailed progress messages
            Graceful fallbacks for missing optimizations

        Memory profile:
        - Base model: ~7.5GB
        - Refiner model: ~3.5GB
        - VAE: ~1.5GB
        - Active during inference: ~12GB total
        """
        target_device:str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"-- CACHING SDXL PIPE >> device: {target_device} --")
        # Load VAE (shared between base and refiner)
        logger.info("-- LOADING VAE --")
        vae = self.load_vae(target_device=target_device, model=default_sdxl_vae)
        if vae is None:
            raise HTTPLoggedException(status_code=500, detail="-- failed to load VAE --")
        
        # Load base model
        logger.info("-- LOADING SDXL BASE MODEL --")
        pipe = self.load_sdxl_pipe(vae=vae, target_device=target_device)
        if pipe is None:
            raise HTTPLoggedException(status_code=500, detail="-- unable to load the default SD-XL pipe --")
        self._cached_pipe = pipe
        self.cached_model = sdxl_base_model
        # Load refiner model
        logger.info("-- LOADING SDXL REFINER MODEL --")
        refiner = self.load_refiner(vae=vae, target_device=target_device)
        if refiner is None:
            raise HTTPLoggedException(status_code=500, detail="-- unable to load the default SD-XL refiner --")
        self._cached_refiner = refiner
        
        # Apply memory optimizations to both pipelines
        logger.info("-- APPLYING MEMORY OPTIMIZATIONS --")
        self._apply_pipe_optimizations(self._cached_pipe, target_device)
        self._apply_pipe_optimizations(self._cached_refiner, target_device)
        
        logger.info("-- SDXL PIPELINE CACHED AND OPTIMIZED --")

    def _cache_sd3_pipe(self):
        target_device:str = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = self.get_pipe_for_model(model=sd3_default_model, device=target_device, with_auto_offload=self._enable_auto_offload)
        self._cached_pipe = pipe
        self._cached_pipe_device = target_device
        self.cached_model = sd3_default_model

    def load_vae(self, model:str, target_device:str = "cuda") -> AutoencoderKL:
        """
        Load VAE with RTX 4060 Ti optimizations
        
        Using float16 for RTX 4060 Ti (bfloat16 support is limited)
        VAE is shared between base and refiner models
        """
        try:
            # float16 is preferred over bfloat16 for RTX 4060 Ti
            torch_dtype = torch.float16 if target_device == "cuda" else torch.float32
            logger.info(f"-- Loading VAE: {model} (dtype: {torch_dtype}) --")
            
            vae = AutoencoderKL.from_pretrained(
                model, 
                torch_dtype=torch_dtype, 
                token=self._hf_token
            )
            vae.to(target_device)
            vae.eval()  # Set to evaluation mode to disable training-specific ops
            
            logger.info(f"✓ VAE loaded successfully")
            logger.info(f"VAE memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")
            return vae
        except Exception as e:
            logger.error(f"-- error loading VAE model {model}: {e} --")
            return None
        
    def load_sdxl_pipe(self, vae:str, target_device:str = "cuda") -> DiffusionPipeline:
        """
        Load SDXL base model with RTX 4060 Ti optimizations
        
        Key optimizations:
        - float16 precision (better for RTX 4060 Ti)
        - low_cpu_mem_usage enabled (reduce temporary allocations)
        - fp16 variant (reduces model size)
        - No safety checker (saves ~1.5GB VRAM)
        """
        try:
            # float16 is preferred over bfloat16 for RTX 4060 Ti
            torch_dtype = torch.float16 if target_device == "cuda" else torch.float32
            logger.info(f"-- Loading SDXL Base Model (dtype: {torch_dtype}) --")
            
            pipe = DiffusionPipeline.from_pretrained(
                sdxl_base_model,
                token=self._hf_token,
                vae=vae,
                torch_dtype=torch_dtype,
                variant="fp16",  # Load fp16 variant to reduce model size
                low_cpu_mem_usage=True,  # Stream layers to VRAM as needed
                safety_check=None  # Disable safety checker (~1.5GB saved)
            )
            pipe.to(target_device)
            logger.info(f"✓ SDXL Base Model loaded successfully")
            logger.info(f"Base model Pipeline memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")
            return pipe
        except Exception as e:
            logger.error(f"-- error loading SDXL base model: {e} --")
            return None
    
    def load_refiner(self, vae:str, target_device:str = "cuda") -> DiffusionPipeline:
        """
        Load SDXL refiner model with RTX 4060 Ti optimizations
        
        Key optimizations:
        - float16 precision (better for RTX 4060 Ti)
        - low_cpu_mem_usage enabled
        - fp16 variant (reduces model size)
        - No safety checker (saves ~1.5GB VRAM)
        - Shared VAE reduces memory footprint
        """
        try:
            # float16 is preferred over bfloat16 for RTX 4060 Ti
            torch_dtype = torch.float16 if target_device == "cuda" else torch.float32
            logger.info(f"-- Loading SDXL Refiner Model (dtype: {torch_dtype}) --")
            
            refiner = DiffusionPipeline.from_pretrained(
                sdxl_refiner_model,
                token=self._hf_token,
                vae=vae,
                torch_dtype=torch_dtype,
                variant="fp16",  # Load fp16 variant to reduce model size
                low_cpu_mem_usage=True,  # Stream layers to VRAM as needed
                safety_check=None  # Disable safety checker (~1.5GB saved)
            )
            refiner.to(target_device)
            logger.info(f"✓ SDXL Refiner Model loaded successfully")
            logger.info(f"Refiner Pipeline memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")
            return refiner
        except Exception as e:
            logger.error(f"-- error loading SDXL refiner model: {e} --")
            return None
        
    def load_sd3_pipe(self, model:str, device:str = "cuda") -> StableDiffusion3Pipeline:
        if "stable-diffusion-3" not in model:
            logger.warning(f"-- The requested model ({model}) is not a Stable Diffusion 3 model --")
            return None
        logger.info(f"-- Loading Stable Diffusion 3 model: {model} --")
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        logger.info(f"-- stable diffusion pipeline for {model} on {device} using {torch_dtype} --")
        if self._current_quant_cfg is not None:
            self._quantization_cfg = self.get_quantization_cfg(default_quant_type)
            if self._quantization_cfg is None:
                raise HTTPLoggedException(status_code=500, detail=f"-- 4bit quantization config not found --")
            transformer_cfg = self._quantization_cfg.quant_mapping["transformer"]
            if transformer_cfg is None:
                raise HTTPLoggedException(status_code=500, detail="-- no 'transformer' 4bit config has been found --")
            model_nf4 = SD3Transformer2DModel.from_pretrained(
                model,
                subfolder="transformer",
                quantization_config=transformer_cfg,
                torch_dtype=torch.bfloat16
            )
            logger.info(f"-- loading {model} with transformer 4bit quantization --")
            return StableDiffusion3Pipeline.from_pretrained(model, transformer=model_nf4, token=self._hf_token, torch_dtype=torch_dtype)
        else:
            return StableDiffusion3Pipeline.from_pretrained(model, token=self._hf_token, torch_dtype=torch_dtype if self._quantization_cfg is None else torch.float32)
