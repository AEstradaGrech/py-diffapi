default_conn_str:str = "mongodb://localhost:27017/"
default_db_name:str = "GenImages"
default_collection_name:str = "FreeGenerations" 

sd3_default_model:str = "stabilityai/stable-diffusion-3.5-medium"
sdxl_base_model:str = "stabilityai/stable-diffusion-xl-base-1.0"
sdxl_refiner_model:str = "stabilityai/stable-diffusion-xl-refiner-1.0"
default_sdxl_vae: str = "madebyollin/sdxl-vae-fp16-fix"
default_integration_name:str = "stablediffusion"
default_flux_model:str = "black-forest-labs/FLUX.1-schnell"
default_sd_model:str = "stabilityai/stable-diffusion-3.5-medium"
default_quant_type:str = "4bit"

quantization_types: [str] = ["4bit", "4bit-full", "4bit+", "4bit++", "transformer_only"]