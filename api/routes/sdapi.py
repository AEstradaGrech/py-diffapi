
from typing import List

from fastapi import APIRouter, Request
from loguru import logger
from api.diffusion_module.integrations.base_integration import ModelIntegration
from api.diffusion_module.integrations.sd_integration import SD_Integration
from api.routes.models.api_schemas import GenerateImageResponse
from api.diffusion_module.schemas import GenerateRefinedImageRequest, StableDiffusionImageRequest
from api.services.images_mgmt_service import ImagesMgmtService
from api.utils.helpers import HTTPLoggedException

import torch

router = APIRouter(prefix="/stablediffusion")

@router.post(
    "/prompt/generate",
    summary="Generates an image from the text passed along in the request",
    responses={
        200:{"description":"Successful response with the generated image in base64 format, the name and the promt used to generate the image"}
    }
)
async def generate_from_prompt(req: StableDiffusionImageRequest, request:Request) -> GenerateImageResponse:
    logger.info('-- on prompt req -- ')
    print(req)
    svc = ImagesMgmtService()
    diffusor: SD_Integration = get_sd_provider(request)
    if diffusor is None:
        raise HTTPLoggedException(status_code=500, detail="An error has occured while accessing the diffusor integration")
    current_req = req if not req.use_refiner else GenerateRefinedImageRequest(name=req.name, diffuser=req.diffuser_name, prompt=req.prompt, inference_steps=req.inference_steps, guidance=req.guidance, file_save=req.file_save, db_save=req.db_save, cache_diffusion_pipe=req.cache_diffusion_pipe)
    image = diffusor.refined_pipe(current_req) if req.use_refiner else diffusor.generate_image(current_req)
    if image is None:
        raise HTTPLoggedException("-- an error has occured while generating the image --")
    return await svc.save_request(req=req, generated_image=image)

@router.post(
    "/refined-pipe/prompt/generate",
    summary="Generates an image from the text passed along in the request using a refining pipeline",
    responses={
        200:{"description":"Successful response with the generated image in base64 format, the name and the promt used to generate the image"}
    }
)
async def generate_refined_from_prompt(req: GenerateRefinedImageRequest, request:Request) -> GenerateImageResponse:
    logger.info('-- on refined pipe prompt req -- ')
    print(req)
    svc = ImagesMgmtService()
    diffusor: SD_Integration = get_sd_provider(request)
    if diffusor is None:
        raise HTTPLoggedException(status_code=500, detail="An error has occured while accessing the diffusor integration")
    logger.info("-- ON REFINED PIPE GENERATE IMAGE ENDPOINT HIT --")    
    image = diffusor.refined_pipe(req)
    if image is None:
        raise HTTPLoggedException("-- an error has occured while generating the image --")
    return await svc.save_request(req=req, generated_image=image)

def get_sd_provider(request:Request) -> SD_Integration:
    if request.app.model_provider.is_available() is False:
        raise HTTPLoggedException(status_code=500, detail="Model Provider not initialized or no diffusor models available for the current integration") 
    if request.app.model_provider.current_integration_name != "stablediffusion":
        raise HTTPLoggedException(status_code=500, detail=f"Current diffusor integration ({request.app.model_provider.current_integration_name}) is not valid")
    integration: ModelIntegration = request.app.model_provider.integration
    if integration is None:
        raise HTTPLoggedException(status_code=500, detail="No diffusor integration has been found")
    if isinstance(integration, SD_Integration) is False:
        raise HTTPLoggedException(status_code=500, detail=f"An error has occured while accessing the diffusor integration. Loaded class is: {type(integration)}")
    return integration