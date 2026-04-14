
from typing import List

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from api.diffusion_module.integrations.base_integration import ModelIntegration
from api.diffusion_module.integrations.flux_integration import Flux_Integration
from api.diffusion_module.integrations.sd_integration import SD_Integration
from api.diffusion_module.schemas import GenerateImageRequest
from api.routes.models.api_schemas import  GenerateImageResponse
from api.services.images_mgmt_service import ImagesMgmtService
from api.utils.helpers import HTTPLoggedException

router = APIRouter(prefix="/flux")

@router.post(
    "/prompt/generate",
    summary="Generates an image from the text passed along in the request",
    responses={
        200:{"description":"Successful response with the generated image in base64 format, the name and the promt used to generate the image"}
    }
)
async def generate_from_prompt(req: GenerateImageRequest, request:Request) -> List[GenerateImageResponse]:
    logger.info('-- on prompt req -- ')
    print(req)
    svc = ImagesMgmtService()
    diffusor: Flux_Integration = get_sd_provider(request)
    if diffusor is None:
        raise HTTPLoggedException(status_code=500, detail="An error has occured while accessing the diffusor integration")
    image = diffusor.generate_image(req)
    return [] if image is None else [await svc.save_request(req=req, generated_image=image)] if image is not None else []

def get_sd_provider(request:Request) -> Flux_Integration:
    if request.app.model_provider.is_available() is False:
        raise HTTPLoggedException(status_code=500, detail="Model Provider not initialized or no diffusor models available for the current integration") 
    if request.app.model_provider.current_integration_name != "flux":
        raise HTTPLoggedException(status_code=500, detail=f"Current diffusor integration ({request.app.model_provider.current_integration_name}) is not valid")
    integration: ModelIntegration = request.app.model_provider.integration
    if integration is None:
        raise HTTPLoggedException(status_code=500, detail="No diffusor integration has been found")
    if isinstance(integration, Flux_Integration) is False:
        raise HTTPLoggedException(status_code=500, detail=f"An error has occured while accessing the diffusor integration. Loaded class is: {type(integration)}")
    return integration