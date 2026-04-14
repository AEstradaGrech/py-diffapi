from typing import List

from fastapi import APIRouter, Request
from loguru import logger

from api.diffusion_module.schemas import IntegrationSettings, ProviderSettings, SetProviderRequest, UpdateQuantConfigRequest
from api.utils.helpers import HTTPLoggedException
from api.utils.statics import quantization_types

router = APIRouter(prefix="/mgmt")

@router.get(
    "/default-quantizations", 
    summary="Returns the default quantization types",
    responses={
        200:{"description":"Successful response with a list of default quantization configuration names"}
    }
)
async def get_default_quantizations() -> List[str]:
    return quantization_types

@router.get(
    "/provider/settings", 
    summary="Returns the current provider settings",
    responses={
        200:{"description":"Successful response with the available integrations, status data and current integration settings"}
    }
)
async def get_settings(request: Request) -> ProviderSettings:
    if request.app.model_provider is None:
        raise HTTPLoggedException(status_code=500, detail="-- No model provider instance has been found --")
    return request.app.model_provider.get_current_settings() 

@router.get(
    "/integration/settings", 
    summary="Returns the current integration settings",
    responses={
        200:{"description":"Successful response with the current integration settings dto"}
    }
)
async def get_current_integration_settings(request: Request) -> IntegrationSettings:
    if request.app.model_provider is None:
        raise HTTPLoggedException(status_code=500, detail="-- No model provider instance has been found --")
    if request.app.model_provider.init is False:
        raise HTTPLoggedException(status_code=500, detail="-- Model provider has not been initialized. No current integration available --")
    return request.app.model_provider.get_current_integration_settings()

@router.post(
    "/set-integration",
    summary="Sets the integration for the model provider",
    responses={
        200:{"description":"Successful response with the updated integration"}
    }
)
async def set_integration(request:Request, integration_req: SetProviderRequest) -> IntegrationSettings:
    if request.app.model_provider.init is False:
        raise HTTPLoggedException(status_code=500, detail="Model Provider not initialized") 
    try:
        new_settings = request.app.model_provider.set_integration_model(integration_req)
        if new_settings is None:
            raise HTTPLoggedException(status_code=500, detail=f"-- An error has occured while setting model: {integration_req.cache_model} --")
        return new_settings
    except Exception as e:
        logger.error(f"An error has occured while setting the integration: {e}")
        raise HTTPLoggedException(status_code=500, detail=f"An error has occured while setting the integration: {e}")
    
@router.post(
    "/quantization/set-config/default",
    summary="Sets the quantization config for the model provider",
    responses={
        200:{"description":"Successful response with the updated quantization config"}
    }
)
async def set_quantization_config(request:Request, req: UpdateQuantConfigRequest) -> IntegrationSettings:
    if request.app.model_provider.init is False:
        raise HTTPLoggedException(status_code=500, detail="Model Provider not initialized") 
    try:
        new_settings = request.app.model_provider.update_quant_config(req)
        if new_settings is None:
            raise HTTPLoggedException(status_code=500, detail=f"-- An error has occured while setting quant configuration: {req.config_name} --")
        return new_settings
    except Exception as e:
        logger.error(f"An error has occured while setting the quantization config: {e}")
        raise HTTPLoggedException(status_code=500, detail=f"An error has occured while setting the quantization config: {e}")