import gc
from typing import List
from fastapi import FastAPI
import torch
import uvicorn
from contextlib import asynccontextmanager

# from api.routes.vacatiapi import router as vacati_router
from api.routes.sdapi import router as sd_router
from api.routes.fluxapi import router as flux_router
from api.routes.mgmtapi import router as mgmt_router
from api.utils.statics import default_quant_type, default_integration_name
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from api.diffusion_module.diffusor_provider import ModelProvider
from dotenv import load_dotenv
import os
load_dotenv()

origins = [
    "http://localhost",
    "http://localhost:4200",
]
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("-- on app start --")
    gc.collect()
    torch.cuda.empty_cache()
    initial_integration:str = default_integration_name
    model_singleton = ModelProvider(app, 
        integration_name=initial_integration, 
        with_cached_diffusor=False, 
        with_auto_offload=False if initial_integration == "stablediffusion" else False,
        quantization=default_quant_type)
    
    #model_singleton.set_model(model_name="llama3.1:8b", provider="ollama", cpu_threads=6, verbose=True, ctx_len=4096, ngl=4352)
    status = "Not initialized"
    if model_singleton.init:
        status = "Initialized"
        logger.info("Models cache initialized...")

        logger.info("current provider: " + model_singleton.current_integration_name)
        app.model_provider = model_singleton
        print("-- on app start cache status -- "+ status)
    else:
        raise SystemExit()
    yield
    logger.info("-- on app shutdown --")
    
    model_singleton.clear()
    logger.info("Clearing API cache...")
    gc.collect()
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)

# app.include_router(vacati_router, prefix="/vacati-api")

app.include_router(sd_router)
app.include_router(flux_router)
app.include_router(mgmt_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
"""
The main purpose of the __name__ == "__main__" is to have some code that is executed when your file is called with:

    -- python main.py --

but is not called when another file imports it, like in:

    -- from myapp import app --
"""
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)

