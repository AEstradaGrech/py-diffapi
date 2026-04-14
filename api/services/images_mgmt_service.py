
from datetime import datetime
from uuid import uuid4
from PIL import Image
from api.diffusion_module.schemas import GenerateImageRequest
from api.infrastructure.repositories.mongo_repositories import GenImagesRepository
from api.infrastructure.models.db_schemas import GeneratedImageDoc
from api.routes.models.api_schemas import  GenerateImageResponse
from api.utils.statics import default_db_name
from api.utils.helpers import pil_to_base64
from loguru import logger

class ImagesMgmtService:
    _repo: GenImagesRepository

    def __init__(self, repo_name:str = "default", db_collection:str = "GeneratedImages"):
        if repo_name is None:
            repo_name = "default"
        self._repo = GenImagesRepository(default_db_name if repo_name == "default" or repo_name == "" else repo_name, db_collection)

    async def db_save_base64(self, req: GenerateImageRequest, b64:str, extra_tag:str = None) -> str:
        logger.info("-- IMAGES SERVICE >> saving db BASE64 --")
        filename:str = self.get_formatted_filename(req=req, append_tag=False, extra_tag=extra_tag)
        try:
            doc = GeneratedImageDoc(
                    name=filename,
                    diffuser=req.diffuser_name,
                    tag=req.tag,
                    prompt=req.prompt,
                    height=req.height,
                    width=req.width,
                    guidance=req.guidance,
                    inference_steps=req.inference_steps,
                    base64=b64)
            doc.creation_date = datetime.now()
            await self._repo.create(doc)
            logger.info(f"-- IMAGES SERVICE >> DB IMAGE SAVED >> filename: {filename} --")
            return b64
        except Exception as e:
            logger.error(f"-- an error has occured while saving B64 to DB >> Exception: {e}--")
            return None
        
    async def db_save_image(self, req: GenerateImageRequest, image:Image, extra_tag:str = None) -> str:
        logger.info("-- IMAGES SERVICE >> saving db PIL image --")
        return await self.db_save_base64(req=req, b64=pil_to_base64(image), extra_tag=extra_tag)
        
    def disk_save_image(self, req: GenerateImageRequest, image: Image, extra_tag:str = None) -> str:
        filename:str = self.get_formatted_filename(req=req, append_tag=True, extra_tag=extra_tag)
        logger.info(f'saving image : {filename}')
        try:
            image.save(f"generated_images/{filename}.png")
            logger.info(f"-- IMAGES SERVICE >> DISK IMAGE SAVED >> filename: {filename} --")
            return filename
        except:
            logger.error(f'an error has occured while saving image: {filename} to disk')
            return None
        
    def get_formatted_filename(self, req: GenerateImageRequest, append_tag:bool, extra_tag:str = None) -> str:
        if req.name is None or len(req.name) == 0:
            req.name = uuid4()
        filename:str = f"{req.name}-{req.tag if len(req.tag) > 0 else uuid4()}" if append_tag else req.name
        filetag:str = extra_tag if extra_tag is not None and len(extra_tag) > 0 else None
        if filetag is not None:
            filename = f"{filename}-{filetag}"
        return filename

    async def save_request(self, req: GenerateImageRequest, generated_image:Image, extra_tag:str = None) -> GenerateImageResponse:
        if req.name is None or len(req.name) == 0:
            req.name = uuid4()
        b64:str = None
        if req.db_save:
            b64 = await self.db_save_image(req=req, image=generated_image, extra_tag=extra_tag)
            if b64 is not None:
                logger.info(f"-- request generated image saved to MONGO >> filename: {req.name} --")
        if req.file_save:
            png_filename:str = self.disk_save_image(req=req, image=generated_image, extra_tag=extra_tag)
            if png_filename is not None:
                logger.info(f"-- request generated image saved to PNG file >> FILENAME {png_filename} --")
        return GenerateImageResponse(name=req.name, diffuser=req.diffuser_name, prompt=req.prompt, height=req.height, width=req.width, base64=b64)