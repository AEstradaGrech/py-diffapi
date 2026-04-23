
from datetime import datetime
import math
from typing import List
from uuid import uuid4
from PIL import Image
from api.diffusion_module.schemas import GenerateImageRequest
from api.infrastructure.repositories.mongo_repositories import GenImagesRepository
from api.infrastructure.models.db_schemas import GeneratedImageDoc
from api.routes.models.api_schemas import  CollectionResponse, FileSaveResponse, GenerateImageResponse, GeneratedImageDto, PatchImageRequest, QueryFilter
from api.utils.mappers import generatedImageToDto, imageDtoToDocument
from api.utils.statics import default_db_name
from api.utils.helpers import HTTPLoggedException, pil_to_base64, base64_to_pil
from loguru import logger

class ImagesMgmtService:
    _repo: GenImagesRepository

    def __init__(self, repo_name:str = "default", db_collection:str = "GeneratedImages"):
        if repo_name is None:
            repo_name = "default"
        self._repo = GenImagesRepository(default_db_name if repo_name == "default" or repo_name == "" else repo_name, db_collection)

    async def db_save_base64(self, req: GenerateImageRequest, b64:str, extra_tag:str = None) -> GeneratedImageDoc:
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
            logger.info(f"-- IMAGES SERVICE >> DB IMAGE SAVED >> filename: {filename} --")
            return await self._repo.create(doc)
        except Exception as e:
            logger.error(f"-- an error has occured while saving B64 to DB >> Exception: {e}--")
            return None
        
    async def db_save_image(self, req: GenerateImageRequest, image:Image, extra_tag:str = None) -> GeneratedImageDoc:
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
        dto: GeneratedImageDto = None
        if req.db_save:
            dto = generatedImageToDto(await self.db_save_image(req=req, image=generated_image, extra_tag=extra_tag))
            if dto is not None:
                logger.info(f"-- request generated image saved to MONGO >> filename: {req.name} --")
        if req.file_save:
            png_filename:str = self.disk_save_image(req=req, image=generated_image, extra_tag=extra_tag)
            if png_filename is not None:
                logger.info(f"-- request generated image saved to PNG file >> FILENAME {png_filename} --")
        return GenerateImageResponse(name=req.name, diffuser=req.diffuser_name, prompt=req.prompt, height=req.height, width=req.width, db_doc=dto, base64=pil_to_base64(generated_image))
    
    async def save_image(self, dto: GeneratedImageDto) -> GeneratedImageDto:
        if dto.base64 is None:
            raise HTTPLoggedException(status_code=400, detail="No Base64 string to save")
        dto.id = None
        return generatedImageToDto(await self._repo.create(imageDtoToDocument(dto)))

    async def file_save_image(self, dto: GeneratedImageDto) -> FileSaveResponse:
        if dto.base64 == "":
            dto.base64 = None
        if dto.base64 is None:
            raise HTTPLoggedException(status_code=400, detail="No Base64 string to convert")
        if dto.name == "":
            dto.name = None
        filename = dto.name if dto.name is not None else uuid4()
        if dto.tag is not None and dto.tag != "":
            filename = f"{filename}-{dto.tag}"
        image = base64_to_pil(dto.base64)
        filepath = f"generated_images/{filename}.png"
        image.save(filepath)
        return FileSaveResponse(filename=filename, output_path=filepath)

    async def query_images(self, filter: QueryFilter) -> List[GeneratedImageDto]:
        docs = await self._repo.query(filter.conditions) if len(filter.conditions) > 0 else await self._repo.stringy_query({})
        results = []
        if len(docs) > 0:
            if filter.page is None and filter.page_size is None:
                results = [generatedImageToDto(doc) for doc in docs]
                return CollectionResponse(data=results, page=0, total_pages=1, total_records=len(docs))
            if filter.page_size is None:
                filter.page_size = 10
            for i in range(filter.page * filter.page_size, filter.page * filter.page_size + filter.page_size):
                if i < len(docs):
                    results.append(generatedImageToDto(docs[i]))
            pages = math.ceil(len(docs) / filter.page_size) if len(docs) > 0 else 0
            return CollectionResponse(data=results, page=filter.page, total_pages=pages, total_records=len(docs))
        return CollectionResponse(data=results, page=0, total_pages=0, total_records=0)

    async def delete_image(self, id: str) -> GeneratedImageDto:
        record = await self._repo.get_by_id(id=id)
        if record is None:
            raise HTTPLoggedException(status_code=404, detail=f"No document has been found with ID: {id}")
        if not await self._repo.delete_by_id(id=id):
            raise HTTPLoggedException(status_code=500, detail="An error has occured while deleting the document")
        return generatedImageToDto(record)

    async def patch_image(self, patch_req: PatchImageRequest) -> GeneratedImageDto:
        record = await self._repo.get_by_id(id=patch_req.id)
        if record is None:
            raise HTTPLoggedException(status_code=404, detail=f"No document has been found with ID: {patch_req.id}")
        doc = GeneratedImageDoc.model_validate(record)
        doc.name = patch_req.name
        doc.tag = patch_req.tag
        if not await self._repo.update(doc.id, doc):
            raise HTTPLoggedException(status_code=500, detail=f"An error has occured while updating the document with ID: {doc.id}")
        return generatedImageToDto(doc)
    
    async def update_image(self, dto: GeneratedImageDto) -> GeneratedImageDto:
        record = await self._repo.get_by_id(id=dto.id)
        if record is None:
            raise HTTPLoggedException(status_code=404, detail=f"No document has been found with ID: {dto.id}")
        doc = imageDtoToDocument(dto)
        if not await self._repo.update(dto.id, doc):
            raise HTTPLoggedException(status_code=500, detail=f"An error has occured while updating the document with ID: {dto.id}")
        return dto