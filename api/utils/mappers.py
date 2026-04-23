from typing import Any

from bson import ObjectId
from datetime import datetime
from api.infrastructure.models.db_schemas import GeneratedImageDoc
from api.routes.models.api_schemas import GeneratedImageDto


def generatedImageToDto(document: Any) -> GeneratedImageDto:
    model:GeneratedImageDoc = GeneratedImageDoc.model_validate(document)
    return GeneratedImageDto(
        id=model.id,
        name=model.name,
        diffuser=model.diffuser,
        prompt=model.prompt,
        height=model.height,
        width=model.width,
        guidance=model.guidance,
        inference_steps=model.inference_steps,
        tag=model.tag,
        base64=model.base64
    )

def imageDtoToDocument(dto: GeneratedImageDto) -> GeneratedImageDoc:
    return GeneratedImageDoc(
        id=dto.id if dto.id is not None else ObjectId(),
        name=dto.name,
        diffuser=dto.diffuser,
        prompt=dto.prompt,
        height=dto.height,
        width=dto.width,
        guidance=dto.guidance,
        inference_steps=dto.inference_steps,
        tag=dto.tag,
        base64=dto.base64,
        creation_date=datetime.now() 
    )