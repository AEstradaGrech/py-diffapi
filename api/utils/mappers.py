from typing import Any

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