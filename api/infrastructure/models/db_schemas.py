from datetime import datetime
from typing import Annotated, Optional

from bson import ObjectId
from pydantic import BaseModel, BeforeValidator, Field
PyObjectId=Annotated[str, BeforeValidator(str)]

class GeneratedImageDoc(BaseModel):
    id:Optional[PyObjectId] = Field(description="Db identifier for a user chat session", alias="_id", default_factory=ObjectId)
    name: str = Field(description="Name of the image file. A guid will be used instead if no name is provided")
    diffuser: str = Field(description="The name of the diffuser or inference endpoint to use")
    prompt: str = Field(description="The text that will be used to generate the image")
    height: float = Field(description="Height of the images that will be generated")
    width: float = Field(description="Width of the images that will be generated")
    guidance: float = Field(description="Parameter to set how much does the model follow the user prompt")
    inference_steps:Optional[int] = Field(description="Number of inference steps used to generate the image. None if it was generated through the Inference Endpoint", default=None) 
    tag: Optional[str] = Field(None, description="Tags a chat session with something meaningful for the user to provide more context about the image")
    creation_date: datetime = Field(description="self-explanatory", default=datetime.now())
    base64: str = Field(description="image in base64 format")