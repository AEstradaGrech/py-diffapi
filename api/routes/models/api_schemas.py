from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class QueryCondition(BaseModel):
    field: str = Field(description="Name of the desired document field to use for filtering")
    value: Any = Field(description="The value for this condition")
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )
class QueryFilter(BaseModel):
    conditions: List[QueryCondition] = Field(description="Array of conditions to build the query", default=[])
    page_size: Optional[int] = Field(description="Number of elements to return for the query. All of them if empty", default=None)
    page:Optional[int] = Field(description="Requested page", default=None)
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )
  
class GenerateImageResponse(BaseModel):
    name: str = Field(description="Name of the image file. A guid will be used instead if no name is provided")
    diffuser: str = Field(description="The name of the diffuser or inference endpoint to use")
    prompt: str = Field(description="The text that will be used to generate the image")
    height: float = Field(description="Height of the images that will be generated")
    width: float = Field(description="Width of the images that will be generated")
    base64: str = Field(description="image in base64 format")

class GeneratedImageDto(BaseModel):
    id:str = Field(description="Db identifier for the image")
    name: str = Field(description="Name of the image file. A guid will be used instead if no name is provided")
    diffuser: str = Field(description="The name of the diffuser or inference endpoint to use")
    prompt: str = Field(description="The text that will be used to generate the image")
    height: float = Field(description="Height of the images that will be generated")
    width: float = Field(description="Width of the images that will be generated")
    guidance: float = Field(description="Parameter to set how much does the model follow the user prompt")
    inference_steps:Optional[int] = Field(description="Number of inference steps used to generate the image. None if it was generated through the Inference Endpoint", default=None) 
    tag: Optional[str] = Field(None, description="Tags a chat session with something meaningful for the user to provide more context about the image")
    base64: str = Field(description="image in base64 format")
    
class PatchImageRequest(BaseModel):
    id:str = Field(description="Db identifier for the image")
    name: Optional[str] = Field(description="Name of the image file", default=None)
    tag: Optional[str] = Field(description="New tag for the image record", default=None)