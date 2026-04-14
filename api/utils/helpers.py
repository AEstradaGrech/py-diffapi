import base64
import io
from PIL import Image
from fastapi import HTTPException
from loguru import logger

def pil_to_base64(image, format:str='png') -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue())

def base64_to_pil(img_base64):
    base64_decoded = base64.b64decode(img_base64)
    byte_stream = io.BytesIO(base64_decoded)
    pil_image = Image.open(byte_stream)
    return pil_image

class HTTPLoggedException(HTTPException):
    def __init__(self,status_code:int, detail:str):
        logger.error(f"<< APP EXCEPTION >> {status_code}:  {detail}")
        super().__init__(status_code=status_code, detail=detail)