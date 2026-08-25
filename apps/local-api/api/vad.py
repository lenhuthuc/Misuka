from fastapi import APIRouter, Depends

from api.dependencies import get_container
from core.container import ServiceContainer
from schemas.vad import VADRequest, VADResponse

router = APIRouter(prefix="/vad", tags=["VAD"])


@router.post("", response_model=VADResponse)
def predict(body: VADRequest, container: ServiceContainer = Depends(get_container)):
    # Legacy contract: signed [-1, 1], via TextVADService.predict() (an alias
    # for predict_signed()) — unchanged for any existing caller of this route.
    v, a, d = container.text_vad.predict(body.text)
    return VADResponse(v=v, a=a, d=d)
