from fastapi import APIRouter, Depends

from schemas.vad import VADRequest, VADResponse
from service.vad_service import VADService


router = APIRouter(prefix="/vad", tags=["VAD"])


def get_service() -> VADService:
    # overridden in main.py via app.dependency_overrides
    raise NotImplementedError


@router.post("", response_model=VADResponse)
def predict(body: VADRequest, svc: VADService = Depends(get_service)):
    v, a, d = svc.predict(body.text)
    return VADResponse(v=v, a=a, d=d)
