from pydantic import BaseModel


class CaptionResponse(BaseModel):
    """`caption` is "" when captioning is disabled or the ingest timed out —
    the endpoint degrades rather than raising, so an empty string is the
    caller's signal to fall back to no context."""
    caption: str
