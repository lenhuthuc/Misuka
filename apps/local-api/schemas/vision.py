from pydantic import BaseModel


class CaptionResponse(BaseModel):
    """`caption` is "" when captioning is disabled or every VLM backend failed
    to load — `CaptionService` degrades to a placeholder rather than raising,
    so an empty string is the caller's signal to fall back to no context."""
    caption: str
