"""Shared FastAPI dependency getters.

Routes depend on `get_container` (or its per-service `get_*` slices below)
instead of importing/instantiating services directly, so the whole service
graph is swappable per-request via `app.dependency_overrides` in tests
without patching global module state.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from core.container import ServiceContainer


def get_container(request: Request) -> "ServiceContainer":
    return request.app.state.container
