"""Regression for settings that resolve paths by directory depth from
`brain/config.py`'s own `__file__` — these silently point at the wrong
directory (and, for `piper_models_dir`, make TTS report zero voices found)
if the app's directory nesting ever changes without updating the `parents[N]`
index to match. See REFACTOR_PLAN.md Phase 6: this exact bug was introduced
and caught when `VAD/` was renamed to `apps/local-api/` (one extra level of
nesting under the repo root).
"""
from pathlib import Path

from brain.config import Settings


def test_piper_models_dir_defaults_to_assets_models_voices():
    settings = Settings()
    # apps/local-api/brain/config.py -> brain -> local-api -> apps -> <repo root>
    expected_root = Path(__file__).resolve().parents[4]
    assert settings.piper_models_dir == expected_root / "assets" / "models" / "voices"


def test_piper_models_dir_default_actually_contains_the_shipped_onnx_voices():
    settings = Settings()
    onnx_files = list(settings.piper_models_dir.glob("*.onnx"))
    assert onnx_files, f"expected at least one .onnx voice under {settings.piper_models_dir}"


def test_resolved_vad_model_path_stays_inside_apps_local_api():
    settings = Settings()
    assert settings.resolved_vad_model_path == settings.base_dir / "model" / "vad_bert_final.pt"


def test_whisper_models_dir_stays_inside_apps_local_api():
    settings = Settings()
    assert settings.whisper_models_dir == settings.base_dir / "models"
