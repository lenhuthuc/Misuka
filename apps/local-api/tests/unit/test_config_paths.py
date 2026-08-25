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


def test_resolved_text_vad_checkpoint_path_stays_inside_apps_local_api():
    settings = Settings()
    assert settings.resolved_text_vad_checkpoint_path == settings.base_dir / "model" / "best_text_vad.pt"


def test_resolved_multimodal_vad_checkpoint_path_stays_inside_apps_local_api():
    settings = Settings()
    assert settings.resolved_multimodal_vad_checkpoint_path == settings.base_dir / "model" / "best_multimodal_vad.pt"


def test_sherpa_onnx_model_dir_defaults_to_assets_models():
    settings = Settings()
    expected_root = Path(__file__).resolve().parents[4]
    assert settings.sherpa_onnx_model_dir == (
        expected_root / "assets" / "models" / "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"
    )


def test_resolved_sherpa_paths_fall_back_to_the_model_dir_when_unset():
    settings = Settings(sherpa_onnx_tokens="", sherpa_onnx_encoder="", sherpa_onnx_decoder="", sherpa_onnx_joiner="")
    assert settings.resolved_sherpa_tokens == str(settings.sherpa_onnx_model_dir / "tokens.txt")
    assert settings.resolved_sherpa_encoder == str(settings.sherpa_onnx_model_dir / "encoder.int8.onnx")
    assert settings.resolved_sherpa_decoder == str(settings.sherpa_onnx_model_dir / "decoder.onnx")
    assert settings.resolved_sherpa_joiner == str(settings.sherpa_onnx_model_dir / "joiner.int8.onnx")


def test_resolved_sherpa_paths_prefer_explicit_env_override():
    settings = Settings(sherpa_onnx_tokens="/custom/tokens.txt")
    assert settings.resolved_sherpa_tokens == "/custom/tokens.txt"
