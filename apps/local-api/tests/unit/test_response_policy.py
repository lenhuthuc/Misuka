from brain.response_policy import derive_response_policy
from schemas.vad import VADScores


def test_high_arousal_low_dominance_prioritizes_short_grounding_response():
    policy = derive_response_policy(
        VADScores(valence=-0.7, arousal=0.9, dominance=-0.6), default_max_tokens=1024,
    )

    assert policy.max_tokens == 256
    assert policy.temperature == 0.55
    assert policy.stream_pace == "immediate"
    assert "brief" in policy.instruction
    assert "one concrete next step" in policy.instruction
    assert "-0.7" not in policy.instruction


def test_no_vad_leaves_generation_defaults_untouched():
    policy = derive_response_policy(None, default_max_tokens=1024)

    assert not policy.is_active
    assert policy.options(0.7, 1024) == {"temperature": 0.7, "num_predict": 1024}
