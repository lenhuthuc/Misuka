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


def test_grounded_knowledge_decodes_colder_and_longer_than_chat():
    policy = derive_response_policy(None, default_max_tokens=320).for_grounded_knowledge(0.30, 480)

    assert policy.temperature == 0.30
    assert policy.max_tokens == 480
    assert policy.options(0.65, 320) == {"temperature": 0.30, "num_predict": 480}


def test_an_activated_user_still_gets_a_short_reply_on_a_grounded_turn():
    # The two axes compose conservatively: the arousal cap exists for the
    # user's sake, so it survives the knowledge headroom.
    aroused = derive_response_policy(
        VADScores(valence=-0.7, arousal=0.9, dominance=-0.6), default_max_tokens=320,
    )

    policy = aroused.for_grounded_knowledge(0.30, 480)

    assert policy.max_tokens == 256      # arousal's cap, not knowledge's 480
    assert policy.temperature == 0.30    # knowledge's floor, not arousal's 0.55
    assert "brief" in policy.instruction
