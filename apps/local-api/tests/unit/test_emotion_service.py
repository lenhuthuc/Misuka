from brain.emotion_mapper import map_vad_to_emotion
from brain.emotion_service import EmotionReading, EmotionService, extract_memory_vads


def test_map_vad_to_emotion_neutral_at_origin():
    assert map_vad_to_emotion(0.0, 0.0, 0.0) == "neutral"


def test_map_vad_to_emotion_nearest_prototype_wins():
    # Close to "joy" (0.76, 0.48, 0.35) but far from every other prototype.
    assert map_vad_to_emotion(0.75, 0.47, 0.34) == "joy"


def test_extract_memory_vads_skips_docs_missing_any_vad_key():
    docs = [
        {"metadata": {"valence": 0.1, "arousal": 0.2, "dominance": 0.3}},
        {"metadata": {"valence": 0.1}},  # missing arousal/dominance
        {"metadata": {}},
        {},  # no metadata key at all
    ]
    assert extract_memory_vads(docs) == [(0.1, 0.2, 0.3)]


def test_current_state_returns_reading_unchanged_when_no_memories():
    svc = EmotionService(vad_service=None)
    current = EmotionReading.from_vad(0.5, 0.5, 0.5)
    assert svc.current_state(current, []) == current


def test_current_state_blends_toward_memory_mean():
    svc = EmotionService(vad_service=None, current_weight=0.5)
    current = EmotionReading.from_vad(1.0, 1.0, 1.0)
    blended = svc.current_state(current, [(0.0, 0.0, 0.0)])
    assert blended.vad == (0.5, 0.5, 0.5)
