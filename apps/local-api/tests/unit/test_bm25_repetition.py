from brain.bm25_repetition import BM25RepetitionIndex, take_complete_sentences
from brain.memory_service import MemoryService


def test_high_bm25_sentence_is_removed_and_novel_sentence_is_kept(tmp_path):
    index = BM25RepetitionIndex(tmp_path / "bm25_index.json", threshold=0.75)
    index.rebuild([
        (1, "Bạn có thể kể cho mình nghe điều khiến bạn hạnh phúc nhất hôm nay không?"),
        (2, "Mình đang ngồi cạnh cửa sổ và nghe tiếng mưa."),
    ])

    filtered, removed = index.filter_text(
        "Bạn có thể kể cho mình nghe điều khiến bạn hạnh phúc nhất hôm nay không? "
        "Một ngày bình thường cũng hoàn toàn ổn."
    )

    assert removed == 1
    assert "kể cho mình nghe" not in filtered
    assert filtered == "Một ngày bình thường cũng hoàn toàn ổn."


def test_index_is_persisted_and_loaded(tmp_path):
    path = tmp_path / "bm25_index.json"
    first = BM25RepetitionIndex(path)
    first.rebuild([(7, "Đây là một câu đủ dài để được lập chỉ mục.")])

    second = BM25RepetitionIndex(path)
    assert second.load() is True
    assert second.size == 1
    assert second.similarity("Đây là một câu đủ dài để được lập chỉ mục.") >= 0.78


def test_excluding_the_source_message_prevents_self_match(tmp_path):
    index = BM25RepetitionIndex(tmp_path / "bm25_index.json")
    index.rebuild([(9, "Đây là một câu duy nhất không hề lặp lại ở nơi khác.")])

    assert index.similarity(
        "Đây là một câu duy nhất không hề lặp lại ở nơi khác.",
        exclude_message_id=9,
    ) == 0.0


def test_stream_buffer_only_releases_complete_sentences():
    completed, tail = take_complete_sentences("Câu thứ nhất. Câu thứ")

    assert completed == ["Câu thứ nhất."]
    assert tail == " Câu thứ"


async def test_memory_service_creates_and_updates_persistent_bm25_file(tmp_path):
    memory = MemoryService(tmp_path / "brain.db")
    await memory.initialize()
    try:
        assert (tmp_path / "bm25_index.json").exists()
        await memory.save_message(
            "assistant",
            "Bạn có thể kể cho mình nghe điều khiến bạn vui nhất hôm nay không?",
        )
    finally:
        await memory.close()

    loaded = BM25RepetitionIndex(tmp_path / "bm25_index.json")
    assert loaded.load() is True
    assert loaded.size == 1
