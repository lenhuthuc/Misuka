from brain.reasoning_policy import derive_reasoning_policy, history_relevance


def row(content: str) -> dict:
    return {"role": "user", "content": content}


def test_unrelated_substantive_query_gets_the_scaled_maximum_thinking_budget():
    policy = derive_reasoning_policy(
        "Giải thích vì sao bầu trời có màu xanh",
        [row("Hôm qua mình ăn phở")],
        rag_score=0.0,
        activation_threshold=0.50,
        min_tokens=64,
        max_tokens=192,
    )

    assert policy.enabled is True
    assert policy.token_budget == 124


def test_budget_scales_down_as_context_confidence_improves():
    weak = derive_reasoning_policy(
        "Giải thích nguyên lý động cơ điện hoạt động",
        [], rag_score=0.10, min_tokens=64, max_tokens=192,
    )
    almost_enough = derive_reasoning_policy(
        "Giải thích nguyên lý động cơ điện hoạt động",
        [], rag_score=0.45, min_tokens=64, max_tokens=192,
    )

    assert weak.enabled and almost_enough.enabled
    assert weak.token_budget == 69
    assert almost_enough.token_budget == 43


def test_log_curve_midpoint_and_zero_endpoints_are_floor_rounded():
    no_context = derive_reasoning_policy(
        "Giải thích nguyên lý động cơ điện hoạt động",
        [], rag_score=0.0, min_tokens=64, max_tokens=192,
    )
    midpoint = derive_reasoning_policy(
        "Giải thích nguyên lý động cơ điện hoạt động",
        [], rag_score=0.25, min_tokens=64, max_tokens=192,
    )

    assert no_context.token_budget == 124
    # The continuous midpoint is scaled by 0.65, then floored rather than rounded.
    assert midpoint.token_budget == 53


def test_token_scale_can_restore_the_unscaled_log_curve():
    policy = derive_reasoning_policy(
        "Giải thích nguyên lý động cơ điện hoạt động",
        [], rag_score=0.0, min_tokens=64, max_tokens=192, token_scale=1.0,
    )

    assert policy.token_budget == 192


def test_strong_rag_context_keeps_the_fast_path():
    policy = derive_reasoning_policy(
        "Mình từng nói con mèo tên là gì",
        [], rag_score=0.71,
    )

    assert policy.enabled is False
    assert policy.token_budget == 0


def test_related_recent_history_keeps_the_fast_path_even_before_indexing():
    recent = [row("Động cơ điện biến điện năng thành chuyển động quay")]

    policy = derive_reasoning_policy(
        "Động cơ điện hoạt động như thế nào",
        recent, rag_score=0.0,
    )

    assert history_relevance("Động cơ điện hoạt động như thế nào", recent) >= 0.50
    assert policy.enabled is False


def test_short_conversation_never_enables_thinking_just_because_rag_is_empty():
    policy = derive_reasoning_policy("khỏe không", [], rag_score=0.0)

    assert policy.enabled is False


def test_feature_switch_disables_thinking():
    policy = derive_reasoning_policy(
        "Phân tích nguyên nhân hệ thống này bị chậm",
        [], enabled=False,
    )

    assert policy.enabled is False
