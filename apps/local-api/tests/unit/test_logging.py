import json
import logging

import pytest

from core.logging import (
    _JsonFormatter,
    bind_request_id,
    bind_turn_id,
    get_request_id,
    get_turn_id,
    log_duration,
)


def test_bind_request_id_scopes_and_resets():
    assert get_request_id() is None
    with bind_request_id("req-1"):
        assert get_request_id() == "req-1"
    assert get_request_id() is None


def test_bind_turn_id_scopes_and_resets():
    assert get_turn_id() is None
    with bind_turn_id("turn-1"):
        assert get_turn_id() == "turn-1"
    assert get_turn_id() is None


def test_nested_request_and_turn_ids_both_visible():
    with bind_request_id("req-1"), bind_turn_id("turn-1"):
        assert get_request_id() == "req-1"
        assert get_turn_id() == "turn-1"


def _make_record(**extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello %s", args=("world",), exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_includes_core_fields():
    formatter = _JsonFormatter(service="mitsuka-api", environment="test")
    record = _make_record(request_id="req-1", turn_id="turn-1", operation="chat", duration_ms=12.5, outcome="success")

    payload = json.loads(formatter.format(record))

    assert payload["service"] == "mitsuka-api"
    assert payload["environment"] == "test"
    assert payload["level"] == "INFO"
    assert payload["message"] == "hello world"
    assert payload["request_id"] == "req-1"
    assert payload["turn_id"] == "turn-1"
    assert payload["operation"] == "chat"
    assert payload["duration_ms"] == 12.5
    assert payload["outcome"] == "success"


def test_json_formatter_omits_absent_correlation_ids():
    formatter = _JsonFormatter(service="mitsuka-api", environment="test")
    record = _make_record(request_id=None, turn_id=None)

    payload = json.loads(formatter.format(record))

    assert "request_id" not in payload
    assert "turn_id" not in payload


async def test_log_duration_success_logs_duration_and_outcome(caplog):
    logger = logging.getLogger("test.log_duration.success")
    with caplog.at_level(logging.INFO, logger=logger.name):
        with log_duration(logger, "unit_test_op"):
            pass

    record = caplog.records[-1]
    assert record.operation == "unit_test_op"
    assert record.outcome == "success"
    assert isinstance(record.duration_ms, float)


async def test_log_duration_failure_logs_exception_and_reraises(caplog):
    logger = logging.getLogger("test.log_duration.failure")
    with caplog.at_level(logging.INFO, logger=logger.name):
        with pytest.raises(RuntimeError):
            with log_duration(logger, "unit_test_op"):
                raise RuntimeError("boom")

    record = caplog.records[-1]
    assert record.operation == "unit_test_op"
    assert record.outcome == "error"
    assert record.error_type == "RuntimeError"
    assert record.exc_info is not None  # logger.exception attached a traceback
