"""
Tests for robustness, cross-platform fallbacks, and input validations.
Run:
    python -m pytest tests/test_robustness.py -v
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException

# Test targets
from nanochat.engine import timeout, eval_with_timeout
from scripts.chat_web import validate_chat_request, ChatRequest, ChatMessage
from nanochat.execution import time_limit, reliability_guard
from nanochat.report import get_system_info

def test_cross_platform_timeout_success():
    """Test that the timeout context manager works when the block finishes quickly."""
    with timeout(1.0, "quick_formula"):
        x = 1 + 1
    assert x == 2

def test_cross_platform_timeout_failure():
    """Test that timeout actually interrupts running code that blocks too long."""
    import time
    with pytest.raises((TimeoutError, KeyboardInterrupt)):
        with timeout(0.1, "hanging_formula"):
            # Sleep longer than the timeout to trigger it
            time.sleep(0.5)

def test_eval_with_timeout_safe():
    """Test that eval_with_timeout executes valid formulas and recovers from errors."""
    assert eval_with_timeout("2 + 2") == 4
    assert eval_with_timeout("invalid syntax") is None
    # Division by zero
    assert eval_with_timeout("1 / 0") is None

def test_report_username_resolution():
    """Test that report resolves USERNAME environment variable on Windows/fallback."""
    with patch.dict(os.environ, {"USER": "test_unix_user"}):
        sys_info = get_system_info()
        assert sys_info["user"] == "test_unix_user"

    with patch.dict(os.environ, {}, clear=True):
        with patch.dict(os.environ, {"USERNAME": "test_windows_user"}):
            sys_info = get_system_info()
            assert sys_info["user"] == "test_windows_user"

def test_validate_chat_request_valid():
    """Test that valid chat requests pass validation without throwing exceptions."""
    req = ChatRequest(
        messages=[
            ChatMessage(role="user", content="Hello assistant"),
            ChatMessage(role="assistant", content="Hello, how can I help you?")
        ],
        temperature=0.7,
        top_k=40,
        max_tokens=256
    )
    # Should not raise any exception
    validate_chat_request(req)

def test_validate_chat_request_invalid_messages():
    """Test validation errors for invalid or empty message schemas."""
    # Empty messages list
    req_empty = ChatRequest(messages=[])
    with pytest.raises(HTTPException) as exc_info:
        validate_chat_request(req_empty)
    assert exc_info.value.status_code == 400
    assert "At least one message" in exc_info.value.detail

    # Empty content
    req_empty_content = ChatRequest(messages=[ChatMessage(role="user", content="   ")])
    with pytest.raises(HTTPException) as exc_info:
        validate_chat_request(req_empty_content)
    assert exc_info.value.status_code == 400
    assert "empty or whitespace-only" in exc_info.value.detail

def test_validate_chat_request_invalid_role():
    """Test validation of message roles."""
    req_bad_role = ChatRequest(messages=[ChatMessage(role="system", content="Act as a helper")])
    with pytest.raises(HTTPException) as exc_info:
        validate_chat_request(req_bad_role)
    assert exc_info.value.status_code == 400
    assert "invalid role" in exc_info.value.detail
    assert "Must be 'user' or 'assistant'" in exc_info.value.detail

def test_validate_chat_request_out_of_bounds_parameters():
    """Test parameter bounds enforcement."""
    # Temperature too high
    req_temp = ChatRequest(
        messages=[ChatMessage(role="user", content="Hi")],
        temperature=3.5
    )
    with pytest.raises(HTTPException) as exc_info:
        validate_chat_request(req_temp)
    assert exc_info.value.status_code == 400
    assert "Temperature must be between" in exc_info.value.detail

    # Top-K out of range
    req_topk = ChatRequest(
        messages=[ChatMessage(role="user", content="Hi")],
        top_k=-5
    )
    with pytest.raises(HTTPException) as exc_info:
        validate_chat_request(req_topk)
    assert exc_info.value.status_code == 400
    assert "top_k must be between" in exc_info.value.detail

    # Max tokens too high
    req_tokens = ChatRequest(
        messages=[ChatMessage(role="user", content="Hi")],
        max_tokens=99999
    )
    with pytest.raises(HTTPException) as exc_info:
        validate_chat_request(req_tokens)
    assert exc_info.value.status_code == 400
    assert "max_tokens must be between" in exc_info.value.detail

def test_time_limit_cross_platform_safety():
    """Verify that time_limit executes safely and catches exceptions."""
    with time_limit(2.0):
        val = 123
    assert val == 123

def _run_guard():
    import os
    reliability_guard(maximum_memory_bytes=100000)
    assert os.chdir is None

def test_reliability_guard_cross_platform_safety():
    """Verify that reliability_guard runs without crashing due to missing resource module on Windows."""
    import multiprocessing
    p = multiprocessing.Process(target=_run_guard)
    p.start()
    p.join()
    assert p.exitcode == 0


