"""Tests for autonomous operation mode."""

import pytest

from job_mcp.models.schemas import OperationMode, ToolResponse


class TestOperationMode:
    def test_enum_values(self):
        assert OperationMode.SUPERVISED == "supervised"
        assert OperationMode.AUTONOMOUS == "autonomous"

    def test_enum_from_string(self):
        assert OperationMode("supervised") == OperationMode.SUPERVISED
        assert OperationMode("autonomous") == OperationMode.AUTONOMOUS

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            OperationMode("invalid_mode")


class TestSetOperationMode:
    @pytest.mark.asyncio
    async def test_set_autonomous_mode(self):
        from job_mcp.main import set_operation_mode, _operation_mode
        result = await set_operation_mode(mode="autonomous")
        assert result["success"] is True
        assert "autonomous" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_set_supervised_mode(self):
        from job_mcp.main import set_operation_mode
        result = await set_operation_mode(mode="supervised")
        assert result["success"] is True
        assert "supervised" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_invalid_mode_returns_error(self):
        from job_mcp.main import set_operation_mode
        result = await set_operation_mode(mode="yolo")
        assert result["success"] is False
        assert result["error_code"] == "INVALID_MODE"

    @pytest.mark.asyncio
    async def test_get_current_mode_in_response(self):
        from job_mcp.main import set_operation_mode
        result = await set_operation_mode(mode="autonomous")
        assert result["data"]["mode"] == "autonomous"


class TestServerInstructionsAutonomy:
    def test_instructions_mention_autonomous(self):
        from job_mcp.main import SERVER_INSTRUCTIONS
        assert "autonomous" in SERVER_INSTRUCTIONS.lower()

    def test_instructions_mention_supervised(self):
        from job_mcp.main import SERVER_INSTRUCTIONS
        assert "supervised" in SERVER_INSTRUCTIONS.lower()

    def test_instructions_require_confirm_for_apply(self):
        from job_mcp.main import SERVER_INSTRUCTIONS
        assert "confirm_auto_apply" in SERVER_INSTRUCTIONS
