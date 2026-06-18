from __future__ import annotations

import json
import logging
import time
from typing import Any, Sequence

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

logger = logging.getLogger(__name__)


class GLMAsyncChatModel(BaseChatModel):
    """LangChain ChatModel wrapper for BigModel GLM async chat completions."""

    api_key: str
    model: str = "glm-5.2"
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    async_submit_path: str = "/async/chat/completions"
    async_result_path: str = "/async-result/{id}"
    poll_interval_s: float = 1.0
    max_poll_s: float = 30.0
    temperature: float = 1.0
    request_timeout_s: float = 30.0

    @property
    def _llm_type(self) -> str:
        return "glm_async_chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "async_submit_path": self.async_submit_path,
            "async_result_path": self.async_result_path,
        }

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | BaseTool | Any],
        *,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        bind_kwargs: dict[str, Any] = {"tools": formatted_tools, **kwargs}
        if tool_choice is not None:
            bind_kwargs["tool_choice"] = tool_choice
        return self.bind(**bind_kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._message_to_dict(message) for message in messages],
            "temperature": kwargs.pop("temperature", self.temperature),
        }
        if stop:
            payload["stop"] = stop
        for key in ("tools", "tool_choice", "max_tokens", "top_p", "thinking"):
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]

        raw = self._submit_and_poll(payload)
        message_payload = self._extract_message_payload(raw)
        ai_message = self._ai_message_from_payload(message_payload)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def _submit_and_poll(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.request_timeout_s) as client:
            submit_resp = client.post(self._url(self.async_submit_path), json=payload, headers=headers)
            submit_resp.raise_for_status()
            submit_data = submit_resp.json()

            task_id = self._extract_task_id(submit_data)
            deadline = time.monotonic() + self.max_poll_s
            last_payload: dict[str, Any] = submit_data

            while time.monotonic() < deadline:
                result_resp = client.get(
                    self._url(self.async_result_path.format(id=task_id)),
                    headers=headers,
                )
                result_resp.raise_for_status()
                result_data = result_resp.json()
                last_payload = result_data

                status = self._extract_status(result_data)
                if status in {"SUCCESS", "SUCCEEDED", "DONE"}:
                    return result_data
                if status in {"FAIL", "FAILED", "ERROR"}:
                    raise RuntimeError(f"GLM async task failed: {result_data}")

                time.sleep(self.poll_interval_s)

            raise TimeoutError(
                f"GLM async task timed out after {self.max_poll_s}s: task_id={task_id}, last={last_payload}"
            )

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _extract_task_id(self, data: dict[str, Any]) -> str:
        task_id = data.get("id") or data.get("task_id") or data.get("request_id")
        if not task_id and isinstance(data.get("data"), dict):
            inner = data["data"]
            task_id = inner.get("id") or inner.get("task_id") or inner.get("request_id")
        if not task_id:
            raise RuntimeError(f"GLM async submit response missing task id: {data}")
        return str(task_id)

    def _extract_status(self, data: dict[str, Any]) -> str:
        status = (
            data.get("task_status")
            or data.get("status")
            or data.get("state")
            or data.get("taskStatus")
            or ""
        )
        return str(status).strip().upper()

    def _extract_message_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        candidates: list[Any] = [data]
        for key in ("result", "data", "output"):
            if isinstance(data.get(key), dict):
                candidates.append(data[key])

        for candidate in candidates:
            choices = candidate.get("choices") if isinstance(candidate, dict) else None
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    message = first.get("message")
                    if isinstance(message, dict):
                        return message
                    if isinstance(first.get("delta"), dict):
                        return first["delta"]
        raise RuntimeError(f"GLM async result missing assistant message: {data}")

    def _ai_message_from_payload(self, payload: dict[str, Any]) -> AIMessage:
        content = payload.get("content") or ""
        raw_tool_calls = payload.get("tool_calls") or []
        tool_calls: list[dict[str, Any]] = []
        if isinstance(raw_tool_calls, list):
            for raw_call in raw_tool_calls:
                call = self._normalize_tool_call(raw_call)
                if call:
                    tool_calls.append(call)
        return AIMessage(
            content=content,
            tool_calls=tool_calls,
            additional_kwargs={"tool_calls": raw_tool_calls} if raw_tool_calls else {},
        )

    def _normalize_tool_call(self, raw_call: Any) -> dict[str, Any] | None:
        if not isinstance(raw_call, dict):
            return None
        function = raw_call.get("function") or {}
        if not isinstance(function, dict):
            return None
        name = str(function.get("name") or "").strip()
        if not name:
            return None
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                logger.warning("[GLM_TOOL_ARGS_PARSE_FAILED] name=%s args=%r", name, arguments)
                arguments = {"_raw_arguments": arguments}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        return {
            "name": name,
            "args": arguments,
            "id": str(raw_call.get("id") or ""),
        }

    def _message_to_dict(self, message: BaseMessage) -> dict[str, Any]:
        content = message.content
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": content}
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": content}
        if isinstance(message, ToolMessage):
            return {
                "role": "tool",
                "content": content,
                "tool_call_id": message.tool_call_id,
            }
        if isinstance(message, FunctionMessage):
            return {"role": "function", "name": message.name, "content": content}
        if isinstance(message, AIMessage):
            payload: dict[str, Any] = {"role": "assistant", "content": content}
            raw_tool_calls = message.additional_kwargs.get("tool_calls")
            if raw_tool_calls:
                payload["tool_calls"] = raw_tool_calls
            return payload
        return {"role": message.type, "content": content}
