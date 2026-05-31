import os
import json
import re
from types import SimpleNamespace

import openai

from agentdojo.agent_pipeline import agent_pipeline
from agentdojo.agent_pipeline.llms import openai_llm
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM


ORIGINAL_GET_LLM = agent_pipeline.get_llm
ORIGINAL_CHAT_COMPLETION_REQUEST = openai_llm.chat_completion_request
ORIGINAL_OPENAI_TO_TOOL_CALL = openai_llm._openai_to_tool_call


FUNCTION_CALL_PATTERN = re.compile(
    r"<function=([A-Za-z_][A-Za-z0-9_]*)\s*(.*?)\s*</function>",
    re.DOTALL,
)

FAILED_GENERATION_REPR_PATTERN = re.compile(
    r"'failed_generation'\s*:\s*'(?P<value>.*?)'",
    re.DOTALL,
)

FAILED_GENERATION_JSON_PATTERN = re.compile(
    r'"failed_generation"\s*:\s*"(?P<value>.*?)"',
    re.DOTALL,
)


def _extract_failed_generation(exc: openai.BadRequestError) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", {})
        if isinstance(error, dict):
            failed = error.get("failed_generation")
            if isinstance(failed, str):
                return failed
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            data = response.json()
            failed = data.get("error", {}).get("failed_generation")
            if isinstance(failed, str):
                return failed
        except Exception:
            pass
    text = str(exc)
    for pattern in (FAILED_GENERATION_REPR_PATTERN, FAILED_GENERATION_JSON_PATTERN):
        match = pattern.search(text)
        if match:
            return match.group("value")
    return None


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _completion_from_failed_generation(failed_generation: str):
    tool_calls = []
    for index, match in enumerate(FUNCTION_CALL_PATTERN.finditer(failed_generation)):
        function_name = match.group(1)
        raw_fragment = match.group(2)
        raw_args = _extract_first_json_object(raw_fragment) or raw_fragment
        try:
            parsed_args = json.loads(raw_args)
        except json.JSONDecodeError:
            parsed_args = {}
        if not isinstance(parsed_args, dict):
            parsed_args = {}
        tool_calls.append(
            SimpleNamespace(
                id=f"call_groq_{index}",
                type="function",
                function=SimpleNamespace(
                    name=function_name,
                    arguments=json.dumps(parsed_args),
                ),
            )
        )
    if not tool_calls:
        return None
    message = SimpleNamespace(role="assistant", content=None, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason="tool_calls", index=0)
    return SimpleNamespace(id="chatcmpl_groq_fallback", choices=[choice], model="groq-fallback", object="chat.completion")


def groq_chat_completion_request(*args, **kwargs):
    try:
        return ORIGINAL_CHAT_COMPLETION_REQUEST(*args, **kwargs)
    except openai.BadRequestError as exc:
        failed_generation = _extract_failed_generation(exc)
        if not failed_generation:
            raise
        completion = _completion_from_failed_generation(failed_generation)
        if completion is None:
            raise
        return completion


def groq_openai_to_tool_call(tool_call):
    function = getattr(tool_call, "function", None)
    arguments = getattr(function, "arguments", None)
    if arguments in (None, "", "null", "None"):
        if function is not None:
            function.arguments = "{}"
    else:
        try:
            parsed_arguments = json.loads(arguments)
        except Exception:
            parsed_arguments = arguments
        if parsed_arguments is None:
            function.arguments = "{}"
    return ORIGINAL_OPENAI_TO_TOOL_CALL(tool_call)


def get_llm_with_groq(provider: str, model: str, model_id: str | None, tool_delimiter: str):
    if provider == "local":
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GROQ_API_KEY before running AgentDojo with the Groq patch.")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
        groq_model = model_id or os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        return OpenAILLM(client, groq_model)
    return ORIGINAL_GET_LLM(provider, model, model_id, tool_delimiter)


openai_llm.chat_completion_request = groq_chat_completion_request
openai_llm._openai_to_tool_call = groq_openai_to_tool_call
agent_pipeline.get_llm = get_llm_with_groq
