"""Record generation prompts and adapt model-specific API request parameters."""
import json
import os
from pathlib import Path
import uuid

from openevolve.llm.openai import OpenAILLM


class AuditedOpenAILLM(OpenAILLM):
    async def _call_api(self, params):
        if str(self.model).lower().startswith("gpt-6"):
            params = params.copy()
            if "max_tokens" in params:
                params["max_completion_tokens"] = params.pop("max_tokens")
            # GPT-6 defaults to reasoning; sampling controls require effort=none.
            if params.get("reasoning_effort", "medium") != "none":
                params.pop("temperature", None)
                params.pop("top_p", None)
        return await super()._call_api(params)

    async def generate_with_context(self, system_message, messages, **kwargs):
        directory = Path(os.environ["NKI_SEARCH_OUTPUT"]) / "prompts"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"request_{uuid.uuid4().hex}.json"
        path.write_text(json.dumps({"model": self.model,
                                   "messages": [{"role": "system", "content": system_message}] + messages},
                                  indent=2))
        return await super().generate_with_context(system_message, messages, **kwargs)


def init_audited_llm(model_cfg):
    return AuditedOpenAILLM(model_cfg)
