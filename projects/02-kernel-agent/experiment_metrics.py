"""Additive experimental telemetry; reported usage never inferred from text."""
from contextvars import ContextVar

CASE_RESULTS = ContextVar('kernel_case_results', default=None)
GRADE_DIRECTORY = ContextVar('kernel_grade_directory', default=None)


class ModelReply(str):
    def __new__(cls, text, metadata):
        result = super().__new__(cls, text)
        result.metadata = metadata
        return result


def enabled(options):
    return (getattr(options, 'instrument', False)
            or getattr(options, 'candidate_policy', 'standard') != 'standard'
            or getattr(options, 'repair_policy', 'standard') != 'standard'
            or getattr(options, 'generation_policy', 'standard') != 'standard'
            or getattr(options, 'feedback_policy', 'legacy') != 'legacy'
            or getattr(options, 'example_policy', 'off') != 'off'
            or getattr(options, 'adaptive_repair', False))


def response_metadata(payload, elapsed=None):
    usage = payload.get('usage') or {}
    def number(key):
        value = usage.get(key)
        return value if type(value) is int and value >= 0 else None
    finish = payload.get('choices', [{}])[0].get('finish_reason')
    return dict(prompt_tokens=number('prompt_tokens'), completion_tokens=number('completion_tokens'),
                total_tokens=number('total_tokens'), usage_source='endpoint' if usage else 'unavailable',
                finish_reason=finish, truncated=(finish == 'length') if finish is not None else None,
                generation_seconds=elapsed)


def record_case(case, **result):
    target = CASE_RESULTS.get()
    if target is not None:
        target.append(dict(case=case, evaluated=True, **result))
