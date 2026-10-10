"""Exception isolation around the unchanged official checker, with no new score gate."""
import math

import pdecheck


def grade_candidate(problem, answer):
    """Keep official scores intact; an exception has no invented numerical score."""
    try:
        result = pdecheck.check(problem, answer)
    except Exception as exc:
        return dict(reward=None, original_reward=None, parts={}, original_parts={},
                    expr=None, start_error=None, grading_policy='original_checker',
                    evaluation_error=dict(type=type(exc).__name__, message=str(exc)),
                    feedback='The original checker could not evaluate this answer '
                    f'({type(exc).__name__}). Return one scalar expression on the '
                    'u(x, t) = ... line, not a list or tuple of terms.')
    result.update(original_reward=result['reward'], original_parts=dict(result['parts']),
                  grading_policy='original_checker', evaluation_error=None)
    # Only normalize log representation; reward and acceptance are unchanged.
    if result['start_error'] is not None and not math.isfinite(result['start_error']):
        result['start_error'] = None
    return result
