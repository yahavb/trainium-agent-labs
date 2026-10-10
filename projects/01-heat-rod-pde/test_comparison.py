import json
from pathlib import Path
import tempfile
import unittest
import sympy as sp
import compare_agents
import decay_repair
import level0_heatrod
import pdecheck


class ComparisonTests(unittest.TestCase):
    def summarize(self, row, problem):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'attempts.jsonl'
            path.write_text(json.dumps(row)+'\n')
            return compare_agents.summarize_attempts([path],problem)

    def test_original_full_score_is_the_reported_success(self):
        problem=level0_heatrod.make(1)
        answer=sp.sstr(problem['exact'])+f" + exp(-1000000*t)*sin(1600*pi*x/{problem['L']})"
        result=self.summarize(dict(answer=answer,round=0,trace=[]),problem)
        self.assertTrue(result['executed_original_solved'])
        self.assertNotIn('validated_solved', result)

    def test_repaired_output_is_distinct_from_model_success_and_tokens(self):
        problem=level0_heatrod.make(1)
        raw='u(x, t) = '+sp.sstr(problem['f'])
        fixed=decay_repair.repair(raw,problem['k'])['answer']
        row=dict(answer=raw,executed_answer=fixed,model_grade=pdecheck.check(problem,raw),round=0,
                 trace=[dict(type='model',usage=dict(completion_tokens=27),finish_reason='stop'),
                        dict(type='decay_repair',accepted=True)])
        result=self.summarize(row,problem)
        self.assertFalse(result['model_original_solved'])
        self.assertTrue(result['executed_original_solved'])
        self.assertEqual(result['model_requests'],1)
        self.assertEqual(result['completion_tokens'],27)
        self.assertEqual(result['accepted_repairs'],1)


if __name__=='__main__':
    unittest.main()
