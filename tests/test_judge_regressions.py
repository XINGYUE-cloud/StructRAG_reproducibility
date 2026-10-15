import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src import evaluate_loong_llm_judge as judge

class JudgeRegressionTests(unittest.TestCase):
    def test_runner_answer_is_in_prompt(self):
        prompt = judge.build_prompt({'question':'Q', 'gold_answer':'GOLD', 'answer':'ACTUAL'})
        self.assertIn('ACTUAL', prompt[-1]['content'])
        self.assertIn('GOLD', prompt[-1]['content'])

    def test_official_rating_format(self):
        self.assertEqual(judge.parse_judge_response('Evaluation evidence: Correct.\nRating: [[100]]')['score'],100)
        with self.assertRaises(ValueError):
            judge.parse_judge_response('Rating: [[999]]')

    def test_dry_run_cannot_supply_live_scores(self):
        rows=[{'id':'x','question':'Q','gold_answer':'G','answer':'P','judge_response':{'score':100,'reason':'fixture'}}]
        with tempfile.TemporaryDirectory() as d, patch.object(judge.request,'urlopen',side_effect=AssertionError('network')):
            judge.evaluate(rows,Path(d),model='gpt-5.6terra',dry_run=True)
            with self.assertRaises(ValueError):
                judge.evaluate(rows,Path(d),model='gpt-5.6terra',dry_run=False)

    def test_changed_answer_changes_cache_identity(self):
        a={'id':'x','question':'Q','gold_answer':'G','answer':'A'}
        b={**a,'answer':'B'}
        self.assertNotEqual(judge.cache_key(a,'model','https://example.test/v1',False),judge.cache_key(b,'model','https://example.test/v1',False))

    def test_configuration_preserves_model_and_never_executes_content(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'.env.judge.local'
            p.write_text('OPENAI_MODEL=gpt-5.6terra\nOPENAI_API_KEY=\nOPENAI_BASE_URL=\n',encoding='utf-8')
            config=judge.load_config(p,{})
            self.assertEqual(config['OPENAI_MODEL'],'gpt-5.6terra')
            self.assertEqual(config['OPENAI_API_KEY'],'')

if __name__=='__main__': unittest.main()
