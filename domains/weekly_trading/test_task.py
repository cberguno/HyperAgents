import json, tempfile, unittest
from pathlib import Path
from task import validate_package, copy_package, score
class Contract(unittest.TestCase):
    def setUp(self): self.p={'schema':'weekly-series-v1','columns':['weekly_return','mean_feature'],'series':[[[0.01, 0.2]] for _ in range(50)]}
    def test_copy(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'in').write_text(json.dumps(self.p)); copy_package(p/'in',p/'out'); self.assertEqual(json.loads((p/'out').read_text()),self.p)
            with self.assertRaises(FileExistsError): copy_package(p/'in',p/'out')
    def test_schema_and_selection(self):
        self.p['future'] = 5
        with self.assertRaises(ValueError): validate_package(self.p)
        for bad in ([0,0], [50], [True]):
            with self.assertRaises(ValueError): score(bad, [[0.01]*50])
    def test_equal_and_cash(self):
        a=score(list(range(50)),[[0.01]*50]); self.assertAlmostEqual(a['excess_return'],0)
        b=score([],[[0.01]*50]); self.assertEqual(b['strategy_return'],0); self.assertAlmostEqual(b['benchmark_return'],0.008995)
if __name__ == '__main__': unittest.main()
