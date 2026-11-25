import json, unittest
from features import transform
class Features(unittest.TestCase):
 def test_causal_and_missing(self):
  d={'schema':'weekly-series-v1','columns':['weekly_return','mean_feature'],'series':[[[0.1,0.2],[None,0.3],[0.5,0.4]] for _ in range(28)]}
  a=transform(json.dumps(d)); self.assertEqual(a['series'][0][0],[0.1,0.2,0.1,0.0]); self.assertIsNone(a['series'][0][1][2])
  for s in d['series']:s[2][0]=100
  b=transform(json.dumps(d)); self.assertEqual(a['series'][0][:2],b['series'][0][:2])
 def test_shape(self):
  with self.assertRaises(ValueError):transform('{}')
if __name__=='__main__':unittest.main()
