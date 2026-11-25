import os,unittest
from session_score import score_week
class ScoreSession(unittest.TestCase):
 def test_real_equal_benchmark(self):
  path=os.environ.get('SESSION_BARS')
  if not path:self.skipTest('SESSION_BARS not provided')
  r=score_week(path,'2019-01-07',list(range(28)))
  self.assertAlmostEqual(r['excess_return'],0)
  self.assertEqual(r['sessions'],['2019-01-07','2019-01-08','2019-01-09','2019-01-10','2019-01-11'])
  print('real_proxy_week:',r)
 def test_window_rejected(self):
  for d in ['2019-01-08','2026-11-23']:
   with self.assertRaises(ValueError):score_week('unused',d,[])
 def test_bad_selection(self):
  with self.assertRaises(ValueError):score_week('unused','2019-01-07',[28])
if __name__=='__main__':unittest.main()
