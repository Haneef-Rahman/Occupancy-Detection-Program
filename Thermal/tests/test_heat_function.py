import unittest
from dataclasses import replace
import numpy as np
from heat_function import HeatConfig,HeatFunction

def row(t,mode='faster'):
 return dict(time=t,mode=mode,base_detections=[{},{}],observed=[{}],estimate=3)
class HeatTests(unittest.TestCase):
 def test_band_and_vertical_weight(self):
  f=HeatFunction(HeatConfig(warmup_s=0,area=1))
  r=row(0);f.update(np.array([[30,24],[39,30.]],float),r)
  self.assertEqual(r['heat_band_pixels'],2);self.assertEqual(r['raw_heat'],4)
  self.assertEqual(r['blend_detected'],5)
  self.assertEqual(r['heat'],4);self.assertIsInstance(r['heat'],int)
  self.assertEqual(r['detected'],6);self.assertEqual(r['detected'],r['heat']+len(r['base_detections']))
 def test_floor_is_not_round_and_outputs_do_not_feed_back(self):
  f=HeatFunction(HeatConfig(warmup_s=0,area=1000))
  r=row(0);f.update(np.zeros((2,2)),r)
  self.assertEqual(r['heat'],3) # floor(.75 * 5) = 3, not 4
  self.assertEqual(r['detected'],5)
  again=row(1);f.update(np.zeros((2,2)),again)
  self.assertEqual(again['heat'],3);self.assertEqual(again['detected'],5)
 def test_top_higher_than_bottom_and_raw_unchanged(self):
  c=HeatConfig(warmup_s=0,area=1);a=np.array([[30.,0],[0,0]]);b=np.flipud(a).copy();original=a.copy()
  self.assertGreater(HeatFunction(c).update(a,row(0))['raw_heat'],HeatFunction(c).update(b,row(0))['raw_heat'])
  np.testing.assert_array_equal(a,original)
 def test_warmup_precise_and_switching(self):
  f=HeatFunction(HeatConfig(warmup_s=2,switch_window_s=10,max_precise_entries=0));a=np.full((2,2),30.)
  self.assertFalse(f.update(a,row(0))['heat_active']);self.assertTrue(f.update(a,row(2))['heat_active'])
  self.assertIsNone(f.update(a,row(3,'precise'))['heat']);f.update(a,row(4))
  self.assertFalse(f.update(a,row(6))['heat_active']);self.assertTrue(f.update(a,row(14))['heat_active'])
 def test_reverse_clock_restarts_gate(self):
  f=HeatFunction();a=np.full((2,2),30.)
  f.update(a,row(0));self.assertTrue(f.update(a,row(6))['heat_active'])
  self.assertFalse(f.update(a,row(0))['heat_active'])
 def test_invalid_settings(self):
  for values in [dict(a=.5,b=.5),dict(a=.2,b=.9),dict(area=0),dict(low_c=40,high_c=20),dict(far_weight=.5),dict(power=float('nan'))]:
   with self.assertRaises(ValueError):HeatConfig(**values)
 def test_nan_does_not_count(self):
  r=HeatFunction(HeatConfig(warmup_s=0)).update(np.array([[np.nan,np.inf],[25.,38.]]),row(0))
  self.assertEqual(r['heat_band_pixels'],2)
if __name__=='__main__':unittest.main()
