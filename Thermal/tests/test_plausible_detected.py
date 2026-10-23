import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from heat_function import HeatConfig,HeatFunction,PlausibleDetected
from occupancy_chart import OccupancyHistory

def row(t,mode='faster'):
    return dict(time=t,frame=int(t*10)+1,mode=mode,base_detections=[{},{}],observed=[{}],estimate=3)

class PlausibleTests(unittest.TestCase):
    def test_hold_detection_peak_when_heat_is_unchanged(self):
        p=PlausibleDetected()
        self.assertEqual([p.update(d,5.) for d in [8,5,6,10,4]],[8,8,8,10,10])

    def test_signed_whole_units_and_fractional_accumulation(self):
        p=PlausibleDetected()
        self.assertEqual(p.update(8,5.),8)
        for heat in [5.2,5.4,5.6,5.8]:self.assertEqual(p.update(4,heat),8)
        self.assertEqual(p.update(4,6.),9)
        self.assertEqual(p.update(4,5.4),9)
        self.assertEqual(p.update(4,5.),8)
        self.assertEqual(p.update(4,2.),5)

    def test_fractional_noise_does_not_accumulate_path_length(self):
        p=PlausibleDetected();p.update(8,5.)
        for heat in [5.6,4.4]*20:self.assertEqual(p.update(4,heat),8)

    def test_current_detected_is_always_a_floor_and_blocked_drop_is_consumed(self):
        p=PlausibleDetected();p.update(10,5.)
        self.assertEqual(p.update(10,4.),10)
        self.assertEqual(p.update(9,4.),10) # no delayed drop without a new heat change
        self.assertEqual(p.update(9,3.),9)

    def test_detection_peak_reanchors_partial_heat_change(self):
        p=PlausibleDetected();p.update(8,5.)
        self.assertEqual(p.update(10,5.8),10)
        self.assertEqual(p.update(8,6.),10)
        self.assertEqual(p.update(8,6.8),11)

    def test_roundoff_at_exact_unit_does_not_delay_step(self):
        p=PlausibleDetected();p.update(8,5.1)
        self.assertEqual(p.update(4,6.1),9)
        self.assertEqual(p.update(4,5.1),8)

    def test_nonnegative_integer_and_reset(self):
        p=PlausibleDetected();p.update(2,5.)
        self.assertEqual(p.update(0,1.),0)
        self.assertIsInstance(p.value,int)
        p.reset();self.assertEqual(p.update(3,1.),3)

    def test_heat_gate_reset_and_history_propagation(self):
        f=HeatFunction(HeatConfig(warmup_s=0));data=np.full((2,2),30.)
        r=row(0);f.update(data,r)
        self.assertEqual(r['plausible_detected'],r['detected'])
        f.plausible.value=30
        precise=row(1,'precise');f.update(data,precise)
        self.assertIsNone(precise['plausible_detected']);self.assertIsNone(f.plausible.value)
        r=row(2);f.update(data,r);self.assertEqual(r['plausible_detected'],r['detected'])
        h=OccupancyHistory();h.append(r)
        self.assertEqual(h.samples[-1]['plausible_detected'],r['plausible_detected'])
        self.assertEqual(h.overview[-1]['plausible_detected'],r['plausible_detected'])
        self.assertEqual(r['estimate'],3);self.assertEqual(len(r['observed']),1)

    def test_time_restart_and_new_settings_reset_state(self):
        data=np.full((2,2),30.);f=HeatFunction(HeatConfig(warmup_s=0))
        f.update(data,row(2));f.plausible.value=40
        r=row(0);f.update(data,r);self.assertEqual(r['plausible_detected'],r['detected'])
        new=HeatFunction(HeatConfig(warmup_s=0,a=.2,b=.8))
        r=row(3);new.update(data,r);self.assertEqual(r['plausible_detected'],r['detected'])

if __name__=='__main__':unittest.main()
