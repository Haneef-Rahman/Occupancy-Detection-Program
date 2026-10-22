from pathlib import Path
import sys
import unittest
import tempfile
from types import SimpleNamespace
import numpy as np
import cv2
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from occupancy_chart import OccupancyHistory
from launcher_controls import LauncherControls,DatasetCapture
from integrated_launcher_v2 import TrackingSession


def row(frame,t,estimate=4,observed=2,mode='precise'):
    return dict(frame=frame,time=t,estimate=estimate,observed=[{}]*observed,
                unresolved=estimate-observed,processing_ms=25.,mode=mode,origin='synthetic-test')

class HistoryTests(unittest.TestCase):
    def test_bounded_and_original_timestamps(self):
        h=OccupancyHistory()
        for i in range(135):h.append(row(i+1,100+i*.17))
        self.assertEqual(len(h.samples),120)
        self.assertEqual(h.samples[0]['frame'],16)
        self.assertAlmostEqual(h.samples[-1]['t'],134*.17)
        self.assertEqual(h.samples[-1]['estimate'],4)
        self.assertEqual(h.samples[-1]['observed'],2)

    def test_duplicate_pause_and_restart(self):
        h=OccupancyHistory();r=row(1,5);h.append(r)
        for _ in range(10):h.append(r)
        self.assertEqual(len(h.samples),1)
        h.append(row(2,5.3));h.append(row(1,0))
        self.assertEqual(len(h.samples),1);self.assertEqual(h.samples[0]['t'],0)
        with self.assertRaises(ValueError):h.append(row(2,float('nan')))

    def test_render_hover_and_monochrome(self):
        with tempfile.TemporaryDirectory() as temp:
            session=TrackingSession(SimpleNamespace(conf=.374,low_conf=.18))
            ui=LauncherControls(session,DatasetCapture(temp,'test.pt'),live=False)
            ui.boxes=False;ui.raw=False;ui.show_search=False
            frames=[row(1,50),row(2,51,5,3),row(3,60,3,3,'faster')]
            for r in frames:ui.observe(r)
            data=np.full((120,160),25,np.float32)
            image=ui.render(data,frames[-1],5)
            for _ in range(4):ui.render(data,frames[-1],5)
            self.assertEqual(len(ui.history.samples),3)
            x0,y0,x1,y1=ui.chart_bounds
            # 1 second into a 10-second span must select frame 2 at 10% width.
            ui.mouse(cv2.EVENT_MOUSEMOVE,round(x0+(x1-x0)*.1),(y0+y1)//2,0,None)
            image=ui.render(data,frames[-1],5)
            self.assertEqual(ui.chart_hover['frame'],2)
            self.assertEqual(ui.chart_hover['estimate'],5)
            self.assertEqual(ui.chart_hover['observed'],3)
            crop=image[92:284,24:856]
            np.testing.assert_array_equal(crop[:,:,0],crop[:,:,1])
            np.testing.assert_array_equal(crop[:,:,1],crop[:,:,2])
            ui.command(ord('o'));ui.render(data,frames[-1],5)
            self.assertIsNone(ui.chart_bounds);self.assertEqual(len(ui.history.samples),3)
            ui.command(ord('x'));self.assertEqual(len(ui.history.samples),0)
            ui.command(ord('o'));ui.render(data,frames[-1],5) # empty chart stays valid
            ui.observe(frames[0]);ui.render(data,frames[0],5) # single sample stays valid

if __name__=='__main__':unittest.main()
