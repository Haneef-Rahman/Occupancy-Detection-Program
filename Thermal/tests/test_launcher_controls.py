import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import cv2
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from integrated_launcher_v2 import TrackingSession
from launcher_controls import DatasetCapture,LauncherControls
from capture_replay import CaptureSequence


def det(x=60,conf=.8):
    return dict(bbox=(x-3,41,6,8),centroid=(x,45),conf=conf,source='base',body_h=8/.3)
class Detector:
    conf=.374;low_conf=.18
    def base(self,image):return [det()]
    def recover(self,image,regions):return []

class ControlsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.session=TrackingSession(Detector())
        self.rec=DatasetCapture(self.root,'test.pt','test note')
        self.ui=LauncherControls(self.session,self.rec,live=False)
        self.data=np.full((120,160),30,np.float32)
    def tearDown(self):self.rec.close();self.tmp.cleanup()
    def frame(self,t=0):return self.session.process(self.data,t,'test-recording')

    def test_confidence_updates_roi_filter_and_preserves_tracks(self):
        self.session.set_mode('faster')
        self.frame();self.frame(.1)
        old=self.session.router.fast.tracks[0]
        self.ui.command(ord("'"))
        self.assertAlmostEqual(self.session.detector.low_conf,.19)
        self.assertAlmostEqual(self.session.router.fast.config.low_conf,.19)
        self.assertIs(self.session.router.fast.tracks[0],old)
        self.assertIs(old.config,self.session.config)
        for _ in range(200):self.ui.command(ord(';'))
        self.assertEqual(self.session.detector.low_conf,.01)
        for _ in range(200):self.ui.command(ord('['))
        self.assertGreater(self.session.detector.conf,self.session.detector.low_conf)
        for _ in range(200):self.ui.command(ord("'"))
        self.assertLess(self.session.detector.low_conf,self.session.detector.conf)
        with self.assertRaises(ValueError):self.session.set_confidence(.1,.2)

    def test_policy_handover_keeps_observed_identity(self):
        self.frame();a=self.frame(.1);identity=a['observed'][0]['id']
        self.ui.command(ord('3'));b=self.frame(.2)
        self.assertEqual(b['mode'],'faster');self.assertEqual(b['observed'][0]['id'],identity)
        self.ui.command(ord('2'));c=self.frame(.3)
        self.assertEqual(c['mode'],'precise');self.assertEqual(c['observed'][0]['id'],identity)
        self.ui.command(ord('1'));self.assertEqual(self.session.mode,'auto')
        self.ui.command(ord('t'));self.assertEqual(self.session.mode,'precise')

    def test_recording_layout_base_only_and_pipeline_discovery(self):
        row=self.frame();row['recovery_detections']=[det(100,.2)]
        row['observed']=[dict(id=99,bbox=[90,40,10,10],conf=.2,source='recovery')]
        self.ui.command(ord('r'),self.data,row)
        cap=self.rec.continuous
        self.assertTrue(self.rec.write(cap,self.data,row))
        self.ui.command(ord('s'),self.data,row)
        self.assertEqual(cap['n'],1,'S must not duplicate an already recorded frame')
        # A changed threshold while paused must not alter labels/provenance of this row.
        self.ui.command(ord(']'));row2=self.frame(.1);self.rec.write(cap,self.data,row2)
        self.ui.command(ord('r'),self.data,row2)
        path=cap['path']
        self.assertIsNone(self.rec.continuous)
        self.assertEqual((path/'classes.txt').read_text(),'person\nhead_shoulder\n')
        labels=(path/'labels/cap_000000.txt').read_text().splitlines()
        self.assertEqual(len(labels),1);self.assertEqual(labels[0].split()[0],'1')
        self.assertAlmostEqual(float(labels[0].split()[1]),60/160)
        png=cv2.imread(str(path/'png/cap_000000.png'),cv2.IMREAD_UNCHANGED)
        self.assertEqual(png.dtype,np.uint16);self.assertEqual(png.shape,(120,160))
        self.assertEqual(int(png[0,0]),32767)
        self.assertTrue((path/'review/cap_000000.png').is_file())
        np.testing.assert_array_equal(np.load(path/'npy/cap_000000.npy'),self.data)
        with (path/'manifest.csv').open() as fh:entries=list(csv.DictReader(fh))
        self.assertEqual(entries[0]['omega_count'],'1')
        details=[json.loads(line) for line in (path/'capture_frames.jsonl').read_text().splitlines()]
        self.assertEqual(details[0]['normal_confidence'],.374)
        self.assertAlmostEqual(details[1]['normal_confidence'],.384)
        np.testing.assert_allclose(CaptureSequence(path).times,[0,.1])
        import dataset_pipeline as pipeline
        from unittest.mock import patch
        with patch.object(pipeline,'LOG_DIR',str(self.root)):
            discovered=pipeline.list_captures()
        self.assertEqual(len(discovered),1);self.assertTrue(discovered[0]['usable']);self.assertTrue(discovered[0]['omega_only'])

    def test_shots_continuous_and_restart_never_overwrite(self):
        row=self.frame()
        self.ui.command(ord('s'),self.data,row);self.ui.command(ord('s'),self.data,row)
        self.assertEqual(self.rec.total,1)
        for i in range(2):
            self.ui.command(ord('r'),self.data,row)
            self.rec.write(self.rec.continuous,self.data,row)
            self.ui.command(ord('r'),self.data,row)
        self.assertEqual(len(set(self.rec.folders)),3)
        self.assertEqual(self.rec.total,3)
        for path in self.rec.folders:self.assertTrue((path/'npy/cap_000000.npy').exists())

    def test_native_dashboard_mouse_and_display_only_toggle(self):
        row=self.frame()
        before=self.session.router
        image=self.ui.render(self.data,row,5)
        self.assertEqual(image.shape,(840,1240,3))
        for (x0,y0,x1,y1),key in self.ui.buttons:
            self.assertTrue(0<=x0<x1<=1240 and 0<=y0<y1<=840)
            if key=='c':
                self.ui.mouse(cv2.EVENT_LBUTTONUP,x0+2,y0+2,0,None)
        self.ui.command(self.ui.actions.popleft())
        self.assertTrue(self.ui.show_search);self.assertIs(self.session.router,before)
        self.ui.command(ord(' '));self.assertTrue(self.ui.paused)
        self.ui.command(ord('n'));self.assertTrue(self.ui.step)
        self.ui.command(ord('x'));self.assertIsNot(self.session.router,before)

class ThemeInteractionTests(unittest.TestCase):
    setUp=ControlsTests.setUp
    tearDown=ControlsTests.tearDown
    frame=ControlsTests.frame
    # Dedicated regressions for the two draggable controls.
    def test_slider_preview_commit_bounds_and_keyboard(self):
        row=self.frame();self.ui.render(self.data,row,5)
        original=self.session.detector.conf
        x0,x1,y=self.ui.sliders['conf']['rail']
        self.ui.mouse(cv2.EVENT_LBUTTONDOWN,x0+int((x1-x0)*.7),y,0,None)
        self.assertEqual(self.ui.dragging,'conf')
        self.ui.render(self.data,row,5)
        self.assertEqual(self.session.detector.conf,original,'Drag preview must not change inference yet')
        self.ui.mouse(cv2.EVENT_MOUSEMOVE,x1+200,y,0,None)
        self.ui.mouse(cv2.EVENT_LBUTTONUP,x1+200,y,0,None)
        self.assertIsNone(self.ui.dragging)
        self.assertEqual(len(self.ui.actions),1)
        self.ui.command(self.ui.actions.popleft(),self.data,row)
        self.assertEqual(self.session.detector.conf,.99)
        self.ui.render(self.data,row,5)
        x0,x1,y=self.ui.sliders['low_conf']['rail']
        self.ui.mouse(cv2.EVENT_LBUTTONDOWN,x1,y,0,None)
        self.ui.mouse(cv2.EVENT_LBUTTONUP,x1,y,0,None)
        self.ui.command(self.ui.actions.popleft(),self.data,row)
        self.assertEqual(self.session.detector.low_conf,.98)
        self.assertEqual(self.session.router.fast.config.low_conf,.98)
        self.ui.command(ord(';'))
        self.assertAlmostEqual(self.session.detector.low_conf,.97)
        self.assertEqual(self.ui.bubble_name,'low_conf')
        self.assertEqual(row['normal_confidence'],original,'Displayed row keeps its inference threshold')

    def test_all_buttons_and_sliders_stay_inside_supported_sizes(self):
        row=self.frame()
        for scale in (1,3,5,8):
            image=self.ui.render(self.data,row,scale)
            h,w=image.shape[:2]
            self.assertEqual({key for _,key in self.ui.buttons},set('q123rs nxhbicgpvo'))
            for (x0,y0,x1,y1),key in self.ui.buttons:
                self.assertTrue(0<=x0<x1<=w and 0<=y0<y1<=h,(scale,key))
            for slider in self.ui.sliders.values():
                x0,y0,x1,y1=slider['bounds'];self.assertTrue(0<=x0<x1<=w and 0<=y0<y1<=h)

class ContrastTests(unittest.TestCase):
    setUp=ControlsTests.setUp
    tearDown=ControlsTests.tearDown
    frame=ControlsTests.frame

    def test_inferno_matches_annotation_presets_without_changing_input(self):
        from dataset_pipeline import colorize_view,VIEWS
        from launcher_theme import thermal_view
        # Varying, flat and outlier-heavy data exercise percentile/fallback behavior.
        arrays=[np.linspace(18,37,19200,dtype=np.float32).reshape(120,160),
                np.full((120,160),26,np.float32)]
        outliers=arrays[0].copy();outliers[0,0]=-20;outliers[-1,-1]=100;arrays.append(outliers)
        row=self.frame();self.ui.boxes=False;self.ui.raw=False;self.ui.show_search=False
        for data in arrays:
            before=data.copy()
            for mode in range(len(VIEWS)):
                self.ui.contrast=mode
                actual=np.asarray(thermal_view(self.ui,data,row,160))
                np.testing.assert_array_equal(actual,cv2.cvtColor(colorize_view(data,mode),cv2.COLOR_BGR2RGB))
                np.testing.assert_array_equal(data,before)

    def test_contrast_key_button_cycle_and_palette_return(self):
        row=self.frame();router=self.session.router
        conf=self.session.detector.conf;low=self.session.detector.low_conf
        self.ui.render(self.data,row,5)
        bounds=next(bounds for bounds,key in self.ui.buttons if key=='v')
        self.ui.mouse(cv2.EVENT_LBUTTONUP,bounds[0]+3,bounds[1]+3,0,None)
        self.ui.command(self.ui.actions.popleft(),self.data,row)
        self.assertEqual(self.ui.contrast,1)
        self.ui.command(ord('v'));self.assertEqual(self.ui.contrast,2)
        self.ui.command(ord('v'));self.assertEqual(self.ui.contrast,0)
        self.ui.command(ord('p'));self.assertEqual(self.ui.palette,1)
        self.ui.command(ord('v'));self.assertEqual(self.ui.palette,0)
        self.assertIs(self.session.router,router)
        self.assertEqual((self.session.detector.conf,self.session.detector.low_conf),(conf,low))

if __name__=='__main__':unittest.main()
