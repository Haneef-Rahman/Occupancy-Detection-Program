import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from capture_replay import CaptureSequence
import integrated_launcher_v2 as engine
from tracking_simulator import Simulator


def capture(root, n=4):
    root.mkdir(); (root/'npy').mkdir()
    for i in range(n):
        np.save(root/'npy'/f'cap_{i:06d}.npy',np.full((120,160),25+i/10,np.float32))
    (root/'manifest.csv').write_text('file,timestamp\n'+''.join(
        f'cap_{i:06d},2026-09-21T10:07:06.{i*100:03d}\n' for i in range(n)))
    return root


class Tensor:
    def __init__(self,a):self.a=np.asarray(a)
    def cpu(self):return self
    def numpy(self):return self.a


class FakeModel:
    names={0:'person',1:'head_shoulder'}
    def __init__(self,*_):pass
    def predict(self,images,**_):
        def result():
            return SimpleNamespace(boxes=SimpleNamespace(
                xyxy=Tensor([[57.,41.,63.,49.]]),conf=Tensor([.8]),cls=Tensor([1])))
        return [result() for _ in images] if isinstance(images,list) else [result()]


def wait_for(sim,predicate,timeout=5):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        s=sim.snapshot()
        if s.get('error'):raise AssertionError(s['error'])
        if predicate(s):return s
        time.sleep(.01)
    raise AssertionError(sim.snapshot())


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_root_or_npy_load_same_real_timing(self):
        p=capture(self.root/'capture_a')
        for folder in [p,p/'npy']:
            seq=CaptureSequence(folder)
            np.testing.assert_allclose(seq.times,[0,.1,.2,.3])
            self.assertEqual(seq.load(1).shape,(120,160))
            self.assertIn('manifest',seq.timing)

    def test_missing_timestamps_require_explicit_override(self):
        p=capture(self.root/'capture_a');(p/'manifest.csv').unlink()
        with self.assertRaises(ValueError):CaptureSequence(p)
        np.testing.assert_allclose(CaptureSequence(p,fps=2).times,[0,.5,1,1.5])

    def test_bad_fps_rejected(self):
        p=capture(self.root/'capture_a')
        for value in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):CaptureSequence(p,value)

    def test_duplicate_and_reversed_timing_rejected(self):
        p=capture(self.root/'capture_a',2)
        for timestamps in [('06.100','06.100'),('06.200','06.100')]:
            (p/'manifest.csv').write_text('file,timestamp\n'+''.join(
                f'cap_{i:06d},2026-09-21T10:07:{stamp}\n' for i,stamp in enumerate(timestamps)))
            with self.assertRaises(ValueError):CaptureSequence(p)

    def test_missing_frame_timestamp_rejected(self):
        p=capture(self.root/'capture_a')
        (p/'manifest.csv').write_text('file,timestamp\ncap_000000,2026-09-21T10:07:06.000\n')
        with self.assertRaises(ValueError):CaptureSequence(p)

    def test_raw_counts_and_nonfinite_not_interpreted_as_celsius(self):
        p=capture(self.root/'capture_a')
        for array in [np.zeros((120,160),np.uint16),np.full((120,160),np.nan),np.zeros((60,80),float)]:
            np.save(p/'npy/cap_000000.npy',array)
            with self.assertRaises(ValueError):CaptureSequence(p)

    def test_paired_and_launcher_timestamp_formats(self):
        p=capture(self.root/'capture_a',2)
        (p/'pairs.json').write_text(json.dumps({'pairs':[
            {'stem':'cap_000000','thermal_host_receipt_epoch_us':1000000},
            {'stem':'cap_000001','thermal_host_receipt_epoch_us':1250000}]}))
        np.testing.assert_allclose(CaptureSequence(p).times,[0,.25])
        (p/'timestamps.json').write_text(json.dumps([
            {'stem':'cap_000000','relative_time_s':20},
            {'stem':'cap_000001','relative_time_s':20.35}]))
        np.testing.assert_allclose(CaptureSequence(p).times,[0,.35])

    def test_shared_engine_accepts_injected_detector(self):
        class Detector:
            def base(self,image):return []
            def recover(self,image,regions):return []
        session=engine.TrackingSession(Detector())
        row=session.process(np.full((120,160),25,np.float32),0)
        self.assertEqual(row['frame'],1);self.assertEqual(row['observed'],[])
        with self.assertRaises(ValueError):session.process(np.zeros((120,160),np.uint16),.1)

    def test_play_step_seek_and_source_preservation(self):
        p=capture(self.root/'capture_a'); weights=self.root/'weights.pt';weights.write_bytes(b'test-only')
        original={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file()}
        sim=Simulator(self.root,self.root/'results',weights,model_factory=FakeModel)
        try:
            sim.submit({'action':'load','folder':str(p)})
            wait_for(sim,lambda s:s['loaded'])
            sim.submit({'action':'step'});s=wait_for(sim,lambda s:s['index']==0)
            self.assertFalse(s['playing']);self.assertEqual(s['row']['time'],0)
            self.assertIsInstance(sim.session,engine.TrackingSession)
            first_run=s['run']
            sim.submit({'action':'speed','value':16});sim.submit({'action':'play'})
            s=wait_for(sim,lambda s:s['status'].startswith('Finished'))
            rows=[json.loads(line) for line in (Path(s['run'])/'frames.jsonl').read_text().splitlines()]
            np.testing.assert_allclose([r['time'] for r in rows],[0,.1,.2,.3])
            self.assertEqual(len(rows),4)
            self.assertEqual(s['speed'],16)
            sim.submit({'action':'seek','index':1})
            s=wait_for(sim,lambda s:s['status']=='Paused at selected frame')
            self.assertNotEqual(s['run'],first_run)
            self.assertEqual(s['row']['observed'][0]['id'],rows[1]['observed'][0]['id'])
            self.assertEqual(s['index'],1)
            self.assertFalse(s['playing'])
            sim.submit({'action':'restart'})
            wait_for(sim,lambda s:s['index']==-1)
            self.assertEqual(original,{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file()})
        finally:sim.close()


if __name__=='__main__':unittest.main()
