"""Cross-platform camera contracts, tracker parity, replay/recording and HTTP."""
from pathlib import Path
import base64,hashlib,json,os,sys,tempfile,threading,time,unittest,urllib.request,urllib.error
from dataclasses import asdict
from unittest.mock import patch
import cv2
import numpy as np

HERE=Path(__file__).resolve().parents[1]
THERMAL=HERE if (HERE/'tracking_router.py').is_file() else Path(os.environ.get('THERMAL_TEST_ROOT','/Users/haneefrahman2650/Developer/Occupancy-Detection-Program/Thermal'))
sys.path[:0]=[str(HERE),str(THERMAL)]
from thermal_web.camera import raw_to_celsius,WindowsY16,CameraStream,open_adapter
from thermal_web.processor import Processor
from thermal_web.workspace import Workspace
from thermal_web.server import Application,create_server
from heat_function import HeatConfig


class Detector:
    def __init__(self,conf=.374,low=.18):self.conf=conf;self.low_conf=low;self.i=0
    def base(self,image):
        self.i+=1
        return [dict(bbox=(14+20*j+self.i*.1,40,6,8),centroid=(17+20*j+self.i*.1,44),conf=.8,source='base',body_h=8/.3) for j in range(5)]
    def recover(self,image,regions):return []


def wait(worker,predicate,timeout=6,ignore_existing_error=False):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        state=worker.snapshot()
        if state['error'] and not ignore_existing_error:raise AssertionError(state['error'])
        if predicate(state):return state
        time.sleep(.01)
    raise AssertionError(worker.snapshot())


def make_capture(root):
    source=root/'logs'/'capture_test';(source/'npy').mkdir(parents=True)
    for i in range(4):np.save(source/'npy'/f'cap_{i:06d}.npy',np.full((120,160),30+i*.1,np.float32))
    (source/'timestamps.json').write_text(json.dumps([dict(stem=f'cap_{i:06d}',relative_time_s=i*.2) for i in range(4)]))
    return source


class CameraTests(unittest.TestCase):
    def test_fixed_scale_and_eight_bit_refusal(self):
        a=np.full((120,160),30000,np.uint16)
        np.testing.assert_allclose(raw_to_celsius(a,.01),26.85,atol=1e-4)
        np.testing.assert_allclose(raw_to_celsius(a//10,.1),26.85,atol=1e-4)
        for bad in [a.astype(np.uint8),np.zeros((120,160,3),np.uint8),np.zeros((120,320),np.uint16)]:
            with self.assertRaises(ValueError):raw_to_celsius(bad,.01)
        with self.assertRaises(ValueError):raw_to_celsius(a,1.)

    def test_telemetry_header_and_footer(self):
        image=np.full((120,160),30000,np.uint16);telemetry=np.zeros((2,160),np.uint16)
        for raw in [np.r_[telemetry,image],np.r_[image,telemetry]]:
            np.testing.assert_allclose(raw_to_celsius(raw,.01),26.85,atol=1e-4)

    def test_windows_y16_success_and_8bit_driver_rejection(self):
        caps=[]
        class Cap:
            def __init__(self,dtype):self.dtype=dtype;self.released=False;self.settings=[];caps.append(self)
            def isOpened(self):return True
            def set(self,*args):self.settings.append(args);return True
            def read(self):return True,np.full((120,160),100 if self.dtype==np.uint8 else 30000,self.dtype)
            def release(self):self.released=True
        with self.assertRaisesRegex(RuntimeError,'16-bit'):
            WindowsY16(factory=lambda *_:Cap(np.uint8))
        self.assertTrue(all(c.released for c in caps))
        camera=WindowsY16(scale=.01,factory=lambda *_:Cap(np.uint16))
        self.assertEqual(camera.backend,'DirectShow');self.assertEqual(camera.read().shape,(120,160))
        self.assertIn((cv2.CAP_PROP_CONVERT_RGB,0),caps[-1].settings)
        camera.close();self.assertTrue(caps[-1].released)

    def test_platform_dispatch(self):
        with patch('thermal_web.camera.WindowsY16') as windows,patch('thermal_web.camera.LibUVCRaw') as mac:
            open_adapter(2,.1,system='Windows');windows.assert_called_once_with(2,.1)
            open_adapter(0,.01,system='Darwin');mac.assert_called_once_with(.01)

    def test_latest_frame_queue_and_release(self):
        class Adapter:
            backend='test';n=0;closed=False
            def read(self):
                self.n+=1;time.sleep(.001);return np.full((120,160),float(self.n),np.float32)
            def close(self):self.closed=True
        adapter=Adapter();stream=CameraStream(opener=lambda **_:adapter)
        try:
            end=time.monotonic()+2
            while adapter.n<20 and time.monotonic()<end:time.sleep(.005)
            data,t=stream.read();self.assertGreater(float(data[0,0]),10);self.assertLessEqual(stream.frames.qsize(),1)
        finally:stream.close()
        self.assertTrue(adapter.closed);self.assertFalse(stream.thread.is_alive())


class ProcessorTests(unittest.TestCase):
    def test_matches_native_shared_tracker_outputs(self):
        # Reference import is test-only: runtime web modules have no launcher dependency.
        from integrated_launcher_v2 import TrackingSession
        from tracker import KalmanTrack
        results=[]
        for factory in [TrackingSession,Processor]:
            KalmanTrack._next_id=1;p=factory(Detector(),mode='faster');rows=[]
            for i in range(5):
                row=p.process(np.full((120,160),30.,np.float32),i*.1,'same')
                rows.append({k:v for k,v in row.items() if k not in ['base_ms','recovery_ms','tracking_ms','processing_ms']})
            results.append(rows)
        self.assertEqual(results[0],results[1])


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.source=make_capture(self.root)
        self.original={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.rglob('*') if p.is_file()}
        self.w=Workspace(self.root,'simulator',detector_factory=Detector)
    def tearDown(self):self.w.close();self.tmp.cleanup()
    def load(self):
        self.w.submit(dict(action='heat',values=asdict(HeatConfig(warmup_s=0))))
        self.w.submit(dict(action='mode',value='faster'))
        self.w.submit(dict(action='load',path=str(self.source)))
        return wait(self.w,lambda s:s['index']==0 and s['row'] is not None)

    def test_replay_pause_speed_seek_export_and_heat_metrics(self):
        first=self.load();self.assertTrue(first['paused']);self.assertIsInstance(first['row']['plausible_detected'],int)
        self.w.submit(dict(action='speed',value=16));self.w.submit(dict(action='pause',value=False))
        done=wait(self.w,lambda s:s['index']==3)
        self.assertEqual(done['speed'],16);self.assertEqual(len(done['history']),4)
        exported=self.w.export();self.assertEqual([r['time'] for r in exported['frames']],[0,.2,.4,.6000000000000001])
        self.assertTrue(all(r['plausible_detected']>=r['detected'] for r in exported['frames']))
        self.w.submit(dict(action='seek',index=1));seek=wait(self.w,lambda s:s['index']==1 and not s['seeking'])
        self.assertTrue(seek['paused']);self.assertEqual(len(seek['history']),2)
        self.assertEqual(self.original,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.rglob('*') if p.is_file()})

    def test_confidence_display_is_not_inference_and_recording_schema(self):
        first=self.load();frame=first['row']['frame'];temp=self.w.data.copy()
        self.w.submit(dict(action='confidence',conf=.5,low_conf=.2));wait(self.w,lambda s:s['conf']==.5)
        self.w.submit(dict(action='display',heat_band=True));changed=wait(self.w,lambda s:s['heat_band'])
        self.assertEqual(changed['row']['frame'],frame);np.testing.assert_array_equal(temp,self.w.data)
        self.w.submit(dict(action='shot'));saved=wait(self.w,lambda s:s['saved']==1)
        folder=self.w.recorder.folders[0]
        self.assertIn('web_workspace.py',(folder/'source.txt').read_text())
        self.assertEqual((folder/'classes.txt').read_text(),'person\nhead_shoulder\n')
        np.testing.assert_array_equal(np.load(folder/'npy/cap_000000.npy'),temp)
        png=cv2.imread(str(folder/'png/cap_000000.png'),cv2.IMREAD_UNCHANGED);self.assertEqual(png.dtype,np.uint16)
        self.assertTrue(all(line.startswith('1 ') for line in (folder/'labels/cap_000000.txt').read_text().splitlines()))
        self.w.submit(dict(action='shot'));time.sleep(.05);self.assertEqual(self.w.snapshot()['saved'],1)

    def test_bad_command_does_not_kill_worker_or_change_configuration(self):
        self.load();self.w.submit(dict(action='confidence',conf=.1,low_conf=.2))
        end=time.monotonic()+2
        while not self.w.snapshot()['error'] and time.monotonic()<end:time.sleep(.01)
        self.assertTrue(self.w.snapshot()['error']);self.assertTrue(self.w.thread.is_alive())
        self.assertEqual(self.w.conf,.374)
        self.w.submit(dict(action='mode',value='precise'));wait(self.w,lambda s:s['policy']=='precise' and not s['error'],ignore_existing_error=True)
        self.w.submit(dict(action='step'));next_state=wait(self.w,lambda s:s['index']==1)
        self.assertIsNone(next_state['row']['plausible_detected'])

    def test_missing_camera_error_and_cleanup(self):
        def fail(**kwargs):raise RuntimeError('Test camera not connected')
        other=Workspace(self.root,'launcher',detector_factory=Detector,camera_factory=fail)
        try:
            other.submit(dict(action='connect',scale=.01));end=time.monotonic()+2
            while not other.snapshot()['error'] and time.monotonic()<end:time.sleep(.01)
            self.assertIn('not connected',other.snapshot()['error']);self.assertFalse(other.snapshot()['loaded']);self.assertIsNone(other.camera)
        finally:other.close()

    def test_live_pause_step_record_and_release(self):
        class Camera:
            backend='mock native Y16';closed=False;n=0
            def __init__(self,**kwargs):pass
            def read(self):
                self.n+=1
                return np.full((120,160),30.,np.float32),time.monotonic()
            def close(self):self.closed=True
        other=Workspace(self.root,'launcher',detector_factory=Detector,camera_factory=Camera)
        try:
            other.submit(dict(action='connect',scale=.01))
            wait(other,lambda s:s['row'] is not None)
            camera=other.camera
            other.submit(dict(action='pause',value=True))
            paused=wait(other,lambda s:s['paused']);index=paused['index']
            other.submit(dict(action='record'));wait(other,lambda s:s['recording'])
            other.submit(dict(action='step'))
            stepped=wait(other,lambda s:s['index']==index+1 and s['saved']==1)
            self.assertTrue(stepped['paused']);self.assertEqual(stepped['source']['kelvin_per_count'],.01)
            other.submit(dict(action='disconnect'));wait(other,lambda s:not s['loaded'])
            self.assertTrue(camera.closed);self.assertFalse(other.snapshot()['recording'])
        finally:other.close()

    def test_options_reset_and_snapshot_ownership(self):
        self.load()
        self.w.submit(dict(action='confidence',conf=.5,low_conf=.2))
        self.w.submit(dict(action='tracker_options',values=dict(recovery_seconds=2,max_radius=18,max_speed=40)))
        state=wait(self.w,lambda s:s['status']=='Tracker options applied; history restarted')
        self.assertEqual(state['index'],0);self.assertEqual(state['tracker_config']['low_conf'],.2)
        self.assertEqual(self.w.processor.detector.low_conf,.2)
        mutable=[dict(value=1)];self.w.publish(ownership_test=mutable);mutable[0]['value']=2
        self.assertEqual(self.w.snapshot()['ownership_test'][0]['value'],1)


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);make_capture(self.root)
        self.app=Application(self.root,worker_factory=lambda root,kind,weights:Workspace(root,kind,detector_factory=Detector))
        self.server=create_server(self.app,0);self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_address[1]}'
    def tearDown(self):self.server.shutdown();self.server.server_close();self.app.close();self.thread.join();self.tmp.cleanup()
    def request(self,path,body=None,**headers):
        request=urllib.request.Request(self.url+path,data=json.dumps(body).encode() if body is not None else None,headers=headers)
        return urllib.request.urlopen(request)

    def test_assets_and_unchanged_state(self):
        for path in ['/','/static/app.js','/static/style.css','/api/meta']:
            with self.request(path) as response:self.assertEqual(response.status,200)
        with self.request('/api/state?workspace=launcher') as response:state=json.load(response)
        with self.request('/api/state?workspace=launcher&since='+str(state['version'])) as response:self.assertEqual(response.status,204)
        with self.assertRaises(urllib.error.HTTPError):self.request('/../../AGENTS.md')

    def test_control_auth_origin_and_valid_command(self):
        body=dict(workspace='launcher',command=dict(action='mode',value='faster'))
        with self.assertRaises(urllib.error.HTTPError) as error:self.request('/api/command',body)
        self.assertEqual(error.exception.code,403)
        with self.assertRaises(urllib.error.HTTPError):self.request('/api/command',body,**{'X-Workspace-Token':self.app.token,'Origin':'https://unrelated.invalid'})
        with self.request('/api/command',body,**{'X-Workspace-Token':self.app.token,'Origin':self.url}) as response:self.assertEqual(response.status,202)
        wait(self.app.workspaces['launcher'],lambda s:s['policy']=='faster')


if __name__=='__main__':unittest.main()
