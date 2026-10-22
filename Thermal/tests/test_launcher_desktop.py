"""Thread-boundary regressions; no display or camera needed."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import threading
import unittest
import numpy as np
from launcher_desktop import DesktopBridge

class BridgeTests(unittest.TestCase):
    def test_published_frame_owns_its_data(self):
        bridge=DesktopBridge();data=np.ones((120,160),np.float32);row={'observed':[{'bbox':[1,2,3,4]}]}
        bridge.publish(row=row,data=data)
        snapshot=bridge.snapshot();data[:]=7;row['observed'][0]['bbox'][0]=99
        self.assertEqual(float(snapshot['data'][0,0]),1)
        self.assertEqual(snapshot['row']['observed'][0]['bbox'][0],1)
        with self.assertRaises(ValueError):snapshot['data'][0,0]=2
        bridge.publish(message='changed')
        self.assertEqual(snapshot['message'],'Loading model…')

    def test_commands_preserve_order_while_worker_is_busy(self):
        bridge=DesktopBridge();busy=threading.Event();release=threading.Event();seen=[]
        def worker(b):
            busy.set();release.wait(2)
            while not b.commands.empty():seen.append(b.commands.get_nowait())
            b.publish(ack=seen[-1][0])
        bridge.start(worker);self.assertTrue(busy.wait(1))
        first=bridge.submit({'slider':'conf','value':.6});second=bridge.submit(ord('r'))
        self.assertEqual(bridge.snapshot()['ack'],0)
        release.set();bridge.thread.join(2)
        self.assertEqual([v[0] for v in seen],[first,second]);self.assertEqual(bridge.snapshot()['ack'],second)
        self.assertTrue(bridge.snapshot()['done'])

    def test_error_reaches_ui_and_cleanup_runs(self):
        bridge=DesktopBridge();cleaned=[]
        def worker(b):
            try:raise ValueError('test inference failure')
            finally:cleaned.append(True)
        bridge.start(worker);bridge.thread.join(2)
        self.assertEqual(cleaned,[True]);self.assertTrue(bridge.snapshot()['done'])
        self.assertIn('test inference failure',bridge.snapshot()['error'])

if __name__=='__main__':unittest.main()
