import ast
import math
from pathlib import Path
import sys
import unittest
import tempfile
import json
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from faster_k import FastConfig, FasterK, touching, clean_detections
from tracking_router import TrackingRouter
from precise_k import PreciseK
from tracker import MultiTracker, KalmanTrack
from crowd_detector import CrowdDetector, thermal_image


def detection(x=60, y=45, conf=.8, w=6, h=8):
    return dict(centroid=(x, y), bbox=(x-w/2, y-h/2, w, h),
                conf=conf, src='cnn', source='base', body_h=h/.3)


class TrackingTests(unittest.TestCase):
    def setUp(self):
        KalmanTrack._next_id = 1

    def seeded(self, config=None):
        k = FasterK(config=config)
        k.begin([detection()], 0); k.finish([])
        k.begin([detection()], .1); k.finish([])
        return k

    def test_dropout_draws_no_predicted_box(self):
        k = self.seeded(); k.begin([], .2); k.finish([])
        self.assertEqual(k.visible(), [])
        self.assertEqual(len(k.unresolved()), 1)
        self.assertEqual(len(k.regions), 1)

    def test_low_requires_two_observations_and_keeps_id(self):
        k = self.seeded(); identity = k.visible()[0].id
        k.begin([], .2); k.finish([detection(61, conf=.2)])
        self.assertEqual(k.visible(), [])
        k.begin([], .3); k.finish([detection(61, conf=.2)])
        self.assertEqual([t.id for t in k.visible()], [identity])
        self.assertEqual(k.visible()[0].bbox, detection(61)['bbox'])

    def test_low_never_births(self):
        k = FasterK(); k.begin([], 0); k.finish([detection(conf=.2)])
        self.assertEqual(k.tracks, [])

    def test_low_outside_circle_rejected(self):
        k = self.seeded(); k.begin([], .2); k.finish([detection(120, conf=.2)])
        self.assertEqual(k.stats['recovered'], 0)

    def test_low_below_recovery_threshold_rejected(self):
        k = self.seeded(); k.begin([], .2); k.finish([detection(conf=.01)])
        self.assertEqual(k.stats['recovered'], 0)

    def test_roi_square_corner_is_not_circle(self):
        k = self.seeded(); r = k.begin([], .2)[0]
        k.finish([detection(r['cx']+.9*r['radius'], r['cy']+.9*r['radius'], .2)])
        self.assertEqual(k.stats['recovered'], 0)

    def test_low_feedback_cannot_extend_deadline(self):
        k = self.seeded()
        for i in range(2, 23):
            k.begin([], i/10); k.finish([detection(conf=.2)])
        self.assertEqual(k.tracks, [])

    def test_gap_breaks_low_confirmation(self):
        k = self.seeded(); k.begin([], .2); k.finish([detection(conf=.2)])
        k.begin([], .3); k.finish([])
        k.begin([], .4); k.finish([detection(conf=.2)])
        self.assertEqual(k.visible(), [])

    def test_base_reconfirmation_renews_deadline(self):
        k = self.seeded(); k.begin([detection()], 1.9); k.finish([])
        k.begin([], 2.5); k.finish([])
        self.assertEqual(len(k.tracks), 1)

    def test_teleport_does_not_inherit_identity(self):
        k = self.seeded(); old = k.visible()[0].id
        k.begin([detection(110)], .2); k.finish([])
        self.assertTrue(any(t.id != old and t.det['centroid'][0] == 110 for t in k.tracks))
        self.assertTrue(next(t for t in k.tracks if t.id == old).misses)

    def test_size_jump_not_recovered(self):
        k = self.seeded(); k.begin([], .2)
        k.finish([detection(w=30, h=40, conf=.2)])
        self.assertEqual(k.stats['recovered'], 0)

    def test_speed_gate_uses_seconds(self):
        cfg = FastConfig(max_speed=10, jitter=1)
        k = self.seeded(cfg); k.begin([], .2)
        self.assertTrue(math.isinf(k.cost(k.tracks[0], detection(65))))
        # Five px is allowed after half a second; covariance/radius also pass.
        k.finish([]); k.begin([], .6)
        self.assertTrue(math.isfinite(k.cost(k.tracks[0], detection(65))))

    def test_recovery_radius_is_capped(self):
        k = self.seeded(); k.begin([], 1.9)
        self.assertLessEqual(k.regions[0]['radius'], k.config.max_radius)

    def test_edge_loss_no_local_lowering(self):
        k = FasterK()
        for now in [0, .1]:
            k.begin([detection(3)], now); k.finish([])
        self.assertEqual(k.begin([], .2), [])

    def test_ambiguous_merge_not_forced_and_no_birth(self):
        k = FasterK()
        for now in [0, .1]:
            k.begin([detection(58), detection(62)], now); k.finish([])
        k.begin([detection(60)], .2); k.finish([])
        self.assertEqual(k.visible(), [])
        self.assertEqual(len(k.tracks), 2)
        self.assertGreater(k.stats['ambiguous'], 0)

    def test_high_match_cannot_be_stolen_by_recovery(self):
        k = FasterK()
        for now in [0, .1]:
            k.begin([detection(55), detection(65)], now); k.finish([])
        k.begin([detection(55)], .2); k.finish([detection(55, conf=.2)])
        self.assertEqual(k.stats['recovered'], 0)
        self.assertEqual(len(k.visible()), 1)

    def test_identical_roi_candidates_do_not_duplicate(self):
        k = self.seeded(); k.begin([], .2)
        k.finish([detection(conf=.2), detection(conf=.22)])
        self.assertEqual(k.stats['recovered'], 1)
        self.assertEqual(len(k.tracks), 1)

    def test_max_rois_and_fair_rotation(self):
        k = FasterK(config=FastConfig(max_rois=2))
        for now in [0, .1]:
            k.begin([detection(x) for x in [20, 40, 60, 80, 100]], now); k.finish([])
        first = {r['id'] for r in k.begin([], .2)}; k.finish([])
        second = {r['id'] for r in k.begin([], .3)}
        self.assertEqual(len(first), 2)
        self.assertFalse(first & second)

    def test_covariance_positive_and_velocity_bounded(self):
        k = self.seeded()
        for i in range(2, 100):
            k.begin([detection(60+min(i*.3, 25))], i*.1); k.finish([])
            for t in k.tracks:
                self.assertGreaterEqual(np.linalg.eigvalsh(t.P).min(), -1e-8)
                self.assertLessEqual(np.linalg.norm(t.x[2:]), k.config.max_speed+1e-8)

    def test_nonmonotonic_timestamp_rejected(self):
        k = self.seeded()
        for now in [.1, float('nan')]:
            with self.assertRaises(ValueError):
                k.begin([], now)

    def test_invalid_observations_rejected(self):
        self.assertEqual(clean_detections([detection(w=0), detection(float('nan')), detection(200)], (160, 120)), [])

    def test_two_close_heads_not_aggressively_deduplicated(self):
        self.assertEqual(len(clean_detections([detection(60), detection(63)], (160, 120))), 2)
        self.assertTrue(touching([detection(60), detection(63)]))

    def test_routing_count_and_touching(self):
        router = TrackingRouter()
        router.begin([detection(x) for x in [15, 40, 65, 90]], 0); router.finish([])
        self.assertEqual(router.mode, 'precise')
        router.begin([detection(x) for x in [15, 40, 65, 90, 115]], .1); router.finish([])
        self.assertEqual(router.mode, 'faster')
        other = TrackingRouter(); other.begin([detection(60), detection(63)], 0)
        self.assertEqual(other.mode, 'faster')

    def test_falling_count_with_missing_people_stays_fast(self):
        router = TrackingRouter()
        for i in range(2):
            router.begin([detection(x) for x in [15, 40, 65, 90, 115]], i*.1); router.finish([])
        for i in range(2, 15):
            router.begin([detection(x) for x in [15, 40, 65, 90]], i*.1); router.finish([])
        self.assertEqual(router.mode, 'faster')

    def test_handover_preserves_ids_and_returns_after_settling(self):
        router = TrackingRouter()
        for i in range(3):
            router.begin([detection(20)], i*.1); router.finish([])
        identity = router.observed()[0].id
        router.begin([detection(x) for x in [20, 45, 70, 95, 120]], .3); router.finish([])
        self.assertIn(identity, [t.id for t in router.observed()])
        for i in range(4, 48):
            router.begin([detection(20)], i*.1); router.finish([])
        self.assertEqual(router.mode, 'precise')
        self.assertEqual([t.id for t in router.observed()], [identity])

    def test_legacy_equal_distance_tie_does_not_crash(self):
        p = PreciseK(min_hits=1, dup_confirm=1, gate_px=1)
        p.update([detection(50), detection(70)])
        p.update([detection(50), detection(70), detection(60)])
        self.assertEqual(len(p.tracks), 3)

    def test_precise_extraction_matches_legacy_sparse_trace(self):
        # Import just the original class; no camera, UI, or model side effects.
        original = Path(__import__('tracker').__file__).with_name('integrated_launcher.py')
        tree = ast.parse(original.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StickyTracker')
        namespace = dict(np=np, MultiTracker=MultiTracker, KalmanTrack=KalmanTrack)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(original), 'exec'), namespace)
        def run(factory):
            KalmanTrack._next_id = 1
            p = factory(max_misses=8, min_hits=2, gate_px=30., ghost_frames=45)
            rows = []
            for i in range(40):
                dets = [detection(40+i*.2)] if not 12 <= i < 24 else []
                p.update(dets); p.resolve_occluded(dets, True)
                rows.append(([(t.id, t.x.tolist(), t.misses) for t in p.tracks], len(p.occluded), p.present()))
            return rows
        self.assertEqual(run(PreciseK), run(namespace['StickyTracker']))


class Tensor:
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a


def result(boxes, scores, classes):
    return SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor(boxes), conf=Tensor(scores), cls=Tensor(classes)))


class DetectorTests(unittest.TestCase):
    def test_recovery_is_crop_only_with_circle_and_offset(self):
        class Model:
            names = {0: 'person', 1: 'head_shoulder'}
            def __init__(self): self.calls = []
            def predict(self, images, **kwargs):
                self.calls.append((images, kwargs))
                if not isinstance(images, list):
                    return [result([], [], [])]
                # Crop origin (42,27), circle (60,45), r=10. First inside;
                # second inside crop but outside circle; third wrong class.
                return [result([[16,16,20,20], [0,0,4,4], [16,16,20,20]], [.2,.8,.9], [1,1,0])]
        model = Model(); d = CrowdDetector(model)
        image = np.zeros((120,160,3), np.uint8)
        d.base(image)
        low = d.recover(image, [dict(id=1, cx=60., cy=45., radius=10.)])
        self.assertEqual(len(low), 1)
        self.assertEqual(low[0]['centroid'], (60.,45.))
        self.assertEqual(model.calls[0][1]['conf'], .374)
        self.assertEqual(model.calls[1][1]['conf'], .18)
        self.assertEqual(model.calls[1][0][0].shape, (36,36,3))
        self.assertEqual(d.conf, .374)
        self.assertEqual(d.recover(image, []), [])
        self.assertEqual(len(model.calls), 2)

    def test_rejects_body_only_model(self):
        with self.assertRaises(ValueError):
            CrowdDetector(SimpleNamespace(names={0:'person'}))

    def test_encoding_fixed_celsius_span(self):
        pixels = thermal_image(np.array([[15.,30.,45.]], np.float32))
        np.testing.assert_array_equal(pixels[0,:,0], [0,127,255])

    def test_replay_uses_real_timestamp_gaps(self):
        from integrated_launcher_v2 import replay_frames
        with tempfile.TemporaryDirectory() as root:
            p = Path(root); (p/'npy').mkdir()
            for i in range(2):
                np.save(p/'npy'/f'cap_{i:06d}.npy', np.zeros((120,160), np.float32))
            (p/'timestamps.json').write_text(json.dumps([
                {'stem':'cap_000000','relative_time_s':10.},
                {'stem':'cap_000001','relative_time_s':10.3}]))
            frames = list(replay_frames(p/'npy'))
            self.assertEqual(frames[0][1], 0.)
            self.assertAlmostEqual(frames[1][1], .3)
            (p/'timestamps.json').unlink()
            (p/'pairs.json').write_text(json.dumps({'pairs': [
                {'stem':'cap_000000','thermal_host_receipt_epoch_us':10000000},
                {'stem':'cap_000001','thermal_host_receipt_epoch_us':10500000}]}))
            self.assertAlmostEqual(list(replay_frames(p/'npy'))[1][1], .5)

    def test_fast_display_never_draws_a_missing_box(self):
        from unittest.mock import patch
        from integrated_launcher_v2 import draw
        router = TrackingRouter(mode='faster')
        for now in [0, .1]:
            router.begin([detection()], now); router.finish([])
        router.begin([], .2); router.finish([])
        with patch('integrated_launcher_v2.cv2.rectangle') as rectangle:
            draw(np.zeros((120,160), np.float32), router, router.regions)
            rectangle.assert_not_called()


if __name__ == '__main__':
    unittest.main()
