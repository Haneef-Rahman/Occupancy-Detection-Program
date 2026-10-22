"""Read-only capture indexing. Never infer sampling time from file mtimes."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def natural_key(path):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', path.name)]


class CaptureSequence:
    def __init__(self, folder, fps=None):
        folder = Path(folder).expanduser().resolve()
        self.directory = folder/'npy' if (folder/'npy').is_dir() else folder
        self.root = self.directory.parent if self.directory.name == 'npy' else folder
        self.paths = sorted(self.directory.glob('*.npy'), key=natural_key)
        if not self.paths:
            raise ValueError('Choose a capture folder containing npy/ thermal frames.')
        self.source_hash = None
        if fps is not None:
            fps = float(fps)
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError('Assumed FPS must be finite and positive.')
            self.times = np.arange(len(self.paths), dtype=float)/fps
            self.timing = f'Explicit assumed rate: {fps:g} FPS (not measured timestamps)'
        else:
            times = {}
            if (self.root/'timestamps.json').is_file():
                source = self.root/'timestamps.json'
                rows = json.loads(source.read_text())
                pairs = [(r['stem'], float(r['relative_time_s'])) for r in rows]
                self.timing = 'Recorded launcher frame times'
            elif (self.root/'pairs.json').is_file():
                source = self.root/'pairs.json'
                rows = json.loads(source.read_text())['pairs']
                pairs = [(r['stem'], float(r['thermal_host_receipt_epoch_us'])/1e6) for r in rows]
                self.timing = 'Recorded thermal host receipt times (not exposure times)'
            elif (self.root/'manifest.csv').is_file():
                source = self.root/'manifest.csv'
                with source.open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                # Compare datetimes directly: naive local timestamps do not need
                # an invented timezone to recover their within-session intervals.
                parsed = [(Path(r['file']).stem, datetime.fromisoformat(r['timestamp'])) for r in rows]
                if not parsed:
                    raise ValueError('Empty manifest.')
                origin = parsed[0][1]
                pairs = [(stem, (stamp-origin).total_seconds()) for stem, stamp in parsed]
                self.timing = 'Recorded manifest save times (not exact exposure times)'
            else:
                raise ValueError('No timestamps found. Enter an explicit assumed FPS to test this folder.')
            for stem, stamp in pairs:
                if stem in times:
                    raise ValueError(f'Duplicate timestamp entry for {stem}')
                times[stem] = stamp
            missing = [p.stem for p in self.paths if p.stem not in times]
            if missing:
                raise ValueError(f'Missing timestamps for {len(missing)} frame(s), starting with {missing[0]}. Use an explicit FPS override only if intended.')
            self.times = np.array([times[p.stem] for p in self.paths], dtype=float)
            self.times -= self.times[0]
            self.source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError('Frame timestamps must be finite and strictly increasing; do not guess across duplicates or clock reversals.')
        self.load(0)

    def __len__(self):
        return len(self.paths)

    def load(self, i):
        data = np.load(self.paths[i], allow_pickle=False)
        if data.shape != (120, 160) or data.dtype.kind != 'f' or not np.isfinite(data).all():
            raise ValueError(f'{self.paths[i].name}: expected finite 120x160 float Celsius data; use npy/, not raw counts or preview PNGs.')
        return data

    def info(self):
        gaps = np.diff(self.times)
        return dict(folder=str(self.root), frames=len(self), duration_s=float(self.times[-1]),
                    median_interval_s=float(np.median(gaps)) if len(gaps) else None,
                    maximum_gap_s=float(gaps.max()) if len(gaps) else None,
                    timing=self.timing, timing_metadata_sha256=self.source_hash)


def discover(root):
    found = []
    for folder in sorted(Path(root).glob('capture_*'), reverse=True):
        count = sum(1 for _ in (folder/'npy').glob('*.npy'))
        if count:
            found.append({'name': folder.name, 'path': str(folder.resolve()), 'frames': count})
    return found
