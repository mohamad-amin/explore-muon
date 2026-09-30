"""Hash this completed study's artifacts without modifying measured inputs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / 'MANIFEST.json'
assert json.loads((HERE / 'finish_status.json').read_text())['status'] == 'complete'
assert 'Draft:' not in (HERE / 'REPORT.md').read_text()
assert not TARGET.exists(), 'Preserve the prior manifest; do not silently overwrite it'
records = []
for path in sorted(HERE.rglob('*')):
    if not path.is_file() or path == TARGET:
        continue
    relative = path.relative_to(HERE)
    if '__pycache__' in relative.parts or any(part.startswith('.mplconfig') for part in relative.parts):
        continue
    if path.name.endswith('.tmp'):
        continue
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            digest.update(chunk)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), path
    records.append(dict(path=str(relative), bytes=after.st_size, sha256=digest.hexdigest()))
manifest = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
                files=len(records), bytes=sum(r['bytes'] for r in records), records=records,
                excludes=['MANIFEST.json itself', '__pycache__', '.mplconfig* caches', '*.tmp'])
with TARGET.open('x') as f:
    json.dump(manifest, f, indent=2)
    f.write('\n')
print(json.dumps({k: v for k, v in manifest.items() if k != 'records'}))
