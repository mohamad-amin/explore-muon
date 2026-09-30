"""Finish the already-authorized saved-data analyses after the existing watcher.

This script starts no producer, model evaluation, training, or restart.
"""
from pathlib import Path
import csv
import json
import os
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
RUN = HERE / 'run1'


def rows(name):
    with (RUN / name).open() as f:
        return list(csv.DictReader(f))


def main():
    start = time.monotonic()
    while True:
        for path in (RUN / 'failure.json', RUN / 'analysis/failure.json'):
            if path.exists():
                raise RuntimeError(f'Prior stage failed: {path}; no restart')
        status = HERE / 'postprocess_status.json'
        if status.exists() and json.loads(status.read_text())['status'] == 'complete':
            break
        if time.monotonic() - start > 10800:
            raise TimeoutError('Existing postprocessor did not finish within three hours; no restart')
        time.sleep(5)
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[key] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'

    def run(script, args=()):
        print('Starting', script, flush=True)
        subprocess.run([sys.executable, str(HERE / script), *map(str, args)],
                       check=True, cwd=HERE.parents[1], env=env)
        print('Finished', script, flush=True)

    run('analyze_transmission.py')
    run('verify_results.py', ('--out', RUN / 'verification.json'))
    transmission = rows('transmission_analysis/metrics.csv')
    curvature = rows('analysis/curvature.csv')
    moments = rows('geometry_analysis/radius_moments.csv')
    for row in transmission:
        for key in ('preactivation_energy', 'GN', 'transmission_gain',
                    'GN_per_preactivation_energy', 'GN_per_residual_energy'):
            assert row[key] and float(row[key]) > 0, (key, row)
    for row in curvature:
        for key in ('hessian_over_gn', 'mean_abs_token_hessian_minus_gn_over_mean_gn'):
            assert row[key], (key, row)
    for row in moments:
        if row['bank'] == 'all' and row['direction'] == 'actual':
            assert row['observed_to_spherical_relative_variance'], row
    (RUN / 'report_ratio_qualification.json').write_text(json.dumps(dict(
        status='passed', transmission_rows=len(transmission), curvature_rows=len(curvature),
        note='Selected summary fields are nonblank and transmission denominators strictly positive; raw negative Hessian values remain valid.'), indent=2) + '\n')
    run('report_tables.py')
    run('plot_overview.py')
    (HERE / 'finish_status.json').write_text(json.dumps(dict(status='complete',
        seconds_including_wait=time.monotonic()-start), indent=2) + '\n')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        (HERE / 'finish_failure.json').write_text(json.dumps(dict(type=type(exc).__name__,
            message=str(exc), traceback=traceback.format_exc()), indent=2) + '\n')
        raise
