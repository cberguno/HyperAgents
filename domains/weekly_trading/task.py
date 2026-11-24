# Pure task contract. Never executes candidate code.
import json
import math

def validate_package(package):
    if not isinstance(package, dict) or set(package) != {'schema', 'columns', 'series'} or package['schema'] != 'weekly-series-v1' or package['columns'] != ['weekly_return', 'mean_feature']:
        raise ValueError('invalid package schema')
    series = package['series']
    if not isinstance(series, list) or len(series) != 50 or not all(isinstance(s, list) for s in series): raise ValueError('expected 50 series')
    if not series[0] or any(len(s) != len(series[0]) for s in series): raise ValueError('expected equal nonempty histories')
    for s in series:
        for pair in s:
            if not isinstance(pair, list) or len(pair) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in pair): raise ValueError('invalid history value')
    return package

def copy_package(source, destination):
    # Caller must supply the already-created single package, never a data root.
    with open(source, 'rb') as f: raw = f.read(20_000_001)
    if len(raw) > 20_000_000: raise ValueError('package too large')
    validate_package(json.loads(raw))
    with open(destination, 'xb') as f: f.write(raw)

def score(selected, sessions, cost_bps=5):
    # Trusted scorer only. Sessions are held back from the candidate.
    if not isinstance(selected, list) or any(type(i) is not int or not 0 <= i < 50 for i in selected) or len(set(selected)) != len(selected): raise ValueError('invalid selection')
    if not isinstance(sessions, list) or not sessions or any(not isinstance(day, list) or len(day)!=50 or any(type(x) not in (float,int) or not math.isfinite(x) or x <= -1 for x in day) for day in sessions): raise ValueError('invalid session returns')
    if type(cost_bps) not in (int,float) or not math.isfinite(cost_bps) or not 0 <= cost_bps < 5000: raise ValueError('invalid costs')
    c=cost_bps/10000; strategy=benchmark=1.0
    for day in sessions:
        base=sum(day)/50; benchmark *= 1 + base - c*(2+base)
        if selected:
            gross=sum(day[j] for j in selected)/len(selected)
            strategy *= 1 + gross - c*(2+gross)
    return {'strategy_return':strategy-1, 'benchmark_return':benchmark-1, 'excess_return':strategy-benchmark}
