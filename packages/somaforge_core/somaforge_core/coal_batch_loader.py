"""Build once per source/ABI/library identity; never substitute a solver.

The extension uses the existing Coal Python geometry instances and the exact
library that owns them. Compilation happens at scene initialization, not query.
Build products live in the persistent runtime cache.
"""
from functools import lru_cache
from pathlib import Path
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import sysconfig


@lru_cache(maxsize=1)
def load_coal_batch():
    import coal

    source = Path(__file__).with_name('coal_batch.cpp')
    prefix = next((p for p in Path(coal.__file__).resolve().parents
                   if (p/'include/coal/distance.h').is_file() and
                   (p/'lib/libcoal.so').is_file()), None)
    if prefix is None:
        raise RuntimeError('Native Coal batch requires matching Coal headers and libcoal.so')
    boost = f'boost_python{sys.version_info.major}{sys.version_info.minor}'
    libraries = [prefix/'lib/libcoal.so', prefix/f'lib/lib{boost}.so']
    if not all(p.is_file() for p in libraries):
        raise RuntimeError('Native Coal batch requires the matching Boost.Python ABI')
    eigen = next((p for p in (prefix/'include/eigen3', Path('/usr/include/eigen3'))
                  if (p/'Eigen/Core').is_file()), None)
    if eigen is None:
        raise RuntimeError('Native Coal batch requires Eigen headers')
    identity = dict(python=sys.version, executable=sys.executable,
                    compiler=os.environ.get('CXX', 'c++'), prefix=str(prefix),
                    libraries=[(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                               for p in libraries])
    digest = hashlib.sha256(source.read_bytes()+json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    project = next((p for p in source.parents if (p/'scripts/source_somaforge.sh').is_file()), None)
    if project is None:
        raise RuntimeError('Native Coal build cache requires the SomaForge workspace')
    cache = project/'runtime/current/cache/coal_batch'/digest
    cache.mkdir(parents=True, exist_ok=True)
    output = cache/('coal_batch'+sysconfig.get_config_var('EXT_SUFFIX'))
    with (cache/'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.exists():
            temporary = cache/(output.name+'.building')
            command = [identity['compiler'], '-O3', '-shared', '-std=c++17', '-fPIC', str(source),
                       '-I'+str(prefix/'include'), '-I'+str(eigen),
                       '-I'+sysconfig.get_path('include'), '-L'+str(prefix/'lib'),
                       '-Wl,-rpath,'+str(prefix/'lib'), '-lcoal', '-l'+boost, '-o', str(temporary)]
            with (cache/'build.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            temporary.replace(output)
            (cache/'identity.json').write_text(json.dumps(identity, indent=2))
    spec = importlib.util.spec_from_file_location('coal_batch', output)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
