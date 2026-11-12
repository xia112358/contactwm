"""Portable launcher must not require the author's temporary experiment tree."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def test_handoff_dry_run_uses_explicit_assets(tmp_path):
    assets=tmp_path/'assets';output=tmp_path/'run'
    proc=subprocess.run([sys.executable,str(ROOT/'scripts/run_handoff.py'),'--assets',str(assets),'--output',str(output),'--gpu','2','--dry-run'],cwd=ROOT,check=True,capture_output=True,text=True)
    plan=json.loads(proc.stdout);argv=plan['argv']
    assert argv[2]==str(ROOT/'packages/supermimic/supermimic/handoff/entry.py')
    assert argv[argv.index('--single_stage_anchor')+1]==str(assets/'contact-init.pt')
    assert argv[argv.index('--iterations')+1]=='100'
    assert '--compile_contact_fk' in argv
    assert '--check_dmo' not in argv
    assert argv[argv.index('--dmo_microbatch')+1]=='1024'
    assert plan['environment']['CUDA_VISIBLE_DEVICES']=='2'
    assert plan['environment']['SOMAFORGE_PUBLISHED_DATA_PATHS']==str(assets/'paths.json')
    assert not output.exists()

def test_handoff_recipe_has_no_author_tmp_paths():
    recipe=json.loads((ROOT/'configs/contact_learning/handoff-ten-stage.json').read_text())
    assert all('/tmp/' not in x and '/home/zhouxia' not in x and '/data1/zhouxia' not in x for x in recipe['arguments'])
    for p in (ROOT/'packages/supermimic/supermimic/handoff').glob('*.py'):
        compile(p.read_text(),str(p),'exec')
        assert 'bottleneck-handoff-retention-20261108' not in p.read_text()
