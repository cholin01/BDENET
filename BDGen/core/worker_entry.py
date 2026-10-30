"""Run one unchanged CLI job and retain its peak CPU memory and wall time."""
import sys,runpy,time,resource,json
from pathlib import Path
start=time.monotonic();args=sys.argv[1:];folder=Path(args[args.index('--output_dir')+1]);folder.mkdir(parents=True,exist_ok=True)
sys.argv=[str(Path(__file__).with_name('run_bde_frag_gt.py')),*args]
status='failed'
try:
    runpy.run_path(sys.argv[0],run_name='__main__');status='completed'
finally:
    (folder/'runtime_resources.json').write_text(json.dumps({'status':status,'seconds':time.monotonic()-start,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2))
