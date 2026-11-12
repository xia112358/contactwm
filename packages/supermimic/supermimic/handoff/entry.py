import runpy,sys,importlib.util
from pathlib import Path
here=Path(__file__).parent;sys.path.insert(0,str(here))
from bottleneck_actor import BottleneckContactActor
import supermimic,supermimic.contact_control as control
control.ContactActor=BottleneckContactActor
for name in ('contact_task','contact_dmo','contact_sequence'):
 spec=importlib.util.spec_from_file_location('supermimic.'+name,here/(name+'.py'));module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;setattr(supermimic,name,module);spec.loader.exec_module(module)
spec=importlib.util.spec_from_file_location('supermimic.contact_collision_gpu',here/'contact_collision_gpu.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
from task import ReferenceProducerTask
supermimic.contact_task.ContactTask=ReferenceProducerTask
runpy.run_path(str(here/'train.py'),run_name='__main__')
