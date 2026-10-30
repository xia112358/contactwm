"""Supplement missing self-pair rows with explicit live-model separation evidence.

Raw Newton rows, contact activation and allocation are never changed. Only
convex mesh pairs are supported; unsupported or overlapping projections remain
unknown. Geometry is taken from the initialized, asset-validated Newton model.
"""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from contact_solver.separation_certificate import vertex_separation


def install(c):
    model=c['query'].model
    types=model.shape_type.numpy();bodies=model.shape_body.numpy()
    transforms=model.shape_transform.numpy();scales=model.shape_scale.numpy()
    margins=model.shape_margin.numpy();cache={};history={};events=[];original=c['observe']
    from newton import GeoType
    def local_vertices(shape):
        if shape in cache:return cache[shape]
        if int(types[shape])!=int(GeoType.CONVEX_MESH):return None
        source=model.shape_source[shape]
        pointer=int(model.shape_source_ptr.numpy()[shape])
        meshes=[*model._mesh_keep_alive,*source._finalized_meshes.values()]
        mesh=next((m for m in meshes if int(m.id)==pointer),None)
        if mesh is None:raise ValueError('Cannot resolve actual Newton mesh source pointer')
        vertices=mesh.points.numpy().astype(np.float64)*scales[shape]
        t=transforms[shape]
        vertices=vertices@Rotation.from_quat(t[3:]).as_matrix().T+t[:3]
        cache[shape]=vertices
        np.savez(c['out']/f'certificate_shape_{shape}.npz',vertices_body_local=vertices,shape_id=shape,body_id=bodies[shape],source_pointer=pointer,model_fingerprint=c['query'].provenance['model_fingerprint'])
        return vertices
    def observe(q,label):
        raw=original(q,label);pairs=raw['pairs'];present=set()
        for i in (pairs['full_kind']==1).nonzero().flatten().tolist():
            a,b=int(pairs['shape0'][i]),int(pairs['shape1'][i]);key=(a,b);present.add(key)
            history[key]=pairs['normal_w'][i].detach().cpu().numpy().copy()
        state=c['query'].state.body_q.numpy();certificates=[]
        for (a,b),normal in history.items():
            if (a,b) in present:continue
            va,vb=local_vertices(a),local_vertices(b)
            if va is None or vb is None or bodies[a]<0 or bodies[b]<0:continue
            ta,tb=state[bodies[a]],state[bodies[b]]
            wa=va@Rotation.from_quat(ta[3:]).as_matrix().T+ta[:3]
            wb=vb@Rotation.from_quat(tb[3:]).as_matrix().T+tb[:3]
            # Padding makes this stricter than bare shape separation. The margin
            # comes from the actual model; epsilon is only numerical conservatism.
            padding=float(margins[a]+margins[b])+1e-6
            gap=vertex_separation(wa,wb,normal,padding)
            if gap>0:
                certificates.append(dict(shape0=a,shape1=b,gap_m=gap,padding_m=padding,axis=normal.tolist(),body_transform0=ta.tolist(),body_transform1=tb.tolist(),source='live_newton_convex_vertices_separating_axis',solver_contact_status='absent_from_current_query'))
        raw['self_separation_certificates']=certificates
        if certificates:
            events.append(dict(query_id=raw['query_id'],certificates=certificates))
            (c['out']/'self_separation_certificates.json').write_text(json.dumps(dict(model_fingerprint=c['query'].provenance['model_fingerprint'],events=events),indent=2))
        return raw
    c['observe']=observe
    return events
