"""Static step14 research: passive pipeline audit + multi-face soft SQP recovery.

Canonical solid geometry supplies optimization constraints only. Newton alone
supplies realized contact, allocation, and solver collision evidence.
"""
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse as sp
from scipy.spatial import ConvexHull
import osqp
import torch
import trimesh
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics
from contact_solver.device_contact_objective import DeviceWitnessRows
from somaforge_core.robot_assets import canonical_g1_urdf_path, decode_robot_asset_json
from somaforge_core.contact_face_selection import select_contact_pairs
from somaforge_core.newton_contact_sources import physical_contact_signature


def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    return x


def plain(x):
    if torch.is_tensor(x):return x.detach().cpu().tolist()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x


def run(query):
    torch.set_num_threads(2)
    out=Path(os.environ['STEP14_RESEARCH_OUTPUT']);out.mkdir(exist_ok=True,parents=True)
    (out/'queries').mkdir(exist_ok=True)
    saved=json.loads(Path('runtime/current/archives/preserved_inputs_20261010/baseline1000_step14_gradient_20260925_v2/diagnostic.json').read_text())
    decode_robot_asset_json(saved['robot_asset_json'],context='static SQP research')
    device=str(query.model.device)
    fk=CanonicalG1ForwardKinematics().double().to(device)
    q0=torch.tensor(saved['records'][0]['q'],dtype=torch.float64,device=device)
    mesh=trimesh.load(saved['terrain'],force='mesh')
    assert mesh.is_watertight and mesh.is_convex
    normals=np.asarray(mesh.facets_normal)
    centers=np.asarray([mesh.triangles_center[f].mean(0) for f in mesh.facets])
    offsets=(normals*centers).sum(1)
    normal=q0.new_tensor(np.concatenate((normals,[[0,0,1]])))
    offset=q0.new_tensor(np.r_[offsets,0.])
    top=int(np.argmax(normals[:,2]));ground=len(normals)
    scale=q0.new_tensor([.02]*3+[.1]*32)
    asset=canonical_g1_urdf_path();xml=ET.parse(asset).getroot()
    shapes=[];link_names=[]
    for link in xml.findall('link'):
        name=link.get('name')
        for c in link.findall('collision'):
            geom=list(c.find('geometry'))[0]
            origin=c.find('origin')
            mat=np.eye(4)
            if origin is not None:
                mat=trimesh.transformations.euler_matrix(*np.fromstring(origin.get('rpy','0 0 0'),sep=' '))
                mat[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            shape=dict(name=name,kind=geom.tag,rotation=q0.new_tensor(mat[:3,:3]),center=q0.new_tensor(mat[:3,3]))
            if geom.tag=='mesh':
                v=np.asarray(trimesh.load(asset.parent/geom.get('filename'),force='mesh').vertices).copy()
                v*=np.fromstring(geom.get('scale','1 1 1'),sep=' ')
                v=v[ConvexHull(v).vertices] @ mat[:3,:3].T+mat[:3,3]
                shape['vertices']=q0.new_tensor(v)
            elif geom.tag=='sphere':shape['radius']=float(geom.get('radius'))
            elif geom.tag=='cylinder':shape.update(radius=float(geom.get('radius')),half=.5*float(geom.get('length')))
            else:raise ValueError(geom.tag)
            shapes.append(shape)
            if name not in link_names:link_names.append(name)
    for s in shapes:s['link']=link_names.index(s['name'])
    def pose(x,seed):
        d=x*scale;a=torch.cat((d.new_ones(1),d[3:6]*.5));a=a/a.norm();b=seed[3:7]
        quat=torch.cat(((a[0]*b[0]-a[1:]@b[1:]).reshape(1),a[0]*b[1:]+b[0]*a[1:]+torch.cross(a[1:],b[1:],dim=0)))
        return torch.cat((seed[:3]+d[:3],quat/quat.norm(),seed[7:]+d[6:]))
    def separation(q):
        p,r=fk.link_poses(q[None],tuple(link_names));values=[]
        for s in shapes:
            i=s['link'];local=normal@r[0,i]
            if s['kind']=='mesh':v=(s['vertices']@local.T).amin(0)+normal@p[0,i]-offset
            else:
                center=p[0,i]+r[0,i]@s['center'];direction=local@s['rotation']
                support=s['radius'] if s['kind']=='sphere' else s['radius']*direction[:,:2].norm(dim=1)+s['half']*direction[:,2].abs()
                v=normal@center-offset-support
            values.append(v)
        return torch.stack(values)
    query_count=0
    def observe(q,label):
        nonlocal query_count
        raw=query.query_device(q[None].float())
        n=int(raw['count']);fields=('dist','includemargin','type','worldid','active','constraint_allocated','constraint_rows','efc_address','shape0','shape1','body0','body1','source_key','primary_surface','part','task_pair','eligible','upward','full_kind','body_link0','body_link1','normal_w','geometry_point0_w','geometry_point1_w')
        pair={k:raw[k][:n].clone() for k in fields};pair['sample']=torch.zeros(n,dtype=torch.long,device=device)
        observed=dict(schema='newton_device_witness_batch_v1',pairs=pair,link_names=query.tensor_reader.link_names,
            contact_part_mask=raw['contact_part_mask'][:1].clone(),contact_surface=raw['contact_surface'][:1].clone(),contact_position_w=raw['contact_position_w'][:1].clone(),configured_margin=q0.new_tensor([min(query.configured_terrain_includemargins)]))
        # Passive same-pass buffers: no re-run, no collision options changed.
        pipe=query.pipeline;np_=pipe.narrow_phase;contacts=query.contacts
        nb=int(pipe.broad_phase_pair_count.numpy()[0]);nt=int(np_.triangle_pairs_count.numpy()[0]);nc=int(contacts.rigid_contact_count.numpy()[0])
        stage=dict(broad_pairs=pipe.broad_phase_shape_pairs.numpy()[:nb],triangle_pairs=np_.triangle_pairs.numpy()[:nt],
            raw_shape0=contacts.rigid_contact_shape0.numpy()[:nc],raw_shape1=contacts.rigid_contact_shape1.numpy()[:nc],
            raw_to_solver=query.solver._contact_tid_to_cid.numpy()[:nc],shape_aabb_lower=np_.shape_aabb_lower.numpy(),shape_aabb_upper=np_.shape_aabb_upper.numpy())
        torch.save(dict(q=q.detach().cpu(),observed=cpu(observed),stage=stage,label=label,sampling='same fixed-q pass; no integration'),out/'queries'/f'{query_count:04d}.pt')
        observed['query_id']=query_count;query_count+=1
        return observed,stage
    initial,stage0=observe(q0,'original')
    from .step14_midphase import inspect as inspect_midphase
    midphase=[dict(pose='original',**inspect_midphase(query,s)) for s in (69,70,71,72)]
    branch=q0.new_tensor(next(x['q'] for x in saved['records'] if x['label']=='global_single_0.25'))
    observe(branch,'previous_gradient_branch_switch')
    midphase += [dict(pose='previous_gradient_branch_switch',**inspect_midphase(query,s)) for s in (69,70)]
    (out/'midphase_audit.json').write_text(json.dumps(midphase,indent=2))
    initial,_=observe(q0,'original_requery')
    initial_audit=query.audit_current(q0[None].float())
    (out/'initial_audit.json').write_text(json.dumps(plain(initial_audit),indent=2))
    # Material anchors on initially realized primary surfaces. Project embedded
    # witness points to a small distance inside the actual activation margin.
    anchors=[]
    p0,r0=fk.link_poses(q0[None],tuple(initial['link_names']))
    pairs=initial['pairs']
    for part in initial['contact_part_mask'][0].nonzero().flatten().tolist():
        mask=pairs['eligible']&(pairs['part']==part)&(pairs['primary_surface']==initial['contact_surface'][0,part])
        candidates=mask.nonzero().flatten();j=int(candidates[pairs['dist'][candidates].argmin()]);link=int(pairs['body_link1'][j])
        point=pairs['geometry_point1_w'][j].double();local=r0[0,link].T@(point-p0[0,link])
        # All original contacts in this fixture are the actual horizontal top.
        assert abs(float(pairs['normal_w'][j,2])-1)<1e-6
        target=point.clone();target[2]=offset[top]+.1*pairs['includemargin'][j]
        anchors.append(dict(part=part,link=initial['link_names'][link],local=local.detach(),target=target.detach(),margin=float(pairs['includemargin'][j]),surface=int(initial['contact_surface'][0,part])))
    anchor_mode=os.environ.get('STEP14_ANCHOR_MODE','material')
    def anchor_error(q):
        p,r=fk.link_poses(q[None],tuple(a['link'] for a in anchors))
        e=torch.stack([p[0,i]+r[0,i]@a['local']-a['target'] for i,a in enumerate(anchors)])
        if anchor_mode in ('surface','activation_interval'):
            sep=separation(q)
            distances=torch.stack([sep[[j for j,s in enumerate(shapes) if s['name']==a['link']],top].amin() for a in anchors])
            if anchor_mode=='activation_interval':
                # Optimization interval uses actual per-pair margin. It never
                # labels contact: actual active+allocated remains required.
                normal_error=distances.clamp_max(0)+(distances-q0.new_tensor([.9*a['margin'] for a in anchors])).relu()
            else:normal_error=distances-torch.stack([a['target'][2]-offset[top] for a in anchors])
            e=torch.cat((e[:,:2],normal_error[:,None]),dim=1)
        return e.flatten()
    initial_sep=separation(q0).detach();initial_faces=initial_sep[:,:ground].argmax(1)
    foot_ids=[i for i,s in enumerate(shapes) if s['name'].startswith('left_ankle')]
    overlap0=(initial_sep[:,:ground].amax(1)<0)
    initial_required=initial['contact_part_mask'][0]
    records=[]
    def record(label,q,raw,**extra):
        distances=raw['pairs']['dist'];full=raw['pairs']['full_kind']>=0
        depth=max(0.,-float(distances[full].min())) if bool(full.any()) else 0.
        self_mask=raw['pairs']['full_kind']==1
        self_depth=max(0.,-float(distances[self_mask].min())) if bool(self_mask.any()) else 0.
        sep=separation(q).detach();best=sep[:,:ground].amax(1)
        geometric=max(float((-best).relu().max()),float((-sep[:,ground]).relu().max()))
        kept=(raw['contact_part_mask'][0]&initial_required&(raw['contact_surface'][0]==initial['contact_surface'][0]))
        anchor=float(anchor_error(q).reshape(-1,3).norm(dim=1).max())
        joint=float(torch.maximum(fk.joint_lower-q[7:],q[7:]-fk.joint_upper).clamp_min(0).max())
        representative_delta=(raw['contact_position_w'][0]-initial['contact_position_w'][0]).double()
        row=dict(label=label,q=plain(q),query_id=raw['query_id'],newton_depth_cm=depth*100,self_depth_cm=self_depth*100,
            plane_separation_violation_cm=geometric*100,anchor_error_cm=anchor*100,joint_violation_rad=joint,
            actual_contact=plain(raw['contact_part_mask'][0]),actual_surface=plain(raw['contact_surface'][0]),contact_position_w=plain(raw['contact_position_w'][0]),
            required_contact_kept=plain(kept),all_required_kept=bool((kept|~initial_required).all()),
            representative_xy_shift_cm=[float(representative_delta[i,:2].norm())*100 if bool(kept[i]) else None for i in range(6)],
            root_translation_cm=float((q[:3]-q0[:3]).norm())*100,joint_delta_max_deg=float((q[7:]-q0[7:]).abs().max())*180/np.pi,
            static_recovery_pass=bool(depth<=.001 and geometric<=.0002 and joint<=1e-6 and (kept|~initial_required).all() and anchor<=.01),**extra)
        records.append(row)
        print(json.dumps(row|{'q':'saved','contact_position_w':'saved'}),flush=True)
        (out/'progress.json').write_text(json.dumps(plain(dict(records=records)),indent=2))
        return row
    record('original',q0,initial)
    # Face assignment is a disjunctive optimization hypothesis, never a contact label.
    exits=[i for i,n in enumerate(normals) if n[2]>-.5]
    if os.environ.get('STEP14_TOP_ONLY')=='1':exits=[top]
    for face in exits:
        seed=q0.clone();raw=initial;faces=initial_faces.clone();faces[foot_ids]=face
        for a in anchors:
            for i,s in enumerate(shapes):
                if s['name']==a['link']:faces[i]=top
        for it in range(1,41):
            current=raw
            def terms(x):
                q=pose(x,seed);sep=separation(q)
                d=sep[torch.arange(len(shapes),device=device),faces]
                # All original collision shapes are constrained against floor too.
                g=torch.cat((d,sep[:,ground]))/.01
                wr=DeviceWitnessRows(fk,q[None],current)
                sm=current['pairs']['full_kind']==1
                if bool(sm.any()):g=torch.cat((g,wr.distances[sm].double()/.01))
                return torch.cat((g,anchor_error(q)/.005))
            x=torch.zeros(35,dtype=torch.float64,device=device,requires_grad=True)
            t=terms(x);jac=torch.autograd.functional.jacobian(terms,x,vectorize=True)
            d=t[:-len(anchors)*3].detach().cpu().numpy();J=jac[:-len(anchors)*3].detach().cpu().numpy()
            e=t[-len(anchors)*3:].detach().cpu().numpy();E=jac[-len(anchors)*3:].detach().cpu().numpy()
            n=len(d);weight=np.r_[[.5]*6,[.1]*29]
            # Positive slack admits temporarily infeasible local linearizations.
            H=np.diag(weight)+20*E.T@E
            linear=20*E.T@e
            P=sp.block_diag((sp.csc_matrix(H),sp.eye(n)*1e4),format='csc')
            A=sp.vstack((sp.hstack((sp.csc_matrix(J),sp.eye(n))),sp.hstack((sp.csc_matrix((n,35)),sp.eye(n))),sp.hstack((sp.eye(35),sp.csc_matrix((35,n))))),format='csc')
            lower=np.full(35,-1.);upper=np.full(35,1.)
            lower[6:]=np.maximum(lower[6:],((fk.joint_lower-seed[7:])/scale[6:]).cpu().numpy())
            upper[6:]=np.minimum(upper[6:],((fk.joint_upper-seed[7:])/scale[6:]).cpu().numpy())
            solver=osqp.OSQP();solver.setup(P=sp.triu(P,format='csc'),q=np.r_[linear,np.zeros(n)],A=A,
                l=np.r_[-d,np.zeros(n),lower],u=np.r_[np.full(n,np.inf),np.full(n,np.inf),upper],verbose=False,eps_abs=1e-6,eps_rel=1e-6,max_iter=10000,polishing=True)
            sol=solver.solve()
            if sol.x is None or sol.info.status_val not in (1,2):
                record(f'face{face}_qp_failed_{it}',seed,raw,status=sol.info.status);break
            step=q0.new_tensor(sol.x[:35])
            old_merit=1e4*np.square(np.minimum(d,0)).sum()+20*np.square(e).sum()
            accepted=False
            for alpha in (1.,.5,.25,.1,.05):
                q=pose(alpha*step,seed).detach();newraw,_=observe(q,f'face{face}_iter{it}_alpha{alpha}')
                # Recompute actual self witnesses for the nonlinear merit.
                sep=separation(q).detach();gd=torch.cat((sep[torch.arange(len(shapes),device=device),faces],sep[:,ground]))/.01
                sm=newraw['pairs']['full_kind']==1
                if bool(sm.any()):gd=torch.cat((gd,newraw['pairs']['dist'][sm]/.01))
                err=anchor_error(q)/.005
                merit=float(1e4*gd.clamp_max(0).square().sum()+20*err.square().sum())
                if merit<old_merit-1e-7:
                    raw=newraw;seed=q;accepted=True
                    row=record(f'face{face}_iter{it}',seed,raw,exit_face=face,exit_normal=normals[face].tolist(),alpha=alpha,merit=merit)
                    if row['static_recovery_pass']:break
                    break
            if not accepted:
                record(f'face{face}_stalled_{it}',seed,raw,exit_face=face);break
            if row['static_recovery_pass']:break
        # Preserve the actual raw source audit for each final candidate.
        observe(seed,f'face{face}_final_audit')
        (out/f'face{face}_newton_audit.json').write_text(json.dumps(plain(query.audit_current(seed[None].float())),indent=2))
    final=dict(schema='step14_static_multiface_sqp_v1',robot_asset_json=saved['robot_asset_json'],terrain=saved['terrain'],joint_names=saved['joint_names'],
        records=records,provenance=query.provenance,normals=normals.tolist(),offsets=offsets.tolist(),
        anchors=plain(anchors),anchor_mode=anchor_mode,query_count=query_count,canonical_shapes=[dict(name=s['name'],kind=s['kind']) for s in shapes],
        contract='Static recovery only, not dynamics or support proof. Canonical solid-face separation is optimization guidance, never contact truth. Contact evidence uses unmodified Newton primary-face selection. Same-pose pipeline buffers saved before conversion can be overwritten.')
    (out/'diagnostic.json').write_text(json.dumps(plain(final),indent=2))
    print('STEP14_STATIC_RESEARCH_COMPLETE',flush=True)
