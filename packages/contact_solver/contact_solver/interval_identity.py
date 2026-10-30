"""Bind optimization intervals to actual complete-solid component identities.

This identifies geometry only. It never changes Newton contact activation,
allocation, primary faces, margins, shape assets or robot pose.
"""
import numpy as np
import torch
from scipy.spatial import ConvexHull
from somaforge_core import CONTACT_BODY_NAMES_BY_PART
from somaforge_core.contact_face_selection import upward_face_mask


class SurfaceSolidOwners:
    def __init__(self, metadata, solid_scene):
        if metadata['solid_geometry']['model_fingerprint'] != solid_scene.fingerprint:
            raise ValueError('Surface and complete-solid scene identities differ')
        self.fingerprint=solid_scene.fingerprint
        self.link_names=solid_scene.link_names
        shapes={s['shape']:s for s in metadata['solid_geometry']['shapes']}
        self.records={(r['shape'],r['component']):r for r in solid_scene.records}
        self.record_links=np.full((max(k[0] for k in self.records)+1,max(k[1] for k in self.records)+1),-2,dtype=np.int64)
        for key,record in self.records.items():self.record_links[key]=record['link']
        static=[]
        for record in solid_scene.records:
            if record['link']>=0:continue
            shape=shapes[record['shape']]
            if shape['kind'] in ('mesh','convex_mesh'):
                vertices=np.asarray(record['geometry'].points())
            elif shape['kind']=='box':
                import itertools
                vertices=np.asarray(list(itertools.product((-1.,1.),repeat=3)))*np.asarray(shape['scale'])
            else:
                # Curved shapes cannot own this finite polygon-face catalog.
                continue
            vertices=vertices@record['rotation'].T+record['position']
            static.append((record,vertices,ConvexHull(vertices).equations))
        self.owners={}
        catalog=metadata['surface_catalog']
        selected=upward_face_mask([f['normal_w'] for f in catalog])
        for face,keep in zip(catalog,selected,strict=True):
            if not keep:continue
            normal=np.asarray(face['normal_w'],float)
            normal/=np.linalg.norm(normal)
            points=np.asarray(face['triangles_w'],float).reshape(-1,3)
            matches=[]
            for record,vertices,planes in static:
                # Binding tolerance covers float32 source/transform rounding;
                # it is never a contact or penetration tolerance.
                magnitude=max(1.,float(np.abs(np.r_[vertices.ravel(),points.ravel()]).max()))
                tolerance=128*np.finfo(np.float32).eps*magnitude
                on_support=np.abs(points@normal-(vertices@normal).max())<=tolerance
                inside=(points@planes[:,:3].T+planes[:,3]<=tolerance).all()
                if on_support.all() and inside:matches.append((record['shape'],record['component']))
            if len(matches)!=1:
                raise ValueError('Target face needs one unambiguous actual static solid component')
            self.owners[int(face['surface'])]=matches[0]
        if not self.owners:raise ValueError('No actual primary upward target faces')
        by_link={name:part for part,key in enumerate(('left_foot','right_foot','left_hand','right_hand','left_knee','right_knee'))
                 for name in CONTACT_BODY_NAMES_BY_PART[key]}
        self.part_by_link=tuple(by_link.get(name,-1) for name in self.link_names)

    def pair_sides(self,pair,surface,active):
        """Return identity matches, never geometric/contact eligibility."""
        required=('sample','shape0','shape1','component0','component1','body_link0','body_link1','full_kind')
        if any(k not in pair for k in required):
            raise ValueError('Missing actual complete-solid pair identities')
        sample=pair['sample'];count=len(sample);result=torch.zeros(count,2,dtype=torch.bool,device=sample.device)
        if surface.shape!=active.shape or surface.shape[1]!=6:
            raise ValueError('Interval intent must retain part and surface identity')
        for key in required:
            if pair[key].shape!=(count,):raise ValueError('Complete-solid identity row mismatch')
        record_links=torch.as_tensor(self.record_links,device=sample.device)
        part_map=torch.tensor(self.part_by_link,device=sample.device)
        for side in (0,1):
            body=pair[f'body_link{side}'];other=1-side
            shape=pair[f'shape{side}'];component=pair[f'component{side}']
            inside=(shape>=0)&(shape<record_links.shape[0])&(component>=0)&(component<record_links.shape[1])
            if not bool(inside.all()) or not bool((record_links[shape,component]==body).all()):
                raise ValueError('Complete-solid shape/component/link identity mismatch')
            if bool(((body < -1)|(body>=len(self.part_by_link))).any()):
                raise ValueError('Complete-solid link identity is outside the scene')
            part=part_map[body.clamp_min(0)]
            relevant=(body>=0)&(part>=0)&(pair[f'body_link{other}']<0)&(pair['full_kind']==0)
            intent=active[sample,part.clamp_min(0)]
            for face,(shape,component) in self.owners.items():
                result[:,side]|=(relevant & intent & (surface[sample,part.clamp_min(0)]==face)
                    & (pair[f'shape{other}']==shape)&(pair[f'component{other}']==component))
        unknown=active & ~torch.stack([surface==face for face in self.owners]).any(0)
        if bool(unknown.any()):raise ValueError('Unknown or nonprimary intended target face')
        return result
