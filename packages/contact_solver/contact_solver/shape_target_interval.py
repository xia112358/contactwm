"""Route coherent interval queries through initialized scene identities."""
import dataclasses

import torch

from contact_solver.contact_surface_interval import SurfaceIntervalBounds
from contact_solver.interval_identity import SurfaceSolidOwners
from contact_solver.material_surface_interval import material_surface_interval


SHAPE_TARGET_INTERVAL_SCHEMA='coherent_shape_target_material_query_interval_v2'


class ShapeTargetIntervalProvider:
    """Geometry adapters share a query protocol; solver contact stays separate."""
    def __init__(self, metadata, solid_router, *, geometry_query=material_surface_interval):
        if set(metadata)!=set(solid_router.scenes):
            raise ValueError('Interval and solid scene routes differ')
        self.owners={sid:SurfaceSolidOwners(meta,solid_router.scenes[sid]) for sid,meta in metadata.items()}
        self.link_names=tuple(solid_router.link_names)
        self.geometry_query=geometry_query

    def __call__(self, model, rows, active, surface, scene, scene_ids):
        if scene_ids.shape!=(len(rows.q),) or scene_ids.dtype!=torch.long:
            raise ValueError('Interval scene route must identify every sample')
        if tuple(rows.observed['link_names'])!=self.link_names:
            raise ValueError('Interval and actual solid body mappings differ')
        solid=getattr(rows,'solid',None)
        if solid is None:raise ValueError('Interval requires actual complete-solid queries')
        bounds=self.geometry_query(model.fk,model.region_geometry,rows.q,surface,scene,
            rows.observed['configured_margin'])
        if not isinstance(bounds,SurfaceIntervalBounds):
            raise ValueError('Geometry adapter must return coherent interval queries')
        pair=solid.pair
        mask=torch.zeros(len(pair['sample']),2,dtype=torch.bool,device=rows.q.device)
        for sid in scene_ids.unique().tolist():
            if sid not in self.owners:raise ValueError('Unknown initialized interval scene')
            chosen=scene_ids[pair['sample']]==sid
            subset={key:value[chosen] for key,value in pair.items()}
            mask[chosen]=self.owners[sid].pair_sides(subset,surface,active)
        return dataclasses.replace(bounds,solid_target_mask=mask)

    def contract(self):
        return dict(schema=SHAPE_TARGET_INTERVAL_SCHEMA,
            scene_fingerprints={str(sid):owner.fingerprint for sid,owner in self.owners.items()},
            geometry_adapter=self.geometry_query.__module__+'.'+self.geometry_query.__name__,
            shared_constraint='same selected unique interior FK material, static component, link/local point/normal/signed gap',
            query_reduction='complete native/material interval scalars; no cross-query bound mixing',
            independent_constraints='other physical witnesses, self collision and release retained',
            contact_truth='unchanged actual Newton activation/allocation and primary-face selection',
            newton_assets_margin_weights_architecture_changed=False)
