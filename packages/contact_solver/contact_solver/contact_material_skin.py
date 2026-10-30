"""Loss-only contact affordances from actual training Newton material points.

The bank contains link-local geometry, not world targets or corrected poses.
It is fixed before optimization; contact labels still come exclusively from
the current Newton solver. No pose-dependent normal filter switches points in
and out of the scalar objective.
"""
import torch
from torch import nn
from contact_solver.contact_surface_interval import CONTACT_SURFACE_INTERVAL_SCHEMA, finite_face_band_cost


class ContactMaterialSkin(nn.Module):
    schema = 'training_newton_contact_material_skin_v1'

    def __init__(self, regions, link_names):
        super().__init__()
        self.link_names = tuple(link_names)
        self.region_names = regions.names
        self.register_buffer('known', torch.zeros_like(regions.valid), persistent=False)
        self._pending = {}
        self.source_ids = set()
        self.source_contact_count = 0
        self.finalized = False

    def add(self, fk, regions, q, observed, source_ids, selected):
        """Collect allocated primary-face contacts of explicitly selected rows.

        The caller selects the training split, including when one solver batch
        mixes training and validation rows. No held-out material enters the
        bank. Source poses are used only to bind witnesses to their own links.
        """
        if self.finalized:
            raise ValueError('Contact material skin is already finalized')
        if observed.get('schema') != 'newton_device_witness_batch_v1':
            raise ValueError('Contact materials require actual device Newton witnesses')
        if tuple(observed['link_names']) != self.link_names:
            raise ValueError('Contact material source link mapping changed')
        if selected.shape != (len(q),) or selected.dtype != torch.bool or len(source_ids) != len(q):
            raise ValueError('Contact skin needs explicit source rows and training selection')
        p = observed['pairs']
        in_training = selected.to(p['sample'].device)[p['sample']]
        if bool((in_training & p['active'] & ~p['constraint_allocated']).any()):
            raise ValueError('Unallocated active source Newton contact')
        eligible = p['eligible'] & in_training
        if bool((eligible & (~p['active'] | ~p['constraint_allocated'] |
                             (p['body_link0'] >= 0) | (p['body_link1'] < 0) |
                             ~p['task_pair'] | ~p['upward'])).any()):
            raise ValueError('Source material is not allocated primary terrain contact')
        sample, part, link = (p[key][eligible].to(q.device) for key in ('sample', 'part', 'body_link1'))
        point = p['geometry_point1_w'][eligible].to(q)
        if not bool(torch.isfinite(point).all()) or bool(((part < 0) | (part >= 6)).any()):
            raise ValueError('Invalid contact source material')
        if bool((link >= len(self.link_names)).any()):
            raise ValueError('Contact source link index out of range')
        with torch.no_grad():
            position, rotation = fk.link_poses(q.detach(), self.link_names)
            local = torch.einsum('nji,nj->ni', rotation[sample, link], point-position[sample, link])
            region = regions.witness_regions(fk, q, sample, part, point)
        for body in range(6):
            for r in range(len(self.region_names[body])):
                mask = (part == body) & (region == r)
                if bool(mask.any()):
                    self._pending.setdefault((body, r), []).append((link[mask].detach(), local[mask].detach()))
        self.source_ids.update(source_ids[selected.to(source_ids.device)].detach().cpu().tolist())
        self.source_contact_count += int(eligible.sum())

    def finalize(self):
        """Deduplicate exact features and freeze the bank, without quantization."""
        if self.finalized:
            raise ValueError('Contact material skin is already finalized')
        for (part, region), records in self._pending.items():
            links = torch.cat([record[0] for record in records])
            local = torch.cat([record[1] for record in records])
            packed = torch.unique(torch.cat((links[:, None].to(local), local), -1), dim=0)
            self.register_buffer(f'link_{part}_{region}', packed[:, 0].long(), persistent=False)
            self.register_buffer(f'local_{part}_{region}', packed[:, 1:], persistent=False)
            self.known[part, region] = True
        self._pending.clear()
        self.finalized = True

    def region_distance(self, fk, q, surface, scene, margin, fallback):
        """Continuous FK approach distance to the requested finite surface.

        Unknown affordances retain the asset-based approach proxy and are
        explicitly reported by ``known``. Neither proxy establishes contact.
        """
        if not self.finalized:
            raise ValueError('Contact material skin must be finalized before loss evaluation')
        result = fallback.clone()
        position, rotation = fk.link_poses(q, self.link_names)
        top_center = scene['box_center']+scene['box_rotation'][:, :, 2]*scene['box_half_extents'][:, 2:3]
        for part in range(6):
            for region in range(len(self.region_names[part])):
                if not bool(self.known[part, region]):
                    continue
                links = getattr(self, f'link_{part}_{region}')
                local = getattr(self, f'local_{part}_{region}').to(q)
                points = position[:, links]+torch.einsum('bnij,nj->bni', rotation[:, links], local)
                ground_gap = points[..., 2]-scene['ground_height'][:, None]
                box_local = torch.einsum('bvi,bij->bvj', points-top_center[:, None], scene['box_rotation'])
                outside = (box_local[..., :2].abs()-scene['box_half_extents'][:, None, :2]).relu()
                ground = surface[:, part, None] == 0
                gap = torch.where(ground, ground_gap, box_local[..., 2])
                cost = finite_face_band_cost(gap, torch.where(ground[..., None], 0, outside), margin[:, None])
                result[:, part, region] = cost.amin(-1)/(.25*margin).square()
        return result

    def contract(self):
        return dict(schema=self.schema, source='actual activated+allocated primary upward terrain Newton contacts from training rows only',
            link_names=list(self.link_names),
            source_ids=sorted(self.source_ids), source_contact_count=self.source_contact_count,
            known=self.known.cpu().tolist(),
            material_counts=[[len(getattr(self, f'link_{p}_{r}')) if bool(self.known[p, r]) else 0
                              for r in range(4)] for p in range(6)],
            geometry='fixed actual-link-local material features; native-width finite-face interval including edges; exact duplicate removal; no world target pose',
            interval_schema=CONTACT_SURFACE_INTERVAL_SCHEMA,
            unknown_policy='reported asset approach proxy; never a contact label',
            contact_truth_unchanged=True, network_input=False, qp_or_projection=False)
