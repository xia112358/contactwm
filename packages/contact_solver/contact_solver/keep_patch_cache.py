"""Run-local material patches, keyed by exact pose and immutable scene identity.

Only allocated Newton witnesses enter this cache. It caches the input geometry,
never predicted distances, losses, gradients, or contact acceptance decisions.
"""
import torch

from .keep_patch_loss import KeepPatch, bind_keep_patch


class KeepPatchCache:
    def __init__(self, size, pose_template, link_names):
        self.q = pose_template.new_zeros((size, pose_template.shape[-1]))
        self.scene = torch.zeros(size, dtype=torch.long, device=self.q.device)
        self.ready = torch.zeros(size, dtype=torch.bool, device=self.q.device)
        self.count = torch.zeros_like(self.scene)
        self.link_names = tuple(link_names)
        self.fields = {}
        self.width = 0
        self.query_samples = 0
        self.hit_samples = 0
        self.source_samples = 0

    @torch.no_grad()
    def put(self, keys, q, scenes, patch):
        """Replace complete patches, including valid states with zero contacts."""
        if patch.link_names != self.link_names or patch.batch_size != len(keys):
            raise ValueError('Keep cache patch identity/batch mismatch')
        if len(keys.unique()) != len(keys):
            raise ValueError('Keep cache writes require unique state keys')
        counts = torch.bincount(patch.sample, minlength=len(keys))
        width = max(self.width, int(counts.max()) if len(keys) else 0)
        for name in ('part', 'link', 'local', 'position', 'normal'):
            value = getattr(patch, name)
            if name not in self.fields or width != self.width:
                expanded = value.new_zeros((len(self.q), width, *value.shape[1:]))
                if name in self.fields:
                    expanded[:, :self.width] = self.fields[name]
                self.fields[name] = expanded
        self.width = width
        order = torch.argsort(patch.sample, stable=True)
        sample = patch.sample[order]
        starts = counts.cumsum(0)-counts
        column = torch.arange(len(sample), device=sample.device)-starts[sample]
        for name, storage in self.fields.items():
            storage[keys] = 0
            storage[keys[sample], column] = getattr(patch, name)[order].detach()
        self.q[keys], self.scene[keys], self.count[keys] = q.detach(), scenes, counts
        self.ready[keys] = True

    def select(self, keys):
        # Duplicate/reordered read keys intentionally produce separate samples.
        valid = torch.arange(self.width, device=self.q.device)[None] < self.count[keys, None]
        sample = valid.nonzero()[:, 0]
        fields = {name: value[keys][valid] for name, value in self.fields.items()}
        return KeepPatch(sample=sample, **fields, link_names=self.link_names, batch_size=len(keys))

    @torch.no_grad()
    def get(self, keys, q, scenes, fk, query, *, source=None, source_keys=None):
        if not len(keys):
            raise ValueError('Keep cache requires a nonempty state batch')
        matches = self.ready[keys] & (self.scene[keys] == scenes) & (self.q[keys] == q).all(-1)
        missing = (~matches).nonzero().flatten()
        self.hit_samples += len(keys)-len(missing)
        if len(missing):
            # Reference batches can repeat a row. Query each missing key once;
            # callers must associate one pose/scene with each key in a batch.
            unique, inverse = keys[missing].unique(return_inverse=True)
            first = torch.full_like(unique, len(missing))
            first.scatter_reduce_(0, inverse, torch.arange(len(missing), device=q.device),
                                  reduce='amin', include_self=True)
            rows = missing[first]
            if not torch.equal(q[missing], q[rows][inverse]) or not torch.equal(scenes[missing], scenes[rows][inverse]):
                raise ValueError('One keep cache key refers to different states in a batch')
            source_rows = source_keys[rows] if source is not None else None
            reusable = (source is not None and bool((source.ready[source_rows]
                & (source.scene[source_rows] == scenes[rows])
                & (source.q[source_rows] == q[rows]).all(-1)).all()))
            if reusable:
                patch = source.select(source_rows)
                self.source_samples += len(rows)
            else:
                observed = query(q[rows].detach(), scenes[rows])
                patch = bind_keep_patch(fk, q[rows], observed)
                self.query_samples += len(rows)
            self.put(unique, q[rows], scenes[rows], patch)
            if not torch.equal(self.q[keys], q) or not torch.equal(self.scene[keys], scenes):
                raise ValueError('One keep cache key refers to different states in a batch')
        return self.select(keys)

    def storage_bytes(self):
        tensors = (self.q, self.scene, self.ready, self.count, *self.fields.values())
        return sum(value.numel()*value.element_size() for value in tensors)
