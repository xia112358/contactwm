"""Exact spatial broad phase for native finite-face attribution.

Bounding spheres only prune triangles that cannot be nearest. They never decide
contact activation. Final distances/ties use the existing native face semantics.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .newton_contacts import triangle_distance_squared


class NativeSurfaceIndex:
    def __init__(self, faces):
        self.faces = faces
        triangles, owners = [], []
        for index, face in enumerate(faces):
            values = np.asarray(face["triangles_w"], np.float64)
            triangles.extend(values)
            owners.extend([index] * len(values))
        self.triangles = np.asarray(triangles)
        if not len(self.triangles) or not np.isfinite(self.triangles).all():
            raise ValueError("Missing finite native triangle geometry")
        self.owners = np.asarray(owners)
        centers = self.triangles.mean(1)
        radii = np.linalg.norm(self.triangles-centers[:, None], axis=-1).max(-1)
        if (radii <= 0).any():
            raise ValueError("Degenerate native triangle")
        self.tree = cKDTree(centers)
        # Radius buckets prevent a large ground triangle from forcing a search
        # over every small box triangle in a multi-motion shared terrain mesh.
        bucket = np.floor(np.log2(radii)).astype(np.int64)
        self.buckets = []
        for key in np.unique(bucket):
            ids = np.flatnonzero(bucket == key)
            self.buckets.append((ids, cKDTree(centers[ids]), float(radii[ids].max())))
        self.normals = np.asarray([f["normal_w"] for f in faces], np.float64)
        self.surface_ids = np.asarray([f["surface"] for f in faces], np.int64)

    def query(self, points, normals, *, batch_size=4096):
        points, normals = np.asarray(points, np.float64), np.asarray(normals, np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,) or normals.shape != points.shape:
            raise ValueError("Invalid native surface witness shape")
        if not np.isfinite(points).all() or not np.isfinite(normals).all():
            raise ValueError("Nonfinite native surface witness")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Invalid native surface query batch size")
        output = []
        triangle_count, face_count = len(self.triangles), len(self.faces)
        for begin in range(0, len(points), batch_size):
            point = points[begin:begin+batch_size]
            normal = normals[begin:begin+batch_size]
            rows = np.arange(len(point), dtype=np.int64)
            nearest = self.tree.query(point)[1]
            # The existing distance function also accepts paired [N,3] points
            # and [N,3,3] triangles. Its finite-triangle arithmetic is unchanged.
            best = triangle_distance_squared(point, self.triangles[nearest])
            # Match actual_primary_face_indices' squared-distance tie rule.
            bound = np.sqrt(best + 1e-12 + 1e-9*np.abs(best))
            candidates = [rows*triangle_count+nearest]
            for ids, tree, radius in self.buckets:
                neighbors = tree.query_ball_point(point, np.nextafter(bound+radius, np.inf))
                sizes = np.fromiter((len(values) for values in neighbors), np.int64, count=len(rows))
                if sizes.any():
                    candidate_rows = np.repeat(rows, sizes)
                    candidate_triangles = ids[np.concatenate(neighbors).astype(np.int64, copy=False)]
                    candidates.append(candidate_rows*triangle_count+candidate_triangles)
            pairs = np.unique(np.concatenate(candidates))
            candidate_rows, triangles = pairs//triangle_count, pairs%triangle_count
            distance = triangle_distance_squared(point[candidate_rows], self.triangles[triangles])
            starts = np.flatnonzero(np.r_[True, np.diff(candidate_rows) != 0])
            minimum = np.minimum.reduceat(distance, starts)
            tied = np.isclose(distance, minimum[candidate_rows], rtol=1e-9, atol=1e-12)
            # De-duplicate owners exactly as the scalar path did, preserving
            # surface-ID and owner-index ordering for coincident faces.
            face_pairs = np.unique(candidate_rows[tied]*face_count+self.owners[triangles[tied]])
            face_rows, owners = face_pairs//face_count, face_pairs%face_count
            score = np.einsum('ij,ij->i', self.normals[owners], normal[face_rows])
            order = np.lexsort((owners, self.surface_ids[owners], -score, face_rows))
            first = np.r_[True, np.diff(face_rows[order]) != 0]
            winners = owners[order[first]]
            top = score[order[first]]
            # A numerical ambiguity only selects the computation route. Use
            # the original scalar matmul/lexsort for near-equal normal scores;
            # no tolerance is added to the primary-face selection itself.
            eps = 32*np.finfo(np.float64).eps
            ambiguous = np.isclose(score, top[face_rows], rtol=eps, atol=eps)
            ambiguous &= np.any(self.normals[owners] != self.normals[winners[face_rows]], axis=1)
            for row in np.unique(face_rows[ambiguous]):
                a, b = np.searchsorted(face_rows, [row, row+1])
                choices = owners[a:b]
                exact = self.normals[choices] @ normal[row]
                winners[row] = choices[np.lexsort((self.surface_ids[choices], -exact))[0]]
            output.append(winners)
        return np.concatenate(output) if output else np.empty(0, np.int64)
