"""Conservative separating-axis evidence for finite convex vertex sets.

A positive value proves separation along the supplied axis, after padding.
A nonpositive value is inconclusive; this does not classify solver contacts.
"""
import numpy as np


def vertex_separation(vertices0, vertices1, axis, padding=0.):
    a,b,n=(np.asarray(x,dtype=np.float64) for x in (vertices0,vertices1,axis))
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=3 or b.shape[1]!=3 or not len(a) or not len(b):raise ValueError('Need finite nonempty vertex sets')
    if n.shape!=(3,) or not all(np.isfinite(x).all() for x in (a,b,n)) or not np.isfinite(padding) or padding<0:raise ValueError('Invalid separation input')
    length=np.linalg.norm(n)
    if length<1e-12:raise ValueError('Degenerate axis')
    n=n/length;pa=a@n;pb=b@n
    return float(max(pb.min()-pa.max(),pa.min()-pb.max())-padding)
