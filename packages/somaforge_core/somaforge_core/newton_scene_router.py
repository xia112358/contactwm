"""Route native-world queries by actual model fingerprint, never network input."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import numpy as np
from .newton_contact_query import query_contacts


class NewtonSceneRouter:
    def __init__(self, routes, *, query=query_contacts, workers=3):
        self.routes = {}
        for fingerprint, endpoints in routes.items():
            if isinstance(endpoints, str):
                endpoints = (endpoints,)
            else:
                endpoints = tuple(endpoints)
            if not endpoints or any(not isinstance(endpoint, str) or not endpoint for endpoint in endpoints):
                raise ValueError('Each Newton route needs at least one endpoint')
            self.routes[fingerprint] = endpoints
        self.query = query
        self.workers = max(1, int(workers))
        for fingerprint in self.routes:
            if len(fingerprint) != 64 or len(bytes.fromhex(fingerprint)) != 32:
                raise ValueError('Invalid actual model fingerprint')

    def __call__(self, q, fingerprints):
        q = np.asarray(q)
        if q.ndim != 2 or q.shape[1] != 36 or len(q) != len(fingerprints) or not len(q):
            raise ValueError('Invalid routed query batch')
        groups = {}
        for i, fingerprint in enumerate(fingerprints):
            if fingerprint not in self.routes:
                raise ValueError(f'No authoritative Newton route for {fingerprint}')
            groups.setdefault(fingerprint, []).append(i)
        jobs = []
        for fp, indices in groups.items():
            endpoints = self.routes[fp]
            for endpoint, shard in zip(endpoints, np.array_split(indices, min(len(endpoints), len(indices)))):
                if len(shard):
                    jobs.append((fp, shard.tolist(), endpoint))
        def run(item):
            fp, indices, endpoint = item
            result = self.query(q[indices], None, endpoint=endpoint)
            if result.get('provenance', {}).get('model_fingerprint') != fp:
                raise ValueError('Newton route returned a different actual model fingerprint')
            if 'candidate_pairs' not in result:
                raise ValueError('Newton route lacks witness candidates')
            return indices, result
        with ThreadPoolExecutor(max_workers=min(self.workers, len(jobs))) as pool:
            results = list(pool.map(run, jobs))
        # Catalogs are per sample: different scenes must not inherit the first
        # worker's planes or shape mapping. Restore original sample order.
        output = {k:[None]*len(q) for k in ('active','unallocated','position_w','surface',
            'pairs','candidate_pairs','provenance_by_sample','surface_catalog_by_sample',
            'configured_terrain_includemargins_by_sample')}
        schema = parts = sampling = attribution = margin_source = None
        has_full=[('full_robot_separation' in r) for _,r in results]
        if any(has_full) and not all(has_full):raise ValueError('Mixed full-body query capability')
        if all(has_full):output['full_robot_separation']=[None]*len(q)
        for indices, result in results:
            if schema is not None and (result['schema'] != schema or result['parts'] != parts
                                       or result['sampling'] != sampling
                                       or result.get('surface_attribution_schema') != attribution
                                       or result.get('configured_margin_source') != margin_source):
                raise ValueError('Mixed Newton semantics or sampling in routes')
            schema, parts, sampling = result['schema'], result['parts'], result['sampling']
            attribution = result.get('surface_attribution_schema')
            margin_source = result.get('configured_margin_source')
            configured_margins = result.get('configured_terrain_includemargins')
            if margin_source != 'realized_mjwarp_contact_params_margin_sum_v2':
                raise ValueError('Newton route lacks realized configured contact margins')
            if not configured_margins or any(float(value) <= 0.0 for value in configured_margins):
                raise ValueError('Newton route lacks positive configured contact margins')
            for local, original in enumerate(indices):
                for key in ('active','unallocated','position_w','surface','pairs','candidate_pairs'):
                    if len(result[key]) != len(indices):raise ValueError('Routed response batch mismatch')
                    output[key][original] = result[key][local]
                output['provenance_by_sample'][original] = result['provenance']
                output['surface_catalog_by_sample'][original] = result['surface_catalog']
                output['configured_terrain_includemargins_by_sample'][original] = list(configured_margins)
                if all(has_full):
                    if len(result['full_robot_separation'])!=len(indices):raise ValueError('Full-body batch mismatch')
                    output['full_robot_separation'][original]=result['full_robot_separation'][local]
        for key in ('active','unallocated','position_w','surface'):
            output[key] = np.asarray(output[key])
        return dict(output, schema=schema, parts=parts, sampling=sampling,
            surface_attribution_schema=attribution, configured_margin_source=margin_source)


def configured_router():
    path = os.environ.get('SOMAFORGE_NEWTON_SCENE_ROUTES')
    if not path:raise RuntimeError('Set SOMAFORGE_NEWTON_SCENE_ROUTES; no single-scene fallback')
    config = json.loads(Path(path).read_text())
    if config.get('schema') != 'newton_native_scene_routes_v1':
        raise ValueError('Unknown Newton scene route schema')
    return NewtonSceneRouter(config['routes'], workers=config.get('workers',3))
