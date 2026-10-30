"""Expand a fresh MuJoCo-Warp data template on GPU, without batched host copies.

Rigid data is copied from one upstream-initialized template, including contacts.
CPU execution, flex models and explicit global capacities use upstream.
No installed dependency files or physics settings are modified.
"""
import dataclasses
import functools
import os


def install():
    import mujoco_warp
    import warp as wp
    if os.environ.get('SUPERMIMIC_FAST_INIT', '1') == '0':
        return
    if getattr(mujoco_warp.put_data, '_supermimic_fast', False):
        return
    original = mujoco_warp.put_data

    @functools.wraps(original)
    def put_data(mjm, mjd, nworld=1, *args, **kwargs):
        if args:
            return original(mjm, mjd, nworld, *args, **kwargs)
        device = wp.get_device()
        unsupported = (not device.is_cuda or nworld <= 1
                       or mjm.nflex or kwargs.get('naconmax') is not None or kwargs.get('naccdmax') is not None)
        if unsupported:
            return original(mjm, mjd, nworld=nworld, **kwargs)
        import time
        import torch
        from mujoco_warp._src.warp_util import mark_batched
        started = time.perf_counter()
        contacts=int(mjd.ncon)
        template = original(mjm, mjd, nworld=1, **kwargs)
        wp.synchronize_device(device)
        arrays = 0
        allocated = 0

        def expand(obj, path=''):
            nonlocal arrays, allocated
            values = {}
            for field in dataclasses.fields(obj):
                value = getattr(obj, field.name)
                if dataclasses.is_dataclass(value):
                    value = expand(value,path+field.name+'.')
                elif isinstance(value, wp.array):
                    spec = getattr(field.type, 'shape', ())
                    if spec and spec[0] in ('nworld', 'naconmax', 'naccdmax'):
                        shape = (value.shape[0]*nworld,) + value.shape[1:]
                        output = wp.empty(shape, dtype=value.dtype, device=device)
                        if value.size:
                            src = wp.to_torch(value).reshape(1, -1)
                            dst = wp.to_torch(output).reshape(nworld, -1)
                            if path == 'contact.':
                                # Upstream packs all active contacts contiguously,
                                # then pads to the GLOBAL capacity (not per world).
                                target=wp.to_torch(output)
                                target.fill_(-1 if field.name=='efc_address' else 1 if field.name=='type' else 0)
                                if contacts:
                                    active=target[:contacts*nworld]
                                    source=wp.to_torch(value)[:contacts]
                                    active.reshape(nworld,contacts,-1).copy_(source.reshape(1,contacts,-1).expand(nworld,-1,-1))
                                    if field.name=='worldid':
                                        active.reshape(nworld,contacts).copy_(torch.arange(nworld,device=target.device)[:,None].expand(-1,contacts))
                            else:
                                dst.copy_(src.expand_as(dst))
                            allocated += dst.numel()*dst.element_size()
                        value = output
                        arrays += 1
                elif field.name in ('nworld', 'naconmax', 'naccdmax'):
                    value *= nworld
                values[field.name] = value
            return type(obj)(**values)

        with torch.cuda.device(device.ordinal):
            # Keep the template alive until every asynchronous broadcast completes.
            result = expand(template)
            torch.cuda.current_stream().synchronize()
        wp.to_torch(result.nacon).mul_(nworld)
        torch.cuda.synchronize(device.ordinal)
        mark_batched(result)
        print(f'Fast MuJoCo data init: {time.perf_counter()-started:.3f}s; '
              f'{nworld} worlds, {arrays} arrays, {allocated/2**20:.1f} MiB expanded on GPU', flush=True)
        return result

    put_data._supermimic_fast = True
    put_data.original = original
    mujoco_warp.put_data = put_data
