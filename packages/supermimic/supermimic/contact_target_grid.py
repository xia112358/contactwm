"""Nine spatial commands for one validated transfer, preserving native demonstrations."""
import copy
import torch


def expand_target_grid(clock, span, segment):
    """Copy source history; offset only new-contact endpoint commands in world XY.

    The returned data is a command view, never contact ground truth. Physical
    states, actions, sensor evidence, timing and root targets stay unchanged.
    Material-point calibration must continue to use the unmodified recording.
    """
    if span <= 0 or not 0 <= segment < len(clock.event_ends):
        raise ValueError('Positive grid span and a valid source segment required')
    end = clock.event_ends[segment] - 1
    new = clock.target_masks[end] & ~clock.keep_masks[end]
    if not bool(new.any()):
        raise ValueError('Nine-point task requires a new contact')
    length = clock.length
    dev = clock.event_ends.device
    offsets = torch.tensor([(x, y, 0.) for x in (-span, 0., span)
                            for y in (-span, 0., span)], device=dev)
    shift = torch.arange(9, device=dev) * length
    result = copy.copy(clock)
    result.data = {k: v.repeat((9,) + (1,) * (v.ndim - 1)) for k, v in clock.data.items()}
    result.length = length * 9
    result.segment_starts = clock.segment_starts[segment] + shift
    result.segment_stops = clock.segment_stops[segment] + shift
    result.event_ends = clock.event_ends[segment] + shift
    result.segment_motion = clock.segment_motion[segment].repeat(9)
    result.case_ids = clock.case_ids[segment].repeat(9)
    result.segment_local_bounds = [clock.segment_local_bounds[segment]] * 9
    result.bounds = None
    result.ends = torch.cat([clock.ends + i * length for i in range(9)])
    result.frame_source_floor = torch.cat([clock.frame_source_floor + i * length for i in range(9)])
    for key in ('target_masks', 'keep_masks', 'support_masks'):
        setattr(result, key, getattr(clock, key).repeat(9, 1))
    result.data['point'][result.event_ends - 1] += offsets[:, None] * new[None, :, None]
    result.target_grid = dict(span_m=span, offsets_w=offsets.cpu().tolist(),
        source_segment=segment + 1, source_case_id=int(clock.case_ids[segment]),
        source_motion=int(clock.segment_motion[segment]), new_parts=new.nonzero().flatten().tolist(),
        target_points_w=result.data['point'][result.event_ends - 1].cpu().tolist(),
        semantics='command offsets only; original physical states and calibrated material points unchanged')
    return result
