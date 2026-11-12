"""Keep achieved contact during denoising, without changing Newton activation."""
import numpy as np


def support_reference(consensus,reports,masks,surfaces,fused_mask,fused_surface,source_frame_maps,source_indices):
    # Transport full measured source episodes, not the single fragment picked
    # by an audit of interpolated poses.
    qualified=np.zeros_like(masks,dtype=bool)
    for event in consensus['events']:
        part,face=event['part'],event['surface']
        for j,index in enumerate(source_indices):
            observation=event['observations'].get(str(index))
            if observation is None:continue  # This execution did not observe this population event.
            a,b=observation['interval'];clock=source_frame_maps[j]
            qualified[j,:,part] |= (clock>=a)&(clock<b)&(fused_surface[:,part]==face)
    actual=masks&(surfaces==fused_surface[None])
    wanted=(qualified.sum(0)*2>len(masks))&(actual.sum(0)*2>len(masks))&fused_mask
    return dict(wanted=wanted,faces=fused_surface)


def support_loss(reference,mask,surface):
    lost=reference['wanted']&~(mask&(surface==reference['faces']))
    return dict(schema='achieved_population_support_preservation_v1',
        required_part_frames=int(reference['wanted'].sum()),lost_part_frames=int(lost.sum()),
        passed=not bool(lost.any()),lost_frame_parts=np.argwhere(lost).tolist(),
        evidence='same-clock native fusion contacts corroborated by majority aligned sources within support events')
