"""One evidence contract for contact, load, slip and declared keep intent.

Load-bearing contacts may slide. A geometric pivot or a persistent contact
intent never establishes measured load-bearing or stationary support.
"""
import numpy as np

SUPPORT_ASSESSMENT_SCHEMA = 'newton_contact_load_slip_assessment_v1'


def assess_support(contact, normal_force=None, tangent_speed=None, *, load_known=None,
                   speed_limit=None, keep_intent=None):
    contact=np.asarray(contact,bool)
    shape=contact.shape
    known=np.zeros(shape,bool) if load_known is None else np.asarray(load_known,bool)
    if known.shape!=shape:raise ValueError('Support evidence dimensions mismatch')
    force=np.full(shape,np.nan) if normal_force is None else np.asarray(normal_force,float)
    if force.shape!=shape or np.any(known & (~np.isfinite(force) | (force<0))):
        raise ValueError('Missing or invalid observed contact load')
    if np.any(known & ~contact & (force>0)):
        raise ValueError('Load cannot create contact activation')
    force=np.where(known,force,np.nan)
    bearing=known & contact & (force>0)
    state=np.full(shape,'unknown',dtype='<U20')
    state[known & ~contact]='no_contact'
    state[known & contact & ~bearing]='unloaded'
    state[bearing]='load_bearing'
    speed=np.full(shape,np.nan) if tangent_speed is None else np.asarray(tangent_speed,float)
    if speed.shape!=shape:raise ValueError('Support speed dimensions mismatch')
    if np.any(bearing & ((np.isfinite(speed) & (speed<0)) | np.isinf(speed))):
        raise ValueError('Invalid observed loaded-point speed')
    measured=bearing & np.isfinite(speed) & (speed>=0)
    speed=np.where(measured,speed,np.nan)
    slip=np.full(shape,'unknown',dtype='<U40')
    slip[known & ~bearing]='not_load_bearing'
    slip[measured]='measured_without_acceptance_budget'
    if speed_limit is not None:
        limit=np.broadcast_to(np.asarray(speed_limit,float),shape)
        if np.any(~np.isfinite(limit) | (limit<0)):raise ValueError('Invalid calibrated slip budget')
        slip[measured & (speed<=limit)]='within_calibrated_speed_budget'
        slip[measured & (speed>limit)]='slipping'
    intent=np.zeros(shape,bool) if keep_intent is None else np.asarray(keep_intent,bool)
    if intent.shape!=shape:raise ValueError('Keep intent dimensions mismatch')
    return dict(schema=SUPPORT_ASSESSMENT_SCHEMA,contact_activated=contact,load_known=known,
                load_bearing=bearing,load_state=state,normal_force_n=force,
                loaded_motion_known=measured,loaded_tangent_speed_m_s=speed,
                slip_state=slip,keep_intent=intent)


def assess_support_path(observations, start, end, *, speed_limit=None):
    """Report native solve samples without inferring unrecorded substep motion.

    At least one measured loaded region can carry support at each solve sample.
    Absence of a calibrated velocity budget means no stationary-support verdict.
    """
    contact = np.asarray(observations['contact_activated'])[start:end + 1]
    if contact.shape != (end-start+1, 6):
        raise ValueError('Support path outside six-part observation table')
    assessed = assess_support(contact,
        np.asarray(observations['normal_force_n'])[start:end + 1],
        np.asarray(observations['loaded_tangent_speed_m_s'])[start:end + 1],
        load_known=np.asarray(observations['load_known'])[start:end + 1], speed_limit=speed_limit)
    carrying = assessed['load_bearing'].any(-1)
    known = assessed['load_known'].all(-1)
    stationary = (assessed['slip_state'] == 'within_calibrated_speed_budget').any(-1)
    return dict(schema=SUPPORT_ASSESSMENT_SCHEMA,
        load_evidence_complete=bool(known.all()),
        load_coverage_passed=bool(known.all() and carrying.all()),
        unloaded_sample_frames=(np.flatnonzero(known & ~carrying)+start).tolist(),
        unknown_sample_frames=(np.flatnonzero(~known)+start).tolist(),
        stationary_support_status=('unknown_without_calibrated_speed_budget' if speed_limit is None
            else 'passed_at_recorded_solve_samples' if known.all() and stationary.all() else 'not_passed'),
        sampling_scope='recorded solver evaluations, not a full-substep path certificate')
