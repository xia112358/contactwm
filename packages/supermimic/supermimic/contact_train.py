"""Single-segment contact task on SuperMimic's collect/ReplayBuffer/WM/BPTT loop."""
from __future__ import annotations
import argparse
import copy
from contextlib import ExitStack
import json
from pathlib import Path
from supermimic.paths import package_path
import signal
import time


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task_manifest',required=True)
    parser.add_argument('--collision_profile',choices=('legacy_v1','surface_v2'),default='surface_v2')
    parser.add_argument('--contact_response',choices=('legacy','firm_v2','manifold_v1','manifold_v2','manifold_v3','newton_contacts_v1','newton_contacts_v2','newton_contacts_v3','manifold_soft_v1','manifold_balanced_v1'),default='legacy')
    parser.add_argument('--allow_collision_transfer',action='store_true',help='Use legacy tracker weights as an unvalidated initialization; recordings and resumes must match current physics')
    parser.add_argument('--checkpoint',required=True,help='Original tracking teacher checkpoint')
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--slices',type=Path,help='Exact validated cuts used by DAgger')
    parser.add_argument('--initialize_from_dagger',action='store_true',help='Initialize weights on new slices; rebuild frame geometry and reset training state')
    parser.add_argument('--resume',type=Path,required=True,help='Accepted DAgger or contact WM checkpoint')
    parser.add_argument('--prior_checkpoint',type=Path,required=True,help='Accepted frozen DAgger initialization')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--policy_gradient',choices=('imagination','dmo'),default='imagination')
    parser.add_argument('--dmo_microbatch',type=int,default=4096)
    parser.add_argument('--check_dmo',action='store_true')
    parser.add_argument('--duration_halfwidth',type=int,help='Per-batch per-segment success mode +/- frames; discard other attempts from actor update only')
    parser.add_argument('--duration_bootstrap_batches',type=int,default=5)
    parser.add_argument('--duration_min_success',type=int,default=8)
    parser.add_argument('--duration_min_mass',type=float,default=.6)
    parser.add_argument('--duration_prior',type=Path,help='Initial frozen collection duration-distribution.json')
    parser.add_argument('--num_envs',type=int,default=4096)
    parser.add_argument('--iterations',type=int,default=500)
    parser.add_argument('--buffer_steps',type=int,default=128)
    parser.add_argument('--warmup_steps',type=int,default=32)
    parser.add_argument('--utd',type=float,default=.4)
    parser.add_argument('--wm_window',type=int,default=16)
    parser.add_argument('--wm_batches',type=int,default=2)
    parser.add_argument('--wm_batch',type=int,default=512)
    parser.add_argument('--wm_diagnostic_every',type=int,default=8,help='Measure paired pre/post WM fit every N updates; optimizer budget is unchanged')
    parser.add_argument('--wm_learning_rate',type=float,default=2e-5)
    parser.add_argument('--learning_rate',type=float,default=1e-5)
    parser.add_argument('--adaptive_lr',action='store_true')
    parser.add_argument('--min_learning_rate',type=float,default=3e-7)
    parser.add_argument('--adam_beta1',type=float,default=.9)
    parser.add_argument('--update_diagnostics',action='store_true')
    parser.add_argument('--contact_position_loss',choices=('measured_material','collision_surface','surface_region'),default='surface_region')
    parser.add_argument('--stable_contact_frames',type=int,default=2,help='Consecutive native active observations plus stable support pair before loss/completion; zero restores onset behavior')
    parser.add_argument('--surface_samples_per_part',type=int,default=128)
    parser.add_argument('--surface_temperature',type=float,default=.02,help='Dimensionless scalar-cost soft minimum temperature')
    parser.add_argument('--contact_region_radius',type=float,default=0.,help='Explicit target disk radius in metres; zero retains point targets')
    parser.add_argument('--contact_target_normal',type=float,nargs=3,help='Explicit unit world normal for disk targets')
    parser.add_argument('--collision_model',type=Path,help='Optional fingerprinted model; default exports the current solver model')
    parser.add_argument('--target_grid_span',type=float,default=0.,help='World-XY 3x3 offsets for one validated source segment')
    parser.add_argument('--target_grid_segment',type=int,help='1-based source segment for nine-point comparison')
    parser.add_argument('--pose_weight',type=float,default=1.)
    parser.add_argument('--prior_weight',type=float,default=1.)
    parser.add_argument('--segment_grace_steps',type=int,default=10,help='Extra control frames beyond each demonstrated segment length')
    parser.add_argument('--completion_mode',choices=('onset','accumulated_target'),default='onset')
    parser.add_argument('--root_tolerance',type=float,default=.25,help='Root position success tolerance in metres')
    parser.add_argument('--contact_criteria',type=Path,default=package_path('experiments/contact_completion.json'),
                        help='Recording-scoped completion radii; independent of the contact loss scale')
    parser.add_argument('--inherited_band_metrics',type=Path,help='Single-stage metrics.jsonl with native durations collected by the exact resumed actor')
    parser.add_argument('--fixed_inherited_duration',action='store_true',help='Fixed per-stage deadline from resumed single-stage duration history plus halfwidth; no secondary duration filtering')
    parser.add_argument('--continuous_random',action='store_true',help='Random segment starts; continue until failure or trajectory end; independent stage gradients')
    parser.add_argument('--case_gradient_guard',action='store_true',help='Compute each independent segment gradient, cap dominant case norms and project conflicting components')
    parser.add_argument('--full_tracking_diagnostics',action='store_true',help='Also compute unused original tracking rewards/termination for continuous contact worlds')
    parser.add_argument('--sequence_length',type=int,default=1,help='Number of consecutive segments per continuous window')
    parser.add_argument('--segment',type=int,help='Run only this segment (1-based); default runs all segments')
    parser.add_argument('--save_every',type=int,default=50)
    parser.add_argument('--log_every',type=int,default=10)
    parser.add_argument('--progress_interval_seconds',type=float,default=1.,help='Minimum wall time between collection/dashboard snapshots; zero reports every eight steps')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--probe_material_point',action='store_true',help='Compare fixed body-local and true contact points at frame 46 during update probes')
    parser.add_argument('--probe_update_path',type=Path,help='Diagnostic only: probe saved Adam direction with both networks frozen, then restore actor')
    parser.add_argument('--profile_collection',action='store_true',help='Profile one frozen warmup collection then exit without optimization/checkpoint writes')
    parser.add_argument('--audit_contacts',action='store_true',help='One same-solve comparison against original CPU reader')
    parser.add_argument('--contact_evidence_every',type=int,default=1000,help='Export owned full solver evidence every N collected steps; zero disables')
    parser.add_argument('--cache_reset_contacts',action=argparse.BooleanOptionalAction,default=True,help='Reuse per-world native evidence only for identical fixed reference starts on static terrain; disable after model changes until revalidated')
    parser.add_argument('--freeze_wm',action='store_true',help='Freeze the resumed WM; update only the actor')
    parser.add_argument('--freeze_actor',action='store_true',help='Keep the resumed actor unchanged for a matched collection control')
    add_launcher_args(parser);args=parser.parse_args()
    if args.target_grid_span and (not args.slices or args.target_grid_segment is None or args.segment is not None or args.sequence_length!=1 or args.policy_gradient!='dmo'):
        raise ValueError('Target grid requires explicit slices, a source segment, and independent DMO')
    if args.continuous_random:args.sequence_length=19
    if args.case_gradient_guard and (args.policy_gradient!='dmo' or args.sequence_length!=1 or args.freeze_actor):raise ValueError('Case gradient guard requires trainable independent DMO segments')
    if args.adaptive_lr and (args.policy_gradient!='dmo' or args.freeze_actor):raise ValueError('Native LR scheduling requires DMO actor updates')
    if not 0<args.min_learning_rate<=args.learning_rate:raise ValueError('Invalid LR floor')
    if args.headless:args.visualizer=None
    if args.contact_evidence_every<0:
        raise ValueError('Contact evidence export interval must be nonnegative')
    if args.progress_interval_seconds<0:raise ValueError('Progress interval must be nonnegative')
    if args.contact_region_radius and args.contact_position_loss!='surface_region':
        raise ValueError('Region radius requires the surface_region objective')
    if args.stable_contact_frames and args.contact_position_loss!='surface_region':
        raise ValueError('Stable contact scoring requires surface_region terminal guidance')
    if args.stable_contact_frames < 0:
        raise ValueError('Stable contact frames must be nonnegative')
    if args.stable_contact_frames and args.policy_gradient!='dmo':
        raise ValueError('Stable contact scoring requires DMO')
    if min(args.utd,args.iterations,args.wm_batches,args.wm_diagnostic_every,args.log_every,args.save_every)<=0:raise ValueError('Positive training budgets required')
    if args.completion_mode=='accumulated_target' and not args.freeze_actor and args.policy_gradient!='dmo':raise ValueError('Accumulated-target learning requires DMO')
    if args.probe_update_path and not (args.freeze_actor and args.freeze_wm):raise ValueError('Update probe requires both networks frozen')
    if args.profile_collection and not (args.freeze_actor and args.freeze_wm):raise ValueError('Collection profiling requires both networks frozen')
    if args.sequence_length>1 and (args.policy_gradient!='dmo' or args.completion_mode!='accumulated_target'):raise ValueError('Adjacent windows require accumulated-target DMO')
    if args.dmo_microbatch<1:raise ValueError('Positive DMO microbatch required')
    if args.duration_halfwidth is not None and (args.duration_halfwidth<0 or args.duration_min_success<1 or args.policy_gradient!='dmo'):
        raise ValueError('Duration filtering requires DMO and valid bandwidth/sample count')
    if args.buffer_steps<max(args.wm_window,5):raise ValueError('Replay buffer too short')
    if args.root_tolerance<=0:raise ValueError('Root tolerance must be positive')
    if args.segment_grace_steps<0:raise ValueError('Segment grace must be nonnegative')
    if args.duration_halfwidth is not None and (not 0<args.duration_min_mass<=1):raise ValueError('Invalid duration mass threshold')
    import torch
    from supermimic.lowstep import install,checkpoint_metadata,validate_checkpoint
    spec=install(args.task_manifest,contact_model='force6',collision_profile=args.collision_profile,contact_response=args.contact_response)
    from supermimic.buffer import ReplayBuffer
    from supermimic.env import TrackingSim,make_sim
    from supermimic.motion import MotionLib
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP,OBS_DIM,normalize_xi
    from supermimic.networks import Policy,WorldModel
    import supermimic.networks as networks
    from supermimic.robot import default_joint_pos_list
    from supermimic.lowstep_observations import HeightForceContext
    from supermimic.contact_native import NativeContacts
    from supermimic.contact_task import ContactTask
    from supermimic.contact_control import ContactActor,ContactClock,ContactGeometry,wm_loss,wm_one_step_loss,event_policy_loss,load_contact_tolerances
    torch.set_num_threads(4);torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=args.policy_gradient!='dmo'
    device='cuda:0';metadata=checkpoint_metadata(spec)
    from supermimic.contact_io import load_checkpoint_cpu
    source=load_checkpoint_cpu(args.checkpoint)
    initialization_transfer=validate_checkpoint(source,metadata,allow_collision_transfer=args.allow_collision_transfer)
    networks.AMP_DTYPE=torch.bfloat16 if source['args'].get('amp')=='bf16' else None
    recording=torch.load(args.recording,map_location=device,weights_only=False);validate_checkpoint(recording['metadata'],metadata)
    from supermimic.contact_library import build_contact_clock,load_slice_clock,segment_ranges,case_world_counts
    demo=recording['data'];clock=load_slice_clock(recording,args.slices) if args.slices else build_contact_clock(recording,hold=1,role_aware=True)
    case_starts,case_stops=segment_ranges(clock)
    if getattr(clock,'source_evidence_contract',None) is not None:
        from supermimic.contact_library import require_source_replay_capacity
        require_source_replay_capacity(clock,args.buffer_steps,args.segment_grace_steps,args.segment)
    if args.continuous_random:args.sequence_length=len(clock.event_ends)
    contact_tolerances=load_contact_tolerances(clock,recording['metadata'],args.contact_criteria)
    if args.contact_region_radius>min(contact_tolerances):
        raise ValueError('Objective region exceeds a native task completion radius')
    radius_description='; '.join(f'{value*100:g}cm for segments '+','.join(str(i+1) for i,v in enumerate(contact_tolerances) if v==value)
                                 for value in sorted(set(contact_tolerances)))
    resumed=load_checkpoint_cpu(args.resume)
    if args.slices:
        import hashlib
        slices_sha=hashlib.sha256(args.slices.read_bytes()).hexdigest()
        if not args.initialize_from_dagger and resumed['config'].get('slices_sha256')!=slices_sha:raise ValueError('Resume uses different contact slices')
        if args.policy_gradient!='dmo':raise ValueError('Explicit cuts require native DMO')
    validate_checkpoint(resumed.get('source_metadata',{}),metadata)
    if args.initialize_from_dagger and not resumed.get('dagger'):raise ValueError('New-slice initialization requires DAgger weights')
    if resumed['schema']!='supermimic_contact_demo_v1' or (not args.initialize_from_dagger and Path(resumed['config']['recording']).resolve()!=args.recording.resolve()):
        raise ValueError('Resume must be from this exact recorded task')
    if bool(demo['term'].any()) and getattr(clock,'source_evidence_contract',None) is None:
        raise ValueError('Legacy source termination requires migration to per-slice native source-evidence admission')
    actor=ContactActor().to(device);actor.load_state_dict(resumed['actor'])
    frozen_actor_state=None
    if args.freeze_actor:
        actor.requires_grad_(False)
        frozen_actor_state={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
    prior_checkpoint=load_checkpoint_cpu(args.prior_checkpoint)
    if args.slices and not args.initialize_from_dagger and prior_checkpoint['config'].get('slices_sha256')!=slices_sha:raise ValueError('DAgger prior uses different slices')
    validate_checkpoint(prior_checkpoint.get('source_metadata',{}),metadata)
    if not prior_checkpoint.get('dagger') or (not args.initialize_from_dagger and Path(prior_checkpoint['config']['recording']).resolve()!=args.recording.resolve()):
        raise ValueError('Prior must be the accepted task-matched DAgger initialization')
    prior=copy.deepcopy(actor).eval().requires_grad_(False);prior.load_state_dict(prior_checkpoint['actor'])
    prior_checkpoint={'dagger':prior_checkpoint['dagger']}
    context=HeightForceContext(spec['tasks']);wm=WorldModel().to(device)
    original_resume=not args.initialize_from_dagger and resumed['config'].get('wm_implementation')=='original_tracking'
    if not original_resume and int(resumed['wm_updates'])!=0:
        raise ValueError('Restart from DAgger initialization; legacy native-head optimization is retired')
    wm.load_state_dict(resumed['wm'] if original_resume else source['wm'],strict=True)
    if not original_resume:
        assert all(torch.equal(v.cpu(),source['wm'][k]) for k,v in wm.state_dict().items())
    frozen_wm_state=None
    if args.freeze_wm:
        wm.requires_grad_(False)
        frozen_wm_state={k:v.detach().cpu().clone() for k,v in wm.state_dict().items()}
    geometry=ContactGeometry(demo,device)
    resumed_grid=resumed['config'].get('target_grid_span',0.)
    if original_resume and (resumed_grid!=args.target_grid_span or resumed['config'].get('target_grid_segment')!=args.target_grid_segment):
        raise ValueError('Resume must retain its target grid')
    if resumed_grid:geometry.local_by_frame=geometry.local_by_frame.repeat(9,1,1)
    if not args.initialize_from_dagger:geometry.load_state_dict(resumed['geometry'])
    if args.target_grid_span:
        from supermimic.contact_target_grid import expand_target_grid
        selected=args.target_grid_segment-1
        clock=expand_target_grid(clock,args.target_grid_span,selected)
        demo=clock.data
        if not resumed_grid:geometry.local_by_frame=geometry.local_by_frame.repeat(9,1,1)
        case_starts,case_stops=segment_ranges(clock)
        contact_tolerances=[contact_tolerances[selected]]*9

    default_q=torch.tensor(default_joint_pos_list(),device=device)
    kin=G1Kinematics(device);motion=MotionLib.from_dir(spec['motion_dir'],kin,device);mdp=MDP(kin,motion,default_q)
    teacher_hist=source['args'].get('hist',10);goal_frames=source['args'].get('goal_frames',10)
    teacher=Policy(OBS_DIM,73*goal_frames,hist=teacher_hist).to(device).eval().requires_grad_(False);teacher.load_state_dict(source['pi'])
    del source
    actor_opt=torch.optim.Adam(actor.parameters(),lr=args.learning_rate,betas=(args.adam_beta1,.95))
    # DAgger's regression moments do not belong to the new spatial objective.
    if original_resume and resumed['config'].get('policy_gradient','imagination')==args.policy_gradient and resumed['config'].get('sequence_length',1)==args.sequence_length:actor_opt.load_state_dict(resumed['actor_optimizer'])
    for group in actor_opt.param_groups:
        group['lr']=args.learning_rate
        group['betas']=(args.adam_beta1,group['betas'][1])
    previous_actor_gradient=None
    lr_schedule=None
    if args.adaptive_lr:
        from supermimic.contact_lr import NativeSuccessLR
        lr_schedule=NativeSuccessLR(actor_opt,minimum=args.min_learning_rate)
        if original_resume and resumed.get('lr_schedule'):
            lr_schedule.load_state_dict(resumed['lr_schedule'])
            for group in actor_opt.param_groups:group['lr']=resumed['actor_optimizer']['param_groups'][0]['lr']
    wm_opt=torch.optim.Adam(wm.parameters(),lr=args.wm_learning_rate)
    if original_resume:wm_opt.load_state_dict(resumed['wm_optimizer'])
    if args.segment is not None and not 1<=args.segment<=len(clock.event_ends):raise ValueError('Segment out of range')
    if hasattr(clock,'segment_motion'):
        if args.sequence_length!=1 and clock.segment_motion.unique().numel()!=1:raise ValueError('Continuous windows require a single motion and terrain')
        counts=case_world_counts(clock,len(spec['tasks']),args.num_envs,
            None if args.segment is None else args.segment-1)
        selected_motions=clock.segment_motion.unique().tolist() if args.segment is None else [int(clock.segment_motion[args.segment-1])]
        task_policy_worlds=sum(counts[i] for i in selected_motions)
        if args.segment is None:
            for source_motion in selected_motions:
                if counts[source_motion]<int((clock.segment_motion==source_motion).sum()):raise ValueError('Insufficient worlds per source motion')
    else:
        counts=[1]*len(spec['tasks']);counts[recording['metadata']['motion']]=args.num_envs-len(counts)+1
        task_policy_worlds=counts[recording['metadata']['motion']]
    if args.segment is not None and not 1<=args.segment<=len(clock.event_ends):raise ValueError('Segment out of range')
    if min(counts)<1 or (not hasattr(clock,'segment_motion') and task_policy_worlds<(1 if args.segment else len(clock.event_ends))):raise ValueError('Insufficient worlds for selected segments')
    config=dict(vars(args),independent_segments=True,separate_native_evaluation=False,initialization_transfer=initialization_transfer,
        start_iteration=int(resumed['iteration']),
        task_world_counts=counts,student_worlds=task_policy_worlds,background_worlds=args.num_envs-task_policy_worlds,contact_bounds=clock.bounds,arrival_time_input=False,
        segment_start='exact recorded segment start; no randomized start/no target offsets',
        replay_implementation='supermimic.buffer.ReplayBuffer',
        native_contact_selector='allocated native rows; shared primary horizontal upward face eligibility; nearest actual witness per part',
        training_loop='collect -> WM fit -> policy BPTT; existing UTD scheduling',
        contact_tolerances_m=contact_tolerances,contact_loss_scale_m=.05,
        completion=f'actual contact inside per-segment target radius ({radius_description}; native contact; no face matching), root <{args.root_tolerance*100:g}cm/<0.4rad; no historical keep veto; no dwell')
    if getattr(clock,'material_goal_mode',False):
        config['completion_evidence_contract']='first_stable_completion_latched_v1'
        config['metric_sample_scope']='all collected first attempts; duration selection only affects actor loss'
    config['target_grid']=getattr(clock,'target_grid',None)
    if args.target_grid_span:config['segment_start']='nine commands from identical recorded start; world XY target offsets only'
    config['physical_failure_rule']='nonfinite state, root z < 0.15 m, or pelvis up-axis points downward (Rzz <= 0)'
    config['nonfinite_replay_policy']='save offending transition and preceding context, then abort; never sanitize into valid data'
    config['contact_face_matching']=False
    if hasattr(clock,'segment_motion'):
        config['case_source_motions']=clock.segment_motion.tolist()
        config['case_source_frame_bounds']=clock.segment_local_bounds
    config['contact_force_source']='existing force6 sensors; diagnostic only, unused in contact/root success'
    config['wm_implementation']='original_tracking'
    config['wm_source']=str(Path(args.checkpoint).resolve())
    config['policy_objective']='DMO from exact reset; contact error at first native onset with endpoint fallback if missing; root error at endpoint; process action/DAgger regularization'
    config['horizon']=int((case_stops-case_starts)[args.segment-1] if args.segment else (case_stops-case_starts).max())+args.segment_grace_steps
    if args.slices:config.update(slices_sha256=slices_sha,case_ids=clock.case_ids.tolist(),case_starts=case_starts.tolist(),case_stops=case_stops.tolist(),case_goal_indices=(clock.event_ends-1).tolist())
    if args.sequence_length>1:
        windows=len(clock.event_ends) if args.continuous_random else len(clock.event_ends)-args.sequence_length+1
        if args.segment is not None and not 1<=args.segment<=windows:raise ValueError('Sequence start out of range')
        starts=[args.segment-1] if args.segment else range(windows)
        config['horizon']=max(int((case_stops-case_starts)[i:(len(clock.event_ends) if args.continuous_random else i+args.sequence_length)].sum())+(len(clock.event_ends)-i if args.continuous_random else args.sequence_length)*args.segment_grace_steps for i in starts)
        config['segment_start']='exact recorded start of adjacent window; no state/history reset at handoff'
    config['policy_batch']=1 if args.segment else len(clock.event_ends)
    config['terminal_steps_source']='latest completed attempt of first native world per segment; fallback demonstration length + grace'
    config['contact_stop_mode']='requested_part_onset_accumulated'
    config['stop_rule']='new physical contact onset of requested parts (inactive to active), independent of face/placement; multi-part goals accumulate first onsets across the segment; each placement uses its own onset evidence, root uses last onset; initial contacts excluded; otherwise physical failure or deadline'
    if args.completion_mode=='accumulated_target':
        config['contact_stop_mode']='accumulated_target'
        config['stop_rule']=f'accumulate each part contact inside its recorded segment radius ({radius_description}); terminate only on combined success, physical failure, or deadline; no first-touch failure'
    if args.policy_gradient=='dmo':
        if args.buffer_steps<config['horizon']:raise ValueError('DMO replay must retain a complete fresh rollout')
        config['policy_objective']='simulator-forward / original-WM-backward BPTT; per-part onset loss and final root loss; missing-contact terminal FK proxy'
        if args.completion_mode=='accumulated_target':config['policy_objective']='simulator-forward / original-WM-backward; per-part first valid goal contact loss, terminal fallback for missing goals, and final root loss'
        config['training_loop']='WM one-step fit on previous native replay -> fresh full native attempts -> full-batch DMO backward -> one actor update'
        from supermimic.contact_case_gradient import ACTOR_GRADIENT_GROUP_SIZE, ACTOR_GRADIENT_OUTLIER_RATIO
        config['actor_gradient_outlier_cap']=dict(group_size=ACTOR_GRADIENT_GROUP_SIZE,median_ratio=ACTOR_GRADIENT_OUTLIER_RATIO)
        config['collision_objective_version']='all_frame_collision_surface_clearance_v1'
        config['wm_fit']='DMO deterministic one-step residual MSE on real replay; tracking normalization and existing force6 supervision retained'
        config['wm_fit_order']='before fresh actor rollout, following DMO Algorithm 1'
        config['lr_schedule_rule']='mean success over recent 3 native training batches; 3 non-improving observations (absolute improvement >0.01); halve LR to floor' if args.adaptive_lr else 'constant'
        config['policy_batch']=task_policy_worlds
        config['terminal_steps_source']='each current-policy native first attempt, without reset crossing'
    if args.sequence_length>1:
        config['policy_objective']+='; mean over retained stage endpoints; uninterrupted BPTT across command handoff'
        config['actor_optimizer_reset']=True
    if args.continuous_random:
        config['policy_objective']='continuous simulator collection; independent stage BPTT with actual detached entry state/history; equal weight per retained stage'
        config['segment_start']='uniform random segment start on each collection; continue until physical failure, stage timeout, or final segment completion'
    fixed_budgets=None;fixed_centers=None
    if args.fixed_inherited_duration:
        history=resumed.get('duration_history')
        if not args.continuous_random or not history:raise ValueError('Fixed inherited durations require continuous mode and single-stage duration history')
        fixed_centers=list(history['fixed_centers']) if 'fixed_centers' in history else [v if seen else initial for v,seen,initial in zip(history['latest'],history['reliable_seen'],history['initial'])]
        if 'fixed_centers' in history:
            for key in ('fixed_band_source','fixed_band_collected_policy_updates','fixed_band_source_iteration'):
                if key in resumed['config']:config[key]=resumed['config'][key]
        if args.inherited_band_metrics is not None and 'fixed_centers' not in history:
            if args.inherited_band_metrics.resolve().parent!=args.resume.resolve().parent:raise ValueError('Duration metrics must belong to the resumed checkpoint run')
            observations=[json.loads(line) for line in args.inherited_band_metrics.read_text().splitlines() if line.strip()]
            matches=[row for row in observations if row.get('collected_policy_updates')==resumed['policy_updates'] and row.get('duration_filter')]
            if len(matches)!=1:raise ValueError('Expected exactly one native duration batch for this policy checkpoint')
            segments=matches[0]['duration_filter']['segments']
            if sorted(v['segment'] for v in segments)!=list(range(1,len(clock.event_ends)+1)):raise ValueError('Incomplete stage duration centers')
            fixed_centers=[v['center'] for v in sorted(segments,key=lambda x:x['segment'])]
            config['fixed_band_source']=str(args.inherited_band_metrics.resolve())
            config['fixed_band_collected_policy_updates']=matches[0]['collected_policy_updates']
            config['fixed_band_source_iteration']=matches[0]['iteration']
        fixed_budgets=[v+(args.duration_halfwidth if args.duration_halfwidth is not None else 2) for v in fixed_centers]
        config['fixed_stage_budgets']=fixed_budgets
        config['horizon']=sum(fixed_budgets)
        config['duration_loss_scope']='fixed inherited +/-2 bands; exclude short samples from actor; upper bound is actual native deadline'
        config['stop_rule']='actual contact/root success before fixed inherited deadline, else physical failure or timeout'
    duration_prior=None;duration_history=None
    if args.duration_bootstrap_batches<0:raise ValueError('Invalid duration bootstrap batches')
    if args.duration_halfwidth is not None and args.duration_prior is not None:
        prior_report=json.loads(args.duration_prior.read_text())
        if Path(prior_report['checkpoint']).resolve()!=args.resume.resolve():raise ValueError('Duration prior must describe the initial resumed checkpoint')
        prior_rows=sorted(prior_report['segments'],key=lambda r:r['segment'])
        if [r['segment'] for r in prior_rows]!=list(range(1,len(clock.event_ends)+1)):raise ValueError('Missing segment duration priors')
        centers=[int(r['quantiles']['median']) for r in prior_rows]
        duration_prior=torch.tensor(centers,device=device)
        config['duration_prior_centers']=centers
        config['duration_loss_scope']='actor loss on retained attempts only; native metrics and WM fit use all attempts; long attempts use cutoff terminal contact/root loss; short attempts discarded'
    if args.duration_halfwidth is not None:
        config['duration_fallback_rule']='initial prior for first bootstrap batches, then last reliable per-segment center; initial only if no reliable history exists'
        if resumed.get('duration_history') and not args.fixed_inherited_duration:
            from supermimic.contact_duration import DurationBandHistory
            duration_history=DurationBandHistory.from_state_dict(resumed['duration_history'])
            duration_prior=torch.tensor(duration_history.initial,device=device)
            config['duration_prior_centers']=duration_history.initial.copy()
    if fixed_centers is not None:
        from supermimic.contact_duration import FixedDurationBand
        duration_history=FixedDurationBand(fixed_centers)
        duration_prior=torch.tensor(fixed_centers,device=device)
        config['duration_prior_centers']=fixed_centers
        config['duration_fallback_rule']='fixed inherited centers; never recenter'
    config['policy_start']='exact demonstrated segment starts; all selected segments in one differentiable batch'
    config['failure_rule']='nonfinite state or root height <0.15m; no absolute upright threshold'
    config['deadline_rule']='demonstrated segment length + grace steps; success allowed on last frame'
    config['segment_budgets']=(case_stops-case_starts+args.segment_grace_steps).tolist()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'config.json').write_text(json.dumps(config,default=str,indent=2))
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    updates=0 if args.initialize_from_dagger else int(resumed['policy_updates']);wm_updates=0 if args.initialize_from_dagger else int(resumed['wm_updates']);start=0 if args.initialize_from_dagger else int(resumed['iteration'])
    resumed={'config':resumed['config']}
    begin=time.perf_counter();sim_steps=0;task=None;native=None
    def status(phase,iteration,**extra):
        payload=dict(phase=phase,iteration=iteration,policy_updates=updates,wm_updates=wm_updates,
            updated_at=time.time(),elapsed_s=time.perf_counter()-begin,sim_steps=sim_steps,native=native,**extra)
        path=args.output/'status.tmp';path.write_text(json.dumps(payload,allow_nan=False,default=str,indent=2));path.replace(args.output/'status.json')
    def save(iteration,tag):
        torch.save(dict(schema='supermimic_contact_demo_v1',iteration=iteration,actor=actor.state_dict(),wm=wm.state_dict(),
            actor_optimizer=actor_opt.state_dict(),wm_optimizer=wm_opt.state_dict(),policy_updates=updates,wm_updates=wm_updates,
            geometry=geometry.state_dict(),config=config,source_metadata=metadata,dagger=prior_checkpoint['dagger'],
            duration_history=duration_history.state_dict() if duration_history else None,
            lr_schedule=lr_schedule.state_dict() if lr_schedule else None,
            rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),sim_steps=sim_steps),args.output/f'checkpoint_{tag}.pt')
    status('starting',start)
    cfg=make_sim(device,args)
    with launch_simulation(cfg,args), ExitStack() as contact_hooks:
        env=TrackingSim(cfg,args.num_envs,motion,mdp,eval_mode=True,task_world_counts=counts)
        config['solver_numerics']=getattr(env,'contact_solver_numerics',{}).get('contract','upstream')
        reader=NativeContacts(env,direct=True,primary_faces=True)
        (args.output/'contact_binding.json').write_text(json.dumps(dict(
            **reader.evidence_binding.metadata(),provenance=reader.evidence_provenance),indent=2))
        task=ContactTask(env,actor,teacher,motion,mdp,clock,demo,context,default_q,reader,teacher_hist,goal_frames,args.segment_grace_steps,
                         None if args.segment is None else args.segment-1,args.root_tolerance,args.completion_mode,contact_tolerances,
                         sequential=args.sequence_length>1,sequence_length=args.sequence_length if args.sequence_length>1 else None,continuous_random=args.continuous_random,cache_reset_contacts=args.cache_reset_contacts,stable_contact_frames=args.stable_contact_frames)
        if args.stable_contact_frames and clock.fixed_goal_mode:
            contact_hooks.enter_context(task.contact_monitor)
            config['stable_loss_sampling']=dict(version='native_substep_loss_witness_gpu_v7_outside_region',
                physics_dt=task.contact_monitor.dt,substeps_per_control=task.contact_monitor.substeps,
                qualification='native known active + same-solve NativeContactStability',
                sampling='last qualified new support substep; actual entry support excluded until confirmed exit; airborne entry armed immediately',
                initial_support_exit_steps=task.contact_monitor.exit_steps,
                completion='per-part native onset holds only its limb until observed stability; no confirmation timer; all target parts required; original total segment deadline and placement/root thresholds',
                cuda_graph=env.sim.backend_manager._cfg.use_cuda_graph)
        if task.contact_only:
            config['stable_loss_sampling']=dict(task.contact_monitor.monitor.contract,
                completion='new support establishment after entry support exits; no landing-distance or root gate',
                cuda_graph=env.sim.backend_manager._cfg.use_cuda_graph)
        task.skip_student_tracking=args.continuous_random and not args.full_tracking_diagnostics
        config['native_tracking_diagnostics']='background worlds only; unused student tracking rewards are zero and not optimized' if task.skip_student_tracking else 'all worlds'
        if fixed_budgets is not None:
            task.fixed_budgets=torch.as_tensor(fixed_budgets,device=device)
            task.budgets=task.fixed_budgets[task.segment]
        surface=None
        if args.contact_position_loss in ('collision_surface','surface_region'):
            from supermimic.contact_surface import CollisionSurface,SurfaceRegionObjective
            collision_model=reader.export_collision_model(args.output/'solver-collision.mjb') if args.collision_model is None else args.collision_model
            config['resolved_collision_model']=str(collision_model)
            surface=(SurfaceRegionObjective(collision_model,geometry,device,
                samples_per_part=args.surface_samples_per_part,temperature=args.surface_temperature,
                radius=args.contact_region_radius,target_normal=args.contact_target_normal,collision_profile=args.collision_profile)
                if args.contact_position_loss=='surface_region' else CollisionSurface(collision_model,geometry,device,collision_profile=args.collision_profile))
            config['contact_collision_catalog']=surface.catalog
            if hasattr(surface,'contract'):
                config['surface_objective_contract']=surface.contract
                previous=resumed.get('config',{}).get('surface_objective_contract')
                if previous is not None and previous!=surface.contract:
                    raise ValueError('Resumed surface objective differs; select a compatible atlas and target contract')
                config['source_surface_objective_contract']=previous
                import numpy as np
                np.savez_compressed(args.output/'surface-atlas.npz',
                    points_local=surface.atlas.points_local.cpu().numpy(),
                    normals_local=surface.atlas.normals_local.cpu().numpy(),
                    areas=surface.atlas.areas.cpu().numpy(),parts=np.asarray(surface.atlas.parts),
                    fingerprint=np.asarray(surface.atlas.fingerprint))
        if clock.fixed_goal_mode:
            config['contact_loss_contract']=clock.fixed_goal_contract
            config['completion_contract']='unchanged accumulated_target; stability gates loss witnesses only'
        elif args.stable_contact_frames:
            from supermimic.contact_phase import VERSION
            from supermimic.contact_stability import CONTRACT
            config['contact_phase_contract']=dict(version=VERSION,support_qualification=CONTRACT,
                reference_release_starts=clock.contact_phase_release_starts.cpu().tolist(),
                confirmation_frames=args.stable_contact_frames,
                release='known inactive after demonstrated target release boundary',
                witness='same first phase-confirmed native point for loss, completion and metrics',
                missing='terminal surface-region guidance reported separately')
            config['duration_loss_scope']='duration band selects attempts; preserve native terminal only when all target parts have stable witnesses; otherwise truncate'
        config['target_faces_by_segment']=None if task.target_faces_by_segment is None else task.target_faces_by_segment.cpu().tolist()
        (args.output/'config.json').write_text(json.dumps(config,default=str,indent=2))
        if getattr(clock,'material_goal_mode',False) and (args.policy_gradient!='dmo' or not args.stable_contact_frames):
            raise ValueError('Fixed material targets require DMO with native stable-substep sampling')
        if getattr(clock,'material_goal_mode',False):
            config['contact_objective']=clock.material_goal_contract
            config['stable_loss_sampling']['pose_witness']='same qualifying native substep; position and rotation retained together'
            (args.output/'config.json').write_text(json.dumps(config,default=str,indent=2))
        extra=dict(contact_goal=(6,torch.bool),contact_first=(6,torch.bool),next_xi=(demo['xi'].shape[-1],torch.float32),goal_index=((),torch.long),term=((),torch.bool),
            active=(6,torch.bool),known=(6,torch.bool),surface_local=(6,torch.long),point=((6,3),torch.float32),force=((6,3),torch.float32),load=(6,torch.float32),
            paired_force_w=((6,3),torch.float32),contact_count=(6,torch.long))
        if args.stable_contact_frames:
            extra.update(contact_stable=(6,torch.bool),contact_row_stable=(6,torch.bool),contact_material_speed=(6,torch.float32),contact_load=(6,torch.float32),contact_load_stable=(6,torch.bool))
        if task.new_contact_stop:
            extra.update(contact_entry_blocked=(6,torch.bool),collision_mask=(6,torch.bool),collision_local=((6,3),torch.float32),collision_point=((6,3),torch.float32),collision_normal=((6,3),torch.float32),collision_gap=(6,torch.float32),collision_speed=(6,torch.float32),collision_in_region=(6,torch.bool))
            extra.update(action_held=(29,torch.bool),body_endpoint_valid=((),torch.bool),contact_interrupted=((),torch.bool),
                contact_new_event=(6,torch.bool),contact_event_parts=(6,torch.bool),contact_stop_substep=(6,torch.int32),
                contact_confirmation_steps=(6,torch.int32),contact_confirmation_done=(6,torch.bool),contact_confirmation_lost=(6,torch.bool),contact_failed_parts=(6,torch.bool))
        if args.stable_contact_frames and clock.fixed_goal_mode:
            extra['contact_loss_point']=((6,3),torch.float32)
            if getattr(clock,'material_goal_mode',False):
                extra.update(contact_loss_position=((6,3),torch.float32),contact_loss_rotation=((6,3,3),torch.float32))
                if hasattr(clock,'paired_material_local'):extra['contact_loss_candidate']=(6,torch.long)
                if hasattr(clock,'paired_material_local'):extra.update(contact_raw_stable=(6,torch.bool),contact_raw_point=((6,3),torch.float32),contact_raw_position=((6,3),torch.float32),contact_raw_rotation=((6,3,3),torch.float32))
        if args.sequence_length>1:extra.update(stage_step=((),torch.long),stage_done=((),torch.bool),stage_success=((),torch.bool),sequence_live=((),torch.bool))
        buffer=ReplayBuffer(args.buffer_steps,task.N,device,extra_fields=extra)
        profiler=None
        if args.profile_collection:
            from supermimic.contact_profile import CollectionProfiler
            profiler=CollectionProfiler()
            for obj,attr,label in [(task,'step','task.step'),(actor,'forward','actor'),(teacher,'forward','teacher'),
                    (env,'step','env.step'),(env.sim,'step','physics.step'),(env,'state','state_and_sensors'),
                    (mdp,'reward','tracking_reward'),(mdp,'termination','tracking_termination'),
                    (reader,'read_device','native_contacts'),(task,'reset','task.reset'),
                    (env,'reset','env.reset'),(env,'refresh','refresh'),(buffer,'add','replay.add')]:
                profiler.wrap(obj,attr,label)
        audited=False
        collection_initial_history=None
        def collect(steps,iteration,until_first_done=False):
            nonlocal sim_steps,native,audited,collection_initial_history
            completed_first=torch.zeros(task.N,dtype=torch.bool,device=device)
            collected=0
            next_progress=time.perf_counter()+args.progress_interval_seconds
            for k in range(steps):
                if stopped:break
                transition=task.step()
                if k==0:collection_initial_history=transition.get('actor_history')
                if args.contact_evidence_every and sim_steps//task.N%args.contact_evidence_every==0:
                    import numpy as np
                    audit=reader.last_evidence.export()
                    if hasattr(surface,'atlas'):
                        features=reader.surface_features(surface.atlas)
                        audit.update({'surface_'+key:value.detach().cpu().numpy()
                                      for key,value in features.items() if isinstance(value,torch.Tensor)})
                    np.savez_compressed(args.output/f'contact-evidence-{sim_steps//task.N:08d}.npz',
                                        **audit)
                if not bool(torch.isfinite(transition['s1']).all()):
                    bad=(~torch.isfinite(transition['s1']).all(-1)).nonzero().flatten()
                    fields=('s','s1','a','a_prev','xi','next_xi','f','ep_step','goal_index','term','done','active','known','point')
                    trace=dict(schema='native_nonfinite_transition_v1',iteration=iteration,collection_step=k,
                        world_ids=task.ids[bad].cpu(),first_attempt_completed=completed_first[bad].cpu(),
                        source_metadata=metadata,
                        transition={key:transition[key][bad].detach().cpu() for key in fields},
                        preceding={key:value[:,bad].detach().cpu() for key,value in
                            buffer.recent(min(buffer.size,30),fields).items()})
                    torch.save(trace,args.output/'native-nonfinite.pt')
                    raise FloatingPointError('Invalid native state; saved native-nonfinite.pt; refusing replay insertion')
                buffer.add(transition);sim_steps+=task.N;collected+=1
                completed_first|=transition['done']
                if args.audit_contacts and not audited:
                    # Task.step may reset completed worlds; audit a fresh passive
                    # device read and CPU read of the same retained solver solve.
                    device_read=reader.read_device();cpu_read=reader.read()
                    for name in ('active','known','surface_local'):assert torch.equal(device_read[name],cpu_read[name]),name
                    for name in ('point','force','load'):
                        torch.testing.assert_close(device_read[name],cpu_read[name],rtol=2e-4,atol=2e-4,check_dtype=False)
                    print('PASS direct GPU native contacts match independent CPU reduction',flush=True);audited=True
                if k%8==0 and steps>8 and time.perf_counter()>=next_progress:
                    native=task.report(iteration,updates);status('collecting',iteration,warmup_collected=k+1,warmup_total=steps)
                    next_progress=time.perf_counter()+args.progress_interval_seconds
                if until_first_done and bool(completed_first.all()):break
            return collected
        warmup_count=collect(max(args.warmup_steps,args.wm_window+actor.hist,config['horizon'] if args.duration_halfwidth is not None else 0),start,until_first_done=args.continuous_random)
        if args.duration_halfwidth is not None and duration_prior is None and not stopped:
            from supermimic.contact_duration import select_duration_batch
            warmup_rows=(buffer.ptr-warmup_count+torch.arange(warmup_count,device=device))%buffer.T
            initial=buffer.recent(warmup_count,('done','term','goal_index','stage_done','stage_success','stage_step') if args.sequence_length>1 else ('done','term','goal_index'))
            demo_durations=case_stops-case_starts
            if args.sequence_length>1:
                from supermimic.contact_sequence import sequence_plan
                from supermimic.contact_duration import DurationBandHistory
                bootstrap=DurationBandHistory(demo_durations.cpu().tolist(),0)
                _,_,prior_audit=sequence_plan(initial,args.sequence_length,bootstrap,args.duration_halfwidth,args.duration_min_success,args.duration_min_mass)
            else:
                _,_,prior_audit=select_duration_batch(initial,demo_durations,args.duration_halfwidth,args.duration_min_success,args.duration_min_mass)
            duration_prior=demo_durations.clone()
            for item in prior_audit['segments']:duration_prior[item['segment']-1]=item['center']
            config['duration_prior_centers']=duration_prior.cpu().tolist()
            config['duration_prior_source']='initial collection at DAgger/reset policy; demonstrated duration fallback if unreliable'
            config['duration_loss_scope']=('all target parts stable: retain native terminal; otherwise truncate at duration cutoff; short attempts discarded; native metrics and WM use all data' if args.stable_contact_frames else 'long attempts get terminal contact/root loss at cutoff; short attempts discarded; native metrics and WM use all data')
            (args.output/'duration-prior.json').write_text(json.dumps(prior_audit,indent=2))
            (args.output/'config.json').write_text(json.dumps(config,default=str,indent=2))
        if args.duration_halfwidth is not None and duration_history is None and duration_prior is not None:
            from supermimic.contact_duration import DurationBandHistory
            duration_history=DurationBandHistory(duration_prior.cpu().tolist(),args.duration_bootstrap_batches)
        native=task.report(start,updates)
        if profiler is not None:
            profiler.write(args.output/'collection-profile.json')
            (args.output/'contact-reader-counts.json').write_text(json.dumps(reader.direct_counts,indent=2))
            assert all(torch.equal(v.detach().cpu(),frozen_actor_state[k]) for k,v in actor.state_dict().items())
            assert all(torch.equal(v.detach().cpu(),frozen_wm_state[k]) for k,v in wm.state_dict().items())
            status('complete',start,actor_frozen_unchanged=True,wm_frozen_unchanged=True)
            return
        if args.probe_update_path:
            from supermimic.contact_probe import probe_update
            probe_update(args.probe_update_path,actor,wm,prior,task,buffer,clock,geometry,context,normalize_xi,default_q,
                         collect,config['horizon'],start,args.output,material_probe=args.probe_material_point,
                         contact_surface=surface)
            assert all(torch.equal(v.detach().cpu(),frozen_actor_state[k]) for k,v in actor.state_dict().items())
            assert all(torch.equal(v.detach().cpu(),frozen_wm_state[k]) for k,v in wm.state_dict().items())
            status('complete',start,actor_frozen_unchanged=True,wm_frozen_unchanged=True)
            return
        (args.output/f'native_{start:06d}.json').write_text(json.dumps(native,indent=2));save(start,f'{start:06d}')
        n=max(1,round(1/args.utd));iteration=start
        for iteration in range(start+1,start+args.iterations+1):
            if stopped:break
            t0=time.perf_counter()
            fit_fn=wm_one_step_loss if args.policy_gradient=='dmo' else wm_loss
            loss=None;wm_fit_before=[];wm_fit_after=[]
            for wm_batch_index in range(0 if args.freeze_wm else args.wm_batches):
                wm_fields=('s','s1','a','xi','next_xi')+(('sequence_live',) if args.sequence_length>1 else ())
                wm_window=buffer.sample_windows(args.wm_batch,1 if args.policy_gradient=='dmo' else args.wm_window,fields=wm_fields)
                if args.sequence_length>1:
                    valid=wm_window['sequence_live'].all(1)
                    if not valid.any():continue
                    wm_window={k:v[valid] for k,v in wm_window.items()}
                loss=fit_fn(wm,wm_window,normalize_xi)
                measure_fit=(wm_batch_index+1)%args.wm_diagnostic_every==0 or wm_batch_index+1==args.wm_batches
                if measure_fit:wm_fit_before.append(loss.detach())
                wm_opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(wm.parameters(),1.,error_if_nonfinite=True);wm_opt.step();wm_updates+=1
                if measure_fit:
                    with torch.no_grad():wm_fit_after.append(fit_fn(wm,wm_window,normalize_xi).detach())
            if wm_fit_before:
                wm_means=torch.stack((torch.stack(wm_fit_before).mean(),torch.stack(wm_fit_after).mean())).cpu().tolist()
            else:wm_means=[None,None]
            torch.cuda.synchronize();wm_seconds=time.perf_counter()-t0
            collect_begin=time.perf_counter()
            if args.policy_gradient=='dmo':
                if args.sequence_length>1:task.reset_sequence()
                else:task.reset(torch.arange(task.N,device=device))
                collected=collect(config['horizon'],iteration,until_first_done=True)
            else:collect(n,iteration)
            t1=time.perf_counter()
            if stopped:break
            t2=time.perf_counter()
            objective=None;objective_parts=None;native_batch=None;update_diagnostics=None;lr_signal=None
            collected_policy_updates=updates
            duration_filter=None;duration_stops=None;plan=None;loss_aligned_batch=None
            selected=torch.arange(task.N,device=device)
            if args.policy_gradient=='dmo':
                from supermimic.contact_dmo import native_attempt_metrics
                rows=(buffer.ptr-collected+torch.arange(collected,device=device))%buffer.T
                if args.sequence_length>1:
                    from supermimic.contact_sequence import sequence_plan,sequence_attempt_metrics,slice_plan
                    fields=('done','term','s1','goal_index','active','known','point','stage_step','stage_done','stage_success')
                    if getattr(clock,'material_goal_mode',False):fields+=('contact_stable','contact_loss_point','contact_loss_position','contact_loss_rotation')
                    if hasattr(clock,'paired_material_local'):fields+=('contact_loss_candidate',)
                    raw=buffer.recent(collected,fields)
                    native_batch=sequence_attempt_metrics(raw,clock,args.sequence_length,continuous=args.continuous_random)
                    torch.cuda.synchronize();filter_begin=time.perf_counter()
                    selected,plan,duration_filter=sequence_plan(raw,args.sequence_length,duration_history,args.duration_halfwidth or 0,args.duration_min_success,args.duration_min_mass)
                    torch.cuda.synchronize()
                    if duration_filter is not None:duration_filter['seconds']=time.perf_counter()-filter_begin
                else:
                    native_batch=native_attempt_metrics(buffer.recent(collected,('done','term','s1','goal_index','active','known','surface_local','point','contact_first')),clock,task.target_faces_by_segment,first_contact=(args.stable_contact_frames>0 and not clock.fixed_goal_mode) or args.completion_mode=='onset')
                if args.stable_contact_frames and args.sequence_length==1 and not clock.fixed_goal_mode:
                    native_batch.update(contact_metric='reference_stable_support_native_point',
                        scope='first confirmed native point in demonstrated landing phase; same witness as contact loss and completion')
                if args.duration_halfwidth is not None and args.sequence_length==1:
                    from supermimic.contact_duration import select_duration_batch
                    torch.cuda.synchronize();filter_begin=time.perf_counter()
                    selected,duration_stops,duration_filter=duration_history.select(
                        buffer.recent(collected,('done','term','goal_index')),
                        args.duration_halfwidth,args.duration_min_success,args.duration_min_mass)
                    torch.cuda.synchronize();duration_filter['seconds']=time.perf_counter()-filter_begin
            if args.stable_contact_frames and duration_stops is not None:
                from supermimic.contact_loss_sampling import stable_duration_stops
                recent=buffer.recent(collected,('done','contact_stable','goal_index'))
                ends=clock.event_ends[recent['goal_index'][0]]-1
                required=clock.target_masks[ends]&~clock.keep_masks[ends]
                upper=torch.empty(len(clock.event_ends),device=duration_stops.device,dtype=torch.long)
                for row in duration_filter['segments']:upper[row['segment']-1]=row['upper']
                native=recent['done'].long().argmax(0)+1
                cutoffs=torch.minimum(native,upper[recent['goal_index'][0]])
                duration_stops=stable_duration_stops(recent['done'],recent['contact_stable'],required,cutoffs)
            if args.case_gradient_guard:
                # Every segment continues contributing even if its entire batch
                # was discarded by the duration selector; use its real attempts.
                event_all=buffer.data['goal_index'][rows[0]]
                present=torch.unique(event_all[selected]);expected=torch.unique(task.segment)
                missing=expected[~torch.isin(expected,present)]
                recovered=[]
                for case in missing.tolist():
                    slots=(event_all==case).nonzero().flatten()
                    if not len(slots):raise ValueError('A sampled segment has no collected attempts')
                    selected=torch.cat((selected,slots));recovered.append(dict(segment=case+1,attempts=len(slots)))
                if duration_filter is not None:duration_filter['guard_unfiltered_recovered']=recovered
            if args.policy_gradient=='dmo' and args.sequence_length==1 and args.stable_contact_frames and clock.fixed_goal_mode and len(selected):
                from supermimic.contact_dmo import loss_aligned_attempt_metrics
                fields=('done','s1','goal_index','contact_stable','contact_loss_point')
                if task.new_contact_stop:fields+=('contact_interrupted',)
                if getattr(clock,'material_goal_mode',False):fields+=('contact_loss_position','contact_loss_rotation')
                if hasattr(clock,'paired_material_local'):fields+=('contact_loss_candidate',)
                recent=buffer.recent(collected,fields)
                # Metrics cover every collected attempt; duration selection only affects actor loss.
                loss_aligned_batch=loss_aligned_attempt_metrics(recent,clock,task.contact_tolerances_by_segment,task.root_tolerance,task.root_angle_tolerance)
                if getattr(clock,'material_goal_mode',False):native_batch=loss_aligned_batch
            if not args.freeze_actor and len(selected):
                actor_opt.zero_grad(set_to_none=True)
                if args.policy_gradient=='dmo':
                    from supermimic.contact_dmo import replay_loss,native_attempt_metrics
                    if lr_schedule:lr_signal=lr_schedule.observe(native_batch['macro_success_rate'])
                    objective_value=0.;objective_parts={}
                    def ordinary_batches():
                        if args.case_gradient_guard:
                            event_all=buffer.data['goal_index'][rows[0]]
                            slot_groups=[selected[event_all[selected]==case] for case in torch.unique(event_all[selected],sorted=True)]
                        else:
                            slot_groups=[selected[offset:offset+args.dmo_microbatch] for offset in range(0,len(selected),args.dmo_microbatch)]
                        for slots in slot_groups:
                            needed=('s','s1','xi','next_xi','a','a_prev','ep_step','goal_index','done','contact_goal','contact_first','active','known','point')
                            if task.new_contact_stop:needed+=('action_held','body_endpoint_valid','contact_interrupted','contact_entry_blocked','collision_mask','collision_local','collision_point','collision_normal','collision_gap','collision_speed','collision_in_region')
                            if args.stable_contact_frames:needed+=('contact_stable',)
                            if args.stable_contact_frames and clock.fixed_goal_mode:needed+=('contact_loss_point',)
                            if getattr(clock,'material_goal_mode',False):needed+=('contact_loss_position','contact_loss_rotation')
                            if hasattr(clock,'paired_material_local'):needed+=('contact_loss_candidate','contact_raw_stable','contact_raw_point','contact_raw_position','contact_raw_rotation')
                            replay_rows=rows[:,None];replay_slots=slots[None,:]
                            batch={k:buffer.data[k][replay_rows,replay_slots] for k in needed}
                            entry=None if collection_initial_history is None else collection_initial_history[slots]
                            yield batch,None if duration_stops is None else duration_stops[slots],entry,len(slots)/len(selected),len(selected),None if plan is None else slice_plan(plan,slots)
                    if args.continuous_random:
                        from supermimic.contact_sequence import stage_batches
                        batches=(item+(None,) for item in stage_batches(buffer,rows,plan,actor,context,default_q,args.dmo_microbatch))
                    else:batches=ordinary_batches()
                    processed=0;next_training_progress=0.
                    gradient_rows=[];gradient_weights=[]
                    case_gradients=[];case_loss_rows=[];guard_parameters=list(actor.parameters()) if args.case_gradient_guard else []
                    stage_loss_sums=torch.zeros((len(clock.event_ends),4),device=device,dtype=torch.float64) if args.continuous_random else None
                    for batch,cutoffs,entry_history,weight,total,batch_plan in batches:
                        attempt_losses={} if args.continuous_random else None
                        if not args.case_gradient_guard:
                            from supermimic.contact_grouped_gradient import GroupedActorGradients
                            from supermimic.contact_case_gradient import ACTOR_GRADIENT_GROUP_SIZE
                            group_ids=torch.arange(len(batch['goal_index'][0]),device=device)//ACTOR_GRADIENT_GROUP_SIZE
                            grouped=GroupedActorGradients(actor,group_ids)
                            grouped.__enter__()
                        part,details=replay_loss(actor,wm,clock,geometry,context,normalize_xi,default_q,batch,
                            pose_weight=args.pose_weight,prior_actor=prior,prior_weight=args.prior_weight,
                            check=args.check_dmo and iteration==start+1 and (not args.case_gradient_guard or processed==0),completion_mode=args.completion_mode,
                            stop_steps=cutoffs,stage_plan=batch_plan,initial_history=entry_history,per_attempt_losses=attempt_losses,contact_surface=surface,
                            stable_contact_frames=args.stable_contact_frames)
                        processed+=len(batch['goal_index'][0])
                        if attempt_losses is not None:
                            contact=attempt_losses['contact'].detach()
                            root=sum(attempt_losses[k].detach() for k in ('root_position','root_orientation','root_velocity'))
                            stats=torch.stack((torch.ones_like(contact),(batch['ep_step'][0]>0),contact,root),-1).double()
                            stage_loss_sums.index_add_(0,batch['goal_index'][0],stats)
                        if time.perf_counter()>=next_training_progress:
                            if processed==len(batch['goal_index'][0]):native=task.report(iteration,updates)
                            status('training',iteration,dmo_processed=processed,dmo_total=total,duration_filter=duration_filter)
                            next_training_progress=time.perf_counter()+args.progress_interval_seconds
                        if args.case_gradient_guard:
                            actor_opt.zero_grad(set_to_none=True);part.backward()
                            gradient=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in guard_parameters]).detach().clone()
                            case_gradients.append(gradient)
                            case_loss_rows.append(dict(segment=int(batch['goal_index'][0,0])+1,samples=len(batch['goal_index'][0]),loss=float(part.detach()),contact_loss=details['terminal_contact'],gradient_norm=float(gradient.norm())))
                        else:
                            torch.autograd.grad(part,tuple(actor.parameters()),allow_unused=True)
                            matrix=grouped.matrix()
                            grouped.__exit__(None,None,None)
                            gradient_rows.append(matrix)
                            gradient_weights.append(grouped.counts.to(matrix.dtype)*(weight/grouped.batch))
                        objective_value+=float(part.detach())*weight
                        for key,value in details.items():
                            if value is None:continue
                            if key in ('process_collision','collision_prefix','terminal_contact','terminal_body','process_regularization','stable_contact_loss','terminal_guidance_loss','contact_supervision_coverage'):
                                objective_parts[key]=objective_parts.get(key,0.)+weight*value
                            elif key in ('trajectories','missing_contact_proxy_terms','geometric_unobserved_terms','truncated_trajectories','required_contact_terms','stable_contact_terms','terminal_guidance_terms'):
                                objective_parts[key]=objective_parts.get(key,0)+value
                            else:objective_parts[key]=max(objective_parts.get(key,0),value)
                    if args.case_gradient_guard:
                        from supermimic.contact_case_gradient import guarded_case_gradient
                        direction,guard=guarded_case_gradient(torch.stack(case_gradients),rotation=iteration%len(case_gradients))
                        cursor=0
                        for parameter in guard_parameters:
                            size=parameter.numel();parameter.grad=direction[cursor:cursor+size].reshape_as(parameter).clone();cursor+=size
                        guard['segments']=[row['segment'] for row in case_loss_rows]
                        objective_parts['case_gradient_guard']=guard;objective_parts['case_losses']=case_loss_rows
                    if args.continuous_random:
                        objective_parts['stage_losses']=[dict(segment=g+1,samples=int(n),continuation_samples=int(continued),contact=contact/n,root=root/n)
                            for g,(n,continued,contact,root) in enumerate(stage_loss_sums.cpu().tolist()) if n]
                    objective=torch.tensor(objective_value,device=device)
                else:
                    objective,objective_parts=event_policy_loss(actor,wm,clock,geometry,context,normalize_xi,default_q,
                        terminal_steps=task.last_terminal_steps,pose_weight=args.pose_weight,prior_actor=prior,prior_weight=args.prior_weight,
                        segment_indices=None if args.segment is None else [args.segment-1],return_details=True,
                        contact_surface=surface)
                    objective.backward()
                if args.policy_gradient=='dmo' and not args.case_gradient_guard:
                    from supermimic.contact_case_gradient import cap_gradient_outliers
                    direction,gradient_cap=cap_gradient_outliers(torch.cat(gradient_rows),torch.cat(gradient_weights))
                    offset=0
                    for parameter in actor.parameters():
                        width=parameter.numel()
                        parameter.grad=direction[offset:offset+width].view_as(parameter)
                        offset+=width
                    objective_parts['gradient_outlier_cap']=gradient_cap
                raw_grad_norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True)
                if args.update_diagnostics:
                    params=[p for p in actor.parameters() if p.grad is not None]
                    before=torch.cat([p.detach().flatten() for p in params])
                    gradient=torch.cat([p.grad.detach().flatten() for p in params])
                if args.case_gradient_guard:
                    pre_step=torch.cat([parameter.detach().flatten().clone() for parameter in guard_parameters])
                actor_opt.step();updates+=1
                if args.case_gradient_guard:
                    from supermimic.contact_case_gradient import guard_adam_displacement
                    after_step=torch.cat([parameter.detach().flatten() for parameter in guard_parameters])
                    applied,step_guard=guard_adam_displacement(after_step-pre_step,torch.stack(case_gradients))
                    with torch.no_grad():
                        cursor=0
                        for parameter in guard_parameters:
                            size=parameter.numel();parameter.copy_((pre_step[cursor:cursor+size]+applied[cursor:cursor+size]).reshape_as(parameter));cursor+=size
                    objective_parts['case_gradient_guard']['adam_step']=step_guard
                if args.update_diagnostics:
                    delta=torch.cat([p.detach().flatten() for p in params])-before
                    cosine=lambda x,y:float(torch.dot(x,y)/(x.norm()*y.norm()).clamp_min(1e-30))
                    update_diagnostics=dict(raw_gradient_norm=float(raw_grad_norm),step_norm=float(delta.norm()),
                        step_vs_negative_gradient_cosine=cosine(delta,-gradient),
                        gradient_vs_previous_cosine=None if previous_actor_gradient is None else cosine(gradient,previous_actor_gradient),
                        clipped_gradient_dot_step=float(torch.dot(gradient,delta)))
                    previous_actor_gradient=gradient

            torch.cuda.synchronize();t3=time.perf_counter()
            row=dict(iteration=iteration,policy_updates=updates,wm_updates=wm_updates,wm_loss=float(loss.detach()) if loss is not None else None,
                policy_loss=float(objective.detach()) if objective is not None else None,collect_seconds=t1-collect_begin,wm_seconds=wm_seconds,policy_seconds=t3-t2,
                sim_steps=sim_steps,elapsed_s=t3-begin,objective_parts=objective_parts,native_batch=native_batch,collected_policy_updates=collected_policy_updates,update_diagnostics=update_diagnostics,learning_rate=actor_opt.param_groups[0]['lr'],lr_signal=lr_signal,
                wm_fit_before=wm_means[0],wm_fit_after=wm_means[1])
            row['duration_filter']=duration_filter
            if loss_aligned_batch is not None:row['loss_aligned_batch']=loss_aligned_batch
            with (args.output/'metrics.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
            if iteration%args.log_every==0 or iteration==start+1:
                native=task.report(iteration,updates)
                (args.output/f'native_{iteration:06d}.json').write_text(json.dumps(native,indent=2))
                status('training',iteration,wm_loss=row['wm_loss'],policy_loss=row['policy_loss'],timing=row)
                print(json.dumps(dict(**row,training_success_rate=native['success_rate'],attempts=native['attempts'])),flush=True)
            if iteration%args.save_every==0:save(iteration,f'{iteration:06d}')
        native=task.report(iteration,updates)
        unchanged=None
        if args.freeze_wm:
            unchanged=all(torch.equal(v.detach().cpu(),frozen_wm_state[k]) for k,v in wm.state_dict().items())
            if not unchanged:raise AssertionError('Frozen WM state changed')
        actor_unchanged=None
        if args.freeze_actor:
            actor_unchanged=all(torch.equal(v.detach().cpu(),frozen_actor_state[k]) for k,v in actor.state_dict().items())
            if not actor_unchanged:raise AssertionError('Frozen actor state changed')
        save(iteration,f'final_{iteration:06d}')
        torch.save(dict(data=buffer.data,ptr=buffer.ptr,size=buffer.size,N=buffer.N,T=buffer.T),args.output/'replay.pt')
        status('interrupted' if stopped else 'complete',iteration,wm_frozen_unchanged=unchanged,actor_frozen_unchanged=actor_unchanged)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys
        if '--output' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--output')+1])
            if output.exists():
                path=output/'status.json';old=json.loads(path.read_text()) if path.exists() else {}
                path.write_text(json.dumps(dict(old,phase='failed',error=repr(exc),updated_at=time.time()),indent=2))
        raise
