"""Single-segment contact task on SuperMimic's collect/ReplayBuffer/WM/BPTT loop."""
from __future__ import annotations
import argparse
import copy
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
    parser.add_argument('--contact_response',choices=('legacy','firm_v2','manifold_v1','manifold_v2','manifold_v3','newton_contacts_v1','newton_contacts_v2','manifold_soft_v1','manifold_balanced_v1'),default='legacy')
    parser.add_argument('--allow_collision_transfer',action='store_true',help='Use legacy tracker weights as an unvalidated initialization; recordings and resumes must match current physics')
    parser.add_argument('--checkpoint',required=True,help='Original tracking teacher checkpoint')
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--slices',type=Path,help='Exact validated cuts used by DAgger')
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
    parser.add_argument('--wm_learning_rate',type=float,default=2e-5)
    parser.add_argument('--learning_rate',type=float,default=1e-5)
    parser.add_argument('--adaptive_lr',action='store_true')
    parser.add_argument('--min_learning_rate',type=float,default=3e-7)
    parser.add_argument('--adam_beta1',type=float,default=.9)
    parser.add_argument('--update_diagnostics',action='store_true')
    parser.add_argument('--contact_position_loss',choices=('measured_material','collision_surface','surface_region'),default='surface_region')
    parser.add_argument('--surface_samples_per_part',type=int,default=128)
    parser.add_argument('--surface_temperature',type=float,default=.02,help='Dimensionless scalar-cost soft minimum temperature')
    parser.add_argument('--contact_region_radius',type=float,default=0.,help='Explicit target disk radius in metres; zero retains point targets')
    parser.add_argument('--contact_target_normal',type=float,nargs=3,help='Explicit unit world normal for disk targets')
    parser.add_argument('--collision_model',type=Path,help='Optional fingerprinted model; default exports the current solver model')
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
    parser.add_argument('--full_tracking_diagnostics',action='store_true',help='Also compute unused original tracking rewards/termination for continuous contact worlds')
    parser.add_argument('--sequence_length',type=int,choices=(1,2),default=1,help='Independent single segments or all adjacent two-stage windows')
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
    parser.add_argument('--freeze_wm',action='store_true',help='Freeze the resumed WM; update only the actor')
    parser.add_argument('--freeze_actor',action='store_true',help='Keep the resumed actor unchanged for a matched collection control')
    add_launcher_args(parser);args=parser.parse_args()
    if args.continuous_random:args.sequence_length=19
    if args.adaptive_lr and (args.policy_gradient!='dmo' or args.freeze_actor):raise ValueError('Native LR scheduling requires DMO actor updates')
    if not 0<args.min_learning_rate<=args.learning_rate:raise ValueError('Invalid LR floor')
    if args.headless:args.visualizer=None
    if args.contact_evidence_every<0:
        raise ValueError('Contact evidence export interval must be nonnegative')
    if args.progress_interval_seconds<0:raise ValueError('Progress interval must be nonnegative')
    if args.contact_region_radius and args.contact_position_loss!='surface_region':
        raise ValueError('Region radius requires the surface_region objective')
    if min(args.utd,args.iterations,args.wm_batches,args.log_every,args.save_every)<=0:raise ValueError('Positive training budgets required')
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
        if resumed['config'].get('slices_sha256')!=slices_sha:raise ValueError('Resume uses different contact slices')
        if args.sequence_length!=1 or args.policy_gradient!='dmo':raise ValueError('Explicit cuts require independent native DMO')
    validate_checkpoint(resumed.get('source_metadata',{}),metadata)
    if resumed['schema']!='supermimic_contact_demo_v1' or Path(resumed['config']['recording']).resolve()!=args.recording.resolve():
        raise ValueError('Resume must be from this exact recorded task')
    if bool(demo['term'].any()):raise ValueError('Failed source demonstration')
    actor=ContactActor().to(device);actor.load_state_dict(resumed['actor'])
    frozen_actor_state=None
    if args.freeze_actor:
        actor.requires_grad_(False)
        frozen_actor_state={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
    prior_checkpoint=load_checkpoint_cpu(args.prior_checkpoint)
    if args.slices and prior_checkpoint['config'].get('slices_sha256')!=slices_sha:raise ValueError('DAgger prior uses different slices')
    validate_checkpoint(prior_checkpoint.get('source_metadata',{}),metadata)
    if not prior_checkpoint.get('dagger') or Path(prior_checkpoint['config']['recording']).resolve()!=args.recording.resolve():
        raise ValueError('Prior must be the accepted task-matched DAgger initialization')
    prior=copy.deepcopy(actor).eval().requires_grad_(False);prior.load_state_dict(prior_checkpoint['actor'])
    prior_checkpoint={'dagger':prior_checkpoint['dagger']}
    context=HeightForceContext(spec['tasks']);wm=WorldModel().to(device)
    original_resume=resumed['config'].get('wm_implementation')=='original_tracking'
    if not original_resume and int(resumed['wm_updates'])!=0:
        raise ValueError('Restart from DAgger initialization; legacy native-head optimization is retired')
    wm.load_state_dict(resumed['wm'] if original_resume else source['wm'],strict=True)
    if not original_resume:
        assert all(torch.equal(v.cpu(),source['wm'][k]) for k,v in wm.state_dict().items())
    frozen_wm_state=None
    if args.freeze_wm:
        wm.requires_grad_(False)
        frozen_wm_state={k:v.detach().cpu().clone() for k,v in wm.state_dict().items()}
    geometry=ContactGeometry(demo,device);geometry.load_state_dict(resumed['geometry'])
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
    if hasattr(clock,'segment_motion'):
        if args.sequence_length!=1:raise ValueError('Multi-motion contact transfers must use independent episodes')
        counts=case_world_counts(clock,len(spec['tasks']),args.num_envs)
        task_policy_worlds=sum(counts[i] for i in clock.segment_motion.unique().tolist())
        for source_motion in clock.segment_motion.unique().tolist():
            if counts[source_motion]<int((clock.segment_motion==source_motion).sum()):raise ValueError('Insufficient worlds per source motion')
    else:
        counts=[1]*len(spec['tasks']);counts[recording['metadata']['motion']]=args.num_envs-len(counts)+1
        task_policy_worlds=counts[recording['metadata']['motion']]
    if args.segment is not None and not 1<=args.segment<=len(clock.event_ends):raise ValueError('Segment out of range')
    if min(counts)<1 or (not hasattr(clock,'segment_motion') and task_policy_worlds<(1 if args.segment else len(clock.event_ends))):raise ValueError('Insufficient worlds for selected segments')
    config=dict(vars(args),independent_segments=True,separate_native_evaluation=False,initialization_transfer=initialization_transfer,
        start_iteration=int(resumed['iteration']),
        task_world_counts=counts,contact_bounds=clock.bounds,arrival_time_input=False,
        segment_start='exact recorded segment start; no randomized start/no target offsets',
        replay_implementation='supermimic.buffer.ReplayBuffer',
        native_contact_selector='direct MJWarp allocated-contact rows; nearest actual witness per part; no face classification',
        training_loop='collect -> WM fit -> policy BPTT; existing UTD scheduling',
        contact_tolerances_m=contact_tolerances,contact_loss_scale_m=.05,
        completion=f'actual contact inside per-segment target radius ({radius_description}; native contact; no face matching), root <{args.root_tolerance*100:g}cm/<0.4rad; no historical keep veto; no dwell')
    config['contact_face_matching']=False
    if hasattr(clock,'segment_motion'):
        config['case_source_motions']=clock.segment_motion.tolist()
        config['case_source_frame_bounds']=clock.segment_local_bounds
    config['contact_force_source']='existing force6 sensors; diagnostic only, unused in contact/root success'
    config['wm_implementation']='original_tracking'
    config['wm_source']=str(Path(args.checkpoint).resolve())
    config['policy_objective']='full-segment BPTT from exact reset to observed native stop; contact/root errors only at terminal state; process action/DAgger regularization only'
    config['horizon']=int((case_stops-case_starts)[args.segment-1] if args.segment else (case_stops-case_starts).max())+args.segment_grace_steps
    if args.slices:config.update(slices_sha256=slices_sha,case_ids=clock.case_ids.tolist(),case_starts=case_starts.tolist(),case_stops=case_stops.tolist(),case_goal_indices=(clock.event_ends-1).tolist())
    if args.sequence_length>1:
        windows=len(clock.event_ends) if args.continuous_random else len(clock.event_ends)-args.sequence_length+1
        if args.segment is not None and not 1<=args.segment<=windows:raise ValueError('Sequence start out of range')
        starts=[args.segment-1] if args.segment else range(windows)
        config['horizon']=max(clock.bounds[-1]-clock.bounds[i]+(len(clock.event_ends)-i)*args.segment_grace_steps for i in starts) if args.continuous_random else max(clock.bounds[i+args.sequence_length]-clock.bounds[i] for i in starts)+args.sequence_length*args.segment_grace_steps
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
    updates=int(resumed['policy_updates']);wm_updates=int(resumed['wm_updates']);start=int(resumed['iteration'])
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
    with launch_simulation(cfg,args):
        env=TrackingSim(cfg,args.num_envs,motion,mdp,eval_mode=True,task_world_counts=counts)
        reader=NativeContacts(env,direct=True)
        (args.output/'contact_binding.json').write_text(json.dumps(dict(
            **reader.evidence_binding.metadata(),provenance=reader.evidence_provenance),indent=2))
        task=ContactTask(env,actor,teacher,motion,mdp,clock,demo,context,default_q,reader,teacher_hist,goal_frames,args.segment_grace_steps,
                         None if args.segment is None else args.segment-1,args.root_tolerance,args.completion_mode,contact_tolerances,
                         sequential=args.sequence_length>1,sequence_length=args.sequence_length if args.sequence_length>1 else None,continuous_random=args.continuous_random)
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
        config['target_faces_by_segment']=None if task.target_faces_by_segment is None else task.target_faces_by_segment.cpu().tolist()
        (args.output/'config.json').write_text(json.dumps(config,default=str,indent=2))
        extra=dict(contact_goal=(6,torch.bool),contact_first=(6,torch.bool),next_xi=(demo['xi'].shape[-1],torch.float32),goal_index=((),torch.long),term=((),torch.bool),
            active=(6,torch.bool),known=(6,torch.bool),surface_local=(6,torch.long),point=((6,3),torch.float32),force=((6,3),torch.float32),load=(6,torch.float32),
            paired_force_w=((6,3),torch.float32),contact_count=(6,torch.long))
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
                if not bool(torch.isfinite(transition['s1']).all()):raise FloatingPointError('Invalid native state; refusing replay insertion')
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
            config['duration_loss_scope']='long attempts get terminal contact/root loss at cutoff; short attempts discarded; native metrics and WM use all data'
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
            for _ in range(0 if args.freeze_wm else args.wm_batches):
                wm_fields=('s','s1','a','xi','next_xi')+(('sequence_live',) if args.sequence_length>1 else ())
                wm_window=buffer.sample_windows(args.wm_batch,1 if args.policy_gradient=='dmo' else args.wm_window,fields=wm_fields)
                if args.sequence_length>1:
                    valid=wm_window['sequence_live'].all(1)
                    if not valid.any():continue
                    wm_window={k:v[valid] for k,v in wm_window.items()}
                loss=fit_fn(wm,wm_window,normalize_xi)
                wm_fit_before.append(loss.detach())
                wm_opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(wm.parameters(),1.,error_if_nonfinite=True);wm_opt.step();wm_updates+=1
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
            duration_filter=None;duration_stops=None;plan=None
            selected=torch.arange(task.N,device=device)
            if args.policy_gradient=='dmo':
                from supermimic.contact_dmo import native_attempt_metrics
                rows=(buffer.ptr-collected+torch.arange(collected,device=device))%buffer.T
                if args.sequence_length>1:
                    from supermimic.contact_sequence import sequence_plan,sequence_attempt_metrics,slice_plan
                    raw=buffer.recent(collected,('done','term','s1','goal_index','active','known','point','stage_step','stage_done','stage_success'))
                    native_batch=sequence_attempt_metrics(raw,clock,args.sequence_length,continuous=args.continuous_random)
                    torch.cuda.synchronize();filter_begin=time.perf_counter()
                    selected,plan,duration_filter=sequence_plan(raw,args.sequence_length,duration_history,args.duration_halfwidth or 0,args.duration_min_success,args.duration_min_mass)
                    torch.cuda.synchronize()
                    if duration_filter is not None:duration_filter['seconds']=time.perf_counter()-filter_begin
                else:
                    native_batch=native_attempt_metrics(buffer.recent(collected,('done','term','s1','goal_index','active','known','surface_local','point')),clock,task.target_faces_by_segment)
                if args.duration_halfwidth is not None and args.sequence_length==1:
                    from supermimic.contact_duration import select_duration_batch
                    torch.cuda.synchronize();filter_begin=time.perf_counter()
                    selected,duration_stops,duration_filter=duration_history.select(
                        buffer.recent(collected,('done','term','goal_index')),
                        args.duration_halfwidth,args.duration_min_success,args.duration_min_mass)
                    torch.cuda.synchronize();duration_filter['seconds']=time.perf_counter()-filter_begin
            if not args.freeze_actor and len(selected):
                actor_opt.zero_grad(set_to_none=True)
                if args.policy_gradient=='dmo':
                    from supermimic.contact_dmo import replay_loss,native_attempt_metrics
                    if lr_schedule:lr_signal=lr_schedule.observe(native_batch['macro_success_rate'])
                    objective_value=0.;objective_parts={}
                    def ordinary_batches():
                        for offset in range(0,len(selected),args.dmo_microbatch):
                            slots=selected[offset:offset+args.dmo_microbatch]
                            needed=('s','s1','xi','next_xi','a','a_prev','ep_step','goal_index','done','contact_goal','contact_first','active','known','point')
                            replay_rows=rows[:,None];replay_slots=slots[None,:]
                            batch={k:buffer.data[k][replay_rows,replay_slots] for k in needed}
                            entry=None if collection_initial_history is None else collection_initial_history[slots]
                            yield batch,None if duration_stops is None else duration_stops[slots],entry,len(slots)/len(selected),len(selected),None if plan is None else slice_plan(plan,slots)
                    if args.continuous_random:
                        from supermimic.contact_sequence import stage_batches
                        batches=(item+(None,) for item in stage_batches(buffer,rows,plan,actor,context,default_q,args.dmo_microbatch))
                    else:batches=ordinary_batches()
                    processed=0;next_training_progress=0.
                    stage_loss_sums=torch.zeros((len(clock.event_ends),4),device=device,dtype=torch.float64) if args.continuous_random else None
                    for batch,cutoffs,entry_history,weight,total,batch_plan in batches:
                        attempt_losses={} if args.continuous_random else None
                        part,details=replay_loss(actor,wm,clock,geometry,context,normalize_xi,default_q,batch,
                            pose_weight=args.pose_weight,prior_actor=prior,prior_weight=args.prior_weight,
                            check=args.check_dmo and iteration==start+1,completion_mode=args.completion_mode,
                            stop_steps=cutoffs,stage_plan=batch_plan,initial_history=entry_history,per_attempt_losses=attempt_losses,contact_surface=surface)
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
                        (part*weight).backward();objective_value+=float(part.detach())*weight
                        for key,value in details.items():
                            if value is None:continue
                            if key in ('terminal_contact','terminal_body','process_regularization'):
                                objective_parts[key]=objective_parts.get(key,0.)+weight*value
                            elif key in ('trajectories','missing_contact_proxy_terms','geometric_unobserved_terms','truncated_trajectories'):
                                objective_parts[key]=objective_parts.get(key,0)+value
                            else:objective_parts[key]=max(objective_parts.get(key,0),value)
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
                raw_grad_norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True)
                if args.update_diagnostics:
                    params=[p for p in actor.parameters() if p.grad is not None]
                    before=torch.cat([p.detach().flatten() for p in params])
                    gradient=torch.cat([p.grad.detach().flatten() for p in params])
                actor_opt.step();updates+=1
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
