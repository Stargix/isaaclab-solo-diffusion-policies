"""Nominal native-V8 takeover feasibility gate and fixed-cohort collection.

Each six-second attempt: expert hold .5 s, learner .4/.6/.8 s, expert recovery.
No action blending, impulses, skill inputs or reward changes. Failed attempts
remain in the report. HDF5 stores real pre-action state and executed action.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
from contextlib import nullcontext
import numpy as np
import torch
from rsl_rl.modules import ActorCritic  # native dependencies before Kit
from isaaclab.app import AppLauncher

sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from scripts.diffusion_policy.experiments.pronk_intervention_v1.protocol import (
    ROOT, BASE_SHA, EXPERT, EXPERT_SHA, START, STEPS, PREFIXES, SPEEDS,
    DEFAULT_OUTPUT, load_checkpoint, recovery_anchors, gate_passes, sha256)
from scripts.diffusion_policy.experiments.pronk_dagger_v1.protocol import enforce_collection_gate, exposure_increment

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--checkpoint',type=Path,default=ROOT/'best.pt')
ap.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
ap.add_argument('--rounds',type=int,default=4)
ap.add_argument('--seed',type=int,default=443)
ap.add_argument('--evaluate',action='store_true',help='Student-only physical screen; writes no training HDF5.')
ap.add_argument('--expected_sha256',help='Explicit integrity pin for an iterative student checkpoint.')
ap.add_argument('--expert',type=Path,default=EXPERT)
ap.add_argument('--dagger_round',type=int,choices=(0,1,2),help='Opt-in iterative protocol; pilot defaults remain unchanged.')
ap.add_argument('--dagger_fixed_exposure',action='store_true',help='Keep round-zero exposure; record the cohort gate without stopping collection.')
AppLauncher.add_app_launcher_args(ap)
args=ap.parse_args()
if args.dagger_fixed_exposure and (args.dagger_round is None or args.evaluate):
    raise ValueError('Fixed-exposure aggregation requires a DAgger collection round, not evaluation.')
if args.rounds not in (1,4):
    raise ValueError('Use one feasibility cohort or the frozen four-cohort pilot.')
if args.evaluate and args.rounds != 1:
    raise ValueError('Evaluation uses exactly one fixed cohort.')
if args.output.exists():
    raise FileExistsError(args.output)
if args.dagger_round is not None and not args.expected_sha256:
    raise ValueError('Iterative collection requires an explicit checkpoint SHA256.')
student_ck=load_checkpoint(args.checkpoint,args.expected_sha256 or (None if args.evaluate else BASE_SHA))
expert_ck=load_checkpoint(args.expert,EXPERT_SHA)
app=AppLauncher(args).app

import h5py
import gymnasium as gym
import isaaclab_tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.direct.solo12.solo12_pronk_v8 import Solo12PronkV8EnvCfg, Solo12PronkV8PPORunnerCfg
from scripts.diffusion_policy.model.solo12_diffusion_policy import Solo12DiffusionPolicy
from scripts.dppo_diffusion_rl.checkpointing import model_config_from_checkpoint
from scripts.diffusion_policy.train.data.normalization import NormalizerStats
from scripts.diffusion_policy.train.data.obs_utils import proprio_from_env_tensors
from scripts.diffusion_policy.train.conditioning.goal_builder import build_geometric_height_profile_goal_vector
from scripts.diffusion_policy.train.conditioning.geometry import cumulative_xy_lengths
from scripts.diffusion_policy.runtime.controller import DelayedHistory, validate_joint_mapping


def main():
    args.output.mkdir(parents=True)
    # Keep the optional historical task local to this experiment. A clean
    # checkout need not register every intervening pronk training variant.
    if 'solo12-pronk-v8-v0' not in gym.registry:
        gym.register(
            id='solo12-pronk-v8-v0',
            entry_point='isaaclab_tasks.direct.solo12.solo12_pronk_v8:Solo12PronkV8Env',
            disable_env_checker=True,
        )
    cfg=Solo12PronkV8EnvCfg()
    cfg.use_full_difficulty()
    cfg.scene.num_envs=18
    cfg.sim.device=args.device or 'cuda:0'
    cfg.seed=args.seed
    cfg.episode_length_s=8.
    cfg.command_resampling_time_s=1000.
    cfg.events=None
    cfg.enable_observation_corruption=False
    cfg.actuation_delay_range=(0,0)
    cfg.initial_max_delay_steps=0
    cfg.noise_initial_fraction=1.
    cfg.forces_applied_to_base_curriculum=(0.,)
    cfg.base_push_force_xy_range=(0.,0.)
    cfg.base_push_force_z_range=(0.,0.)
    cfg.base_push_interval_range_s=(1e9,1e9)
    cfg.reset_base_lin_vel_range=(0.,0.)
    cfg.reset_base_ang_vel_range=(0.,0.)
    cfg.flexed_initial_joint_pos_noise_range=(0.,0.)
    runner=Solo12PronkV8PPORunnerCfg()
    env=RslRlVecEnvWrapper(gym.make('solo12-pronk-v8-v0',cfg=cfg),clip_actions=runner.clip_actions)
    rows=[]
    report=dict(protocol='pronk_intervention_v1',mode='student_evaluation' if args.evaluate else 'takeover_collection',seed=args.seed,
                base_sha256=sha256(args.checkpoint),expert_sha256=sha256(args.expert),
                dagger_round=args.dagger_round,
                dagger_fixed_exposure=args.dagger_fixed_exposure,
                gate_is_diagnostic=args.dagger_fixed_exposure,
                physics='native V8 nominal, no DR/noise/delay/push; unchanged PD',
                source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')},
                git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
                attempts=rows,gate_passed=False)
    if args.dagger_round is not None:
        report['protocol']='pronk_dagger_v1'
    try:
        raw=env.unwrapped
        device=torch.device(raw.device)
        ids=raw._joint_ids
        validate_joint_mapping([raw._robot.joint_names[i] for i in ids])
        obs=env.get_observations()
        pc=runner.policy.to_dict();pc.pop('class_name')
        teacher=ActorCritic(obs,runner.obs_groups,env.num_actions,**pc).to(device)
        teacher.load_state_dict(expert_ck['model_state_dict']);teacher.eval()
        student=Solo12DiffusionPolicy(model_config_from_checkpoint(student_ck['config'])).to(device)
        student.load_state_dict(student_ck['ema_model_state_dict'])
        student.set_normalizer_stats(NormalizerStats.from_dict(student_ck['normalizer_stats']));student.eval()
        speeds=torch.tensor(SPEEDS,device=device).repeat_interleave(6)
        takeover=torch.tensor([START+x for x in PREFIXES for _ in range(2)],device=device).repeat(3)
        starts=torch.full((18,),START,device=device,dtype=torch.long)
        if args.dagger_round is not None:
            # Half the attempts expose reset/startup; half expose an established gait.
            starts[::2]=0
            prefixes=torch.tensor([20,20,30,30,40,40],device=device).repeat(3)
            takeover=starts+prefixes+exposure_increment(args.dagger_round,fixed_exposure=args.dagger_fixed_exposure)
        with (nullcontext(None) if args.evaluate else h5py.File(args.output/'recovery.hdf5','w')) as f:
            group=None if f is None else f.create_group('data')
            if group is not None:
                group.attrs['convention']='aligned: obs[t] is the proprioceptive state BEFORE executing actions[t]; last_action[t]=actions[t-1].'
                group.attrs['skill_names']=np.asarray([b'pronk'],dtype='S32')
                group.attrs['control_rate_hz']=50.
                group.attrs['collection']='expert_takeover_after_student_prefix_v1'
            for cohort in range(args.rounds):
                with torch.inference_mode():
                    env.reset()
                torch.manual_seed(args.seed+cohort)
                history=DelayedHistory.zeros(18,device=device)
                alive=torch.ones(18,dtype=torch.bool,device=device)
                failures=torch.full((18,),-1.,device=device)
                cycles=np.zeros(18,dtype=int)
                speed_sum=np.zeros(18)
                speed_n=np.zeros(18,dtype=int)
                records=[]
                chunk=None
                for tick in range(STEPS):
                    commands=torch.zeros(18,3,device=device)
                    if tick>=START:commands[:,0]=speeds
                    raw._commands[:]=commands
                    raw._command_steps_left.fill_(100000)
                    d=raw._robot.data
                    p=proprio_from_env_tensors(d.joint_pos[:,ids],d.joint_vel[:,ids],d.root_ang_vel_b,d.projected_gravity_b).clone()
                    pos=d.root_pos_w.detach().cpu().numpy().copy()
                    quat=d.root_quat_w.detach().cpu().numpy().copy()
                    yaw=np.arctan2(2*(quat[:,0]*quat[:,3]+quat[:,1]*quat[:,2]),1-2*(quat[:,2]**2+quat[:,3]**2))
                    goals=[]
                    for i in range(18):
                        distance=np.arange(101,dtype=np.float32)*.02*float(commands[i,0])
                        path=np.tile(pos[i],(101,1))
                        path[:,0]+=distance*np.cos(yaw[i]);path[:,1]+=distance*np.sin(yaw[i])
                        goals.append(build_geometric_height_profile_goal_vector(path,cumulative_xy_lengths(path),
                            np.full(101,.2932,dtype=np.float32),0,100,pos[i],quat[i],
                            yaws_w=np.full(101,yaw[i]),dt=.02,v_avg_clip=2.))
                    g=torch.as_tensor(np.asarray(goals),device=device)
                    if tick==0:
                        history.proprio[:]=p[:,None];history.goal[:]=g[:,None]
                    expert_control=(tick<starts)|(tick>=takeover)
                    if args.evaluate:expert_control=torch.zeros_like(expert_control)
                    active=~expert_control
                    with torch.inference_mode():
                        action=teacher.act_inference(env.get_observations()).clone()
                        # Per-env phase since THIS env's student takeover. A shared
                        # tick%4 index skips replans when half the cohort starts at 0
                        # and half at START, and feeds stale chunk slots to late starters.
                        if args.evaluate:
                            phases=torch.full((18,),tick,device=device,dtype=torch.long)
                        elif args.dagger_round is not None:
                            phases=(tick-starts).clamp(min=0)
                        else:
                            phases=torch.full((18,),tick-START,device=device,dtype=torch.long)
                        if active.any():
                            need=active&((phases%4)==0)
                            if chunk is None:
                                chunk=torch.zeros(18,4,env.num_actions,device=device)
                                need=active.clone()
                            if need.any():
                                pred=student.predict_action_denormalized(history.proprio,history.action,history.goal)
                                chunk[need]=student.executable_chunk(pred,4)[need]
                            action[active]=chunk[active,phases[active]%4]
                        if runner.clip_actions is not None:
                            action=action.clamp(-runner.clip_actions,runner.clip_actions)
                        if not torch.isfinite(action).all():
                            raise FloatingPointError('Nonfinite action; refusing simulator step')
                        record=dict(joint_pos=p[:,:12].cpu().numpy().copy(),joint_vel=p[:,12:24].cpu().numpy().copy(),
                            base_ang_vel=p[:,24:27].cpu().numpy().copy(),projected_gravity=p[:,27:].cpu().numpy().copy(),
                            root_pos_w=pos,root_quat_w=quat,command_speed=commands.cpu().numpy().copy(),
                            last_action=history.previous_action.cpu().numpy().copy(),
                            desired_base_height=np.full(18,.2932,dtype=np.float32),
                            actions=action.cpu().numpy().copy(),expert_control=expert_control.cpu().numpy().copy())
                        _,_,done,_=env.step(action)
                        history.commit_tick(p,action,g)
                    new=alive&done.bool();failures[new]=(tick+1)*.02;alive&=~done.bool()
                    c=raw._pronk_cycle_components
                    record['physical_flight']=c['physical_flight'].cpu().numpy().copy()
                    record['completed']=c['completed'].cpu().numpy().copy()
                    record['flight_duration']=c['flight_duration'].cpu().numpy().copy()
                    record['clearance']=c['clearance_m'].cpu().numpy().copy()
                    record['foot_force']=raw._contact_sensor.data.net_forces_w[:,raw._pronk_foot_ids].cpu().numpy().copy()
                    record['valid']=alive.cpu().numpy().copy()
                    record['height']=(raw._robot.data.root_pos_w[:,2]-raw._terrain.env_origins[:,2]).cpu().numpy().copy()
                    record['vx']=raw._robot.data.root_lin_vel_b[:,0].cpu().numpy().copy()
                    records.append(record)
                    if tick>=100:
                        a=alive.cpu().numpy()
                        cycles+=record['completed']&a
                        speed_sum+=np.where(a,record['vx'],0.)
                        speed_n+=a
                    if args.evaluate and not alive.any():
                        break  # all first attempts failed; never evaluate reset continuations
                survived=alive.cpu().numpy()
                for i in range(18):
                    row=dict(cohort=cohort,env=i,speed=float(speeds[i]),takeover_step=int(takeover[i]),
                             student_start_step=int(starts[i]),
                             survived=bool(survived[i]),failure_time_s=float(failures[i]),cycles=int(cycles[i]),
                             mean_vx=float(speed_sum[i]/max(1,speed_n[i])),saved=False)
                    if not args.evaluate and survived[i] and cycles[i]>=2 and abs(row['mean_vx']-row['speed'])<=.10:
                        demo=group.create_group(f'demo_{len(group)}')
                        obs_group=demo.create_group('obs')
                        for key in records[0]:
                            values=np.asarray([r[key][i] for r in records])
                            if key=='actions':demo.create_dataset(key,data=values)
                            elif key in ('expert_control','physical_flight','completed','height','vx','flight_duration','clearance','foot_force','valid'):
                                demo.require_group('diagnostics').create_dataset(key,data=values)
                            else:obs_group.create_dataset(key,data=values)
                        control=np.asarray([r['expert_control'][i] for r in records])
                        anchors=recovery_anchors(control)
                        demo.create_dataset('training_anchor_steps',data=anchors)
                        demo.create_dataset('skill_idx',data=np.zeros(STEPS,dtype=np.int16))
                        dones=np.zeros(STEPS,dtype=bool);dones[-1]=True
                        demo.create_dataset('dones',data=dones)
                        demo.attrs['cohort']=cohort
                        demo.attrs['takeover_step']=int(takeover[i])
                        row.update(saved=True,demo=demo.name.split('/')[-1],anchors=len(anchors))
                    rows.append(row)
                if cohort==0:report['gate_passed']=gate_passes(rows)
                if f is not None:f.flush()
                np.savez_compressed(args.output/f'cohort_{cohort}.npz',
                    **{key:np.asarray([r[key] for r in records]) for key in ('height','vx','physical_flight','completed','expert_control','flight_duration','clearance','foot_force','valid')})
                (args.output/'collection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                print(json.dumps(dict(cohort=cohort,survived=int(survived.sum()),saved=sum(r['saved'] for r in rows if r['cohort']==cohort),gate_passed=report['gate_passed'])),flush=True)
                if not report['gate_passed'] and enforce_collection_gate(
                        evaluate=args.evaluate,fixed_exposure=args.dagger_fixed_exposure):
                    print('Student-only screen did not pass.' if args.evaluate else
                          'Gate failed: no further collection or training authorized by this protocol.',flush=True)
                    break
        print('REPORT',args.output/'collection.json',flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
