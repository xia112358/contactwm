"""ACP Table S2 startup dynamics and episode sensor offsets on Newton."""
from __future__ import annotations

import math

import torch

from holosoma.managers.randomization.base import RandomizationTermBase


def observation_noise_enabled(env):
    settings = env.acp_settings
    return settings.get('observation_noise', False) and (
        not env.is_evaluating or settings.get('evaluation_randomization', False))


def dynamics_enabled(env):
    settings = env.acp_settings
    return settings.get('domain_randomization', False) and (
        not env.is_evaluating or settings.get('evaluation_randomization', False))


class PaperRandomization(RandomizationTermBase):
    def setup(self):
        env = self.env
        self.enabled = bool(self.cfg.params.get('enabled', True))
        shape = (env.num_envs, env.num_dof)
        env._randomize_ctrl_delay = False  # ACP implements delay at physics substeps.
        env._pending_torque_rfi = (False, 0.)
        env.acp_actuation_delay_s = torch.zeros(env.num_envs, device=env.device)
        env.acp_odometry_delay_s = torch.zeros_like(env.acp_actuation_delay_s)
        env.acp_odometry_drift = torch.zeros(env.num_envs, 3, device=env.device)
        kp = torch.ones(shape, device=env.device)
        kd = torch.ones_like(kp)
        bias = torch.zeros_like(kp)
        if self.enabled:
            kp.uniform_(0.9, 1.1)
            kd.uniform_(0.9, 1.1)
            ankle = torch.tensor(['ankle' in name for name in env.robot_config.dof_names], device=env.device)
            scale = torch.where(ankle, 0.03, 0.01)
            bias = (2*torch.rand_like(bias)-1)*scale
            env.acp_actuation_delay_s.uniform_(0., 0.02)
            env.acp_odometry_delay_s.uniform_(0.05, 0.1)
            self._randomize_native_properties()
        env.default_dof_pos = env.default_dof_pos_base.expand(*shape)+bias
        env.action_manager.get_term('joint_control').update_pd_scales(
            torch.arange(env.num_envs, device=env.device), kp, kd)
        self.startup_parameters = dict(kp_scale=kp.clone(), kd_scale=kd.clone(),
                                      joint_zero_bias=bias.clone(),
                                      actuator_delay_s=env.acp_actuation_delay_s.clone(),
                                      odometry_delay_s=env.acp_odometry_delay_s.clone())

    def _randomize_native_properties(self):
        import warp as wp
        from loguru import logger
        from holosoma.managers.randomization.terms.locomotion import (
            _is_isaaclab_newton_backend, randomize_robot_rigid_body_material_startup)
        env = self.env
        sim = env.simulator
        if not _is_isaaclab_newton_backend(sim):
            raise RuntimeError('ACP dynamics randomization requires the maintained Newton backend')
        randomize_robot_rigid_body_material_startup(env, static_friction_range=(0.3, 1.2),
            dynamic_friction_range=(0.3, 1.2), restitution_range=(0., 0.5))
        robot = sim._robot
        leg_names = [name for name in sim._robot.body_names
                     if name.startswith(('left_', 'right_')) and any(x in name for x in ('hip', 'knee', 'ankle'))]
        if not leg_names:
            raise RuntimeError('Canonical G1 leg bodies are missing from the native articulation')
        for names, interval, operation in (
            (leg_names, (0.85, 1.15), 'scale'), ([env.robot_config.torso_name], (-2., 4.), 'add')):
            body_ids = [robot.body_names.index(name) for name in names]
            original = robot.data.body_mass.torch[:, body_ids].clone()
            sample = torch.empty_like(original).uniform_(*interval)
            masses = original*sample if operation == 'scale' else original+sample
            if (masses <= 0).any():
                raise RuntimeError('ACP mass randomization produced a nonpositive body mass')
            inertia = robot.data.body_inertia.torch[:, body_ids].clone()
            inertia *= (masses/original)[..., None]
            # This pinned Newton/Warp build requires explicit Warp arguments.
            # Keep the compatibility conversion local to ACP startup.
            robot.set_masses_index(masses=wp.from_torch(masses.contiguous()), body_ids=body_ids)
            robot.set_inertias_index(inertias=wp.from_torch(inertia.contiguous()), body_ids=body_ids)
            actual = robot.data.body_mass.torch[:, body_ids]
            if not torch.allclose(actual, masses):
                raise RuntimeError('Native ACP body mass write did not take effect')
            logger.info('ACP native {} mass samples: [{:.4f}, {:.4f}]', operation,
                        sample.min().item(), sample.max().item())
        torso_id = robot.body_names.index(env.robot_config.torso_name)
        bias = torch.empty(env.num_envs, 1, 3, device=env.device).uniform_(-1., 1.)
        bias *= torch.tensor([0.025, 0.05, 0.05], device=env.device)
        coms = sim.default_coms[:, [torso_id], :3].clone()+bias
        robot.set_coms_index(coms=wp.from_torch(coms.contiguous(), dtype=wp.vec3), body_ids=[torso_id])
        sim.base_com_bias.copy_(bias[:, 0])
        armature = robot.data.default_joint_armature.torch.clone()
        armature *= torch.empty_like(armature).uniform_(0.9, 1.1)
        robot.write_joint_armature_to_sim_index(armature=wp.from_torch(armature.contiguous()))
        # Newton has one continuously sampled friction coefficient per shape;
        # it cannot represent independently sampled static/dynamic coefficients.
        env.acp_native_material_contract = 'newton_single_mu_per_shape_uniform_0.3_1.2'

    def reset(self, env_ids):
        env = self.env
        ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
               else torch.as_tensor(env_ids, dtype=torch.long, device=env.device))
        if not self.enabled or not dynamics_enabled(env):
            env.acp_odometry_drift[ids] = 0
            return
        angle = torch.rand(len(ids), device=env.device)*(2*math.pi)
        radius = 0.05*torch.rand(len(ids), device=env.device).sqrt()
        env.acp_odometry_drift[ids, 0] = radius*angle.cos()
        env.acp_odometry_drift[ids, 1] = radius*angle.sin()
        env.acp_odometry_drift[ids, 2] = torch.empty(len(ids), device=env.device).uniform_(-0.05, 0.05)

    def step(self):
        pass
