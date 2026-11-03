"""Continuous landing-process objective and first-arrival diagnostic baseline.

Inputs are persistent material nodes and distances to the actual finite terrain
surfaces, including non-target surfaces. They MUST NOT be selected/gated by
native active rows or proximity to the goal. Units are metres and seconds.
This is an optimization surrogate, not a measured contact location or success.
"""
import torch


class SoftArrival:
    """Streaming first-arrival distribution; no event latch or point winner.

    A survival weight spends probability on early proximity, so subsequent
    sliding cannot replace the early error. Remaining probability pays the
    current geometric error: avoiding all contact cannot simply erase loss.
    Initially supported parts need continuous separation evidence to rearm.
    Only reset/live masks are discrete; neither depends on contact activation.
    """
    def __init__(self, initial_gap, areas, *, initial_supported=None, width=.005, release_distance=.02,
                 arrival_time=.02, error_scale=.05):
        if min(width, release_distance, arrival_time, error_scale) <= 0:
            raise ValueError('Arrival length/time scales must be positive')
        self.width, self.release_distance = width, release_distance
        self.arrival_time, self.error_scale = arrival_time, error_scale
        self.area = areas / areas.sum(-1, keepdim=True)
        # Initial context is held fixed during a replay; gap is unsigned.
        self.initial_gap = initial_gap.detach().amin(-1)
        # Native reset support, when available, is fixed context, not a runtime
        # event switch. It excludes initial support even with coarse atlas nodes.
        self.initial_support = (torch.exp(-(self.initial_gap / width).square())
                                if initial_supported is None else initial_supported.to(initial_gap.dtype))
        self.excursion = torch.zeros_like(self.initial_gap)
        self.survival = torch.ones_like(self.initial_gap)
        self.integral = torch.zeros_like(self.initial_gap)
        self.loss = torch.zeros_like(self.initial_gap)

    def _arrival_rate(self, strength, eligibility, gap, dt, live):
        return eligibility * strength / self.arrival_time

    def _spatial_weights(self, evidence, strength):
        return evidence / strength.unsqueeze(-1)

    def update(self, points, gap, target, *, dt, live=None):
        if dt <= 0:
            raise ValueError('Positive sample interval required')
        if live is None:
            live = torch.ones_like(self.survival, dtype=torch.bool)
        separation = (gap.amin(-1) - self.initial_gap).clamp_min(0)
        excursion = torch.maximum(self.excursion, separation)
        self.excursion = torch.where(live, excursion, self.excursion)
        x = (self.excursion / self.release_distance).clamp(0, 1)
        released = x.square() * (3 - 2*x)
        eligibility = 1 - self.initial_support * (1 - released)
        # Rational tails retain approach gradients well outside the contact
        # band; area normalization prevents duplicate-node count dependence.
        evidence = self.area / (1 + (gap / self.width).square())
        strength = evidence.sum(-1)
        spatial = self._spatial_weights(evidence,strength)
        error = ((points - target.unsqueeze(-2)) / self.error_scale).square().sum(-1)
        spatial_error = (spatial * error).sum(-1)
        approach_error = (evidence / strength.unsqueeze(-1) * error).sum(-1)
        hazard = self._arrival_rate(strength, eligibility, gap, dt, live)
        fraction = -torch.expm1(-hazard * dt)
        mass = torch.where(live, self.survival * fraction, 0)
        self.integral = self.integral + mass * spatial_error
        self.survival = self.survival - mass
        value = self.integral + self.survival * spatial_error
        self.loss = torch.where(live, value, self.loss)
        return dict(loss=self.loss, arrival_mass=1-self.survival,
                    first_weight=mass, eligibility=eligibility,
                    spatial_error=spatial_error, approach_error=approach_error, proximity=strength)


class SoftLanding(SoftArrival):
    """Score the entire landing process, not a frozen first-point distribution.

    Native load continuously ramps the process in; fixed-node force weights
    score spreading, sliding and recontact without picking a witness. Unloaded
    flight supplies endpoint approach guidance but not landing-position history.
    No dropout resets history. Spatial errors are averaged, never positions.
    Load is same-solve measured evidence, not a policy input or a redefinition
    of contact activation. This score is not a calibrated success criterion.
    ``proximity_v1`` and ``incoming`` retain rejected geometric-only ablations.
    """
    def __init__(self, initial_gap, areas, *, dwell_time=.04,
                 phase_mode='load', load_scale=20., node_blend_load=1.,
                 incoming_gain=4., flux_smoothing=.5, **kwargs):
        super().__init__(initial_gap, areas, **kwargs)
        if min(dwell_time,incoming_gain,flux_smoothing,load_scale,node_blend_load) <= 0:
            raise ValueError('Positive landing time/flux scales required')
        if phase_mode not in ('load','incoming','proximity_v1'):
            raise ValueError('Unknown landing phase mode')
        self.phase_mode = phase_mode
        self.incoming_gain, self.flux_smoothing = incoming_gain, flux_smoothing
        self.load_scale, self.node_blend_load = load_scale, node_blend_load
        self.nearness = torch.exp(-.5*(self.initial_gap/self.width).square())
        self.dwell_time = dwell_time
        self.duration = torch.zeros_like(self.survival)
        self.proximity_time = torch.zeros_like(self.survival)
        self.process_error = torch.zeros_like(self.survival)
        self.position_loss = torch.zeros_like(self.survival)
        self.persistence_loss = torch.zeros_like(self.survival)

    def _arrival_rate(self, strength, eligibility, gap, dt, live):
        if self.phase_mode == 'load':
            return eligibility * self.support_fraction / self.arrival_time
        if self.phase_mode == 'proximity_v1':
            return super()._arrival_rate(strength,eligibility,gap,dt,live)
        near = torch.exp(-.5*(gap.amin(-1)/self.width).square())
        incoming = ((near-self.nearness)/dt).clamp_min(0)
        # C1 at zero: departure and constant near-ground hovering earn no
        # process-start credit. Units are change in nearness per second.
        flux = incoming.square()/(incoming+self.flux_smoothing)
        self.nearness = torch.where(live,near,self.nearness)
        return self.incoming_gain * eligibility * flux

    def _spatial_weights(self, evidence, strength):
        if self.phase_mode != 'load':
            return super()._spatial_weights(evidence,strength)
        # Native point forces are conservatively splatted onto the SAME fixed
        # material nodes. Weak new contacts get proportionally weak influence.
        # The geometry term supplies approach gradients when load is absent.
        combined = self.node_load + self.node_blend_load * evidence
        return combined / combined.sum(-1,keepdim=True)

    def update(self, points, gap, target, *, dt, live=None, normal_load=None, node_load=None):
        if self.phase_mode == 'load':
            if normal_load is None or node_load is None:
                raise ValueError('Load-based landing needs same-solve native normal_load and conservative node_load')
            self.support_fraction = normal_load / (normal_load+self.load_scale)
            self.node_load = node_load
        previous_phase = 1-self.survival
        previous_loss = self.loss
        row = super().update(points, gap, target, dt=dt, live=live)
        phase = (previous_phase + row['arrival_mass']) / 2
        if self.phase_mode == 'incoming':
            # Flight/approach contributes endpoint guidance, not accumulated
            # landing-position history. Keep history across any later gaps.
            phase = phase * self.nearness
        elif self.phase_mode == 'load':
            # No accumulated landing-position history without actual load.
            # This does not redefine native activation or latch on a force threshold.
            phase = phase * self.support_fraction
        if live is not None:
            phase = torch.where(live, phase, 0)
        time = phase * dt
        self.duration = self.duration + time
        presence = 1 if self.phase_mode == 'load' else row['proximity']
        self.proximity_time = self.proximity_time + time * presence * row['eligibility']
        self.process_error = self.process_error + time * row['spatial_error']
        # The residual approach term supplies gradients before a landing and
        # decreases smoothly as measured geometric process time accumulates.
        # Never put instantaneous force redistribution into the endpoint term.
        # Geometry guides approach; loaded spatial evidence enters via impulse-
        # weighted time integration only, so a force switch is not a point switch.
        approach = row['approach_error'] if self.phase_mode=='load' else row['spatial_error']
        position = (self.process_error + self.dwell_time * approach) / (self.duration+self.dwell_time)
        persistence = self.dwell_time / (self.proximity_time+self.dwell_time)
        self.position_loss = position if live is None else torch.where(live,position,self.position_loss)
        self.persistence_loss = persistence if live is None else torch.where(live,persistence,self.persistence_loss)
        value = position+persistence
        self.loss = value if live is None else torch.where(live,value,previous_loss)
        return dict(**{k:v for k,v in row.items() if k != 'loss'},loss=self.loss,
                    position_loss=self.position_loss,persistence_loss=self.persistence_loss,
                    process_time=self.duration,proximity_time=self.proximity_time)


def soft_arrival_loss(points, gap, target, areas, initial_gap, *, dt, live=None, **kwargs):
    """Batch reference: [time,batch,part,node,(xyz)]; same rule as streaming."""
    state = SoftArrival(initial_gap, areas, **kwargs)
    rows = []
    for t in range(len(points)):
        rows.append(state.update(points[t], gap[t], target, dt=dt,
                                 live=None if live is None else live[t]))
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


def soft_landing_loss(points, gap, target, areas, initial_gap, *, dt, live=None, normal_load=None,node_load=None, **kwargs):
    """Whole-process reference; the first-arrival function is diagnostic only."""
    state = SoftLanding(initial_gap,areas,**kwargs)
    rows = [state.update(points[t],gap[t],target,dt=dt,live=None if live is None else live[t],
                         normal_load=None if normal_load is None else normal_load[t],
                         node_load=None if node_load is None else node_load[t])
            for t in range(len(points))]
    return {key:torch.stack([row[key] for row in rows]) for key in rows[0]}
