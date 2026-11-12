import torch
import warp as wp
from supermimic.contact_collision_gpu import CollisionMonitor


def test_region_arrival_excludes_side_rows_and_old_support_has_no_penalty():
    with wp.ScopedDevice("cpu"):
        m=CollisionMonitor(1,2,1,'cpu',.002)
        ids=torch.tensor([0]);lo=torch.full((1,6,1,3),-.1);hi=-lo;valid=torch.ones(1,6,1,dtype=torch.bool)
        m.begin(ids,lo,hi,valid)
        groups=wp.array([0,1],dtype=int);query=wp.ones(2,dtype=wp.bool)
        points=wp.zeros(2,dtype=wp.vec3);faces=wp.array([0,1],dtype=wp.int64);upward=wp.array([True,False],dtype=wp.bool)
        m.classify(groups,query,points,faces,upward)
        assert wp.to_torch(m.counts).tolist()==[1,0,0,0,0,0]
        assert wp.to_torch(m.inside).tolist()==[True,True]
        normals=wp.array([[0.,0.,1.],[0.,0.,1.]],dtype=wp.vec3d);gap=wp.array([-.01,-.01],dtype=float)
        required=wp.ones(6,dtype=wp.bool);blocked=wp.array([True,False,False,False,False,False],dtype=wp.bool)
        stable=wp.zeros(6,dtype=wp.bool);bodies=wp.zeros(6,dtype=wp.int64)
        pos=wp.zeros((1,1),dtype=wp.vec3);rot=wp.array([[wp.mat33(1.)]],dtype=wp.mat33)
        m.sample(groups,query,points,normals,gap,required,blocked,bodies,pos,rot,wp.zeros(2,dtype=wp.spatial_vector))
        result=m.summary()
        # Unstable in-region top and side contacts are both excluded from avoidance.
        assert not result['collision_mask'].any()
        wp.to_torch(points)[:,0]=.2
        m.classify(groups,query,points,faces,upward)
        m.sample(groups,query,points,normals,gap,required,blocked,bodies,pos,rot,wp.zeros(2,dtype=wp.spatial_vector))
        result=m.summary()
        assert result['collision_mask'].tolist()==[[False,True,False,False,False,False]]
        assert torch.isclose(result['collision_gap'][0,1],torch.tensor(-.01))
        torch.testing.assert_close(result['collision_point'][0,1],torch.tensor([.2,0.,0.]))


def test_impulse_requires_persistence_and_resets_between_contacts():
    from supermimic.contact_collision_gpu import advance_collision
    with wp.ScopedDevice('cpu'):
        load=wp.array([6000.,300.],dtype=float);present=wp.ones(2,dtype=int)
        ticks=wp.zeros(2,dtype=int);impulse=wp.zeros(2,dtype=float);harmful=wp.zeros(2,dtype=wp.bool)
        def step():
            wp.launch(advance_collision,2,inputs=[load,present,.002,10,4.,ticks,impulse,harmful])
        step()
        # A 12 Ns, 2 ms pulse is not enough; duration is independent evidence.
        assert not wp.to_torch(harmful).any()
        wp.to_torch(present)[0]=0
        for _ in range(8):step()
        assert not wp.to_torch(harmful).any()
        step()
        assert wp.to_torch(harmful).tolist()==[False,True]
        assert wp.to_torch(impulse)[0]==0
        # Once detected, retain the decision even if contact ends later this frame.
        present.zero_();step()
        assert wp.to_torch(harmful).tolist()==[False,True]


def test_monitor_reset_clears_episode_but_control_begin_keeps_it():
    with wp.ScopedDevice('cpu'):
        m=CollisionMonitor(1,1,1,'cpu',.002)
        m.ticks.fill_(9);m.impulse.fill_(3.9);m.harmful.fill_(True)
        ids=torch.tensor([0]);lo=torch.zeros(1,6,1,3)
        m.begin(ids,lo,lo,torch.ones(1,6,1,dtype=torch.bool))
        assert wp.to_torch(m.ticks).eq(9).all()
        assert not wp.to_torch(m.harmful).any()
        m.reset(ids)
        assert not wp.to_torch(m.ticks).any()
        assert not wp.to_torch(m.impulse).any()


def test_low_impulse_persistent_touch_does_not_interrupt():
    from supermimic.contact_collision_gpu import advance_collision
    with wp.ScopedDevice('cpu'):
        load=wp.array([1.],dtype=float);present=wp.ones(1,dtype=int)
        ticks=wp.zeros(1,dtype=int);impulse=wp.zeros(1,dtype=float);harmful=wp.zeros(1,dtype=wp.bool)
        for _ in range(100):
            wp.launch(advance_collision,1,inputs=[load,present,.002,10,4.,ticks,impulse,harmful])
        assert not wp.to_torch(harmful).any()
