import unittest
import numpy as np
from supermimic.contact_body_summary import summarize

class BodySummaryTest(unittest.TestCase):
    def fixture(self):
        world=np.full(21,-1,int);slot=world.copy()
        world[[10,11,20]]=[0,0,1];slot[[10,11,20]]=[0,1,1]
        snap=dict(count=4,active=np.ones(4,bool),constraint_allocated=np.ones(4,bool),worldid=np.array([0,0,1,0]),
            body0=np.array([-1,-1,20,10]),body1=np.array([10,10,-1,11]),
            shape0=np.array([100,100,2,0]),shape1=np.array([0,0,101,1]),
            frame_w=np.repeat(np.eye(3)[None],4,axis=0),dist=np.array([-.001,-.002,-.003,-.004]),
            force_on_body1_w=np.array([[0,0,10],[0,0,20],[0,0,-5],[100,0,0]]),
            position_w=np.array([[0,0,0],[2,0,0],[4,0,0],[9,0,0]],float))
        return snap,world,slot

    def run_summary(self,observed):
        snap,world,slot=self.fixture()
        return summarize(snap,world,slot,np.array([100,101]),2,2,np.array(observed))

    def test_multipoint_force_sign_and_self_collision(self):
        r=self.run_summary([True,True])
        self.assertEqual(r['body_contact_count'].tolist(),[[2,0],[0,1]])
        np.testing.assert_allclose(r['body_contact_force'][0,0],[0,0,30])
        np.testing.assert_allclose(r['body_contact_force'][1,1],[0,0,5])
        np.testing.assert_allclose(r['body_contact_point'][0,0],[1,0,0])
        self.assertAlmostEqual(float(r['body_contact_penetration'][0,0]),.002)

    def test_unobserved_is_unknown(self):
        r=self.run_summary([True,False])
        self.assertFalse(r['body_contact_known'][1].any())
        self.assertFalse(r['body_contact_active'][1].any())
        self.assertTrue(r['body_contact_known'][0].all())

    def test_invalid_constraints_rejected(self):
        for field in ['normal','allocation']:
            s,w,b=self.fixture()
            if field=='normal':s['frame_w'][0]=0
            else:s['constraint_allocated'][0]=False
            with self.assertRaises(ValueError):summarize(s,w,b,np.array([100,101]),2,2,np.ones(2,bool))

if __name__=='__main__':unittest.main()
