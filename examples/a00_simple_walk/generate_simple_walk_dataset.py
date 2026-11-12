#!/usr/bin/env python3
"""Generate a safe 29-DOF A00 walking reference for SuperMimic/MuJoCo."""
import argparse, json
from pathlib import Path
import numpy as np
JOINTS=['right_hip_pitch','right_hip_roll','right_hip_yaw','right_knee','right_ankle_pitch','right_ankle_roll','left_hip_pitch','left_hip_roll','left_hip_yaw','left_knee','left_ankle_pitch','left_ankle_roll','waist_yaw','waist_roll','waist_pitch','left_shoulder_pitch','left_shoulder_roll','left_shoulder_yaw','left_elbow','left_wrist_yaw','left_wrist_roll','left_wrist_pitch','right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow','right_wrist_yaw','right_wrist_roll','right_wrist_pitch']
def make(T=10.0, hz=50.0, speed=0.25):
 t=np.arange(0,T,1/hz); ph=2*np.pi*t/0.6; q=np.zeros((len(t),29)); q[:,0:29]=np.array([.2,0,-.07,.3,-.15,0,-.2,0,.07,-.3,-.15,0]+[0]*17)
 s=np.sin(ph); c=np.cos(ph)
 # legs: opposite phase, conservative amplitudes
 for side,sgn in [(0,1),(6,-1)]:
  z=sgn
  q[:,side+0] += 0.16*z*s
  q[:,side+1] += 0.035*z*c
  q[:,side+3] += 0.18*(1-c)/2
  q[:,side+4] += -0.08*z*s
  q[:,side+5] += 0.025*z*c
 # pelvis and arms counter-swing
 q[:,12] += 0.025*s; q[:,13] += 0.02*c; q[:,14] += 0.03*c
 q[:,15] += -0.10*s; q[:,22] += 0.10*s
 q[:,18] += 0.15; q[:,25] += 0.15
 q[:,16] += 0.02*c; q[:,23] -= 0.02*c
 qd=np.gradient(q,1/hz,axis=0); phase=np.column_stack([np.sin(ph),np.cos(ph)])
 cmd=np.column_stack([np.full(len(t),speed),np.zeros((len(t),2))])
 return t,q,qd,phase,cmd
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--out',default='simple_walk_a00.npz'); ap.add_argument('--seconds',type=float,default=10); a=ap.parse_args()
 t,q,qd,phase,cmd=make(T=a.seconds); np.savez(a.out,time=t,joint_pos=q,joint_vel=qd,phase=phase,command=cmd,actions=q)
 Path(a.out+'.json').write_text(json.dumps({'joint_names':JOINTS,'hz':50,'gait_period_s':0.6,'observation_dim':490,'action_dim':29,'units':'radians'} ,indent=2))
 print(f'wrote {a.out}: {len(t)} frames, 29 joints')
