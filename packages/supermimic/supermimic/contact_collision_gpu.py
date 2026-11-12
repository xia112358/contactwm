"""Native contact rows -> bounded per-part collision witnesses on the GPU."""
import torch
import warp as wp

@wp.kernel
def classify(groups:wp.array[int],query:wp.array[wp.bool],point:wp.array[wp.vec3],
             lower:wp.array[wp.vec3],upper:wp.array[wp.vec3],valid:wp.array[wp.bool],K:int,
             inside:wp.array[wp.bool],counts:wp.array[int],face:wp.array[wp.int64],upward:wp.array[wp.bool]):
    i=wp.tid();inside[i]=False
    if not query[i]:return
    g=groups[i];p=point[i]
    for k in range(K):
        j=g*K+k
        if valid[j] and p[0]>=lower[j][0] and p[0]<=upper[j][0] and p[1]>=lower[j][1] and p[1]<=upper[j][1] and p[2]>=lower[j][2] and p[2]<=upper[j][2]:inside[i]=True
    if inside[i] and face[i]>=0:
        if upward[face[i]]:wp.atomic_add(counts,g,1)

@wp.kernel
def score_rows(groups:wp.array[int],query:wp.array[wp.bool],point:wp.array[wp.vec3],normal:wp.array[wp.vec3d],gap:wp.array[float],
               required:wp.array[wp.bool],blocked:wp.array[wp.bool],inside:wp.array[wp.bool],
               bodies:wp.array[wp.int64],p:wp.array2d[wp.vec3],R:wp.array2d[wp.mat33],oldp:wp.array[wp.vec3],oldR:wp.array[wp.mat33],
               old_valid:wp.array[wp.bool],dt:float,scores:wp.array[float],maximum:wp.array[float]):
    i=wp.tid();scores[i]=-1.0
    if not query[i]:return
    g=groups[i];w=g//6;b=int(bodies[g%6])
    if not required[g] or blocked[g] or inside[i]:return
    clearance_error=wp.max(0.0,.01-gap[i])/.05
    score=clearance_error*clearance_error
    scores[i]=score;wp.atomic_max(maximum,g,score)

@wp.kernel
def choose(groups:wp.array[int],scores:wp.array[float],maximum:wp.array[float],winner:wp.array[int]):
    i=wp.tid()
    if scores[i]<0.0:return
    g=groups[i]
    if scores[i]==maximum[g]:wp.atomic_min(winner,g,i)

@wp.kernel
def witness(winner:wp.array[int],maximum:wp.array[float],best:wp.array[float],capacity:int,
            point:wp.array[wp.vec3],normal:wp.array[wp.vec3d],gap:wp.array[float],inside:wp.array[wp.bool],
            bodies:wp.array[wp.int64],p:wp.array2d[wp.vec3],R:wp.array2d[wp.mat33],oldp:wp.array[wp.vec3],oldR:wp.array[wp.mat33],old_valid:wp.array[wp.bool],dt:float,
            mask:wp.array[wp.bool],local:wp.array[wp.vec3],out_point:wp.array[wp.vec3],out_normal:wp.array[wp.vec3],out_gap:wp.array[float],speed:wp.array[float],in_region:wp.array[wp.bool]):
    g=wp.tid();w=g//6;b=int(bodies[g%6]);i=winner[g]
    if i<capacity and maximum[g]>best[g]:
        best[g]=maximum[g];mask[g]=True;out_point[g]=point[i]
        local[g]=wp.transpose(R[w,b])*(point[i]-p[w,b]);out_normal[g]=wp.vec3(normal[i]);out_gap[g]=gap[i]
        velocity=wp.vec3(0.0)
        if old_valid[g]:velocity=(point[i]-(oldp[g]+oldR[g]*local[g]))/dt
        speed[g]=wp.max(0.0,-wp.dot(velocity,out_normal[g]));in_region[g]=inside[i]
    oldp[g]=p[w,b];oldR[g]=R[w,b];old_valid[g]=True

# Calibrated against successful source scrapes and frozen #127 native replay.
COLLISION_MIN_SECONDS = .020
COLLISION_MIN_IMPULSE = 4.0

@wp.kernel
def sum_collision_force(groups:wp.array[int],query:wp.array[wp.bool],normal:wp.array[wp.vec3d],
                        force:wp.array[wp.spatial_vector],required:wp.array[wp.bool],blocked:wp.array[wp.bool],
                        inside:wp.array[wp.bool],load:wp.array[float],present:wp.array[int]):
    i=wp.tid()
    if not query[i]:return
    g=groups[i]
    if not required[g] or blocked[g] or inside[i]:return
    f=force[i]
    wp.atomic_add(load,g,wp.abs(wp.dot(wp.vec3(f[0],f[1],f[2]),wp.vec3(normal[i]))))
    wp.atomic_add(present,g,1)

@wp.kernel
def advance_collision(load:wp.array[float],present:wp.array[int],dt:float,min_ticks:int,min_impulse:float,
                      ticks:wp.array[int],impulse:wp.array[float],harmful:wp.array[wp.bool]):
    g=wp.tid()
    if present[g]>0:
        ticks[g]+=1
        impulse[g]+=load[g]*dt
        if ticks[g]>=min_ticks and impulse[g]>=min_impulse:harmful[g]=True
    else:
        ticks[g]=0
        impulse[g]=0.0

class CollisionMonitor:
    def __init__(self,worlds,capacity,K,device,dt):
        self.N=worlds;self.G=worlds*6;self.capacity=capacity;self.K=K;self.device=device;self.dt=dt
        alloc=lambda n,t:wp.zeros(n,dtype=t,device=device)
        self.lower=alloc(self.G*K,wp.vec3);self.upper=alloc(self.G*K,wp.vec3);self.valid=alloc(self.G*K,wp.bool)
        self.inside=alloc(capacity,wp.bool);self.counts=alloc(self.G,wp.int32)
        self.scores=alloc(capacity,wp.float32);self.maximum=alloc(self.G,wp.float32);self.winner=alloc(self.G,wp.int32)
        self.best=alloc(self.G,wp.float32);self.mask=alloc(self.G,wp.bool);self.local=alloc(self.G,wp.vec3);self.point=alloc(self.G,wp.vec3);self.normal=alloc(self.G,wp.vec3)
        self.gap=alloc(self.G,wp.float32);self.speed=alloc(self.G,wp.float32);self.in_region=alloc(self.G,wp.bool)
        self.oldp=alloc(self.G,wp.vec3);self.oldR=alloc(self.G,wp.mat33);self.old_valid=alloc(self.G,wp.bool)
        self.load=alloc(self.G,wp.float32);self.present=alloc(self.G,wp.int32)
        self.ticks=alloc(self.G,wp.int32);self.impulse=alloc(self.G,wp.float32);self.harmful=alloc(self.G,wp.bool)
        import math
        self.min_ticks=math.ceil(COLLISION_MIN_SECONDS/dt)
    def reset(self,ids):
        for field in (self.old_valid,self.ticks,self.impulse,self.harmful):
            wp.to_torch(field).reshape(self.N,6)[ids]=0
    def begin(self,ids,lower,upper,valid):
        wp.to_torch(self.lower).reshape(self.N,6,self.K,3)[ids]=lower
        wp.to_torch(self.upper).reshape(self.N,6,self.K,3)[ids]=upper
        wp.to_torch(self.valid).reshape(self.N,6,self.K)[ids]=valid
        self.harmful.zero_();self.best.fill_(-1.0);self.mask.zero_();self.local.zero_();self.point.zero_();self.normal.zero_();self.gap.zero_();self.speed.zero_();self.in_region.zero_()
    def classify(self,groups,query,point,face,upward):
        self.counts.zero_();wp.launch(classify,self.capacity,inputs=[groups,query,point,self.lower,self.upper,self.valid,self.K,self.inside,self.counts,face,upward],device=self.device)
    def sample(self,groups,query,point,normal,gap,required,blocked,bodies,p,R,force):
        self.load.zero_();self.present.zero_()
        wp.launch(sum_collision_force,self.capacity,inputs=[groups,query,normal,force,required,blocked,self.inside,self.load,self.present],device=self.device)
        wp.launch(advance_collision,self.G,inputs=[self.load,self.present,self.dt,self.min_ticks,COLLISION_MIN_IMPULSE,self.ticks,self.impulse,self.harmful],device=self.device)
        self.maximum.fill_(-1.0);self.winner.fill_(self.capacity)
        wp.launch(score_rows,self.capacity,inputs=[groups,query,point,normal,gap,required,blocked,self.inside,bodies,p,R,self.oldp,self.oldR,self.old_valid,self.dt,self.scores,self.maximum],device=self.device)
        wp.launch(choose,self.capacity,inputs=[groups,self.scores,self.maximum,self.winner],device=self.device)
        wp.launch(witness,self.G,inputs=[self.winner,self.maximum,self.best,self.capacity,point,normal,gap,self.inside,bodies,p,R,self.oldp,self.oldR,self.old_valid,self.dt,self.mask,self.local,self.point,self.normal,self.gap,self.speed,self.in_region],device=self.device)
    def summary(self):
        return {'collision_'+k:wp.to_torch(getattr(self,k)).reshape((self.N,6,3) if k in ('local','point','normal') else (self.N,6)).clone() for k in ('mask','local','point','normal','gap','speed','in_region','harmful')}
