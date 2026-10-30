import * as THREE from 'three';
import { clearResources } from '../core/resources';
import type { Review } from '../core/types';

/** Every arrow follows one actual contact material point to the next pose. */
export class SupportSlipOverlay {
  private group = new THREE.Group();
  private review: Review | undefined;
  private gain = 20;
  private frame = 0;
  private details: HTMLElement | null = null;
  constructor(scene: THREE.Scene,private focus:(points:THREE.Vector3[])=>void) { scene.add(this.group); }
  private clear() {
    clearResources(this.group);
  }
  setup(review: Review | undefined) {
    this.clear();this.review=review;this.details=null;
    if (review?.kind !== 'support_slip') return;
    const panel=document.getElementById('reviewPanel')!;panel.hidden=false;
    const notice=document.createElement('p');notice.textContent=review.notice;panel.append(notice);
    const label=document.createElement('label');label.className='stack-field';
    label.append(document.createTextNode('位移箭头倍率（不改变统计值）'));
    const input=document.createElement('input');input.type='number';input.min='1';input.max='100';input.value=String(this.gain);
    input.onchange=()=>{this.gain=Math.max(1,Math.min(100,Number(input.value)||1));this.update(this.frame);};
    label.append(input);panel.append(label);
    const focus=document.createElement('button');focus.textContent='聚焦当前接触点';
    focus.onclick=()=>{
      const points=this.review?.slip_frames?.[this.frame]??[];
      if(points.length)this.focus(points.map(p=>new THREE.Vector3(...p.position)));
    };panel.append(focus);
    this.details=document.createElement('p');panel.append(this.details);
  }
  update(frame: number) {
    this.frame=frame;this.clear();if(this.review?.kind!=='support_slip')return;
    const points=this.review.slip_frames?.[frame]??[];
    const names=['左脚','右脚','左手','右手','左膝','右膝'];
    const key=(p:typeof points[number])=>`${p.part}:${p.surface}`;
    const minima=new Map<string, number>();
    points.forEach(p=>minima.set(key(p),Math.min(minima.get(key(p))??Infinity,p.motion_cm)));
    points.forEach(p=>{
      const color=p.motion_cm===minima.get(key(p))?0x36d99a:0xf3a540;
      const start=new THREE.Vector3(...p.position),delta=new THREE.Vector3(...p.delta);
      const dot=new THREE.Mesh(new THREE.SphereGeometry(.0015,8,6),new THREE.MeshBasicMaterial({color,depthTest:false}));
      dot.position.copy(start);dot.renderOrder=25;this.group.add(dot);
      const length=delta.length()*this.gain;
      if(length>1e-7){
        const arrow=new THREE.ArrowHelper(delta.normalize(),start,length,color,Math.min(.004,length*.25),Math.min(.002,length*.12));
        arrow.traverse((o:any)=>{if(o.material)o.material.depthTest=false;o.renderOrder=25;});this.group.add(arrow);
      }
    });
    const text=[...minima].map(([group,min])=>{
      const [part,surface]=group.split(':').map(Number);
      const values=points.filter(p=>key(p)===group).map(p=>p.motion_cm).sort((a,b)=>a-b);
      const mid=Math.floor(values.length/2),median=values.length%2?values[mid]:(values[mid-1]+values[mid])/2;
      return `${names[part]}·面${surface}：最低几何运动 ${min.toFixed(3)} / 中位 ${median.toFixed(3)} / 最大 ${values.at(-1)!.toFixed(3)} cm`;
    });
    const loads=this.review.support_frames?.[frame];
    const states:Record<string,string>={no_contact:'无有效接触',unloaded:'接触但卸载',load_bearing:'承重',unknown:'未知'};
    const loadText=loads?loads.map(p=>`${names[p.part]} ${states[p.load_state]??'未知'}：${p.normal_force_n?.toFixed(1)??'未知'} N，承重点相对切向速度 ${p.loaded_speed_cm_s?.toFixed(2)??'未知'} cm/s`).join('；'):'承重滑移：未知';
    if(this.details)this.details.textContent=(points.length?`${frame}→${frame+1}；${text.join('；')}。`:
      '此帧无可测几何区间，不能记作零滑移。')+loadText+'。';
  }
}
