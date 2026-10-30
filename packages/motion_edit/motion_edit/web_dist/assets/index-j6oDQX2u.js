var Bu=Object.defineProperty;var zu=(i,t,e)=>t in i?Bu(i,t,{enumerable:!0,configurable:!0,writable:!0,value:e}):i[t]=e;var oe=(i,t,e)=>zu(i,typeof t!="symbol"?t+"":t,e);(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))n(s);new MutationObserver(s=>{for(const r of s)if(r.type==="childList")for(const a of r.addedNodes)a.tagName==="LINK"&&a.rel==="modulepreload"&&n(a)}).observe(document,{childList:!0,subtree:!0});function e(s){const r={};return s.integrity&&(r.integrity=s.integrity),s.referrerPolicy&&(r.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?r.credentials="include":s.crossOrigin==="anonymous"?r.credentials="omit":r.credentials="same-origin",r}function n(s){if(s.ep)return;s.ep=!0;const r=e(s);fetch(s.href,r)}})();/**
 * @license
 * Copyright 2010-2024 Three.js Authors
 * SPDX-License-Identifier: MIT
 */const ml="170",ps={ROTATE:0,DOLLY:1,PAN:2},ds={ROTATE:0,PAN:1,DOLLY_PAN:2,DOLLY_ROTATE:3},Vu=0,Wl=1,Hu=2,Ah=1,Gu=2,Qn=3,ii=0,ln=1,Pn=2,Si=0,ms=1,Xl=2,ql=3,jl=4,Wu=5,zi=100,Xu=101,qu=102,ju=103,Yu=104,$u=200,Ku=201,Ju=202,Zu=203,To=204,wo=205,Qu=206,td=207,ed=208,nd=209,id=210,sd=211,rd=212,ad=213,od=214,Ao=0,Co=1,Ro=2,ys=3,Po=4,Lo=5,No=6,Do=7,Ma=0,ld=1,cd=2,bi=0,hd=1,ud=2,dd=3,Ch=4,fd=5,pd=6,md=7,Yl="attached",gd="detached",Rh=300,Ms=301,Ss=302,Io=303,Uo=304,Sa=306,Hi=1e3,Nn=1001,Fo=1002,mn=1003,_d=1004,br=1005,fn=1006,Ua=1007,ti=1008,si=1009,Ph=1010,Lh=1011,ir=1012,gl=1013,Gi=1014,Bn=1015,hr=1016,_l=1017,vl=1018,bs=1020,Nh=35902,Dh=1021,Ih=1022,Sn=1023,Uh=1024,Fh=1025,gs=1026,Es=1027,Oh=1028,xl=1029,kh=1030,yl=1031,Ml=1033,sa=33776,ra=33777,aa=33778,oa=33779,Oo=35840,ko=35841,Bo=35842,zo=35843,Vo=36196,Ho=37492,Go=37496,Wo=37808,Xo=37809,qo=37810,jo=37811,Yo=37812,$o=37813,Ko=37814,Jo=37815,Zo=37816,Qo=37817,tl=37818,el=37819,nl=37820,il=37821,la=36492,sl=36494,rl=36495,Bh=36283,al=36284,ol=36285,ll=36286,da=2300,cl=2301,Fa=2302,$l=2400,Kl=2401,Jl=2402,vd=2500,xd=3200,yd=3201,ba=0,Md=1,yi="",Pe="srgb",Ds="srgb-linear",Ea="linear",Ee="srgb",Ji=7680,Zl=519,Sd=512,bd=513,Ed=514,zh=515,Td=516,wd=517,Ad=518,Cd=519,Ql=35044,tc="300 es",ei=2e3,fa=2001;class qi{addEventListener(t,e){this._listeners===void 0&&(this._listeners={});const n=this._listeners;n[t]===void 0&&(n[t]=[]),n[t].indexOf(e)===-1&&n[t].push(e)}hasEventListener(t,e){if(this._listeners===void 0)return!1;const n=this._listeners;return n[t]!==void 0&&n[t].indexOf(e)!==-1}removeEventListener(t,e){if(this._listeners===void 0)return;const s=this._listeners[t];if(s!==void 0){const r=s.indexOf(e);r!==-1&&s.splice(r,1)}}dispatchEvent(t){if(this._listeners===void 0)return;const n=this._listeners[t.type];if(n!==void 0){t.target=this;const s=n.slice(0);for(let r=0,a=s.length;r<a;r++)s[r].call(this,t);t.target=null}}}const $e=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"];let ec=1234567;const Zs=Math.PI/180,Ts=180/Math.PI;function Ci(){const i=Math.random()*4294967295|0,t=Math.random()*4294967295|0,e=Math.random()*4294967295|0,n=Math.random()*4294967295|0;return($e[i&255]+$e[i>>8&255]+$e[i>>16&255]+$e[i>>24&255]+"-"+$e[t&255]+$e[t>>8&255]+"-"+$e[t>>16&15|64]+$e[t>>24&255]+"-"+$e[e&63|128]+$e[e>>8&255]+"-"+$e[e>>16&255]+$e[e>>24&255]+$e[n&255]+$e[n>>8&255]+$e[n>>16&255]+$e[n>>24&255]).toLowerCase()}function Xe(i,t,e){return Math.max(t,Math.min(e,i))}function Sl(i,t){return(i%t+t)%t}function Rd(i,t,e,n,s){return n+(i-t)*(s-n)/(e-t)}function Pd(i,t,e){return i!==t?(e-i)/(t-i):0}function Qs(i,t,e){return(1-e)*i+e*t}function Ld(i,t,e,n){return Qs(i,t,1-Math.exp(-e*n))}function Nd(i,t=1){return t-Math.abs(Sl(i,t*2)-t)}function Dd(i,t,e){return i<=t?0:i>=e?1:(i=(i-t)/(e-t),i*i*(3-2*i))}function Id(i,t,e){return i<=t?0:i>=e?1:(i=(i-t)/(e-t),i*i*i*(i*(i*6-15)+10))}function Ud(i,t){return i+Math.floor(Math.random()*(t-i+1))}function Fd(i,t){return i+Math.random()*(t-i)}function Od(i){return i*(.5-Math.random())}function kd(i){i!==void 0&&(ec=i);let t=ec+=1831565813;return t=Math.imul(t^t>>>15,t|1),t^=t+Math.imul(t^t>>>7,t|61),((t^t>>>14)>>>0)/4294967296}function Bd(i){return i*Zs}function zd(i){return i*Ts}function Vd(i){return(i&i-1)===0&&i!==0}function Hd(i){return Math.pow(2,Math.ceil(Math.log(i)/Math.LN2))}function Gd(i){return Math.pow(2,Math.floor(Math.log(i)/Math.LN2))}function Wd(i,t,e,n,s){const r=Math.cos,a=Math.sin,o=r(e/2),c=a(e/2),l=r((t+n)/2),h=a((t+n)/2),u=r((t-n)/2),d=a((t-n)/2),p=r((n-t)/2),g=a((n-t)/2);switch(s){case"XYX":i.set(o*h,c*u,c*d,o*l);break;case"YZY":i.set(c*d,o*h,c*u,o*l);break;case"ZXZ":i.set(c*u,c*d,o*h,o*l);break;case"XZX":i.set(o*h,c*g,c*p,o*l);break;case"YXY":i.set(c*p,o*h,c*g,o*l);break;case"ZYZ":i.set(c*g,c*p,o*h,o*l);break;default:console.warn("THREE.MathUtils: .setQuaternionFromProperEuler() encountered an unknown order: "+s)}}function us(i,t){switch(t.constructor){case Float32Array:return i;case Uint32Array:return i/4294967295;case Uint16Array:return i/65535;case Uint8Array:return i/255;case Int32Array:return Math.max(i/2147483647,-1);case Int16Array:return Math.max(i/32767,-1);case Int8Array:return Math.max(i/127,-1);default:throw new Error("Invalid component type.")}}function tn(i,t){switch(t.constructor){case Float32Array:return i;case Uint32Array:return Math.round(i*4294967295);case Uint16Array:return Math.round(i*65535);case Uint8Array:return Math.round(i*255);case Int32Array:return Math.round(i*2147483647);case Int16Array:return Math.round(i*32767);case Int8Array:return Math.round(i*127);default:throw new Error("Invalid component type.")}}const Mi={DEG2RAD:Zs,RAD2DEG:Ts,generateUUID:Ci,clamp:Xe,euclideanModulo:Sl,mapLinear:Rd,inverseLerp:Pd,lerp:Qs,damp:Ld,pingpong:Nd,smoothstep:Dd,smootherstep:Id,randInt:Ud,randFloat:Fd,randFloatSpread:Od,seededRandom:kd,degToRad:Bd,radToDeg:zd,isPowerOfTwo:Vd,ceilPowerOfTwo:Hd,floorPowerOfTwo:Gd,setQuaternionFromProperEuler:Wd,normalize:tn,denormalize:us};class Lt{constructor(t=0,e=0){Lt.prototype.isVector2=!0,this.x=t,this.y=e}get width(){return this.x}set width(t){this.x=t}get height(){return this.y}set height(t){this.y=t}set(t,e){return this.x=t,this.y=e,this}setScalar(t){return this.x=t,this.y=t,this}setX(t){return this.x=t,this}setY(t){return this.y=t,this}setComponent(t,e){switch(t){case 0:this.x=e;break;case 1:this.y=e;break;default:throw new Error("index is out of range: "+t)}return this}getComponent(t){switch(t){case 0:return this.x;case 1:return this.y;default:throw new Error("index is out of range: "+t)}}clone(){return new this.constructor(this.x,this.y)}copy(t){return this.x=t.x,this.y=t.y,this}add(t){return this.x+=t.x,this.y+=t.y,this}addScalar(t){return this.x+=t,this.y+=t,this}addVectors(t,e){return this.x=t.x+e.x,this.y=t.y+e.y,this}addScaledVector(t,e){return this.x+=t.x*e,this.y+=t.y*e,this}sub(t){return this.x-=t.x,this.y-=t.y,this}subScalar(t){return this.x-=t,this.y-=t,this}subVectors(t,e){return this.x=t.x-e.x,this.y=t.y-e.y,this}multiply(t){return this.x*=t.x,this.y*=t.y,this}multiplyScalar(t){return this.x*=t,this.y*=t,this}divide(t){return this.x/=t.x,this.y/=t.y,this}divideScalar(t){return this.multiplyScalar(1/t)}applyMatrix3(t){const e=this.x,n=this.y,s=t.elements;return this.x=s[0]*e+s[3]*n+s[6],this.y=s[1]*e+s[4]*n+s[7],this}min(t){return this.x=Math.min(this.x,t.x),this.y=Math.min(this.y,t.y),this}max(t){return this.x=Math.max(this.x,t.x),this.y=Math.max(this.y,t.y),this}clamp(t,e){return this.x=Math.max(t.x,Math.min(e.x,this.x)),this.y=Math.max(t.y,Math.min(e.y,this.y)),this}clampScalar(t,e){return this.x=Math.max(t,Math.min(e,this.x)),this.y=Math.max(t,Math.min(e,this.y)),this}clampLength(t,e){const n=this.length();return this.divideScalar(n||1).multiplyScalar(Math.max(t,Math.min(e,n)))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(t){return this.x*t.x+this.y*t.y}cross(t){return this.x*t.y-this.y*t.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(t){const e=Math.sqrt(this.lengthSq()*t.lengthSq());if(e===0)return Math.PI/2;const n=this.dot(t)/e;return Math.acos(Xe(n,-1,1))}distanceTo(t){return Math.sqrt(this.distanceToSquared(t))}distanceToSquared(t){const e=this.x-t.x,n=this.y-t.y;return e*e+n*n}manhattanDistanceTo(t){return Math.abs(this.x-t.x)+Math.abs(this.y-t.y)}setLength(t){return this.normalize().multiplyScalar(t)}lerp(t,e){return this.x+=(t.x-this.x)*e,this.y+=(t.y-this.y)*e,this}lerpVectors(t,e,n){return this.x=t.x+(e.x-t.x)*n,this.y=t.y+(e.y-t.y)*n,this}equals(t){return t.x===this.x&&t.y===this.y}fromArray(t,e=0){return this.x=t[e],this.y=t[e+1],this}toArray(t=[],e=0){return t[e]=this.x,t[e+1]=this.y,t}fromBufferAttribute(t,e){return this.x=t.getX(e),this.y=t.getY(e),this}rotateAround(t,e){const n=Math.cos(e),s=Math.sin(e),r=this.x-t.x,a=this.y-t.y;return this.x=r*n-a*s+t.x,this.y=r*s+a*n+t.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}}class ie{constructor(t,e,n,s,r,a,o,c,l){ie.prototype.isMatrix3=!0,this.elements=[1,0,0,0,1,0,0,0,1],t!==void 0&&this.set(t,e,n,s,r,a,o,c,l)}set(t,e,n,s,r,a,o,c,l){const h=this.elements;return h[0]=t,h[1]=s,h[2]=o,h[3]=e,h[4]=r,h[5]=c,h[6]=n,h[7]=a,h[8]=l,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(t){const e=this.elements,n=t.elements;return e[0]=n[0],e[1]=n[1],e[2]=n[2],e[3]=n[3],e[4]=n[4],e[5]=n[5],e[6]=n[6],e[7]=n[7],e[8]=n[8],this}extractBasis(t,e,n){return t.setFromMatrix3Column(this,0),e.setFromMatrix3Column(this,1),n.setFromMatrix3Column(this,2),this}setFromMatrix4(t){const e=t.elements;return this.set(e[0],e[4],e[8],e[1],e[5],e[9],e[2],e[6],e[10]),this}multiply(t){return this.multiplyMatrices(this,t)}premultiply(t){return this.multiplyMatrices(t,this)}multiplyMatrices(t,e){const n=t.elements,s=e.elements,r=this.elements,a=n[0],o=n[3],c=n[6],l=n[1],h=n[4],u=n[7],d=n[2],p=n[5],g=n[8],_=s[0],m=s[3],f=s[6],R=s[1],C=s[4],x=s[7],N=s[2],P=s[5],L=s[8];return r[0]=a*_+o*R+c*N,r[3]=a*m+o*C+c*P,r[6]=a*f+o*x+c*L,r[1]=l*_+h*R+u*N,r[4]=l*m+h*C+u*P,r[7]=l*f+h*x+u*L,r[2]=d*_+p*R+g*N,r[5]=d*m+p*C+g*P,r[8]=d*f+p*x+g*L,this}multiplyScalar(t){const e=this.elements;return e[0]*=t,e[3]*=t,e[6]*=t,e[1]*=t,e[4]*=t,e[7]*=t,e[2]*=t,e[5]*=t,e[8]*=t,this}determinant(){const t=this.elements,e=t[0],n=t[1],s=t[2],r=t[3],a=t[4],o=t[5],c=t[6],l=t[7],h=t[8];return e*a*h-e*o*l-n*r*h+n*o*c+s*r*l-s*a*c}invert(){const t=this.elements,e=t[0],n=t[1],s=t[2],r=t[3],a=t[4],o=t[5],c=t[6],l=t[7],h=t[8],u=h*a-o*l,d=o*c-h*r,p=l*r-a*c,g=e*u+n*d+s*p;if(g===0)return this.set(0,0,0,0,0,0,0,0,0);const _=1/g;return t[0]=u*_,t[1]=(s*l-h*n)*_,t[2]=(o*n-s*a)*_,t[3]=d*_,t[4]=(h*e-s*c)*_,t[5]=(s*r-o*e)*_,t[6]=p*_,t[7]=(n*c-l*e)*_,t[8]=(a*e-n*r)*_,this}transpose(){let t;const e=this.elements;return t=e[1],e[1]=e[3],e[3]=t,t=e[2],e[2]=e[6],e[6]=t,t=e[5],e[5]=e[7],e[7]=t,this}getNormalMatrix(t){return this.setFromMatrix4(t).invert().transpose()}transposeIntoArray(t){const e=this.elements;return t[0]=e[0],t[1]=e[3],t[2]=e[6],t[3]=e[1],t[4]=e[4],t[5]=e[7],t[6]=e[2],t[7]=e[5],t[8]=e[8],this}setUvTransform(t,e,n,s,r,a,o){const c=Math.cos(r),l=Math.sin(r);return this.set(n*c,n*l,-n*(c*a+l*o)+a+t,-s*l,s*c,-s*(-l*a+c*o)+o+e,0,0,1),this}scale(t,e){return this.premultiply(Oa.makeScale(t,e)),this}rotate(t){return this.premultiply(Oa.makeRotation(-t)),this}translate(t,e){return this.premultiply(Oa.makeTranslation(t,e)),this}makeTranslation(t,e){return t.isVector2?this.set(1,0,t.x,0,1,t.y,0,0,1):this.set(1,0,t,0,1,e,0,0,1),this}makeRotation(t){const e=Math.cos(t),n=Math.sin(t);return this.set(e,-n,0,n,e,0,0,0,1),this}makeScale(t,e){return this.set(t,0,0,0,e,0,0,0,1),this}equals(t){const e=this.elements,n=t.elements;for(let s=0;s<9;s++)if(e[s]!==n[s])return!1;return!0}fromArray(t,e=0){for(let n=0;n<9;n++)this.elements[n]=t[n+e];return this}toArray(t=[],e=0){const n=this.elements;return t[e]=n[0],t[e+1]=n[1],t[e+2]=n[2],t[e+3]=n[3],t[e+4]=n[4],t[e+5]=n[5],t[e+6]=n[6],t[e+7]=n[7],t[e+8]=n[8],t}clone(){return new this.constructor().fromArray(this.elements)}}const Oa=new ie;function Vh(i){for(let t=i.length-1;t>=0;--t)if(i[t]>=65535)return!0;return!1}function sr(i){return document.createElementNS("http://www.w3.org/1999/xhtml",i)}function Xd(){const i=sr("canvas");return i.style.display="block",i}const nc={};function Ys(i){i in nc||(nc[i]=!0,console.warn(i))}function qd(i,t,e){return new Promise(function(n,s){function r(){switch(i.clientWaitSync(t,i.SYNC_FLUSH_COMMANDS_BIT,0)){case i.WAIT_FAILED:s();break;case i.TIMEOUT_EXPIRED:setTimeout(r,e);break;default:n()}}setTimeout(r,e)})}function jd(i){const t=i.elements;t[2]=.5*t[2]+.5*t[3],t[6]=.5*t[6]+.5*t[7],t[10]=.5*t[10]+.5*t[11],t[14]=.5*t[14]+.5*t[15]}function Yd(i){const t=i.elements;t[11]===-1?(t[10]=-t[10]-1,t[14]=-t[14]):(t[10]=-t[10],t[14]=-t[14]+1)}const le={enabled:!0,workingColorSpace:Ds,spaces:{},convert:function(i,t,e){return this.enabled===!1||t===e||!t||!e||(this.spaces[t].transfer===Ee&&(i.r=ni(i.r),i.g=ni(i.g),i.b=ni(i.b)),this.spaces[t].primaries!==this.spaces[e].primaries&&(i.applyMatrix3(this.spaces[t].toXYZ),i.applyMatrix3(this.spaces[e].fromXYZ)),this.spaces[e].transfer===Ee&&(i.r=_s(i.r),i.g=_s(i.g),i.b=_s(i.b))),i},fromWorkingColorSpace:function(i,t){return this.convert(i,this.workingColorSpace,t)},toWorkingColorSpace:function(i,t){return this.convert(i,t,this.workingColorSpace)},getPrimaries:function(i){return this.spaces[i].primaries},getTransfer:function(i){return i===yi?Ea:this.spaces[i].transfer},getLuminanceCoefficients:function(i,t=this.workingColorSpace){return i.fromArray(this.spaces[t].luminanceCoefficients)},define:function(i){Object.assign(this.spaces,i)},_getMatrix:function(i,t,e){return i.copy(this.spaces[t].toXYZ).multiply(this.spaces[e].fromXYZ)},_getDrawingBufferColorSpace:function(i){return this.spaces[i].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(i=this.workingColorSpace){return this.spaces[i].workingColorSpaceConfig.unpackColorSpace}};function ni(i){return i<.04045?i*.0773993808:Math.pow(i*.9478672986+.0521327014,2.4)}function _s(i){return i<.0031308?i*12.92:1.055*Math.pow(i,.41666)-.055}const ic=[.64,.33,.3,.6,.15,.06],sc=[.2126,.7152,.0722],rc=[.3127,.329],ac=new ie().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),oc=new ie().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);le.define({[Ds]:{primaries:ic,whitePoint:rc,transfer:Ea,toXYZ:ac,fromXYZ:oc,luminanceCoefficients:sc,workingColorSpaceConfig:{unpackColorSpace:Pe},outputColorSpaceConfig:{drawingBufferColorSpace:Pe}},[Pe]:{primaries:ic,whitePoint:rc,transfer:Ee,toXYZ:ac,fromXYZ:oc,luminanceCoefficients:sc,outputColorSpaceConfig:{drawingBufferColorSpace:Pe}}});let Zi;class $d{static getDataURL(t){if(/^data:/i.test(t.src)||typeof HTMLCanvasElement>"u")return t.src;let e;if(t instanceof HTMLCanvasElement)e=t;else{Zi===void 0&&(Zi=sr("canvas")),Zi.width=t.width,Zi.height=t.height;const n=Zi.getContext("2d");t instanceof ImageData?n.putImageData(t,0,0):n.drawImage(t,0,0,t.width,t.height),e=Zi}return e.width>2048||e.height>2048?(console.warn("THREE.ImageUtils.getDataURL: Image converted to jpg for performance reasons",t),e.toDataURL("image/jpeg",.6)):e.toDataURL("image/png")}static sRGBToLinear(t){if(typeof HTMLImageElement<"u"&&t instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&t instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&t instanceof ImageBitmap){const e=sr("canvas");e.width=t.width,e.height=t.height;const n=e.getContext("2d");n.drawImage(t,0,0,t.width,t.height);const s=n.getImageData(0,0,t.width,t.height),r=s.data;for(let a=0;a<r.length;a++)r[a]=ni(r[a]/255)*255;return n.putImageData(s,0,0),e}else if(t.data){const e=t.data.slice(0);for(let n=0;n<e.length;n++)e instanceof Uint8Array||e instanceof Uint8ClampedArray?e[n]=Math.floor(ni(e[n]/255)*255):e[n]=ni(e[n]);return{data:e,width:t.width,height:t.height}}else return console.warn("THREE.ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),t}}let Kd=0;class Hh{constructor(t=null){this.isSource=!0,Object.defineProperty(this,"id",{value:Kd++}),this.uuid=Ci(),this.data=t,this.dataReady=!0,this.version=0}set needsUpdate(t){t===!0&&this.version++}toJSON(t){const e=t===void 0||typeof t=="string";if(!e&&t.images[this.uuid]!==void 0)return t.images[this.uuid];const n={uuid:this.uuid,url:""},s=this.data;if(s!==null){let r;if(Array.isArray(s)){r=[];for(let a=0,o=s.length;a<o;a++)s[a].isDataTexture?r.push(ka(s[a].image)):r.push(ka(s[a]))}else r=ka(s);n.url=r}return e||(t.images[this.uuid]=n),n}}function ka(i){return typeof HTMLImageElement<"u"&&i instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&i instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&i instanceof ImageBitmap?$d.getDataURL(i):i.data?{data:Array.from(i.data),width:i.width,height:i.height,type:i.data.constructor.name}:(console.warn("THREE.Texture: Unable to serialize Texture."),{})}let Jd=0;class Ye extends qi{constructor(t=Ye.DEFAULT_IMAGE,e=Ye.DEFAULT_MAPPING,n=Nn,s=Nn,r=fn,a=ti,o=Sn,c=si,l=Ye.DEFAULT_ANISOTROPY,h=yi){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:Jd++}),this.uuid=Ci(),this.name="",this.source=new Hh(t),this.mipmaps=[],this.mapping=e,this.channel=0,this.wrapS=n,this.wrapT=s,this.magFilter=r,this.minFilter=a,this.anisotropy=l,this.format=o,this.internalFormat=null,this.type=c,this.offset=new Lt(0,0),this.repeat=new Lt(1,1),this.center=new Lt(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new ie,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=h,this.userData={},this.version=0,this.onUpdate=null,this.isRenderTargetTexture=!1,this.pmremVersion=0}get image(){return this.source.data}set image(t=null){this.source.data=t}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}clone(){return new this.constructor().copy(this)}copy(t){return this.name=t.name,this.source=t.source,this.mipmaps=t.mipmaps.slice(0),this.mapping=t.mapping,this.channel=t.channel,this.wrapS=t.wrapS,this.wrapT=t.wrapT,this.magFilter=t.magFilter,this.minFilter=t.minFilter,this.anisotropy=t.anisotropy,this.format=t.format,this.internalFormat=t.internalFormat,this.type=t.type,this.offset.copy(t.offset),this.repeat.copy(t.repeat),this.center.copy(t.center),this.rotation=t.rotation,this.matrixAutoUpdate=t.matrixAutoUpdate,this.matrix.copy(t.matrix),this.generateMipmaps=t.generateMipmaps,this.premultiplyAlpha=t.premultiplyAlpha,this.flipY=t.flipY,this.unpackAlignment=t.unpackAlignment,this.colorSpace=t.colorSpace,this.userData=JSON.parse(JSON.stringify(t.userData)),this.needsUpdate=!0,this}toJSON(t){const e=t===void 0||typeof t=="string";if(!e&&t.textures[this.uuid]!==void 0)return t.textures[this.uuid];const n={metadata:{version:4.6,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(t).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(n.userData=this.userData),e||(t.textures[this.uuid]=n),n}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(t){if(this.mapping!==Rh)return t;if(t.applyMatrix3(this.matrix),t.x<0||t.x>1)switch(this.wrapS){case Hi:t.x=t.x-Math.floor(t.x);break;case Nn:t.x=t.x<0?0:1;break;case Fo:Math.abs(Math.floor(t.x)%2)===1?t.x=Math.ceil(t.x)-t.x:t.x=t.x-Math.floor(t.x);break}if(t.y<0||t.y>1)switch(this.wrapT){case Hi:t.y=t.y-Math.floor(t.y);break;case Nn:t.y=t.y<0?0:1;break;case Fo:Math.abs(Math.floor(t.y)%2)===1?t.y=Math.ceil(t.y)-t.y:t.y=t.y-Math.floor(t.y);break}return this.flipY&&(t.y=1-t.y),t}set needsUpdate(t){t===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(t){t===!0&&this.pmremVersion++}}Ye.DEFAULT_IMAGE=null;Ye.DEFAULT_MAPPING=Rh;Ye.DEFAULT_ANISOTROPY=1;class ve{constructor(t=0,e=0,n=0,s=1){ve.prototype.isVector4=!0,this.x=t,this.y=e,this.z=n,this.w=s}get width(){return this.z}set width(t){this.z=t}get height(){return this.w}set height(t){this.w=t}set(t,e,n,s){return this.x=t,this.y=e,this.z=n,this.w=s,this}setScalar(t){return this.x=t,this.y=t,this.z=t,this.w=t,this}setX(t){return this.x=t,this}setY(t){return this.y=t,this}setZ(t){return this.z=t,this}setW(t){return this.w=t,this}setComponent(t,e){switch(t){case 0:this.x=e;break;case 1:this.y=e;break;case 2:this.z=e;break;case 3:this.w=e;break;default:throw new Error("index is out of range: "+t)}return this}getComponent(t){switch(t){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("index is out of range: "+t)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(t){return this.x=t.x,this.y=t.y,this.z=t.z,this.w=t.w!==void 0?t.w:1,this}add(t){return this.x+=t.x,this.y+=t.y,this.z+=t.z,this.w+=t.w,this}addScalar(t){return this.x+=t,this.y+=t,this.z+=t,this.w+=t,this}addVectors(t,e){return this.x=t.x+e.x,this.y=t.y+e.y,this.z=t.z+e.z,this.w=t.w+e.w,this}addScaledVector(t,e){return this.x+=t.x*e,this.y+=t.y*e,this.z+=t.z*e,this.w+=t.w*e,this}sub(t){return this.x-=t.x,this.y-=t.y,this.z-=t.z,this.w-=t.w,this}subScalar(t){return this.x-=t,this.y-=t,this.z-=t,this.w-=t,this}subVectors(t,e){return this.x=t.x-e.x,this.y=t.y-e.y,this.z=t.z-e.z,this.w=t.w-e.w,this}multiply(t){return this.x*=t.x,this.y*=t.y,this.z*=t.z,this.w*=t.w,this}multiplyScalar(t){return this.x*=t,this.y*=t,this.z*=t,this.w*=t,this}applyMatrix4(t){const e=this.x,n=this.y,s=this.z,r=this.w,a=t.elements;return this.x=a[0]*e+a[4]*n+a[8]*s+a[12]*r,this.y=a[1]*e+a[5]*n+a[9]*s+a[13]*r,this.z=a[2]*e+a[6]*n+a[10]*s+a[14]*r,this.w=a[3]*e+a[7]*n+a[11]*s+a[15]*r,this}divide(t){return this.x/=t.x,this.y/=t.y,this.z/=t.z,this.w/=t.w,this}divideScalar(t){return this.multiplyScalar(1/t)}setAxisAngleFromQuaternion(t){this.w=2*Math.acos(t.w);const e=Math.sqrt(1-t.w*t.w);return e<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=t.x/e,this.y=t.y/e,this.z=t.z/e),this}setAxisAngleFromRotationMatrix(t){let e,n,s,r;const c=t.elements,l=c[0],h=c[4],u=c[8],d=c[1],p=c[5],g=c[9],_=c[2],m=c[6],f=c[10];if(Math.abs(h-d)<.01&&Math.abs(u-_)<.01&&Math.abs(g-m)<.01){if(Math.abs(h+d)<.1&&Math.abs(u+_)<.1&&Math.abs(g+m)<.1&&Math.abs(l+p+f-3)<.1)return this.set(1,0,0,0),this;e=Math.PI;const C=(l+1)/2,x=(p+1)/2,N=(f+1)/2,P=(h+d)/4,L=(u+_)/4,I=(g+m)/4;return C>x&&C>N?C<.01?(n=0,s=.707106781,r=.707106781):(n=Math.sqrt(C),s=P/n,r=L/n):x>N?x<.01?(n=.707106781,s=0,r=.707106781):(s=Math.sqrt(x),n=P/s,r=I/s):N<.01?(n=.707106781,s=.707106781,r=0):(r=Math.sqrt(N),n=L/r,s=I/r),this.set(n,s,r,e),this}let R=Math.sqrt((m-g)*(m-g)+(u-_)*(u-_)+(d-h)*(d-h));return Math.abs(R)<.001&&(R=1),this.x=(m-g)/R,this.y=(u-_)/R,this.z=(d-h)/R,this.w=Math.acos((l+p+f-1)/2),this}setFromMatrixPosition(t){const e=t.elements;return this.x=e[12],this.y=e[13],this.z=e[14],this.w=e[15],this}min(t){return this.x=Math.min(this.x,t.x),this.y=Math.min(this.y,t.y),this.z=Math.min(this.z,t.z),this.w=Math.min(this.w,t.w),this}max(t){return this.x=Math.max(this.x,t.x),this.y=Math.max(this.y,t.y),this.z=Math.max(this.z,t.z),this.w=Math.max(this.w,t.w),this}clamp(t,e){return this.x=Math.max(t.x,Math.min(e.x,this.x)),this.y=Math.max(t.y,Math.min(e.y,this.y)),this.z=Math.max(t.z,Math.min(e.z,this.z)),this.w=Math.max(t.w,Math.min(e.w,this.w)),this}clampScalar(t,e){return this.x=Math.max(t,Math.min(e,this.x)),this.y=Math.max(t,Math.min(e,this.y)),this.z=Math.max(t,Math.min(e,this.z)),this.w=Math.max(t,Math.min(e,this.w)),this}clampLength(t,e){const n=this.length();return this.divideScalar(n||1).multiplyScalar(Math.max(t,Math.min(e,n)))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(t){return this.x*t.x+this.y*t.y+this.z*t.z+this.w*t.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(t){return this.normalize().multiplyScalar(t)}lerp(t,e){return this.x+=(t.x-this.x)*e,this.y+=(t.y-this.y)*e,this.z+=(t.z-this.z)*e,this.w+=(t.w-this.w)*e,this}lerpVectors(t,e,n){return this.x=t.x+(e.x-t.x)*n,this.y=t.y+(e.y-t.y)*n,this.z=t.z+(e.z-t.z)*n,this.w=t.w+(e.w-t.w)*n,this}equals(t){return t.x===this.x&&t.y===this.y&&t.z===this.z&&t.w===this.w}fromArray(t,e=0){return this.x=t[e],this.y=t[e+1],this.z=t[e+2],this.w=t[e+3],this}toArray(t=[],e=0){return t[e]=this.x,t[e+1]=this.y,t[e+2]=this.z,t[e+3]=this.w,t}fromBufferAttribute(t,e){return this.x=t.getX(e),this.y=t.getY(e),this.z=t.getZ(e),this.w=t.getW(e),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}}class Zd extends qi{constructor(t=1,e=1,n={}){super(),this.isRenderTarget=!0,this.width=t,this.height=e,this.depth=1,this.scissor=new ve(0,0,t,e),this.scissorTest=!1,this.viewport=new ve(0,0,t,e);const s={width:t,height:e,depth:1};n=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:fn,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1},n);const r=new Ye(s,n.mapping,n.wrapS,n.wrapT,n.magFilter,n.minFilter,n.format,n.type,n.anisotropy,n.colorSpace);r.flipY=!1,r.generateMipmaps=n.generateMipmaps,r.internalFormat=n.internalFormat,this.textures=[];const a=n.count;for(let o=0;o<a;o++)this.textures[o]=r.clone(),this.textures[o].isRenderTargetTexture=!0;this.depthBuffer=n.depthBuffer,this.stencilBuffer=n.stencilBuffer,this.resolveDepthBuffer=n.resolveDepthBuffer,this.resolveStencilBuffer=n.resolveStencilBuffer,this.depthTexture=n.depthTexture,this.samples=n.samples}get texture(){return this.textures[0]}set texture(t){this.textures[0]=t}setSize(t,e,n=1){if(this.width!==t||this.height!==e||this.depth!==n){this.width=t,this.height=e,this.depth=n;for(let s=0,r=this.textures.length;s<r;s++)this.textures[s].image.width=t,this.textures[s].image.height=e,this.textures[s].image.depth=n;this.dispose()}this.viewport.set(0,0,t,e),this.scissor.set(0,0,t,e)}clone(){return new this.constructor().copy(this)}copy(t){this.width=t.width,this.height=t.height,this.depth=t.depth,this.scissor.copy(t.scissor),this.scissorTest=t.scissorTest,this.viewport.copy(t.viewport),this.textures.length=0;for(let n=0,s=t.textures.length;n<s;n++)this.textures[n]=t.textures[n].clone(),this.textures[n].isRenderTargetTexture=!0;const e=Object.assign({},t.texture.image);return this.texture.source=new Hh(e),this.depthBuffer=t.depthBuffer,this.stencilBuffer=t.stencilBuffer,this.resolveDepthBuffer=t.resolveDepthBuffer,this.resolveStencilBuffer=t.resolveStencilBuffer,t.depthTexture!==null&&(this.depthTexture=t.depthTexture.clone()),this.samples=t.samples,this}dispose(){this.dispatchEvent({type:"dispose"})}}class Wi extends Zd{constructor(t=1,e=1,n={}){super(t,e,n),this.isWebGLRenderTarget=!0}}class Gh extends Ye{constructor(t=null,e=1,n=1,s=1){super(null),this.isDataArrayTexture=!0,this.image={data:t,width:e,height:n,depth:s},this.magFilter=mn,this.minFilter=mn,this.wrapR=Nn,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(t){this.layerUpdates.add(t)}clearLayerUpdates(){this.layerUpdates.clear()}}class Qd extends Ye{constructor(t=null,e=1,n=1,s=1){super(null),this.isData3DTexture=!0,this.image={data:t,width:e,height:n,depth:s},this.magFilter=mn,this.minFilter=mn,this.wrapR=Nn,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class Dn{constructor(t=0,e=0,n=0,s=1){this.isQuaternion=!0,this._x=t,this._y=e,this._z=n,this._w=s}static slerpFlat(t,e,n,s,r,a,o){let c=n[s+0],l=n[s+1],h=n[s+2],u=n[s+3];const d=r[a+0],p=r[a+1],g=r[a+2],_=r[a+3];if(o===0){t[e+0]=c,t[e+1]=l,t[e+2]=h,t[e+3]=u;return}if(o===1){t[e+0]=d,t[e+1]=p,t[e+2]=g,t[e+3]=_;return}if(u!==_||c!==d||l!==p||h!==g){let m=1-o;const f=c*d+l*p+h*g+u*_,R=f>=0?1:-1,C=1-f*f;if(C>Number.EPSILON){const N=Math.sqrt(C),P=Math.atan2(N,f*R);m=Math.sin(m*P)/N,o=Math.sin(o*P)/N}const x=o*R;if(c=c*m+d*x,l=l*m+p*x,h=h*m+g*x,u=u*m+_*x,m===1-o){const N=1/Math.sqrt(c*c+l*l+h*h+u*u);c*=N,l*=N,h*=N,u*=N}}t[e]=c,t[e+1]=l,t[e+2]=h,t[e+3]=u}static multiplyQuaternionsFlat(t,e,n,s,r,a){const o=n[s],c=n[s+1],l=n[s+2],h=n[s+3],u=r[a],d=r[a+1],p=r[a+2],g=r[a+3];return t[e]=o*g+h*u+c*p-l*d,t[e+1]=c*g+h*d+l*u-o*p,t[e+2]=l*g+h*p+o*d-c*u,t[e+3]=h*g-o*u-c*d-l*p,t}get x(){return this._x}set x(t){this._x=t,this._onChangeCallback()}get y(){return this._y}set y(t){this._y=t,this._onChangeCallback()}get z(){return this._z}set z(t){this._z=t,this._onChangeCallback()}get w(){return this._w}set w(t){this._w=t,this._onChangeCallback()}set(t,e,n,s){return this._x=t,this._y=e,this._z=n,this._w=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(t){return this._x=t.x,this._y=t.y,this._z=t.z,this._w=t.w,this._onChangeCallback(),this}setFromEuler(t,e=!0){const n=t._x,s=t._y,r=t._z,a=t._order,o=Math.cos,c=Math.sin,l=o(n/2),h=o(s/2),u=o(r/2),d=c(n/2),p=c(s/2),g=c(r/2);switch(a){case"XYZ":this._x=d*h*u+l*p*g,this._y=l*p*u-d*h*g,this._z=l*h*g+d*p*u,this._w=l*h*u-d*p*g;break;case"YXZ":this._x=d*h*u+l*p*g,this._y=l*p*u-d*h*g,this._z=l*h*g-d*p*u,this._w=l*h*u+d*p*g;break;case"ZXY":this._x=d*h*u-l*p*g,this._y=l*p*u+d*h*g,this._z=l*h*g+d*p*u,this._w=l*h*u-d*p*g;break;case"ZYX":this._x=d*h*u-l*p*g,this._y=l*p*u+d*h*g,this._z=l*h*g-d*p*u,this._w=l*h*u+d*p*g;break;case"YZX":this._x=d*h*u+l*p*g,this._y=l*p*u+d*h*g,this._z=l*h*g-d*p*u,this._w=l*h*u-d*p*g;break;case"XZY":this._x=d*h*u-l*p*g,this._y=l*p*u-d*h*g,this._z=l*h*g+d*p*u,this._w=l*h*u+d*p*g;break;default:console.warn("THREE.Quaternion: .setFromEuler() encountered an unknown order: "+a)}return e===!0&&this._onChangeCallback(),this}setFromAxisAngle(t,e){const n=e/2,s=Math.sin(n);return this._x=t.x*s,this._y=t.y*s,this._z=t.z*s,this._w=Math.cos(n),this._onChangeCallback(),this}setFromRotationMatrix(t){const e=t.elements,n=e[0],s=e[4],r=e[8],a=e[1],o=e[5],c=e[9],l=e[2],h=e[6],u=e[10],d=n+o+u;if(d>0){const p=.5/Math.sqrt(d+1);this._w=.25/p,this._x=(h-c)*p,this._y=(r-l)*p,this._z=(a-s)*p}else if(n>o&&n>u){const p=2*Math.sqrt(1+n-o-u);this._w=(h-c)/p,this._x=.25*p,this._y=(s+a)/p,this._z=(r+l)/p}else if(o>u){const p=2*Math.sqrt(1+o-n-u);this._w=(r-l)/p,this._x=(s+a)/p,this._y=.25*p,this._z=(c+h)/p}else{const p=2*Math.sqrt(1+u-n-o);this._w=(a-s)/p,this._x=(r+l)/p,this._y=(c+h)/p,this._z=.25*p}return this._onChangeCallback(),this}setFromUnitVectors(t,e){let n=t.dot(e)+1;return n<Number.EPSILON?(n=0,Math.abs(t.x)>Math.abs(t.z)?(this._x=-t.y,this._y=t.x,this._z=0,this._w=n):(this._x=0,this._y=-t.z,this._z=t.y,this._w=n)):(this._x=t.y*e.z-t.z*e.y,this._y=t.z*e.x-t.x*e.z,this._z=t.x*e.y-t.y*e.x,this._w=n),this.normalize()}angleTo(t){return 2*Math.acos(Math.abs(Xe(this.dot(t),-1,1)))}rotateTowards(t,e){const n=this.angleTo(t);if(n===0)return this;const s=Math.min(1,e/n);return this.slerp(t,s),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(t){return this._x*t._x+this._y*t._y+this._z*t._z+this._w*t._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let t=this.length();return t===0?(this._x=0,this._y=0,this._z=0,this._w=1):(t=1/t,this._x=this._x*t,this._y=this._y*t,this._z=this._z*t,this._w=this._w*t),this._onChangeCallback(),this}multiply(t){return this.multiplyQuaternions(this,t)}premultiply(t){return this.multiplyQuaternions(t,this)}multiplyQuaternions(t,e){const n=t._x,s=t._y,r=t._z,a=t._w,o=e._x,c=e._y,l=e._z,h=e._w;return this._x=n*h+a*o+s*l-r*c,this._y=s*h+a*c+r*o-n*l,this._z=r*h+a*l+n*c-s*o,this._w=a*h-n*o-s*c-r*l,this._onChangeCallback(),this}slerp(t,e){if(e===0)return this;if(e===1)return this.copy(t);const n=this._x,s=this._y,r=this._z,a=this._w;let o=a*t._w+n*t._x+s*t._y+r*t._z;if(o<0?(this._w=-t._w,this._x=-t._x,this._y=-t._y,this._z=-t._z,o=-o):this.copy(t),o>=1)return this._w=a,this._x=n,this._y=s,this._z=r,this;const c=1-o*o;if(c<=Number.EPSILON){const p=1-e;return this._w=p*a+e*this._w,this._x=p*n+e*this._x,this._y=p*s+e*this._y,this._z=p*r+e*this._z,this.normalize(),this}const l=Math.sqrt(c),h=Math.atan2(l,o),u=Math.sin((1-e)*h)/l,d=Math.sin(e*h)/l;return this._w=a*u+this._w*d,this._x=n*u+this._x*d,this._y=s*u+this._y*d,this._z=r*u+this._z*d,this._onChangeCallback(),this}slerpQuaternions(t,e,n){return this.copy(t).slerp(e,n)}random(){const t=2*Math.PI*Math.random(),e=2*Math.PI*Math.random(),n=Math.random(),s=Math.sqrt(1-n),r=Math.sqrt(n);return this.set(s*Math.sin(t),s*Math.cos(t),r*Math.sin(e),r*Math.cos(e))}equals(t){return t._x===this._x&&t._y===this._y&&t._z===this._z&&t._w===this._w}fromArray(t,e=0){return this._x=t[e],this._y=t[e+1],this._z=t[e+2],this._w=t[e+3],this._onChangeCallback(),this}toArray(t=[],e=0){return t[e]=this._x,t[e+1]=this._y,t[e+2]=this._z,t[e+3]=this._w,t}fromBufferAttribute(t,e){return this._x=t.getX(e),this._y=t.getY(e),this._z=t.getZ(e),this._w=t.getW(e),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(t){return this._onChangeCallback=t,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}}class F{constructor(t=0,e=0,n=0){F.prototype.isVector3=!0,this.x=t,this.y=e,this.z=n}set(t,e,n){return n===void 0&&(n=this.z),this.x=t,this.y=e,this.z=n,this}setScalar(t){return this.x=t,this.y=t,this.z=t,this}setX(t){return this.x=t,this}setY(t){return this.y=t,this}setZ(t){return this.z=t,this}setComponent(t,e){switch(t){case 0:this.x=e;break;case 1:this.y=e;break;case 2:this.z=e;break;default:throw new Error("index is out of range: "+t)}return this}getComponent(t){switch(t){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("index is out of range: "+t)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(t){return this.x=t.x,this.y=t.y,this.z=t.z,this}add(t){return this.x+=t.x,this.y+=t.y,this.z+=t.z,this}addScalar(t){return this.x+=t,this.y+=t,this.z+=t,this}addVectors(t,e){return this.x=t.x+e.x,this.y=t.y+e.y,this.z=t.z+e.z,this}addScaledVector(t,e){return this.x+=t.x*e,this.y+=t.y*e,this.z+=t.z*e,this}sub(t){return this.x-=t.x,this.y-=t.y,this.z-=t.z,this}subScalar(t){return this.x-=t,this.y-=t,this.z-=t,this}subVectors(t,e){return this.x=t.x-e.x,this.y=t.y-e.y,this.z=t.z-e.z,this}multiply(t){return this.x*=t.x,this.y*=t.y,this.z*=t.z,this}multiplyScalar(t){return this.x*=t,this.y*=t,this.z*=t,this}multiplyVectors(t,e){return this.x=t.x*e.x,this.y=t.y*e.y,this.z=t.z*e.z,this}applyEuler(t){return this.applyQuaternion(lc.setFromEuler(t))}applyAxisAngle(t,e){return this.applyQuaternion(lc.setFromAxisAngle(t,e))}applyMatrix3(t){const e=this.x,n=this.y,s=this.z,r=t.elements;return this.x=r[0]*e+r[3]*n+r[6]*s,this.y=r[1]*e+r[4]*n+r[7]*s,this.z=r[2]*e+r[5]*n+r[8]*s,this}applyNormalMatrix(t){return this.applyMatrix3(t).normalize()}applyMatrix4(t){const e=this.x,n=this.y,s=this.z,r=t.elements,a=1/(r[3]*e+r[7]*n+r[11]*s+r[15]);return this.x=(r[0]*e+r[4]*n+r[8]*s+r[12])*a,this.y=(r[1]*e+r[5]*n+r[9]*s+r[13])*a,this.z=(r[2]*e+r[6]*n+r[10]*s+r[14])*a,this}applyQuaternion(t){const e=this.x,n=this.y,s=this.z,r=t.x,a=t.y,o=t.z,c=t.w,l=2*(a*s-o*n),h=2*(o*e-r*s),u=2*(r*n-a*e);return this.x=e+c*l+a*u-o*h,this.y=n+c*h+o*l-r*u,this.z=s+c*u+r*h-a*l,this}project(t){return this.applyMatrix4(t.matrixWorldInverse).applyMatrix4(t.projectionMatrix)}unproject(t){return this.applyMatrix4(t.projectionMatrixInverse).applyMatrix4(t.matrixWorld)}transformDirection(t){const e=this.x,n=this.y,s=this.z,r=t.elements;return this.x=r[0]*e+r[4]*n+r[8]*s,this.y=r[1]*e+r[5]*n+r[9]*s,this.z=r[2]*e+r[6]*n+r[10]*s,this.normalize()}divide(t){return this.x/=t.x,this.y/=t.y,this.z/=t.z,this}divideScalar(t){return this.multiplyScalar(1/t)}min(t){return this.x=Math.min(this.x,t.x),this.y=Math.min(this.y,t.y),this.z=Math.min(this.z,t.z),this}max(t){return this.x=Math.max(this.x,t.x),this.y=Math.max(this.y,t.y),this.z=Math.max(this.z,t.z),this}clamp(t,e){return this.x=Math.max(t.x,Math.min(e.x,this.x)),this.y=Math.max(t.y,Math.min(e.y,this.y)),this.z=Math.max(t.z,Math.min(e.z,this.z)),this}clampScalar(t,e){return this.x=Math.max(t,Math.min(e,this.x)),this.y=Math.max(t,Math.min(e,this.y)),this.z=Math.max(t,Math.min(e,this.z)),this}clampLength(t,e){const n=this.length();return this.divideScalar(n||1).multiplyScalar(Math.max(t,Math.min(e,n)))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(t){return this.x*t.x+this.y*t.y+this.z*t.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(t){return this.normalize().multiplyScalar(t)}lerp(t,e){return this.x+=(t.x-this.x)*e,this.y+=(t.y-this.y)*e,this.z+=(t.z-this.z)*e,this}lerpVectors(t,e,n){return this.x=t.x+(e.x-t.x)*n,this.y=t.y+(e.y-t.y)*n,this.z=t.z+(e.z-t.z)*n,this}cross(t){return this.crossVectors(this,t)}crossVectors(t,e){const n=t.x,s=t.y,r=t.z,a=e.x,o=e.y,c=e.z;return this.x=s*c-r*o,this.y=r*a-n*c,this.z=n*o-s*a,this}projectOnVector(t){const e=t.lengthSq();if(e===0)return this.set(0,0,0);const n=t.dot(this)/e;return this.copy(t).multiplyScalar(n)}projectOnPlane(t){return Ba.copy(this).projectOnVector(t),this.sub(Ba)}reflect(t){return this.sub(Ba.copy(t).multiplyScalar(2*this.dot(t)))}angleTo(t){const e=Math.sqrt(this.lengthSq()*t.lengthSq());if(e===0)return Math.PI/2;const n=this.dot(t)/e;return Math.acos(Xe(n,-1,1))}distanceTo(t){return Math.sqrt(this.distanceToSquared(t))}distanceToSquared(t){const e=this.x-t.x,n=this.y-t.y,s=this.z-t.z;return e*e+n*n+s*s}manhattanDistanceTo(t){return Math.abs(this.x-t.x)+Math.abs(this.y-t.y)+Math.abs(this.z-t.z)}setFromSpherical(t){return this.setFromSphericalCoords(t.radius,t.phi,t.theta)}setFromSphericalCoords(t,e,n){const s=Math.sin(e)*t;return this.x=s*Math.sin(n),this.y=Math.cos(e)*t,this.z=s*Math.cos(n),this}setFromCylindrical(t){return this.setFromCylindricalCoords(t.radius,t.theta,t.y)}setFromCylindricalCoords(t,e,n){return this.x=t*Math.sin(e),this.y=n,this.z=t*Math.cos(e),this}setFromMatrixPosition(t){const e=t.elements;return this.x=e[12],this.y=e[13],this.z=e[14],this}setFromMatrixScale(t){const e=this.setFromMatrixColumn(t,0).length(),n=this.setFromMatrixColumn(t,1).length(),s=this.setFromMatrixColumn(t,2).length();return this.x=e,this.y=n,this.z=s,this}setFromMatrixColumn(t,e){return this.fromArray(t.elements,e*4)}setFromMatrix3Column(t,e){return this.fromArray(t.elements,e*3)}setFromEuler(t){return this.x=t._x,this.y=t._y,this.z=t._z,this}setFromColor(t){return this.x=t.r,this.y=t.g,this.z=t.b,this}equals(t){return t.x===this.x&&t.y===this.y&&t.z===this.z}fromArray(t,e=0){return this.x=t[e],this.y=t[e+1],this.z=t[e+2],this}toArray(t=[],e=0){return t[e]=this.x,t[e+1]=this.y,t[e+2]=this.z,t}fromBufferAttribute(t,e){return this.x=t.getX(e),this.y=t.getY(e),this.z=t.getZ(e),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){const t=Math.random()*Math.PI*2,e=Math.random()*2-1,n=Math.sqrt(1-e*e);return this.x=n*Math.cos(t),this.y=e,this.z=n*Math.sin(t),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}}const Ba=new F,lc=new Dn;class Fn{constructor(t=new F(1/0,1/0,1/0),e=new F(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=t,this.max=e}set(t,e){return this.min.copy(t),this.max.copy(e),this}setFromArray(t){this.makeEmpty();for(let e=0,n=t.length;e<n;e+=3)this.expandByPoint(An.fromArray(t,e));return this}setFromBufferAttribute(t){this.makeEmpty();for(let e=0,n=t.count;e<n;e++)this.expandByPoint(An.fromBufferAttribute(t,e));return this}setFromPoints(t){this.makeEmpty();for(let e=0,n=t.length;e<n;e++)this.expandByPoint(t[e]);return this}setFromCenterAndSize(t,e){const n=An.copy(e).multiplyScalar(.5);return this.min.copy(t).sub(n),this.max.copy(t).add(n),this}setFromObject(t,e=!1){return this.makeEmpty(),this.expandByObject(t,e)}clone(){return new this.constructor().copy(this)}copy(t){return this.min.copy(t.min),this.max.copy(t.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(t){return this.isEmpty()?t.set(0,0,0):t.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(t){return this.isEmpty()?t.set(0,0,0):t.subVectors(this.max,this.min)}expandByPoint(t){return this.min.min(t),this.max.max(t),this}expandByVector(t){return this.min.sub(t),this.max.add(t),this}expandByScalar(t){return this.min.addScalar(-t),this.max.addScalar(t),this}expandByObject(t,e=!1){t.updateWorldMatrix(!1,!1);const n=t.geometry;if(n!==void 0){const r=n.getAttribute("position");if(e===!0&&r!==void 0&&t.isInstancedMesh!==!0)for(let a=0,o=r.count;a<o;a++)t.isMesh===!0?t.getVertexPosition(a,An):An.fromBufferAttribute(r,a),An.applyMatrix4(t.matrixWorld),this.expandByPoint(An);else t.boundingBox!==void 0?(t.boundingBox===null&&t.computeBoundingBox(),Er.copy(t.boundingBox)):(n.boundingBox===null&&n.computeBoundingBox(),Er.copy(n.boundingBox)),Er.applyMatrix4(t.matrixWorld),this.union(Er)}const s=t.children;for(let r=0,a=s.length;r<a;r++)this.expandByObject(s[r],e);return this}containsPoint(t){return t.x>=this.min.x&&t.x<=this.max.x&&t.y>=this.min.y&&t.y<=this.max.y&&t.z>=this.min.z&&t.z<=this.max.z}containsBox(t){return this.min.x<=t.min.x&&t.max.x<=this.max.x&&this.min.y<=t.min.y&&t.max.y<=this.max.y&&this.min.z<=t.min.z&&t.max.z<=this.max.z}getParameter(t,e){return e.set((t.x-this.min.x)/(this.max.x-this.min.x),(t.y-this.min.y)/(this.max.y-this.min.y),(t.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(t){return t.max.x>=this.min.x&&t.min.x<=this.max.x&&t.max.y>=this.min.y&&t.min.y<=this.max.y&&t.max.z>=this.min.z&&t.min.z<=this.max.z}intersectsSphere(t){return this.clampPoint(t.center,An),An.distanceToSquared(t.center)<=t.radius*t.radius}intersectsPlane(t){let e,n;return t.normal.x>0?(e=t.normal.x*this.min.x,n=t.normal.x*this.max.x):(e=t.normal.x*this.max.x,n=t.normal.x*this.min.x),t.normal.y>0?(e+=t.normal.y*this.min.y,n+=t.normal.y*this.max.y):(e+=t.normal.y*this.max.y,n+=t.normal.y*this.min.y),t.normal.z>0?(e+=t.normal.z*this.min.z,n+=t.normal.z*this.max.z):(e+=t.normal.z*this.max.z,n+=t.normal.z*this.min.z),e<=-t.constant&&n>=-t.constant}intersectsTriangle(t){if(this.isEmpty())return!1;this.getCenter(Hs),Tr.subVectors(this.max,Hs),Qi.subVectors(t.a,Hs),ts.subVectors(t.b,Hs),es.subVectors(t.c,Hs),hi.subVectors(ts,Qi),ui.subVectors(es,ts),Ni.subVectors(Qi,es);let e=[0,-hi.z,hi.y,0,-ui.z,ui.y,0,-Ni.z,Ni.y,hi.z,0,-hi.x,ui.z,0,-ui.x,Ni.z,0,-Ni.x,-hi.y,hi.x,0,-ui.y,ui.x,0,-Ni.y,Ni.x,0];return!za(e,Qi,ts,es,Tr)||(e=[1,0,0,0,1,0,0,0,1],!za(e,Qi,ts,es,Tr))?!1:(wr.crossVectors(hi,ui),e=[wr.x,wr.y,wr.z],za(e,Qi,ts,es,Tr))}clampPoint(t,e){return e.copy(t).clamp(this.min,this.max)}distanceToPoint(t){return this.clampPoint(t,An).distanceTo(t)}getBoundingSphere(t){return this.isEmpty()?t.makeEmpty():(this.getCenter(t.center),t.radius=this.getSize(An).length()*.5),t}intersect(t){return this.min.max(t.min),this.max.min(t.max),this.isEmpty()&&this.makeEmpty(),this}union(t){return this.min.min(t.min),this.max.max(t.max),this}applyMatrix4(t){return this.isEmpty()?this:(jn[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(t),jn[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(t),jn[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(t),jn[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(t),jn[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(t),jn[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(t),jn[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(t),jn[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(t),this.setFromPoints(jn),this)}translate(t){return this.min.add(t),this.max.add(t),this}equals(t){return t.min.equals(this.min)&&t.max.equals(this.max)}}const jn=[new F,new F,new F,new F,new F,new F,new F,new F],An=new F,Er=new Fn,Qi=new F,ts=new F,es=new F,hi=new F,ui=new F,Ni=new F,Hs=new F,Tr=new F,wr=new F,Di=new F;function za(i,t,e,n,s){for(let r=0,a=i.length-3;r<=a;r+=3){Di.fromArray(i,r);const o=s.x*Math.abs(Di.x)+s.y*Math.abs(Di.y)+s.z*Math.abs(Di.z),c=t.dot(Di),l=e.dot(Di),h=n.dot(Di);if(Math.max(-Math.max(c,l,h),Math.min(c,l,h))>o)return!1}return!0}const tf=new Fn,Gs=new F,Va=new F;class oi{constructor(t=new F,e=-1){this.isSphere=!0,this.center=t,this.radius=e}set(t,e){return this.center.copy(t),this.radius=e,this}setFromPoints(t,e){const n=this.center;e!==void 0?n.copy(e):tf.setFromPoints(t).getCenter(n);let s=0;for(let r=0,a=t.length;r<a;r++)s=Math.max(s,n.distanceToSquared(t[r]));return this.radius=Math.sqrt(s),this}copy(t){return this.center.copy(t.center),this.radius=t.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(t){return t.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(t){return t.distanceTo(this.center)-this.radius}intersectsSphere(t){const e=this.radius+t.radius;return t.center.distanceToSquared(this.center)<=e*e}intersectsBox(t){return t.intersectsSphere(this)}intersectsPlane(t){return Math.abs(t.distanceToPoint(this.center))<=this.radius}clampPoint(t,e){const n=this.center.distanceToSquared(t);return e.copy(t),n>this.radius*this.radius&&(e.sub(this.center).normalize(),e.multiplyScalar(this.radius).add(this.center)),e}getBoundingBox(t){return this.isEmpty()?(t.makeEmpty(),t):(t.set(this.center,this.center),t.expandByScalar(this.radius),t)}applyMatrix4(t){return this.center.applyMatrix4(t),this.radius=this.radius*t.getMaxScaleOnAxis(),this}translate(t){return this.center.add(t),this}expandByPoint(t){if(this.isEmpty())return this.center.copy(t),this.radius=0,this;Gs.subVectors(t,this.center);const e=Gs.lengthSq();if(e>this.radius*this.radius){const n=Math.sqrt(e),s=(n-this.radius)*.5;this.center.addScaledVector(Gs,s/n),this.radius+=s}return this}union(t){return t.isEmpty()?this:this.isEmpty()?(this.copy(t),this):(this.center.equals(t.center)===!0?this.radius=Math.max(this.radius,t.radius):(Va.subVectors(t.center,this.center).setLength(t.radius),this.expandByPoint(Gs.copy(t.center).add(Va)),this.expandByPoint(Gs.copy(t.center).sub(Va))),this)}equals(t){return t.center.equals(this.center)&&t.radius===this.radius}clone(){return new this.constructor().copy(this)}}const Yn=new F,Ha=new F,Ar=new F,di=new F,Ga=new F,Cr=new F,Wa=new F;class ur{constructor(t=new F,e=new F(0,0,-1)){this.origin=t,this.direction=e}set(t,e){return this.origin.copy(t),this.direction.copy(e),this}copy(t){return this.origin.copy(t.origin),this.direction.copy(t.direction),this}at(t,e){return e.copy(this.origin).addScaledVector(this.direction,t)}lookAt(t){return this.direction.copy(t).sub(this.origin).normalize(),this}recast(t){return this.origin.copy(this.at(t,Yn)),this}closestPointToPoint(t,e){e.subVectors(t,this.origin);const n=e.dot(this.direction);return n<0?e.copy(this.origin):e.copy(this.origin).addScaledVector(this.direction,n)}distanceToPoint(t){return Math.sqrt(this.distanceSqToPoint(t))}distanceSqToPoint(t){const e=Yn.subVectors(t,this.origin).dot(this.direction);return e<0?this.origin.distanceToSquared(t):(Yn.copy(this.origin).addScaledVector(this.direction,e),Yn.distanceToSquared(t))}distanceSqToSegment(t,e,n,s){Ha.copy(t).add(e).multiplyScalar(.5),Ar.copy(e).sub(t).normalize(),di.copy(this.origin).sub(Ha);const r=t.distanceTo(e)*.5,a=-this.direction.dot(Ar),o=di.dot(this.direction),c=-di.dot(Ar),l=di.lengthSq(),h=Math.abs(1-a*a);let u,d,p,g;if(h>0)if(u=a*c-o,d=a*o-c,g=r*h,u>=0)if(d>=-g)if(d<=g){const _=1/h;u*=_,d*=_,p=u*(u+a*d+2*o)+d*(a*u+d+2*c)+l}else d=r,u=Math.max(0,-(a*d+o)),p=-u*u+d*(d+2*c)+l;else d=-r,u=Math.max(0,-(a*d+o)),p=-u*u+d*(d+2*c)+l;else d<=-g?(u=Math.max(0,-(-a*r+o)),d=u>0?-r:Math.min(Math.max(-r,-c),r),p=-u*u+d*(d+2*c)+l):d<=g?(u=0,d=Math.min(Math.max(-r,-c),r),p=d*(d+2*c)+l):(u=Math.max(0,-(a*r+o)),d=u>0?r:Math.min(Math.max(-r,-c),r),p=-u*u+d*(d+2*c)+l);else d=a>0?-r:r,u=Math.max(0,-(a*d+o)),p=-u*u+d*(d+2*c)+l;return n&&n.copy(this.origin).addScaledVector(this.direction,u),s&&s.copy(Ha).addScaledVector(Ar,d),p}intersectSphere(t,e){Yn.subVectors(t.center,this.origin);const n=Yn.dot(this.direction),s=Yn.dot(Yn)-n*n,r=t.radius*t.radius;if(s>r)return null;const a=Math.sqrt(r-s),o=n-a,c=n+a;return c<0?null:o<0?this.at(c,e):this.at(o,e)}intersectsSphere(t){return this.distanceSqToPoint(t.center)<=t.radius*t.radius}distanceToPlane(t){const e=t.normal.dot(this.direction);if(e===0)return t.distanceToPoint(this.origin)===0?0:null;const n=-(this.origin.dot(t.normal)+t.constant)/e;return n>=0?n:null}intersectPlane(t,e){const n=this.distanceToPlane(t);return n===null?null:this.at(n,e)}intersectsPlane(t){const e=t.distanceToPoint(this.origin);return e===0||t.normal.dot(this.direction)*e<0}intersectBox(t,e){let n,s,r,a,o,c;const l=1/this.direction.x,h=1/this.direction.y,u=1/this.direction.z,d=this.origin;return l>=0?(n=(t.min.x-d.x)*l,s=(t.max.x-d.x)*l):(n=(t.max.x-d.x)*l,s=(t.min.x-d.x)*l),h>=0?(r=(t.min.y-d.y)*h,a=(t.max.y-d.y)*h):(r=(t.max.y-d.y)*h,a=(t.min.y-d.y)*h),n>a||r>s||((r>n||isNaN(n))&&(n=r),(a<s||isNaN(s))&&(s=a),u>=0?(o=(t.min.z-d.z)*u,c=(t.max.z-d.z)*u):(o=(t.max.z-d.z)*u,c=(t.min.z-d.z)*u),n>c||o>s)||((o>n||n!==n)&&(n=o),(c<s||s!==s)&&(s=c),s<0)?null:this.at(n>=0?n:s,e)}intersectsBox(t){return this.intersectBox(t,Yn)!==null}intersectTriangle(t,e,n,s,r){Ga.subVectors(e,t),Cr.subVectors(n,t),Wa.crossVectors(Ga,Cr);let a=this.direction.dot(Wa),o;if(a>0){if(s)return null;o=1}else if(a<0)o=-1,a=-a;else return null;di.subVectors(this.origin,t);const c=o*this.direction.dot(Cr.crossVectors(di,Cr));if(c<0)return null;const l=o*this.direction.dot(Ga.cross(di));if(l<0||c+l>a)return null;const h=-o*di.dot(Wa);return h<0?null:this.at(h/a,r)}applyMatrix4(t){return this.origin.applyMatrix4(t),this.direction.transformDirection(t),this}equals(t){return t.origin.equals(this.origin)&&t.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}}class Kt{constructor(t,e,n,s,r,a,o,c,l,h,u,d,p,g,_,m){Kt.prototype.isMatrix4=!0,this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],t!==void 0&&this.set(t,e,n,s,r,a,o,c,l,h,u,d,p,g,_,m)}set(t,e,n,s,r,a,o,c,l,h,u,d,p,g,_,m){const f=this.elements;return f[0]=t,f[4]=e,f[8]=n,f[12]=s,f[1]=r,f[5]=a,f[9]=o,f[13]=c,f[2]=l,f[6]=h,f[10]=u,f[14]=d,f[3]=p,f[7]=g,f[11]=_,f[15]=m,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new Kt().fromArray(this.elements)}copy(t){const e=this.elements,n=t.elements;return e[0]=n[0],e[1]=n[1],e[2]=n[2],e[3]=n[3],e[4]=n[4],e[5]=n[5],e[6]=n[6],e[7]=n[7],e[8]=n[8],e[9]=n[9],e[10]=n[10],e[11]=n[11],e[12]=n[12],e[13]=n[13],e[14]=n[14],e[15]=n[15],this}copyPosition(t){const e=this.elements,n=t.elements;return e[12]=n[12],e[13]=n[13],e[14]=n[14],this}setFromMatrix3(t){const e=t.elements;return this.set(e[0],e[3],e[6],0,e[1],e[4],e[7],0,e[2],e[5],e[8],0,0,0,0,1),this}extractBasis(t,e,n){return t.setFromMatrixColumn(this,0),e.setFromMatrixColumn(this,1),n.setFromMatrixColumn(this,2),this}makeBasis(t,e,n){return this.set(t.x,e.x,n.x,0,t.y,e.y,n.y,0,t.z,e.z,n.z,0,0,0,0,1),this}extractRotation(t){const e=this.elements,n=t.elements,s=1/ns.setFromMatrixColumn(t,0).length(),r=1/ns.setFromMatrixColumn(t,1).length(),a=1/ns.setFromMatrixColumn(t,2).length();return e[0]=n[0]*s,e[1]=n[1]*s,e[2]=n[2]*s,e[3]=0,e[4]=n[4]*r,e[5]=n[5]*r,e[6]=n[6]*r,e[7]=0,e[8]=n[8]*a,e[9]=n[9]*a,e[10]=n[10]*a,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,this}makeRotationFromEuler(t){const e=this.elements,n=t.x,s=t.y,r=t.z,a=Math.cos(n),o=Math.sin(n),c=Math.cos(s),l=Math.sin(s),h=Math.cos(r),u=Math.sin(r);if(t.order==="XYZ"){const d=a*h,p=a*u,g=o*h,_=o*u;e[0]=c*h,e[4]=-c*u,e[8]=l,e[1]=p+g*l,e[5]=d-_*l,e[9]=-o*c,e[2]=_-d*l,e[6]=g+p*l,e[10]=a*c}else if(t.order==="YXZ"){const d=c*h,p=c*u,g=l*h,_=l*u;e[0]=d+_*o,e[4]=g*o-p,e[8]=a*l,e[1]=a*u,e[5]=a*h,e[9]=-o,e[2]=p*o-g,e[6]=_+d*o,e[10]=a*c}else if(t.order==="ZXY"){const d=c*h,p=c*u,g=l*h,_=l*u;e[0]=d-_*o,e[4]=-a*u,e[8]=g+p*o,e[1]=p+g*o,e[5]=a*h,e[9]=_-d*o,e[2]=-a*l,e[6]=o,e[10]=a*c}else if(t.order==="ZYX"){const d=a*h,p=a*u,g=o*h,_=o*u;e[0]=c*h,e[4]=g*l-p,e[8]=d*l+_,e[1]=c*u,e[5]=_*l+d,e[9]=p*l-g,e[2]=-l,e[6]=o*c,e[10]=a*c}else if(t.order==="YZX"){const d=a*c,p=a*l,g=o*c,_=o*l;e[0]=c*h,e[4]=_-d*u,e[8]=g*u+p,e[1]=u,e[5]=a*h,e[9]=-o*h,e[2]=-l*h,e[6]=p*u+g,e[10]=d-_*u}else if(t.order==="XZY"){const d=a*c,p=a*l,g=o*c,_=o*l;e[0]=c*h,e[4]=-u,e[8]=l*h,e[1]=d*u+_,e[5]=a*h,e[9]=p*u-g,e[2]=g*u-p,e[6]=o*h,e[10]=_*u+d}return e[3]=0,e[7]=0,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,this}makeRotationFromQuaternion(t){return this.compose(ef,t,nf)}lookAt(t,e,n){const s=this.elements;return un.subVectors(t,e),un.lengthSq()===0&&(un.z=1),un.normalize(),fi.crossVectors(n,un),fi.lengthSq()===0&&(Math.abs(n.z)===1?un.x+=1e-4:un.z+=1e-4,un.normalize(),fi.crossVectors(n,un)),fi.normalize(),Rr.crossVectors(un,fi),s[0]=fi.x,s[4]=Rr.x,s[8]=un.x,s[1]=fi.y,s[5]=Rr.y,s[9]=un.y,s[2]=fi.z,s[6]=Rr.z,s[10]=un.z,this}multiply(t){return this.multiplyMatrices(this,t)}premultiply(t){return this.multiplyMatrices(t,this)}multiplyMatrices(t,e){const n=t.elements,s=e.elements,r=this.elements,a=n[0],o=n[4],c=n[8],l=n[12],h=n[1],u=n[5],d=n[9],p=n[13],g=n[2],_=n[6],m=n[10],f=n[14],R=n[3],C=n[7],x=n[11],N=n[15],P=s[0],L=s[4],I=s[8],M=s[12],S=s[1],U=s[5],k=s[9],H=s[13],J=s[2],Q=s[6],O=s[10],tt=s[14],W=s[3],ot=s[7],ft=s[11],pt=s[15];return r[0]=a*P+o*S+c*J+l*W,r[4]=a*L+o*U+c*Q+l*ot,r[8]=a*I+o*k+c*O+l*ft,r[12]=a*M+o*H+c*tt+l*pt,r[1]=h*P+u*S+d*J+p*W,r[5]=h*L+u*U+d*Q+p*ot,r[9]=h*I+u*k+d*O+p*ft,r[13]=h*M+u*H+d*tt+p*pt,r[2]=g*P+_*S+m*J+f*W,r[6]=g*L+_*U+m*Q+f*ot,r[10]=g*I+_*k+m*O+f*ft,r[14]=g*M+_*H+m*tt+f*pt,r[3]=R*P+C*S+x*J+N*W,r[7]=R*L+C*U+x*Q+N*ot,r[11]=R*I+C*k+x*O+N*ft,r[15]=R*M+C*H+x*tt+N*pt,this}multiplyScalar(t){const e=this.elements;return e[0]*=t,e[4]*=t,e[8]*=t,e[12]*=t,e[1]*=t,e[5]*=t,e[9]*=t,e[13]*=t,e[2]*=t,e[6]*=t,e[10]*=t,e[14]*=t,e[3]*=t,e[7]*=t,e[11]*=t,e[15]*=t,this}determinant(){const t=this.elements,e=t[0],n=t[4],s=t[8],r=t[12],a=t[1],o=t[5],c=t[9],l=t[13],h=t[2],u=t[6],d=t[10],p=t[14],g=t[3],_=t[7],m=t[11],f=t[15];return g*(+r*c*u-s*l*u-r*o*d+n*l*d+s*o*p-n*c*p)+_*(+e*c*p-e*l*d+r*a*d-s*a*p+s*l*h-r*c*h)+m*(+e*l*u-e*o*p-r*a*u+n*a*p+r*o*h-n*l*h)+f*(-s*o*h-e*c*u+e*o*d+s*a*u-n*a*d+n*c*h)}transpose(){const t=this.elements;let e;return e=t[1],t[1]=t[4],t[4]=e,e=t[2],t[2]=t[8],t[8]=e,e=t[6],t[6]=t[9],t[9]=e,e=t[3],t[3]=t[12],t[12]=e,e=t[7],t[7]=t[13],t[13]=e,e=t[11],t[11]=t[14],t[14]=e,this}setPosition(t,e,n){const s=this.elements;return t.isVector3?(s[12]=t.x,s[13]=t.y,s[14]=t.z):(s[12]=t,s[13]=e,s[14]=n),this}invert(){const t=this.elements,e=t[0],n=t[1],s=t[2],r=t[3],a=t[4],o=t[5],c=t[6],l=t[7],h=t[8],u=t[9],d=t[10],p=t[11],g=t[12],_=t[13],m=t[14],f=t[15],R=u*m*l-_*d*l+_*c*p-o*m*p-u*c*f+o*d*f,C=g*d*l-h*m*l-g*c*p+a*m*p+h*c*f-a*d*f,x=h*_*l-g*u*l+g*o*p-a*_*p-h*o*f+a*u*f,N=g*u*c-h*_*c-g*o*d+a*_*d+h*o*m-a*u*m,P=e*R+n*C+s*x+r*N;if(P===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);const L=1/P;return t[0]=R*L,t[1]=(_*d*r-u*m*r-_*s*p+n*m*p+u*s*f-n*d*f)*L,t[2]=(o*m*r-_*c*r+_*s*l-n*m*l-o*s*f+n*c*f)*L,t[3]=(u*c*r-o*d*r-u*s*l+n*d*l+o*s*p-n*c*p)*L,t[4]=C*L,t[5]=(h*m*r-g*d*r+g*s*p-e*m*p-h*s*f+e*d*f)*L,t[6]=(g*c*r-a*m*r-g*s*l+e*m*l+a*s*f-e*c*f)*L,t[7]=(a*d*r-h*c*r+h*s*l-e*d*l-a*s*p+e*c*p)*L,t[8]=x*L,t[9]=(g*u*r-h*_*r-g*n*p+e*_*p+h*n*f-e*u*f)*L,t[10]=(a*_*r-g*o*r+g*n*l-e*_*l-a*n*f+e*o*f)*L,t[11]=(h*o*r-a*u*r-h*n*l+e*u*l+a*n*p-e*o*p)*L,t[12]=N*L,t[13]=(h*_*s-g*u*s+g*n*d-e*_*d-h*n*m+e*u*m)*L,t[14]=(g*o*s-a*_*s-g*n*c+e*_*c+a*n*m-e*o*m)*L,t[15]=(a*u*s-h*o*s+h*n*c-e*u*c-a*n*d+e*o*d)*L,this}scale(t){const e=this.elements,n=t.x,s=t.y,r=t.z;return e[0]*=n,e[4]*=s,e[8]*=r,e[1]*=n,e[5]*=s,e[9]*=r,e[2]*=n,e[6]*=s,e[10]*=r,e[3]*=n,e[7]*=s,e[11]*=r,this}getMaxScaleOnAxis(){const t=this.elements,e=t[0]*t[0]+t[1]*t[1]+t[2]*t[2],n=t[4]*t[4]+t[5]*t[5]+t[6]*t[6],s=t[8]*t[8]+t[9]*t[9]+t[10]*t[10];return Math.sqrt(Math.max(e,n,s))}makeTranslation(t,e,n){return t.isVector3?this.set(1,0,0,t.x,0,1,0,t.y,0,0,1,t.z,0,0,0,1):this.set(1,0,0,t,0,1,0,e,0,0,1,n,0,0,0,1),this}makeRotationX(t){const e=Math.cos(t),n=Math.sin(t);return this.set(1,0,0,0,0,e,-n,0,0,n,e,0,0,0,0,1),this}makeRotationY(t){const e=Math.cos(t),n=Math.sin(t);return this.set(e,0,n,0,0,1,0,0,-n,0,e,0,0,0,0,1),this}makeRotationZ(t){const e=Math.cos(t),n=Math.sin(t);return this.set(e,-n,0,0,n,e,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(t,e){const n=Math.cos(e),s=Math.sin(e),r=1-n,a=t.x,o=t.y,c=t.z,l=r*a,h=r*o;return this.set(l*a+n,l*o-s*c,l*c+s*o,0,l*o+s*c,h*o+n,h*c-s*a,0,l*c-s*o,h*c+s*a,r*c*c+n,0,0,0,0,1),this}makeScale(t,e,n){return this.set(t,0,0,0,0,e,0,0,0,0,n,0,0,0,0,1),this}makeShear(t,e,n,s,r,a){return this.set(1,n,r,0,t,1,a,0,e,s,1,0,0,0,0,1),this}compose(t,e,n){const s=this.elements,r=e._x,a=e._y,o=e._z,c=e._w,l=r+r,h=a+a,u=o+o,d=r*l,p=r*h,g=r*u,_=a*h,m=a*u,f=o*u,R=c*l,C=c*h,x=c*u,N=n.x,P=n.y,L=n.z;return s[0]=(1-(_+f))*N,s[1]=(p+x)*N,s[2]=(g-C)*N,s[3]=0,s[4]=(p-x)*P,s[5]=(1-(d+f))*P,s[6]=(m+R)*P,s[7]=0,s[8]=(g+C)*L,s[9]=(m-R)*L,s[10]=(1-(d+_))*L,s[11]=0,s[12]=t.x,s[13]=t.y,s[14]=t.z,s[15]=1,this}decompose(t,e,n){const s=this.elements;let r=ns.set(s[0],s[1],s[2]).length();const a=ns.set(s[4],s[5],s[6]).length(),o=ns.set(s[8],s[9],s[10]).length();this.determinant()<0&&(r=-r),t.x=s[12],t.y=s[13],t.z=s[14],Cn.copy(this);const l=1/r,h=1/a,u=1/o;return Cn.elements[0]*=l,Cn.elements[1]*=l,Cn.elements[2]*=l,Cn.elements[4]*=h,Cn.elements[5]*=h,Cn.elements[6]*=h,Cn.elements[8]*=u,Cn.elements[9]*=u,Cn.elements[10]*=u,e.setFromRotationMatrix(Cn),n.x=r,n.y=a,n.z=o,this}makePerspective(t,e,n,s,r,a,o=ei){const c=this.elements,l=2*r/(e-t),h=2*r/(n-s),u=(e+t)/(e-t),d=(n+s)/(n-s);let p,g;if(o===ei)p=-(a+r)/(a-r),g=-2*a*r/(a-r);else if(o===fa)p=-a/(a-r),g=-a*r/(a-r);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+o);return c[0]=l,c[4]=0,c[8]=u,c[12]=0,c[1]=0,c[5]=h,c[9]=d,c[13]=0,c[2]=0,c[6]=0,c[10]=p,c[14]=g,c[3]=0,c[7]=0,c[11]=-1,c[15]=0,this}makeOrthographic(t,e,n,s,r,a,o=ei){const c=this.elements,l=1/(e-t),h=1/(n-s),u=1/(a-r),d=(e+t)*l,p=(n+s)*h;let g,_;if(o===ei)g=(a+r)*u,_=-2*u;else if(o===fa)g=r*u,_=-1*u;else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+o);return c[0]=2*l,c[4]=0,c[8]=0,c[12]=-d,c[1]=0,c[5]=2*h,c[9]=0,c[13]=-p,c[2]=0,c[6]=0,c[10]=_,c[14]=-g,c[3]=0,c[7]=0,c[11]=0,c[15]=1,this}equals(t){const e=this.elements,n=t.elements;for(let s=0;s<16;s++)if(e[s]!==n[s])return!1;return!0}fromArray(t,e=0){for(let n=0;n<16;n++)this.elements[n]=t[n+e];return this}toArray(t=[],e=0){const n=this.elements;return t[e]=n[0],t[e+1]=n[1],t[e+2]=n[2],t[e+3]=n[3],t[e+4]=n[4],t[e+5]=n[5],t[e+6]=n[6],t[e+7]=n[7],t[e+8]=n[8],t[e+9]=n[9],t[e+10]=n[10],t[e+11]=n[11],t[e+12]=n[12],t[e+13]=n[13],t[e+14]=n[14],t[e+15]=n[15],t}}const ns=new F,Cn=new Kt,ef=new F(0,0,0),nf=new F(1,1,1),fi=new F,Rr=new F,un=new F,cc=new Kt,hc=new Dn;class cn{constructor(t=0,e=0,n=0,s=cn.DEFAULT_ORDER){this.isEuler=!0,this._x=t,this._y=e,this._z=n,this._order=s}get x(){return this._x}set x(t){this._x=t,this._onChangeCallback()}get y(){return this._y}set y(t){this._y=t,this._onChangeCallback()}get z(){return this._z}set z(t){this._z=t,this._onChangeCallback()}get order(){return this._order}set order(t){this._order=t,this._onChangeCallback()}set(t,e,n,s=this._order){return this._x=t,this._y=e,this._z=n,this._order=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(t){return this._x=t._x,this._y=t._y,this._z=t._z,this._order=t._order,this._onChangeCallback(),this}setFromRotationMatrix(t,e=this._order,n=!0){const s=t.elements,r=s[0],a=s[4],o=s[8],c=s[1],l=s[5],h=s[9],u=s[2],d=s[6],p=s[10];switch(e){case"XYZ":this._y=Math.asin(Xe(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(-h,p),this._z=Math.atan2(-a,r)):(this._x=Math.atan2(d,l),this._z=0);break;case"YXZ":this._x=Math.asin(-Xe(h,-1,1)),Math.abs(h)<.9999999?(this._y=Math.atan2(o,p),this._z=Math.atan2(c,l)):(this._y=Math.atan2(-u,r),this._z=0);break;case"ZXY":this._x=Math.asin(Xe(d,-1,1)),Math.abs(d)<.9999999?(this._y=Math.atan2(-u,p),this._z=Math.atan2(-a,l)):(this._y=0,this._z=Math.atan2(c,r));break;case"ZYX":this._y=Math.asin(-Xe(u,-1,1)),Math.abs(u)<.9999999?(this._x=Math.atan2(d,p),this._z=Math.atan2(c,r)):(this._x=0,this._z=Math.atan2(-a,l));break;case"YZX":this._z=Math.asin(Xe(c,-1,1)),Math.abs(c)<.9999999?(this._x=Math.atan2(-h,l),this._y=Math.atan2(-u,r)):(this._x=0,this._y=Math.atan2(o,p));break;case"XZY":this._z=Math.asin(-Xe(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(d,l),this._y=Math.atan2(o,r)):(this._x=Math.atan2(-h,p),this._y=0);break;default:console.warn("THREE.Euler: .setFromRotationMatrix() encountered an unknown order: "+e)}return this._order=e,n===!0&&this._onChangeCallback(),this}setFromQuaternion(t,e,n){return cc.makeRotationFromQuaternion(t),this.setFromRotationMatrix(cc,e,n)}setFromVector3(t,e=this._order){return this.set(t.x,t.y,t.z,e)}reorder(t){return hc.setFromEuler(this),this.setFromQuaternion(hc,t)}equals(t){return t._x===this._x&&t._y===this._y&&t._z===this._z&&t._order===this._order}fromArray(t){return this._x=t[0],this._y=t[1],this._z=t[2],t[3]!==void 0&&(this._order=t[3]),this._onChangeCallback(),this}toArray(t=[],e=0){return t[e]=this._x,t[e+1]=this._y,t[e+2]=this._z,t[e+3]=this._order,t}_onChange(t){return this._onChangeCallback=t,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}}cn.DEFAULT_ORDER="XYZ";class Wh{constructor(){this.mask=1}set(t){this.mask=(1<<t|0)>>>0}enable(t){this.mask|=1<<t|0}enableAll(){this.mask=-1}toggle(t){this.mask^=1<<t|0}disable(t){this.mask&=~(1<<t|0)}disableAll(){this.mask=0}test(t){return(this.mask&t.mask)!==0}isEnabled(t){return(this.mask&(1<<t|0))!==0}}let sf=0;const uc=new F,is=new Dn,$n=new Kt,Pr=new F,Ws=new F,rf=new F,af=new Dn,dc=new F(1,0,0),fc=new F(0,1,0),pc=new F(0,0,1),mc={type:"added"},of={type:"removed"},ss={type:"childadded",child:null},Xa={type:"childremoved",child:null};class Le extends qi{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:sf++}),this.uuid=Ci(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=Le.DEFAULT_UP.clone();const t=new F,e=new cn,n=new Dn,s=new F(1,1,1);function r(){n.setFromEuler(e,!1)}function a(){e.setFromQuaternion(n,void 0,!1)}e._onChange(r),n._onChange(a),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:t},rotation:{configurable:!0,enumerable:!0,value:e},quaternion:{configurable:!0,enumerable:!0,value:n},scale:{configurable:!0,enumerable:!0,value:s},modelViewMatrix:{value:new Kt},normalMatrix:{value:new ie}}),this.matrix=new Kt,this.matrixWorld=new Kt,this.matrixAutoUpdate=Le.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=Le.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new Wh,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.userData={}}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(t){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(t),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(t){return this.quaternion.premultiply(t),this}setRotationFromAxisAngle(t,e){this.quaternion.setFromAxisAngle(t,e)}setRotationFromEuler(t){this.quaternion.setFromEuler(t,!0)}setRotationFromMatrix(t){this.quaternion.setFromRotationMatrix(t)}setRotationFromQuaternion(t){this.quaternion.copy(t)}rotateOnAxis(t,e){return is.setFromAxisAngle(t,e),this.quaternion.multiply(is),this}rotateOnWorldAxis(t,e){return is.setFromAxisAngle(t,e),this.quaternion.premultiply(is),this}rotateX(t){return this.rotateOnAxis(dc,t)}rotateY(t){return this.rotateOnAxis(fc,t)}rotateZ(t){return this.rotateOnAxis(pc,t)}translateOnAxis(t,e){return uc.copy(t).applyQuaternion(this.quaternion),this.position.add(uc.multiplyScalar(e)),this}translateX(t){return this.translateOnAxis(dc,t)}translateY(t){return this.translateOnAxis(fc,t)}translateZ(t){return this.translateOnAxis(pc,t)}localToWorld(t){return this.updateWorldMatrix(!0,!1),t.applyMatrix4(this.matrixWorld)}worldToLocal(t){return this.updateWorldMatrix(!0,!1),t.applyMatrix4($n.copy(this.matrixWorld).invert())}lookAt(t,e,n){t.isVector3?Pr.copy(t):Pr.set(t,e,n);const s=this.parent;this.updateWorldMatrix(!0,!1),Ws.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?$n.lookAt(Ws,Pr,this.up):$n.lookAt(Pr,Ws,this.up),this.quaternion.setFromRotationMatrix($n),s&&($n.extractRotation(s.matrixWorld),is.setFromRotationMatrix($n),this.quaternion.premultiply(is.invert()))}add(t){if(arguments.length>1){for(let e=0;e<arguments.length;e++)this.add(arguments[e]);return this}return t===this?(console.error("THREE.Object3D.add: object can't be added as a child of itself.",t),this):(t&&t.isObject3D?(t.removeFromParent(),t.parent=this,this.children.push(t),t.dispatchEvent(mc),ss.child=t,this.dispatchEvent(ss),ss.child=null):console.error("THREE.Object3D.add: object not an instance of THREE.Object3D.",t),this)}remove(t){if(arguments.length>1){for(let n=0;n<arguments.length;n++)this.remove(arguments[n]);return this}const e=this.children.indexOf(t);return e!==-1&&(t.parent=null,this.children.splice(e,1),t.dispatchEvent(of),Xa.child=t,this.dispatchEvent(Xa),Xa.child=null),this}removeFromParent(){const t=this.parent;return t!==null&&t.remove(this),this}clear(){return this.remove(...this.children)}attach(t){return this.updateWorldMatrix(!0,!1),$n.copy(this.matrixWorld).invert(),t.parent!==null&&(t.parent.updateWorldMatrix(!0,!1),$n.multiply(t.parent.matrixWorld)),t.applyMatrix4($n),t.removeFromParent(),t.parent=this,this.children.push(t),t.updateWorldMatrix(!1,!0),t.dispatchEvent(mc),ss.child=t,this.dispatchEvent(ss),ss.child=null,this}getObjectById(t){return this.getObjectByProperty("id",t)}getObjectByName(t){return this.getObjectByProperty("name",t)}getObjectByProperty(t,e){if(this[t]===e)return this;for(let n=0,s=this.children.length;n<s;n++){const a=this.children[n].getObjectByProperty(t,e);if(a!==void 0)return a}}getObjectsByProperty(t,e,n=[]){this[t]===e&&n.push(this);const s=this.children;for(let r=0,a=s.length;r<a;r++)s[r].getObjectsByProperty(t,e,n);return n}getWorldPosition(t){return this.updateWorldMatrix(!0,!1),t.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(t){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Ws,t,rf),t}getWorldScale(t){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Ws,af,t),t}getWorldDirection(t){this.updateWorldMatrix(!0,!1);const e=this.matrixWorld.elements;return t.set(e[8],e[9],e[10]).normalize()}raycast(){}traverse(t){t(this);const e=this.children;for(let n=0,s=e.length;n<s;n++)e[n].traverse(t)}traverseVisible(t){if(this.visible===!1)return;t(this);const e=this.children;for(let n=0,s=e.length;n<s;n++)e[n].traverseVisible(t)}traverseAncestors(t){const e=this.parent;e!==null&&(t(e),e.traverseAncestors(t))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale),this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(t){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||t)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,t=!0);const e=this.children;for(let n=0,s=e.length;n<s;n++)e[n].updateMatrixWorld(t)}updateWorldMatrix(t,e){const n=this.parent;if(t===!0&&n!==null&&n.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),e===!0){const s=this.children;for(let r=0,a=s.length;r<a;r++)s[r].updateWorldMatrix(!1,!0)}}toJSON(t){const e=t===void 0||typeof t=="string",n={};e&&(t={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},n.metadata={version:4.6,type:"Object",generator:"Object3D.toJSON"});const s={};s.uuid=this.uuid,s.type=this.type,this.name!==""&&(s.name=this.name),this.castShadow===!0&&(s.castShadow=!0),this.receiveShadow===!0&&(s.receiveShadow=!0),this.visible===!1&&(s.visible=!1),this.frustumCulled===!1&&(s.frustumCulled=!1),this.renderOrder!==0&&(s.renderOrder=this.renderOrder),Object.keys(this.userData).length>0&&(s.userData=this.userData),s.layers=this.layers.mask,s.matrix=this.matrix.toArray(),s.up=this.up.toArray(),this.matrixAutoUpdate===!1&&(s.matrixAutoUpdate=!1),this.isInstancedMesh&&(s.type="InstancedMesh",s.count=this.count,s.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(s.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(s.type="BatchedMesh",s.perObjectFrustumCulled=this.perObjectFrustumCulled,s.sortObjects=this.sortObjects,s.drawRanges=this._drawRanges,s.reservedRanges=this._reservedRanges,s.visibility=this._visibility,s.active=this._active,s.bounds=this._bounds.map(o=>({boxInitialized:o.boxInitialized,boxMin:o.box.min.toArray(),boxMax:o.box.max.toArray(),sphereInitialized:o.sphereInitialized,sphereRadius:o.sphere.radius,sphereCenter:o.sphere.center.toArray()})),s.maxInstanceCount=this._maxInstanceCount,s.maxVertexCount=this._maxVertexCount,s.maxIndexCount=this._maxIndexCount,s.geometryInitialized=this._geometryInitialized,s.geometryCount=this._geometryCount,s.matricesTexture=this._matricesTexture.toJSON(t),this._colorsTexture!==null&&(s.colorsTexture=this._colorsTexture.toJSON(t)),this.boundingSphere!==null&&(s.boundingSphere={center:s.boundingSphere.center.toArray(),radius:s.boundingSphere.radius}),this.boundingBox!==null&&(s.boundingBox={min:s.boundingBox.min.toArray(),max:s.boundingBox.max.toArray()}));function r(o,c){return o[c.uuid]===void 0&&(o[c.uuid]=c.toJSON(t)),c.uuid}if(this.isScene)this.background&&(this.background.isColor?s.background=this.background.toJSON():this.background.isTexture&&(s.background=this.background.toJSON(t).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(s.environment=this.environment.toJSON(t).uuid);else if(this.isMesh||this.isLine||this.isPoints){s.geometry=r(t.geometries,this.geometry);const o=this.geometry.parameters;if(o!==void 0&&o.shapes!==void 0){const c=o.shapes;if(Array.isArray(c))for(let l=0,h=c.length;l<h;l++){const u=c[l];r(t.shapes,u)}else r(t.shapes,c)}}if(this.isSkinnedMesh&&(s.bindMode=this.bindMode,s.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(r(t.skeletons,this.skeleton),s.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){const o=[];for(let c=0,l=this.material.length;c<l;c++)o.push(r(t.materials,this.material[c]));s.material=o}else s.material=r(t.materials,this.material);if(this.children.length>0){s.children=[];for(let o=0;o<this.children.length;o++)s.children.push(this.children[o].toJSON(t).object)}if(this.animations.length>0){s.animations=[];for(let o=0;o<this.animations.length;o++){const c=this.animations[o];s.animations.push(r(t.animations,c))}}if(e){const o=a(t.geometries),c=a(t.materials),l=a(t.textures),h=a(t.images),u=a(t.shapes),d=a(t.skeletons),p=a(t.animations),g=a(t.nodes);o.length>0&&(n.geometries=o),c.length>0&&(n.materials=c),l.length>0&&(n.textures=l),h.length>0&&(n.images=h),u.length>0&&(n.shapes=u),d.length>0&&(n.skeletons=d),p.length>0&&(n.animations=p),g.length>0&&(n.nodes=g)}return n.object=s,n;function a(o){const c=[];for(const l in o){const h=o[l];delete h.metadata,c.push(h)}return c}}clone(t){return new this.constructor().copy(this,t)}copy(t,e=!0){if(this.name=t.name,this.up.copy(t.up),this.position.copy(t.position),this.rotation.order=t.rotation.order,this.quaternion.copy(t.quaternion),this.scale.copy(t.scale),this.matrix.copy(t.matrix),this.matrixWorld.copy(t.matrixWorld),this.matrixAutoUpdate=t.matrixAutoUpdate,this.matrixWorldAutoUpdate=t.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=t.matrixWorldNeedsUpdate,this.layers.mask=t.layers.mask,this.visible=t.visible,this.castShadow=t.castShadow,this.receiveShadow=t.receiveShadow,this.frustumCulled=t.frustumCulled,this.renderOrder=t.renderOrder,this.animations=t.animations.slice(),this.userData=JSON.parse(JSON.stringify(t.userData)),e===!0)for(let n=0;n<t.children.length;n++){const s=t.children[n];this.add(s.clone())}return this}}Le.DEFAULT_UP=new F(0,1,0);Le.DEFAULT_MATRIX_AUTO_UPDATE=!0;Le.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;const Rn=new F,Kn=new F,qa=new F,Jn=new F,rs=new F,as=new F,gc=new F,ja=new F,Ya=new F,$a=new F,Ka=new ve,Ja=new ve,Za=new ve;class Ln{constructor(t=new F,e=new F,n=new F){this.a=t,this.b=e,this.c=n}static getNormal(t,e,n,s){s.subVectors(n,e),Rn.subVectors(t,e),s.cross(Rn);const r=s.lengthSq();return r>0?s.multiplyScalar(1/Math.sqrt(r)):s.set(0,0,0)}static getBarycoord(t,e,n,s,r){Rn.subVectors(s,e),Kn.subVectors(n,e),qa.subVectors(t,e);const a=Rn.dot(Rn),o=Rn.dot(Kn),c=Rn.dot(qa),l=Kn.dot(Kn),h=Kn.dot(qa),u=a*l-o*o;if(u===0)return r.set(0,0,0),null;const d=1/u,p=(l*c-o*h)*d,g=(a*h-o*c)*d;return r.set(1-p-g,g,p)}static containsPoint(t,e,n,s){return this.getBarycoord(t,e,n,s,Jn)===null?!1:Jn.x>=0&&Jn.y>=0&&Jn.x+Jn.y<=1}static getInterpolation(t,e,n,s,r,a,o,c){return this.getBarycoord(t,e,n,s,Jn)===null?(c.x=0,c.y=0,"z"in c&&(c.z=0),"w"in c&&(c.w=0),null):(c.setScalar(0),c.addScaledVector(r,Jn.x),c.addScaledVector(a,Jn.y),c.addScaledVector(o,Jn.z),c)}static getInterpolatedAttribute(t,e,n,s,r,a){return Ka.setScalar(0),Ja.setScalar(0),Za.setScalar(0),Ka.fromBufferAttribute(t,e),Ja.fromBufferAttribute(t,n),Za.fromBufferAttribute(t,s),a.setScalar(0),a.addScaledVector(Ka,r.x),a.addScaledVector(Ja,r.y),a.addScaledVector(Za,r.z),a}static isFrontFacing(t,e,n,s){return Rn.subVectors(n,e),Kn.subVectors(t,e),Rn.cross(Kn).dot(s)<0}set(t,e,n){return this.a.copy(t),this.b.copy(e),this.c.copy(n),this}setFromPointsAndIndices(t,e,n,s){return this.a.copy(t[e]),this.b.copy(t[n]),this.c.copy(t[s]),this}setFromAttributeAndIndices(t,e,n,s){return this.a.fromBufferAttribute(t,e),this.b.fromBufferAttribute(t,n),this.c.fromBufferAttribute(t,s),this}clone(){return new this.constructor().copy(this)}copy(t){return this.a.copy(t.a),this.b.copy(t.b),this.c.copy(t.c),this}getArea(){return Rn.subVectors(this.c,this.b),Kn.subVectors(this.a,this.b),Rn.cross(Kn).length()*.5}getMidpoint(t){return t.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(t){return Ln.getNormal(this.a,this.b,this.c,t)}getPlane(t){return t.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(t,e){return Ln.getBarycoord(t,this.a,this.b,this.c,e)}getInterpolation(t,e,n,s,r){return Ln.getInterpolation(t,this.a,this.b,this.c,e,n,s,r)}containsPoint(t){return Ln.containsPoint(t,this.a,this.b,this.c)}isFrontFacing(t){return Ln.isFrontFacing(this.a,this.b,this.c,t)}intersectsBox(t){return t.intersectsTriangle(this)}closestPointToPoint(t,e){const n=this.a,s=this.b,r=this.c;let a,o;rs.subVectors(s,n),as.subVectors(r,n),ja.subVectors(t,n);const c=rs.dot(ja),l=as.dot(ja);if(c<=0&&l<=0)return e.copy(n);Ya.subVectors(t,s);const h=rs.dot(Ya),u=as.dot(Ya);if(h>=0&&u<=h)return e.copy(s);const d=c*u-h*l;if(d<=0&&c>=0&&h<=0)return a=c/(c-h),e.copy(n).addScaledVector(rs,a);$a.subVectors(t,r);const p=rs.dot($a),g=as.dot($a);if(g>=0&&p<=g)return e.copy(r);const _=p*l-c*g;if(_<=0&&l>=0&&g<=0)return o=l/(l-g),e.copy(n).addScaledVector(as,o);const m=h*g-p*u;if(m<=0&&u-h>=0&&p-g>=0)return gc.subVectors(r,s),o=(u-h)/(u-h+(p-g)),e.copy(s).addScaledVector(gc,o);const f=1/(m+_+d);return a=_*f,o=d*f,e.copy(n).addScaledVector(rs,a).addScaledVector(as,o)}equals(t){return t.a.equals(this.a)&&t.b.equals(this.b)&&t.c.equals(this.c)}}const Xh={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},pi={h:0,s:0,l:0},Lr={h:0,s:0,l:0};function Qa(i,t,e){return e<0&&(e+=1),e>1&&(e-=1),e<1/6?i+(t-i)*6*e:e<1/2?t:e<2/3?i+(t-i)*6*(2/3-e):i}class $t{constructor(t,e,n){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(t,e,n)}set(t,e,n){if(e===void 0&&n===void 0){const s=t;s&&s.isColor?this.copy(s):typeof s=="number"?this.setHex(s):typeof s=="string"&&this.setStyle(s)}else this.setRGB(t,e,n);return this}setScalar(t){return this.r=t,this.g=t,this.b=t,this}setHex(t,e=Pe){return t=Math.floor(t),this.r=(t>>16&255)/255,this.g=(t>>8&255)/255,this.b=(t&255)/255,le.toWorkingColorSpace(this,e),this}setRGB(t,e,n,s=le.workingColorSpace){return this.r=t,this.g=e,this.b=n,le.toWorkingColorSpace(this,s),this}setHSL(t,e,n,s=le.workingColorSpace){if(t=Sl(t,1),e=Xe(e,0,1),n=Xe(n,0,1),e===0)this.r=this.g=this.b=n;else{const r=n<=.5?n*(1+e):n+e-n*e,a=2*n-r;this.r=Qa(a,r,t+1/3),this.g=Qa(a,r,t),this.b=Qa(a,r,t-1/3)}return le.toWorkingColorSpace(this,s),this}setStyle(t,e=Pe){function n(r){r!==void 0&&parseFloat(r)<1&&console.warn("THREE.Color: Alpha component of "+t+" will be ignored.")}let s;if(s=/^(\w+)\(([^\)]*)\)/.exec(t)){let r;const a=s[1],o=s[2];switch(a){case"rgb":case"rgba":if(r=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(r[4]),this.setRGB(Math.min(255,parseInt(r[1],10))/255,Math.min(255,parseInt(r[2],10))/255,Math.min(255,parseInt(r[3],10))/255,e);if(r=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(r[4]),this.setRGB(Math.min(100,parseInt(r[1],10))/100,Math.min(100,parseInt(r[2],10))/100,Math.min(100,parseInt(r[3],10))/100,e);break;case"hsl":case"hsla":if(r=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(r[4]),this.setHSL(parseFloat(r[1])/360,parseFloat(r[2])/100,parseFloat(r[3])/100,e);break;default:console.warn("THREE.Color: Unknown color model "+t)}}else if(s=/^\#([A-Fa-f\d]+)$/.exec(t)){const r=s[1],a=r.length;if(a===3)return this.setRGB(parseInt(r.charAt(0),16)/15,parseInt(r.charAt(1),16)/15,parseInt(r.charAt(2),16)/15,e);if(a===6)return this.setHex(parseInt(r,16),e);console.warn("THREE.Color: Invalid hex color "+t)}else if(t&&t.length>0)return this.setColorName(t,e);return this}setColorName(t,e=Pe){const n=Xh[t.toLowerCase()];return n!==void 0?this.setHex(n,e):console.warn("THREE.Color: Unknown color "+t),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(t){return this.r=t.r,this.g=t.g,this.b=t.b,this}copySRGBToLinear(t){return this.r=ni(t.r),this.g=ni(t.g),this.b=ni(t.b),this}copyLinearToSRGB(t){return this.r=_s(t.r),this.g=_s(t.g),this.b=_s(t.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(t=Pe){return le.fromWorkingColorSpace(Ke.copy(this),t),Math.round(Xe(Ke.r*255,0,255))*65536+Math.round(Xe(Ke.g*255,0,255))*256+Math.round(Xe(Ke.b*255,0,255))}getHexString(t=Pe){return("000000"+this.getHex(t).toString(16)).slice(-6)}getHSL(t,e=le.workingColorSpace){le.fromWorkingColorSpace(Ke.copy(this),e);const n=Ke.r,s=Ke.g,r=Ke.b,a=Math.max(n,s,r),o=Math.min(n,s,r);let c,l;const h=(o+a)/2;if(o===a)c=0,l=0;else{const u=a-o;switch(l=h<=.5?u/(a+o):u/(2-a-o),a){case n:c=(s-r)/u+(s<r?6:0);break;case s:c=(r-n)/u+2;break;case r:c=(n-s)/u+4;break}c/=6}return t.h=c,t.s=l,t.l=h,t}getRGB(t,e=le.workingColorSpace){return le.fromWorkingColorSpace(Ke.copy(this),e),t.r=Ke.r,t.g=Ke.g,t.b=Ke.b,t}getStyle(t=Pe){le.fromWorkingColorSpace(Ke.copy(this),t);const e=Ke.r,n=Ke.g,s=Ke.b;return t!==Pe?`color(${t} ${e.toFixed(3)} ${n.toFixed(3)} ${s.toFixed(3)})`:`rgb(${Math.round(e*255)},${Math.round(n*255)},${Math.round(s*255)})`}offsetHSL(t,e,n){return this.getHSL(pi),this.setHSL(pi.h+t,pi.s+e,pi.l+n)}add(t){return this.r+=t.r,this.g+=t.g,this.b+=t.b,this}addColors(t,e){return this.r=t.r+e.r,this.g=t.g+e.g,this.b=t.b+e.b,this}addScalar(t){return this.r+=t,this.g+=t,this.b+=t,this}sub(t){return this.r=Math.max(0,this.r-t.r),this.g=Math.max(0,this.g-t.g),this.b=Math.max(0,this.b-t.b),this}multiply(t){return this.r*=t.r,this.g*=t.g,this.b*=t.b,this}multiplyScalar(t){return this.r*=t,this.g*=t,this.b*=t,this}lerp(t,e){return this.r+=(t.r-this.r)*e,this.g+=(t.g-this.g)*e,this.b+=(t.b-this.b)*e,this}lerpColors(t,e,n){return this.r=t.r+(e.r-t.r)*n,this.g=t.g+(e.g-t.g)*n,this.b=t.b+(e.b-t.b)*n,this}lerpHSL(t,e){this.getHSL(pi),t.getHSL(Lr);const n=Qs(pi.h,Lr.h,e),s=Qs(pi.s,Lr.s,e),r=Qs(pi.l,Lr.l,e);return this.setHSL(n,s,r),this}setFromVector3(t){return this.r=t.x,this.g=t.y,this.b=t.z,this}applyMatrix3(t){const e=this.r,n=this.g,s=this.b,r=t.elements;return this.r=r[0]*e+r[3]*n+r[6]*s,this.g=r[1]*e+r[4]*n+r[7]*s,this.b=r[2]*e+r[5]*n+r[8]*s,this}equals(t){return t.r===this.r&&t.g===this.g&&t.b===this.b}fromArray(t,e=0){return this.r=t[e],this.g=t[e+1],this.b=t[e+2],this}toArray(t=[],e=0){return t[e]=this.r,t[e+1]=this.g,t[e+2]=this.b,t}fromBufferAttribute(t,e){return this.r=t.getX(e),this.g=t.getY(e),this.b=t.getZ(e),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}}const Ke=new $t;$t.NAMES=Xh;let lf=0;class In extends qi{static get type(){return"Material"}get type(){return this.constructor.type}set type(t){}constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:lf++}),this.uuid=Ci(),this.name="",this.blending=ms,this.side=ii,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=To,this.blendDst=wo,this.blendEquation=zi,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new $t(0,0,0),this.blendAlpha=0,this.depthFunc=ys,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=Zl,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=Ji,this.stencilZFail=Ji,this.stencilZPass=Ji,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(t){this._alphaTest>0!=t>0&&this.version++,this._alphaTest=t}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(t){if(t!==void 0)for(const e in t){const n=t[e];if(n===void 0){console.warn(`THREE.Material: parameter '${e}' has value of undefined.`);continue}const s=this[e];if(s===void 0){console.warn(`THREE.Material: '${e}' is not a property of THREE.${this.type}.`);continue}s&&s.isColor?s.set(n):s&&s.isVector3&&n&&n.isVector3?s.copy(n):this[e]=n}}toJSON(t){const e=t===void 0||typeof t=="string";e&&(t={textures:{},images:{}});const n={metadata:{version:4.6,type:"Material",generator:"Material.toJSON"}};n.uuid=this.uuid,n.type=this.type,this.name!==""&&(n.name=this.name),this.color&&this.color.isColor&&(n.color=this.color.getHex()),this.roughness!==void 0&&(n.roughness=this.roughness),this.metalness!==void 0&&(n.metalness=this.metalness),this.sheen!==void 0&&(n.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(n.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(n.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(n.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(n.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(n.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(n.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(n.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(n.shininess=this.shininess),this.clearcoat!==void 0&&(n.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(n.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(n.clearcoatMap=this.clearcoatMap.toJSON(t).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(n.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(t).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(n.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(t).uuid,n.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.dispersion!==void 0&&(n.dispersion=this.dispersion),this.iridescence!==void 0&&(n.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(n.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(n.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(n.iridescenceMap=this.iridescenceMap.toJSON(t).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(n.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(t).uuid),this.anisotropy!==void 0&&(n.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(n.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(n.anisotropyMap=this.anisotropyMap.toJSON(t).uuid),this.map&&this.map.isTexture&&(n.map=this.map.toJSON(t).uuid),this.matcap&&this.matcap.isTexture&&(n.matcap=this.matcap.toJSON(t).uuid),this.alphaMap&&this.alphaMap.isTexture&&(n.alphaMap=this.alphaMap.toJSON(t).uuid),this.lightMap&&this.lightMap.isTexture&&(n.lightMap=this.lightMap.toJSON(t).uuid,n.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(n.aoMap=this.aoMap.toJSON(t).uuid,n.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(n.bumpMap=this.bumpMap.toJSON(t).uuid,n.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(n.normalMap=this.normalMap.toJSON(t).uuid,n.normalMapType=this.normalMapType,n.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(n.displacementMap=this.displacementMap.toJSON(t).uuid,n.displacementScale=this.displacementScale,n.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(n.roughnessMap=this.roughnessMap.toJSON(t).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(n.metalnessMap=this.metalnessMap.toJSON(t).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(n.emissiveMap=this.emissiveMap.toJSON(t).uuid),this.specularMap&&this.specularMap.isTexture&&(n.specularMap=this.specularMap.toJSON(t).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(n.specularIntensityMap=this.specularIntensityMap.toJSON(t).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(n.specularColorMap=this.specularColorMap.toJSON(t).uuid),this.envMap&&this.envMap.isTexture&&(n.envMap=this.envMap.toJSON(t).uuid,this.combine!==void 0&&(n.combine=this.combine)),this.envMapRotation!==void 0&&(n.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(n.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(n.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(n.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(n.gradientMap=this.gradientMap.toJSON(t).uuid),this.transmission!==void 0&&(n.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(n.transmissionMap=this.transmissionMap.toJSON(t).uuid),this.thickness!==void 0&&(n.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(n.thicknessMap=this.thicknessMap.toJSON(t).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(n.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(n.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(n.size=this.size),this.shadowSide!==null&&(n.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(n.sizeAttenuation=this.sizeAttenuation),this.blending!==ms&&(n.blending=this.blending),this.side!==ii&&(n.side=this.side),this.vertexColors===!0&&(n.vertexColors=!0),this.opacity<1&&(n.opacity=this.opacity),this.transparent===!0&&(n.transparent=!0),this.blendSrc!==To&&(n.blendSrc=this.blendSrc),this.blendDst!==wo&&(n.blendDst=this.blendDst),this.blendEquation!==zi&&(n.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(n.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(n.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(n.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(n.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(n.blendAlpha=this.blendAlpha),this.depthFunc!==ys&&(n.depthFunc=this.depthFunc),this.depthTest===!1&&(n.depthTest=this.depthTest),this.depthWrite===!1&&(n.depthWrite=this.depthWrite),this.colorWrite===!1&&(n.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(n.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==Zl&&(n.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(n.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(n.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==Ji&&(n.stencilFail=this.stencilFail),this.stencilZFail!==Ji&&(n.stencilZFail=this.stencilZFail),this.stencilZPass!==Ji&&(n.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(n.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(n.rotation=this.rotation),this.polygonOffset===!0&&(n.polygonOffset=!0),this.polygonOffsetFactor!==0&&(n.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(n.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(n.linewidth=this.linewidth),this.dashSize!==void 0&&(n.dashSize=this.dashSize),this.gapSize!==void 0&&(n.gapSize=this.gapSize),this.scale!==void 0&&(n.scale=this.scale),this.dithering===!0&&(n.dithering=!0),this.alphaTest>0&&(n.alphaTest=this.alphaTest),this.alphaHash===!0&&(n.alphaHash=!0),this.alphaToCoverage===!0&&(n.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(n.premultipliedAlpha=!0),this.forceSinglePass===!0&&(n.forceSinglePass=!0),this.wireframe===!0&&(n.wireframe=!0),this.wireframeLinewidth>1&&(n.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(n.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(n.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(n.flatShading=!0),this.visible===!1&&(n.visible=!1),this.toneMapped===!1&&(n.toneMapped=!1),this.fog===!1&&(n.fog=!1),Object.keys(this.userData).length>0&&(n.userData=this.userData);function s(r){const a=[];for(const o in r){const c=r[o];delete c.metadata,a.push(c)}return a}if(e){const r=s(t.textures),a=s(t.images);r.length>0&&(n.textures=r),a.length>0&&(n.images=a)}return n}clone(){return new this.constructor().copy(this)}copy(t){this.name=t.name,this.blending=t.blending,this.side=t.side,this.vertexColors=t.vertexColors,this.opacity=t.opacity,this.transparent=t.transparent,this.blendSrc=t.blendSrc,this.blendDst=t.blendDst,this.blendEquation=t.blendEquation,this.blendSrcAlpha=t.blendSrcAlpha,this.blendDstAlpha=t.blendDstAlpha,this.blendEquationAlpha=t.blendEquationAlpha,this.blendColor.copy(t.blendColor),this.blendAlpha=t.blendAlpha,this.depthFunc=t.depthFunc,this.depthTest=t.depthTest,this.depthWrite=t.depthWrite,this.stencilWriteMask=t.stencilWriteMask,this.stencilFunc=t.stencilFunc,this.stencilRef=t.stencilRef,this.stencilFuncMask=t.stencilFuncMask,this.stencilFail=t.stencilFail,this.stencilZFail=t.stencilZFail,this.stencilZPass=t.stencilZPass,this.stencilWrite=t.stencilWrite;const e=t.clippingPlanes;let n=null;if(e!==null){const s=e.length;n=new Array(s);for(let r=0;r!==s;++r)n[r]=e[r].clone()}return this.clippingPlanes=n,this.clipIntersection=t.clipIntersection,this.clipShadows=t.clipShadows,this.shadowSide=t.shadowSide,this.colorWrite=t.colorWrite,this.precision=t.precision,this.polygonOffset=t.polygonOffset,this.polygonOffsetFactor=t.polygonOffsetFactor,this.polygonOffsetUnits=t.polygonOffsetUnits,this.dithering=t.dithering,this.alphaTest=t.alphaTest,this.alphaHash=t.alphaHash,this.alphaToCoverage=t.alphaToCoverage,this.premultipliedAlpha=t.premultipliedAlpha,this.forceSinglePass=t.forceSinglePass,this.visible=t.visible,this.toneMapped=t.toneMapped,this.userData=JSON.parse(JSON.stringify(t.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(t){t===!0&&this.version++}onBuild(){console.warn("Material: onBuild() has been removed.")}}class Un extends In{static get type(){return"MeshBasicMaterial"}constructor(t){super(),this.isMeshBasicMaterial=!0,this.color=new $t(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new cn,this.combine=Ma,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.map=t.map,this.lightMap=t.lightMap,this.lightMapIntensity=t.lightMapIntensity,this.aoMap=t.aoMap,this.aoMapIntensity=t.aoMapIntensity,this.specularMap=t.specularMap,this.alphaMap=t.alphaMap,this.envMap=t.envMap,this.envMapRotation.copy(t.envMapRotation),this.combine=t.combine,this.reflectivity=t.reflectivity,this.refractionRatio=t.refractionRatio,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.wireframeLinecap=t.wireframeLinecap,this.wireframeLinejoin=t.wireframeLinejoin,this.fog=t.fog,this}}const Oe=new F,Nr=new Lt;class gn{constructor(t,e,n=!1){if(Array.isArray(t))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,this.name="",this.array=t,this.itemSize=e,this.count=t!==void 0?t.length/e:0,this.normalized=n,this.usage=Ql,this.updateRanges=[],this.gpuType=Bn,this.version=0}onUploadCallback(){}set needsUpdate(t){t===!0&&this.version++}setUsage(t){return this.usage=t,this}addUpdateRange(t,e){this.updateRanges.push({start:t,count:e})}clearUpdateRanges(){this.updateRanges.length=0}copy(t){return this.name=t.name,this.array=new t.array.constructor(t.array),this.itemSize=t.itemSize,this.count=t.count,this.normalized=t.normalized,this.usage=t.usage,this.gpuType=t.gpuType,this}copyAt(t,e,n){t*=this.itemSize,n*=e.itemSize;for(let s=0,r=this.itemSize;s<r;s++)this.array[t+s]=e.array[n+s];return this}copyArray(t){return this.array.set(t),this}applyMatrix3(t){if(this.itemSize===2)for(let e=0,n=this.count;e<n;e++)Nr.fromBufferAttribute(this,e),Nr.applyMatrix3(t),this.setXY(e,Nr.x,Nr.y);else if(this.itemSize===3)for(let e=0,n=this.count;e<n;e++)Oe.fromBufferAttribute(this,e),Oe.applyMatrix3(t),this.setXYZ(e,Oe.x,Oe.y,Oe.z);return this}applyMatrix4(t){for(let e=0,n=this.count;e<n;e++)Oe.fromBufferAttribute(this,e),Oe.applyMatrix4(t),this.setXYZ(e,Oe.x,Oe.y,Oe.z);return this}applyNormalMatrix(t){for(let e=0,n=this.count;e<n;e++)Oe.fromBufferAttribute(this,e),Oe.applyNormalMatrix(t),this.setXYZ(e,Oe.x,Oe.y,Oe.z);return this}transformDirection(t){for(let e=0,n=this.count;e<n;e++)Oe.fromBufferAttribute(this,e),Oe.transformDirection(t),this.setXYZ(e,Oe.x,Oe.y,Oe.z);return this}set(t,e=0){return this.array.set(t,e),this}getComponent(t,e){let n=this.array[t*this.itemSize+e];return this.normalized&&(n=us(n,this.array)),n}setComponent(t,e,n){return this.normalized&&(n=tn(n,this.array)),this.array[t*this.itemSize+e]=n,this}getX(t){let e=this.array[t*this.itemSize];return this.normalized&&(e=us(e,this.array)),e}setX(t,e){return this.normalized&&(e=tn(e,this.array)),this.array[t*this.itemSize]=e,this}getY(t){let e=this.array[t*this.itemSize+1];return this.normalized&&(e=us(e,this.array)),e}setY(t,e){return this.normalized&&(e=tn(e,this.array)),this.array[t*this.itemSize+1]=e,this}getZ(t){let e=this.array[t*this.itemSize+2];return this.normalized&&(e=us(e,this.array)),e}setZ(t,e){return this.normalized&&(e=tn(e,this.array)),this.array[t*this.itemSize+2]=e,this}getW(t){let e=this.array[t*this.itemSize+3];return this.normalized&&(e=us(e,this.array)),e}setW(t,e){return this.normalized&&(e=tn(e,this.array)),this.array[t*this.itemSize+3]=e,this}setXY(t,e,n){return t*=this.itemSize,this.normalized&&(e=tn(e,this.array),n=tn(n,this.array)),this.array[t+0]=e,this.array[t+1]=n,this}setXYZ(t,e,n,s){return t*=this.itemSize,this.normalized&&(e=tn(e,this.array),n=tn(n,this.array),s=tn(s,this.array)),this.array[t+0]=e,this.array[t+1]=n,this.array[t+2]=s,this}setXYZW(t,e,n,s,r){return t*=this.itemSize,this.normalized&&(e=tn(e,this.array),n=tn(n,this.array),s=tn(s,this.array),r=tn(r,this.array)),this.array[t+0]=e,this.array[t+1]=n,this.array[t+2]=s,this.array[t+3]=r,this}onUpload(t){return this.onUploadCallback=t,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){const t={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(t.name=this.name),this.usage!==Ql&&(t.usage=this.usage),t}}class qh extends gn{constructor(t,e,n){super(new Uint16Array(t),e,n)}}class jh extends gn{constructor(t,e,n){super(new Uint32Array(t),e,n)}}class he extends gn{constructor(t,e,n){super(new Float32Array(t),e,n)}}let cf=0;const yn=new Kt,to=new Le,os=new F,dn=new Fn,Xs=new Fn,We=new F;class Ie extends qi{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:cf++}),this.uuid=Ci(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={}}getIndex(){return this.index}setIndex(t){return Array.isArray(t)?this.index=new(Vh(t)?jh:qh)(t,1):this.index=t,this}setIndirect(t){return this.indirect=t,this}getIndirect(){return this.indirect}getAttribute(t){return this.attributes[t]}setAttribute(t,e){return this.attributes[t]=e,this}deleteAttribute(t){return delete this.attributes[t],this}hasAttribute(t){return this.attributes[t]!==void 0}addGroup(t,e,n=0){this.groups.push({start:t,count:e,materialIndex:n})}clearGroups(){this.groups=[]}setDrawRange(t,e){this.drawRange.start=t,this.drawRange.count=e}applyMatrix4(t){const e=this.attributes.position;e!==void 0&&(e.applyMatrix4(t),e.needsUpdate=!0);const n=this.attributes.normal;if(n!==void 0){const r=new ie().getNormalMatrix(t);n.applyNormalMatrix(r),n.needsUpdate=!0}const s=this.attributes.tangent;return s!==void 0&&(s.transformDirection(t),s.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}applyQuaternion(t){return yn.makeRotationFromQuaternion(t),this.applyMatrix4(yn),this}rotateX(t){return yn.makeRotationX(t),this.applyMatrix4(yn),this}rotateY(t){return yn.makeRotationY(t),this.applyMatrix4(yn),this}rotateZ(t){return yn.makeRotationZ(t),this.applyMatrix4(yn),this}translate(t,e,n){return yn.makeTranslation(t,e,n),this.applyMatrix4(yn),this}scale(t,e,n){return yn.makeScale(t,e,n),this.applyMatrix4(yn),this}lookAt(t){return to.lookAt(t),to.updateMatrix(),this.applyMatrix4(to.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(os).negate(),this.translate(os.x,os.y,os.z),this}setFromPoints(t){const e=this.getAttribute("position");if(e===void 0){const n=[];for(let s=0,r=t.length;s<r;s++){const a=t[s];n.push(a.x,a.y,a.z||0)}this.setAttribute("position",new he(n,3))}else{for(let n=0,s=e.count;n<s;n++){const r=t[n];e.setXYZ(n,r.x,r.y,r.z||0)}t.length>e.count&&console.warn("THREE.BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),e.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new Fn);const t=this.attributes.position,e=this.morphAttributes.position;if(t&&t.isGLBufferAttribute){console.error("THREE.BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new F(-1/0,-1/0,-1/0),new F(1/0,1/0,1/0));return}if(t!==void 0){if(this.boundingBox.setFromBufferAttribute(t),e)for(let n=0,s=e.length;n<s;n++){const r=e[n];dn.setFromBufferAttribute(r),this.morphTargetsRelative?(We.addVectors(this.boundingBox.min,dn.min),this.boundingBox.expandByPoint(We),We.addVectors(this.boundingBox.max,dn.max),this.boundingBox.expandByPoint(We)):(this.boundingBox.expandByPoint(dn.min),this.boundingBox.expandByPoint(dn.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&console.error('THREE.BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new oi);const t=this.attributes.position,e=this.morphAttributes.position;if(t&&t.isGLBufferAttribute){console.error("THREE.BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new F,1/0);return}if(t){const n=this.boundingSphere.center;if(dn.setFromBufferAttribute(t),e)for(let r=0,a=e.length;r<a;r++){const o=e[r];Xs.setFromBufferAttribute(o),this.morphTargetsRelative?(We.addVectors(dn.min,Xs.min),dn.expandByPoint(We),We.addVectors(dn.max,Xs.max),dn.expandByPoint(We)):(dn.expandByPoint(Xs.min),dn.expandByPoint(Xs.max))}dn.getCenter(n);let s=0;for(let r=0,a=t.count;r<a;r++)We.fromBufferAttribute(t,r),s=Math.max(s,n.distanceToSquared(We));if(e)for(let r=0,a=e.length;r<a;r++){const o=e[r],c=this.morphTargetsRelative;for(let l=0,h=o.count;l<h;l++)We.fromBufferAttribute(o,l),c&&(os.fromBufferAttribute(t,l),We.add(os)),s=Math.max(s,n.distanceToSquared(We))}this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&console.error('THREE.BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){const t=this.index,e=this.attributes;if(t===null||e.position===void 0||e.normal===void 0||e.uv===void 0){console.error("THREE.BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}const n=e.position,s=e.normal,r=e.uv;this.hasAttribute("tangent")===!1&&this.setAttribute("tangent",new gn(new Float32Array(4*n.count),4));const a=this.getAttribute("tangent"),o=[],c=[];for(let I=0;I<n.count;I++)o[I]=new F,c[I]=new F;const l=new F,h=new F,u=new F,d=new Lt,p=new Lt,g=new Lt,_=new F,m=new F;function f(I,M,S){l.fromBufferAttribute(n,I),h.fromBufferAttribute(n,M),u.fromBufferAttribute(n,S),d.fromBufferAttribute(r,I),p.fromBufferAttribute(r,M),g.fromBufferAttribute(r,S),h.sub(l),u.sub(l),p.sub(d),g.sub(d);const U=1/(p.x*g.y-g.x*p.y);isFinite(U)&&(_.copy(h).multiplyScalar(g.y).addScaledVector(u,-p.y).multiplyScalar(U),m.copy(u).multiplyScalar(p.x).addScaledVector(h,-g.x).multiplyScalar(U),o[I].add(_),o[M].add(_),o[S].add(_),c[I].add(m),c[M].add(m),c[S].add(m))}let R=this.groups;R.length===0&&(R=[{start:0,count:t.count}]);for(let I=0,M=R.length;I<M;++I){const S=R[I],U=S.start,k=S.count;for(let H=U,J=U+k;H<J;H+=3)f(t.getX(H+0),t.getX(H+1),t.getX(H+2))}const C=new F,x=new F,N=new F,P=new F;function L(I){N.fromBufferAttribute(s,I),P.copy(N);const M=o[I];C.copy(M),C.sub(N.multiplyScalar(N.dot(M))).normalize(),x.crossVectors(P,M);const U=x.dot(c[I])<0?-1:1;a.setXYZW(I,C.x,C.y,C.z,U)}for(let I=0,M=R.length;I<M;++I){const S=R[I],U=S.start,k=S.count;for(let H=U,J=U+k;H<J;H+=3)L(t.getX(H+0)),L(t.getX(H+1)),L(t.getX(H+2))}}computeVertexNormals(){const t=this.index,e=this.getAttribute("position");if(e!==void 0){let n=this.getAttribute("normal");if(n===void 0)n=new gn(new Float32Array(e.count*3),3),this.setAttribute("normal",n);else for(let d=0,p=n.count;d<p;d++)n.setXYZ(d,0,0,0);const s=new F,r=new F,a=new F,o=new F,c=new F,l=new F,h=new F,u=new F;if(t)for(let d=0,p=t.count;d<p;d+=3){const g=t.getX(d+0),_=t.getX(d+1),m=t.getX(d+2);s.fromBufferAttribute(e,g),r.fromBufferAttribute(e,_),a.fromBufferAttribute(e,m),h.subVectors(a,r),u.subVectors(s,r),h.cross(u),o.fromBufferAttribute(n,g),c.fromBufferAttribute(n,_),l.fromBufferAttribute(n,m),o.add(h),c.add(h),l.add(h),n.setXYZ(g,o.x,o.y,o.z),n.setXYZ(_,c.x,c.y,c.z),n.setXYZ(m,l.x,l.y,l.z)}else for(let d=0,p=e.count;d<p;d+=3)s.fromBufferAttribute(e,d+0),r.fromBufferAttribute(e,d+1),a.fromBufferAttribute(e,d+2),h.subVectors(a,r),u.subVectors(s,r),h.cross(u),n.setXYZ(d+0,h.x,h.y,h.z),n.setXYZ(d+1,h.x,h.y,h.z),n.setXYZ(d+2,h.x,h.y,h.z);this.normalizeNormals(),n.needsUpdate=!0}}normalizeNormals(){const t=this.attributes.normal;for(let e=0,n=t.count;e<n;e++)We.fromBufferAttribute(t,e),We.normalize(),t.setXYZ(e,We.x,We.y,We.z)}toNonIndexed(){function t(o,c){const l=o.array,h=o.itemSize,u=o.normalized,d=new l.constructor(c.length*h);let p=0,g=0;for(let _=0,m=c.length;_<m;_++){o.isInterleavedBufferAttribute?p=c[_]*o.data.stride+o.offset:p=c[_]*h;for(let f=0;f<h;f++)d[g++]=l[p++]}return new gn(d,h,u)}if(this.index===null)return console.warn("THREE.BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;const e=new Ie,n=this.index.array,s=this.attributes;for(const o in s){const c=s[o],l=t(c,n);e.setAttribute(o,l)}const r=this.morphAttributes;for(const o in r){const c=[],l=r[o];for(let h=0,u=l.length;h<u;h++){const d=l[h],p=t(d,n);c.push(p)}e.morphAttributes[o]=c}e.morphTargetsRelative=this.morphTargetsRelative;const a=this.groups;for(let o=0,c=a.length;o<c;o++){const l=a[o];e.addGroup(l.start,l.count,l.materialIndex)}return e}toJSON(){const t={metadata:{version:4.6,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(t.uuid=this.uuid,t.type=this.type,this.name!==""&&(t.name=this.name),Object.keys(this.userData).length>0&&(t.userData=this.userData),this.parameters!==void 0){const c=this.parameters;for(const l in c)c[l]!==void 0&&(t[l]=c[l]);return t}t.data={attributes:{}};const e=this.index;e!==null&&(t.data.index={type:e.array.constructor.name,array:Array.prototype.slice.call(e.array)});const n=this.attributes;for(const c in n){const l=n[c];t.data.attributes[c]=l.toJSON(t.data)}const s={};let r=!1;for(const c in this.morphAttributes){const l=this.morphAttributes[c],h=[];for(let u=0,d=l.length;u<d;u++){const p=l[u];h.push(p.toJSON(t.data))}h.length>0&&(s[c]=h,r=!0)}r&&(t.data.morphAttributes=s,t.data.morphTargetsRelative=this.morphTargetsRelative);const a=this.groups;a.length>0&&(t.data.groups=JSON.parse(JSON.stringify(a)));const o=this.boundingSphere;return o!==null&&(t.data.boundingSphere={center:o.center.toArray(),radius:o.radius}),t}clone(){return new this.constructor().copy(this)}copy(t){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;const e={};this.name=t.name;const n=t.index;n!==null&&this.setIndex(n.clone(e));const s=t.attributes;for(const l in s){const h=s[l];this.setAttribute(l,h.clone(e))}const r=t.morphAttributes;for(const l in r){const h=[],u=r[l];for(let d=0,p=u.length;d<p;d++)h.push(u[d].clone(e));this.morphAttributes[l]=h}this.morphTargetsRelative=t.morphTargetsRelative;const a=t.groups;for(let l=0,h=a.length;l<h;l++){const u=a[l];this.addGroup(u.start,u.count,u.materialIndex)}const o=t.boundingBox;o!==null&&(this.boundingBox=o.clone());const c=t.boundingSphere;return c!==null&&(this.boundingSphere=c.clone()),this.drawRange.start=t.drawRange.start,this.drawRange.count=t.drawRange.count,this.userData=t.userData,this}dispose(){this.dispatchEvent({type:"dispose"})}}const _c=new Kt,Ii=new ur,Dr=new oi,vc=new F,Ir=new F,Ur=new F,Fr=new F,eo=new F,Or=new F,xc=new F,kr=new F;class Ae extends Le{constructor(t=new Ie,e=new Un){super(),this.isMesh=!0,this.type="Mesh",this.geometry=t,this.material=e,this.updateMorphTargets()}copy(t,e){return super.copy(t,e),t.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=t.morphTargetInfluences.slice()),t.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},t.morphTargetDictionary)),this.material=Array.isArray(t.material)?t.material.slice():t.material,this.geometry=t.geometry,this}updateMorphTargets(){const e=this.geometry.morphAttributes,n=Object.keys(e);if(n.length>0){const s=e[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,a=s.length;r<a;r++){const o=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=r}}}}getVertexPosition(t,e){const n=this.geometry,s=n.attributes.position,r=n.morphAttributes.position,a=n.morphTargetsRelative;e.fromBufferAttribute(s,t);const o=this.morphTargetInfluences;if(r&&o){Or.set(0,0,0);for(let c=0,l=r.length;c<l;c++){const h=o[c],u=r[c];h!==0&&(eo.fromBufferAttribute(u,t),a?Or.addScaledVector(eo,h):Or.addScaledVector(eo.sub(e),h))}e.add(Or)}return e}raycast(t,e){const n=this.geometry,s=this.material,r=this.matrixWorld;s!==void 0&&(n.boundingSphere===null&&n.computeBoundingSphere(),Dr.copy(n.boundingSphere),Dr.applyMatrix4(r),Ii.copy(t.ray).recast(t.near),!(Dr.containsPoint(Ii.origin)===!1&&(Ii.intersectSphere(Dr,vc)===null||Ii.origin.distanceToSquared(vc)>(t.far-t.near)**2))&&(_c.copy(r).invert(),Ii.copy(t.ray).applyMatrix4(_c),!(n.boundingBox!==null&&Ii.intersectsBox(n.boundingBox)===!1)&&this._computeIntersections(t,e,Ii)))}_computeIntersections(t,e,n){let s;const r=this.geometry,a=this.material,o=r.index,c=r.attributes.position,l=r.attributes.uv,h=r.attributes.uv1,u=r.attributes.normal,d=r.groups,p=r.drawRange;if(o!==null)if(Array.isArray(a))for(let g=0,_=d.length;g<_;g++){const m=d[g],f=a[m.materialIndex],R=Math.max(m.start,p.start),C=Math.min(o.count,Math.min(m.start+m.count,p.start+p.count));for(let x=R,N=C;x<N;x+=3){const P=o.getX(x),L=o.getX(x+1),I=o.getX(x+2);s=Br(this,f,t,n,l,h,u,P,L,I),s&&(s.faceIndex=Math.floor(x/3),s.face.materialIndex=m.materialIndex,e.push(s))}}else{const g=Math.max(0,p.start),_=Math.min(o.count,p.start+p.count);for(let m=g,f=_;m<f;m+=3){const R=o.getX(m),C=o.getX(m+1),x=o.getX(m+2);s=Br(this,a,t,n,l,h,u,R,C,x),s&&(s.faceIndex=Math.floor(m/3),e.push(s))}}else if(c!==void 0)if(Array.isArray(a))for(let g=0,_=d.length;g<_;g++){const m=d[g],f=a[m.materialIndex],R=Math.max(m.start,p.start),C=Math.min(c.count,Math.min(m.start+m.count,p.start+p.count));for(let x=R,N=C;x<N;x+=3){const P=x,L=x+1,I=x+2;s=Br(this,f,t,n,l,h,u,P,L,I),s&&(s.faceIndex=Math.floor(x/3),s.face.materialIndex=m.materialIndex,e.push(s))}}else{const g=Math.max(0,p.start),_=Math.min(c.count,p.start+p.count);for(let m=g,f=_;m<f;m+=3){const R=m,C=m+1,x=m+2;s=Br(this,a,t,n,l,h,u,R,C,x),s&&(s.faceIndex=Math.floor(m/3),e.push(s))}}}}function hf(i,t,e,n,s,r,a,o){let c;if(t.side===ln?c=n.intersectTriangle(a,r,s,!0,o):c=n.intersectTriangle(s,r,a,t.side===ii,o),c===null)return null;kr.copy(o),kr.applyMatrix4(i.matrixWorld);const l=e.ray.origin.distanceTo(kr);return l<e.near||l>e.far?null:{distance:l,point:kr.clone(),object:i}}function Br(i,t,e,n,s,r,a,o,c,l){i.getVertexPosition(o,Ir),i.getVertexPosition(c,Ur),i.getVertexPosition(l,Fr);const h=hf(i,t,e,n,Ir,Ur,Fr,xc);if(h){const u=new F;Ln.getBarycoord(xc,Ir,Ur,Fr,u),s&&(h.uv=Ln.getInterpolatedAttribute(s,o,c,l,u,new Lt)),r&&(h.uv1=Ln.getInterpolatedAttribute(r,o,c,l,u,new Lt)),a&&(h.normal=Ln.getInterpolatedAttribute(a,o,c,l,u,new F),h.normal.dot(n.direction)>0&&h.normal.multiplyScalar(-1));const d={a:o,b:c,c:l,normal:new F,materialIndex:0};Ln.getNormal(Ir,Ur,Fr,d.normal),h.face=d,h.barycoord=u}return h}class Is extends Ie{constructor(t=1,e=1,n=1,s=1,r=1,a=1){super(),this.type="BoxGeometry",this.parameters={width:t,height:e,depth:n,widthSegments:s,heightSegments:r,depthSegments:a};const o=this;s=Math.floor(s),r=Math.floor(r),a=Math.floor(a);const c=[],l=[],h=[],u=[];let d=0,p=0;g("z","y","x",-1,-1,n,e,t,a,r,0),g("z","y","x",1,-1,n,e,-t,a,r,1),g("x","z","y",1,1,t,n,e,s,a,2),g("x","z","y",1,-1,t,n,-e,s,a,3),g("x","y","z",1,-1,t,e,n,s,r,4),g("x","y","z",-1,-1,t,e,-n,s,r,5),this.setIndex(c),this.setAttribute("position",new he(l,3)),this.setAttribute("normal",new he(h,3)),this.setAttribute("uv",new he(u,2));function g(_,m,f,R,C,x,N,P,L,I,M){const S=x/L,U=N/I,k=x/2,H=N/2,J=P/2,Q=L+1,O=I+1;let tt=0,W=0;const ot=new F;for(let ft=0;ft<O;ft++){const pt=ft*U-H;for(let Ct=0;Ct<Q;Ct++){const Et=Ct*S-k;ot[_]=Et*R,ot[m]=pt*C,ot[f]=J,l.push(ot.x,ot.y,ot.z),ot[_]=0,ot[m]=0,ot[f]=P>0?1:-1,h.push(ot.x,ot.y,ot.z),u.push(Ct/L),u.push(1-ft/I),tt+=1}}for(let ft=0;ft<I;ft++)for(let pt=0;pt<L;pt++){const Ct=d+pt+Q*ft,Et=d+pt+Q*(ft+1),X=d+(pt+1)+Q*(ft+1),K=d+(pt+1)+Q*ft;c.push(Ct,Et,K),c.push(Et,X,K),W+=6}o.addGroup(p,W,M),p+=W,d+=tt}}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new Is(t.width,t.height,t.depth,t.widthSegments,t.heightSegments,t.depthSegments)}}function ws(i){const t={};for(const e in i){t[e]={};for(const n in i[e]){const s=i[e][n];s&&(s.isColor||s.isMatrix3||s.isMatrix4||s.isVector2||s.isVector3||s.isVector4||s.isTexture||s.isQuaternion)?s.isRenderTargetTexture?(console.warn("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),t[e][n]=null):t[e][n]=s.clone():Array.isArray(s)?t[e][n]=s.slice():t[e][n]=s}}return t}function en(i){const t={};for(let e=0;e<i.length;e++){const n=ws(i[e]);for(const s in n)t[s]=n[s]}return t}function uf(i){const t=[];for(let e=0;e<i.length;e++)t.push(i[e].clone());return t}function Yh(i){const t=i.getRenderTarget();return t===null?i.outputColorSpace:t.isXRRenderTarget===!0?t.texture.colorSpace:le.workingColorSpace}const df={clone:ws,merge:en};var ff=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,pf=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;class Ti extends In{static get type(){return"ShaderMaterial"}constructor(t){super(),this.isShaderMaterial=!0,this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=ff,this.fragmentShader=pf,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,t!==void 0&&this.setValues(t)}copy(t){return super.copy(t),this.fragmentShader=t.fragmentShader,this.vertexShader=t.vertexShader,this.uniforms=ws(t.uniforms),this.uniformsGroups=uf(t.uniformsGroups),this.defines=Object.assign({},t.defines),this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.fog=t.fog,this.lights=t.lights,this.clipping=t.clipping,this.extensions=Object.assign({},t.extensions),this.glslVersion=t.glslVersion,this}toJSON(t){const e=super.toJSON(t);e.glslVersion=this.glslVersion,e.uniforms={};for(const s in this.uniforms){const a=this.uniforms[s].value;a&&a.isTexture?e.uniforms[s]={type:"t",value:a.toJSON(t).uuid}:a&&a.isColor?e.uniforms[s]={type:"c",value:a.getHex()}:a&&a.isVector2?e.uniforms[s]={type:"v2",value:a.toArray()}:a&&a.isVector3?e.uniforms[s]={type:"v3",value:a.toArray()}:a&&a.isVector4?e.uniforms[s]={type:"v4",value:a.toArray()}:a&&a.isMatrix3?e.uniforms[s]={type:"m3",value:a.toArray()}:a&&a.isMatrix4?e.uniforms[s]={type:"m4",value:a.toArray()}:e.uniforms[s]={value:a}}Object.keys(this.defines).length>0&&(e.defines=this.defines),e.vertexShader=this.vertexShader,e.fragmentShader=this.fragmentShader,e.lights=this.lights,e.clipping=this.clipping;const n={};for(const s in this.extensions)this.extensions[s]===!0&&(n[s]=!0);return Object.keys(n).length>0&&(e.extensions=n),e}}class $h extends Le{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new Kt,this.projectionMatrix=new Kt,this.projectionMatrixInverse=new Kt,this.coordinateSystem=ei}copy(t,e){return super.copy(t,e),this.matrixWorldInverse.copy(t.matrixWorldInverse),this.projectionMatrix.copy(t.projectionMatrix),this.projectionMatrixInverse.copy(t.projectionMatrixInverse),this.coordinateSystem=t.coordinateSystem,this}getWorldDirection(t){return super.getWorldDirection(t).negate()}updateMatrixWorld(t){super.updateMatrixWorld(t),this.matrixWorldInverse.copy(this.matrixWorld).invert()}updateWorldMatrix(t,e){super.updateWorldMatrix(t,e),this.matrixWorldInverse.copy(this.matrixWorld).invert()}clone(){return new this.constructor().copy(this)}}const mi=new F,yc=new Lt,Mc=new Lt;class Je extends $h{constructor(t=50,e=1,n=.1,s=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=t,this.zoom=1,this.near=n,this.far=s,this.focus=10,this.aspect=e,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(t,e){return super.copy(t,e),this.fov=t.fov,this.zoom=t.zoom,this.near=t.near,this.far=t.far,this.focus=t.focus,this.aspect=t.aspect,this.view=t.view===null?null:Object.assign({},t.view),this.filmGauge=t.filmGauge,this.filmOffset=t.filmOffset,this}setFocalLength(t){const e=.5*this.getFilmHeight()/t;this.fov=Ts*2*Math.atan(e),this.updateProjectionMatrix()}getFocalLength(){const t=Math.tan(Zs*.5*this.fov);return .5*this.getFilmHeight()/t}getEffectiveFOV(){return Ts*2*Math.atan(Math.tan(Zs*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(t,e,n){mi.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),e.set(mi.x,mi.y).multiplyScalar(-t/mi.z),mi.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),n.set(mi.x,mi.y).multiplyScalar(-t/mi.z)}getViewSize(t,e){return this.getViewBounds(t,yc,Mc),e.subVectors(Mc,yc)}setViewOffset(t,e,n,s,r,a){this.aspect=t/e,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=t,this.view.fullHeight=e,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const t=this.near;let e=t*Math.tan(Zs*.5*this.fov)/this.zoom,n=2*e,s=this.aspect*n,r=-.5*s;const a=this.view;if(this.view!==null&&this.view.enabled){const c=a.fullWidth,l=a.fullHeight;r+=a.offsetX*s/c,e-=a.offsetY*n/l,s*=a.width/c,n*=a.height/l}const o=this.filmOffset;o!==0&&(r+=t*o/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+s,e,e-n,t,this.far,this.coordinateSystem),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(t){const e=super.toJSON(t);return e.object.fov=this.fov,e.object.zoom=this.zoom,e.object.near=this.near,e.object.far=this.far,e.object.focus=this.focus,e.object.aspect=this.aspect,this.view!==null&&(e.object.view=Object.assign({},this.view)),e.object.filmGauge=this.filmGauge,e.object.filmOffset=this.filmOffset,e}}const ls=-90,cs=1;class mf extends Le{constructor(t,e,n){super(),this.type="CubeCamera",this.renderTarget=n,this.coordinateSystem=null,this.activeMipmapLevel=0;const s=new Je(ls,cs,t,e);s.layers=this.layers,this.add(s);const r=new Je(ls,cs,t,e);r.layers=this.layers,this.add(r);const a=new Je(ls,cs,t,e);a.layers=this.layers,this.add(a);const o=new Je(ls,cs,t,e);o.layers=this.layers,this.add(o);const c=new Je(ls,cs,t,e);c.layers=this.layers,this.add(c);const l=new Je(ls,cs,t,e);l.layers=this.layers,this.add(l)}updateCoordinateSystem(){const t=this.coordinateSystem,e=this.children.concat(),[n,s,r,a,o,c]=e;for(const l of e)this.remove(l);if(t===ei)n.up.set(0,1,0),n.lookAt(1,0,0),s.up.set(0,1,0),s.lookAt(-1,0,0),r.up.set(0,0,-1),r.lookAt(0,1,0),a.up.set(0,0,1),a.lookAt(0,-1,0),o.up.set(0,1,0),o.lookAt(0,0,1),c.up.set(0,1,0),c.lookAt(0,0,-1);else if(t===fa)n.up.set(0,-1,0),n.lookAt(-1,0,0),s.up.set(0,-1,0),s.lookAt(1,0,0),r.up.set(0,0,1),r.lookAt(0,1,0),a.up.set(0,0,-1),a.lookAt(0,-1,0),o.up.set(0,-1,0),o.lookAt(0,0,1),c.up.set(0,-1,0),c.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+t);for(const l of e)this.add(l),l.updateMatrixWorld()}update(t,e){this.parent===null&&this.updateMatrixWorld();const{renderTarget:n,activeMipmapLevel:s}=this;this.coordinateSystem!==t.coordinateSystem&&(this.coordinateSystem=t.coordinateSystem,this.updateCoordinateSystem());const[r,a,o,c,l,h]=this.children,u=t.getRenderTarget(),d=t.getActiveCubeFace(),p=t.getActiveMipmapLevel(),g=t.xr.enabled;t.xr.enabled=!1;const _=n.texture.generateMipmaps;n.texture.generateMipmaps=!1,t.setRenderTarget(n,0,s),t.render(e,r),t.setRenderTarget(n,1,s),t.render(e,a),t.setRenderTarget(n,2,s),t.render(e,o),t.setRenderTarget(n,3,s),t.render(e,c),t.setRenderTarget(n,4,s),t.render(e,l),n.texture.generateMipmaps=_,t.setRenderTarget(n,5,s),t.render(e,h),t.setRenderTarget(u,d,p),t.xr.enabled=g,n.texture.needsPMREMUpdate=!0}}class Kh extends Ye{constructor(t,e,n,s,r,a,o,c,l,h){t=t!==void 0?t:[],e=e!==void 0?e:Ms,super(t,e,n,s,r,a,o,c,l,h),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(t){this.image=t}}class gf extends Wi{constructor(t=1,e={}){super(t,t,e),this.isWebGLCubeRenderTarget=!0;const n={width:t,height:t,depth:1},s=[n,n,n,n,n,n];this.texture=new Kh(s,e.mapping,e.wrapS,e.wrapT,e.magFilter,e.minFilter,e.format,e.type,e.anisotropy,e.colorSpace),this.texture.isRenderTargetTexture=!0,this.texture.generateMipmaps=e.generateMipmaps!==void 0?e.generateMipmaps:!1,this.texture.minFilter=e.minFilter!==void 0?e.minFilter:fn}fromEquirectangularTexture(t,e){this.texture.type=e.type,this.texture.colorSpace=e.colorSpace,this.texture.generateMipmaps=e.generateMipmaps,this.texture.minFilter=e.minFilter,this.texture.magFilter=e.magFilter;const n={uniforms:{tEquirect:{value:null}},vertexShader:`

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,fragmentShader:`

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`},s=new Is(5,5,5),r=new Ti({name:"CubemapFromEquirect",uniforms:ws(n.uniforms),vertexShader:n.vertexShader,fragmentShader:n.fragmentShader,side:ln,blending:Si});r.uniforms.tEquirect.value=e;const a=new Ae(s,r),o=e.minFilter;return e.minFilter===ti&&(e.minFilter=fn),new mf(1,10,this).update(t,a),e.minFilter=o,a.geometry.dispose(),a.material.dispose(),this}clear(t,e,n,s){const r=t.getRenderTarget();for(let a=0;a<6;a++)t.setRenderTarget(this,a),t.clear(e,n,s);t.setRenderTarget(r)}}const no=new F,_f=new F,vf=new ie;class vi{constructor(t=new F(1,0,0),e=0){this.isPlane=!0,this.normal=t,this.constant=e}set(t,e){return this.normal.copy(t),this.constant=e,this}setComponents(t,e,n,s){return this.normal.set(t,e,n),this.constant=s,this}setFromNormalAndCoplanarPoint(t,e){return this.normal.copy(t),this.constant=-e.dot(this.normal),this}setFromCoplanarPoints(t,e,n){const s=no.subVectors(n,e).cross(_f.subVectors(t,e)).normalize();return this.setFromNormalAndCoplanarPoint(s,t),this}copy(t){return this.normal.copy(t.normal),this.constant=t.constant,this}normalize(){const t=1/this.normal.length();return this.normal.multiplyScalar(t),this.constant*=t,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(t){return this.normal.dot(t)+this.constant}distanceToSphere(t){return this.distanceToPoint(t.center)-t.radius}projectPoint(t,e){return e.copy(t).addScaledVector(this.normal,-this.distanceToPoint(t))}intersectLine(t,e){const n=t.delta(no),s=this.normal.dot(n);if(s===0)return this.distanceToPoint(t.start)===0?e.copy(t.start):null;const r=-(t.start.dot(this.normal)+this.constant)/s;return r<0||r>1?null:e.copy(t.start).addScaledVector(n,r)}intersectsLine(t){const e=this.distanceToPoint(t.start),n=this.distanceToPoint(t.end);return e<0&&n>0||n<0&&e>0}intersectsBox(t){return t.intersectsPlane(this)}intersectsSphere(t){return t.intersectsPlane(this)}coplanarPoint(t){return t.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(t,e){const n=e||vf.getNormalMatrix(t),s=this.coplanarPoint(no).applyMatrix4(t),r=this.normal.applyMatrix3(n).normalize();return this.constant=-s.dot(r),this}translate(t){return this.constant-=t.dot(this.normal),this}equals(t){return t.normal.equals(this.normal)&&t.constant===this.constant}clone(){return new this.constructor().copy(this)}}const Ui=new oi,zr=new F;class bl{constructor(t=new vi,e=new vi,n=new vi,s=new vi,r=new vi,a=new vi){this.planes=[t,e,n,s,r,a]}set(t,e,n,s,r,a){const o=this.planes;return o[0].copy(t),o[1].copy(e),o[2].copy(n),o[3].copy(s),o[4].copy(r),o[5].copy(a),this}copy(t){const e=this.planes;for(let n=0;n<6;n++)e[n].copy(t.planes[n]);return this}setFromProjectionMatrix(t,e=ei){const n=this.planes,s=t.elements,r=s[0],a=s[1],o=s[2],c=s[3],l=s[4],h=s[5],u=s[6],d=s[7],p=s[8],g=s[9],_=s[10],m=s[11],f=s[12],R=s[13],C=s[14],x=s[15];if(n[0].setComponents(c-r,d-l,m-p,x-f).normalize(),n[1].setComponents(c+r,d+l,m+p,x+f).normalize(),n[2].setComponents(c+a,d+h,m+g,x+R).normalize(),n[3].setComponents(c-a,d-h,m-g,x-R).normalize(),n[4].setComponents(c-o,d-u,m-_,x-C).normalize(),e===ei)n[5].setComponents(c+o,d+u,m+_,x+C).normalize();else if(e===fa)n[5].setComponents(o,u,_,C).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+e);return this}intersectsObject(t){if(t.boundingSphere!==void 0)t.boundingSphere===null&&t.computeBoundingSphere(),Ui.copy(t.boundingSphere).applyMatrix4(t.matrixWorld);else{const e=t.geometry;e.boundingSphere===null&&e.computeBoundingSphere(),Ui.copy(e.boundingSphere).applyMatrix4(t.matrixWorld)}return this.intersectsSphere(Ui)}intersectsSprite(t){return Ui.center.set(0,0,0),Ui.radius=.7071067811865476,Ui.applyMatrix4(t.matrixWorld),this.intersectsSphere(Ui)}intersectsSphere(t){const e=this.planes,n=t.center,s=-t.radius;for(let r=0;r<6;r++)if(e[r].distanceToPoint(n)<s)return!1;return!0}intersectsBox(t){const e=this.planes;for(let n=0;n<6;n++){const s=e[n];if(zr.x=s.normal.x>0?t.max.x:t.min.x,zr.y=s.normal.y>0?t.max.y:t.min.y,zr.z=s.normal.z>0?t.max.z:t.min.z,s.distanceToPoint(zr)<0)return!1}return!0}containsPoint(t){const e=this.planes;for(let n=0;n<6;n++)if(e[n].distanceToPoint(t)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}}function Jh(){let i=null,t=!1,e=null,n=null;function s(r,a){e(r,a),n=i.requestAnimationFrame(s)}return{start:function(){t!==!0&&e!==null&&(n=i.requestAnimationFrame(s),t=!0)},stop:function(){i.cancelAnimationFrame(n),t=!1},setAnimationLoop:function(r){e=r},setContext:function(r){i=r}}}function xf(i){const t=new WeakMap;function e(o,c){const l=o.array,h=o.usage,u=l.byteLength,d=i.createBuffer();i.bindBuffer(c,d),i.bufferData(c,l,h),o.onUploadCallback();let p;if(l instanceof Float32Array)p=i.FLOAT;else if(l instanceof Uint16Array)o.isFloat16BufferAttribute?p=i.HALF_FLOAT:p=i.UNSIGNED_SHORT;else if(l instanceof Int16Array)p=i.SHORT;else if(l instanceof Uint32Array)p=i.UNSIGNED_INT;else if(l instanceof Int32Array)p=i.INT;else if(l instanceof Int8Array)p=i.BYTE;else if(l instanceof Uint8Array)p=i.UNSIGNED_BYTE;else if(l instanceof Uint8ClampedArray)p=i.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+l);return{buffer:d,type:p,bytesPerElement:l.BYTES_PER_ELEMENT,version:o.version,size:u}}function n(o,c,l){const h=c.array,u=c.updateRanges;if(i.bindBuffer(l,o),u.length===0)i.bufferSubData(l,0,h);else{u.sort((p,g)=>p.start-g.start);let d=0;for(let p=1;p<u.length;p++){const g=u[d],_=u[p];_.start<=g.start+g.count+1?g.count=Math.max(g.count,_.start+_.count-g.start):(++d,u[d]=_)}u.length=d+1;for(let p=0,g=u.length;p<g;p++){const _=u[p];i.bufferSubData(l,_.start*h.BYTES_PER_ELEMENT,h,_.start,_.count)}c.clearUpdateRanges()}c.onUploadCallback()}function s(o){return o.isInterleavedBufferAttribute&&(o=o.data),t.get(o)}function r(o){o.isInterleavedBufferAttribute&&(o=o.data);const c=t.get(o);c&&(i.deleteBuffer(c.buffer),t.delete(o))}function a(o,c){if(o.isInterleavedBufferAttribute&&(o=o.data),o.isGLBufferAttribute){const h=t.get(o);(!h||h.version<o.version)&&t.set(o,{buffer:o.buffer,type:o.type,bytesPerElement:o.elementSize,version:o.version});return}const l=t.get(o);if(l===void 0)t.set(o,e(o,c));else if(l.version<o.version){if(l.size!==o.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");n(l.buffer,o,c),l.version=o.version}}return{get:s,remove:r,update:a}}class Ta extends Ie{constructor(t=1,e=1,n=1,s=1){super(),this.type="PlaneGeometry",this.parameters={width:t,height:e,widthSegments:n,heightSegments:s};const r=t/2,a=e/2,o=Math.floor(n),c=Math.floor(s),l=o+1,h=c+1,u=t/o,d=e/c,p=[],g=[],_=[],m=[];for(let f=0;f<h;f++){const R=f*d-a;for(let C=0;C<l;C++){const x=C*u-r;g.push(x,-R,0),_.push(0,0,1),m.push(C/o),m.push(1-f/c)}}for(let f=0;f<c;f++)for(let R=0;R<o;R++){const C=R+l*f,x=R+l*(f+1),N=R+1+l*(f+1),P=R+1+l*f;p.push(C,x,P),p.push(x,N,P)}this.setIndex(p),this.setAttribute("position",new he(g,3)),this.setAttribute("normal",new he(_,3)),this.setAttribute("uv",new he(m,2))}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new Ta(t.width,t.height,t.widthSegments,t.heightSegments)}}var yf=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,Mf=`#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`,Sf=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,bf=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,Ef=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,Tf=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,wf=`#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT ) 
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN ) 
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`,Af=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,Cf=`#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec3 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 ).rgb;
	}
#endif`,Rf=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,Pf=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,Lf=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,Nf=`float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`,Df=`#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`,If=`#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`,Uf=`#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`,Ff=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,Of=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,kf=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,Bf=`#if defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#elif defined( USE_COLOR )
	diffuseColor.rgb *= vColor;
#endif`,zf=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR )
	varying vec3 vColor;
#endif`,Vf=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec3 vColor;
#endif`,Hf=`#if defined( USE_COLOR_ALPHA )
	vColor = vec4( 1.0 );
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec3( 1.0 );
#endif
#ifdef USE_COLOR
	vColor *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.xyz *= instanceColor.xyz;
#endif
#ifdef USE_BATCHING_COLOR
	vec3 batchingColor = getBatchingColor( getIndirectIndex( gl_DrawID ) );
	vColor.xyz *= batchingColor.xyz;
#endif`,Gf=`#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}
mat3 transposeMat3( const in mat3 m ) {
	mat3 tmp;
	tmp[ 0 ] = vec3( m[ 0 ].x, m[ 1 ].x, m[ 2 ].x );
	tmp[ 1 ] = vec3( m[ 0 ].y, m[ 1 ].y, m[ 2 ].y );
	tmp[ 2 ] = vec3( m[ 0 ].z, m[ 1 ].z, m[ 2 ].z );
	return tmp;
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`,Wf=`#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`,Xf=`vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
	#ifdef FLIP_SIDED
		transformedTangent = - transformedTangent;
	#endif
#endif`,qf=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,jf=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,Yf=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,$f=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,Kf="gl_FragColor = linearToOutputTexel( gl_FragColor );",Jf=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,Zf=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );
	#else
		vec4 envColor = vec4( 0.0 );
	#endif
	#ifdef ENVMAP_BLENDING_MULTIPLY
		outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_MIX )
		outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_ADD )
		outgoingLight += envColor.xyz * specularStrength * reflectivity;
	#endif
#endif`,Qf=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
	
#endif`,tp=`#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`,ep=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,np=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`,ip=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,sp=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,rp=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,ap=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,op=`#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`,lp=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,cp=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,hp=`varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,up=`uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif`,dp=`#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, roughness * roughness) );
			reflectVec = inverseTransformDirection( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
#endif`,fp=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,pp=`varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,mp=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,gp=`varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,_p=`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb * ( 1.0 - metalnessFactor );
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = mix( min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.07, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`,vp=`struct PhysicalMaterial {
	vec3 diffuseColor;
	float roughness;
	vec3 specularColor;
	float specularF90;
	float dispersion;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		float v = 0.5 / ( gv + gl );
		return saturate(v);
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColor;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transposeMat3( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float a = roughness < 0.25 ? -339.2 * r2 + 161.4 * roughness - 25.9 : -8.48 * r2 + 14.3 * roughness - 9.95;
	float b = roughness < 0.25 ? 44.0 * r2 - 23.7 * roughness + 3.26 : 1.97 * r2 - 3.27 * roughness + 0.72;
	float DG = exp( a * dotNV + b ) + ( roughness < 0.25 ? 0.0 : 0.1 * ( roughness - 0.25 ) );
	return saturate( DG * RECIPROCAL_PI );
}
vec2 DFGApprox( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	const vec4 c0 = vec4( - 1, - 0.0275, - 0.572, 0.022 );
	const vec4 c1 = vec4( 1, 0.0425, 1.04, - 0.04 );
	vec4 r = roughness * c0 + c1;
	float a004 = min( r.x * r.x, exp2( - 9.28 * dotNV ) ) * r.x + r.y;
	vec2 fab = vec2( - 1.04, 1.04 ) * a004 + r.zw;
	return fab;
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX( directLight.direction, geometryViewDir, geometryNormal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
	#endif
	vec3 singleScattering = vec3( 0.0 );
	vec3 multiScattering = vec3( 0.0 );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnel, material.roughness, singleScattering, multiScattering );
	#else
		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScattering, multiScattering );
	#endif
	vec3 totalScattering = singleScattering + multiScattering;
	vec3 diffuse = material.diffuseColor * ( 1.0 - max( max( totalScattering.r, totalScattering.g ), totalScattering.b ) );
	reflectedLight.indirectSpecular += radiance * singleScattering;
	reflectedLight.indirectSpecular += multiScattering * cosineWeightedIrradiance;
	reflectedLight.indirectDiffuse += diffuse * cosineWeightedIrradiance;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`,xp=`
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnel = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,yp=`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD ) && defined( ENVMAP_TYPE_CUBE_UV )
		iblIrradiance += getIBLIrradiance( geometryNormal );
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,Mp=`#if defined( RE_IndirectDiffuse )
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,Sp=`#if defined( USE_LOGDEPTHBUF )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,bp=`#if defined( USE_LOGDEPTHBUF )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,Ep=`#ifdef USE_LOGDEPTHBUF
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,Tp=`#ifdef USE_LOGDEPTHBUF
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,wp=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,Ap=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,Cp=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`,Rp=`#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,Pp=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,Lp=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,Np=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,Dp=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,Ip=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,Up=`#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`,Fp=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,Op=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`,kp=`#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`,Bp=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,zp=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,Vp=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`,Hp=`#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`,Gp=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,Wp=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,Xp=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,qp=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,jp=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,Yp=`vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return depth * ( near - far ) - near;
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return ( near * far ) / ( ( far - near ) * depth - far );
}`,$p=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,Kp=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,Jp=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,Zp=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,Qp=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,tm=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,em=`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform sampler2D pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	float texture2DCompare( sampler2D depths, vec2 uv, float compare ) {
		return step( compare, unpackRGBAToDepth( texture2D( depths, uv ) ) );
	}
	vec2 texture2DDistribution( sampler2D shadow, vec2 uv ) {
		return unpackRGBATo2Half( texture2D( shadow, uv ) );
	}
	float VSMShadow (sampler2D shadow, vec2 uv, float compare ){
		float occlusion = 1.0;
		vec2 distribution = texture2DDistribution( shadow, uv );
		float hard_shadow = step( compare , distribution.x );
		if (hard_shadow != 1.0 ) {
			float distance = compare - distribution.x ;
			float variance = max( 0.00000, distribution.y * distribution.y );
			float softness_probability = variance / (variance + distance * distance );			softness_probability = clamp( ( softness_probability - 0.3 ) / ( 0.95 - 0.3 ), 0.0, 1.0 );			occlusion = clamp( max( hard_shadow, softness_probability ), 0.0, 1.0 );
		}
		return occlusion;
	}
	float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
		float shadow = 1.0;
		shadowCoord.xyz /= shadowCoord.w;
		shadowCoord.z += shadowBias;
		bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
		bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
		if ( frustumTest ) {
		#if defined( SHADOWMAP_TYPE_PCF )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx0 = - texelSize.x * shadowRadius;
			float dy0 = - texelSize.y * shadowRadius;
			float dx1 = + texelSize.x * shadowRadius;
			float dy1 = + texelSize.y * shadowRadius;
			float dx2 = dx0 / 2.0;
			float dy2 = dy0 / 2.0;
			float dx3 = dx1 / 2.0;
			float dy3 = dy1 / 2.0;
			shadow = (
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy1 ), shadowCoord.z )
			) * ( 1.0 / 17.0 );
		#elif defined( SHADOWMAP_TYPE_PCF_SOFT )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx = texelSize.x;
			float dy = texelSize.y;
			vec2 uv = shadowCoord.xy;
			vec2 f = fract( uv * shadowMapSize + 0.5 );
			uv -= f * texelSize;
			shadow = (
				texture2DCompare( shadowMap, uv, shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( dx, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( 0.0, dy ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + texelSize, shadowCoord.z ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, 0.0 ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 0.0 ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, dy ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( 0.0, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 0.0, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( texture2DCompare( shadowMap, uv + vec2( dx, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( dx, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( mix( texture2DCompare( shadowMap, uv + vec2( -dx, -dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, -dy ), shadowCoord.z ),
						  f.x ),
					 mix( texture2DCompare( shadowMap, uv + vec2( -dx, 2.0 * dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 2.0 * dy ), shadowCoord.z ),
						  f.x ),
					 f.y )
			) * ( 1.0 / 9.0 );
		#elif defined( SHADOWMAP_TYPE_VSM )
			shadow = VSMShadow( shadowMap, shadowCoord.xy, shadowCoord.z );
		#else
			shadow = texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z );
		#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	vec2 cubeToUV( vec3 v, float texelSizeY ) {
		vec3 absV = abs( v );
		float scaleToCube = 1.0 / max( absV.x, max( absV.y, absV.z ) );
		absV *= scaleToCube;
		v *= scaleToCube * ( 1.0 - 2.0 * texelSizeY );
		vec2 planar = v.xy;
		float almostATexel = 1.5 * texelSizeY;
		float almostOne = 1.0 - almostATexel;
		if ( absV.z >= almostOne ) {
			if ( v.z > 0.0 )
				planar.x = 4.0 - v.x;
		} else if ( absV.x >= almostOne ) {
			float signX = sign( v.x );
			planar.x = v.z * signX + 2.0 * signX;
		} else if ( absV.y >= almostOne ) {
			float signY = sign( v.y );
			planar.x = v.x + 2.0 * signY + 2.0;
			planar.y = v.z * signY - 2.0;
		}
		return vec2( 0.125, 0.25 ) * planar + vec2( 0.375, 0.75 );
	}
	float getPointShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		
		float lightToPositionLength = length( lightToPosition );
		if ( lightToPositionLength - shadowCameraFar <= 0.0 && lightToPositionLength - shadowCameraNear >= 0.0 ) {
			float dp = ( lightToPositionLength - shadowCameraNear ) / ( shadowCameraFar - shadowCameraNear );			dp += shadowBias;
			vec3 bd3D = normalize( lightToPosition );
			vec2 texelSize = vec2( 1.0 ) / ( shadowMapSize * vec2( 4.0, 2.0 ) );
			#if defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_PCF_SOFT ) || defined( SHADOWMAP_TYPE_VSM )
				vec2 offset = vec2( - 1, 1 ) * shadowRadius * texelSize.y;
				shadow = (
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxx, texelSize.y ), dp )
				) * ( 1.0 / 9.0 );
			#else
				shadow = texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp );
			#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
#endif`,nm=`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`,im=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	vec3 shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`,sm=`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`,rm=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,am=`#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`,om=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,lm=`#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`,cm=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,hm=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,um=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,dm=`#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`,fm=`#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = inverseTransformDirection( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseColor, material.specularColor, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`,pm=`#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
		
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
		
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		
		#else
		
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`,mm=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,gm=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,_m=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`,vm=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`;const xm=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,ym=`uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Mm=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Sm=`#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float flipEnvMap;
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, backgroundRotation * vec3( flipEnvMap * vWorldDirection.x, vWorldDirection.yz ) );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,bm=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Em=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Tm=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`,wm=`#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	float fragCoordZ = 0.5 * vHighPrecisionZW[0] / vHighPrecisionZW[1] + 0.5;
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`,Am=`#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`,Cm=`#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main () {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = packDepthToRGBA( dist );
}`,Rm=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,Pm=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Lm=`uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,Nm=`uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,Dm=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`,Im=`uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Um=`#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Fm=`#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Om=`#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`,km=`#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Bm=`#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`,zm=`#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <packing>
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( packNormalToRGB( normal ), diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`,Vm=`#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Hm=`#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Gm=`#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`,Wm=`#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
		float sheenEnergyComp = 1.0 - 0.157 * max3( material.sheenColor );
		outgoingLight = outgoingLight * sheenEnergyComp + sheenSpecularDirect + sheenSpecularIndirect;
	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Xm=`#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,qm=`#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,jm=`uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`,Ym=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,$m=`#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Km=`uniform vec3 color;
uniform float opacity;
#include <common>
#include <packing>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,Jm=`uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,Zm=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,se={alphahash_fragment:yf,alphahash_pars_fragment:Mf,alphamap_fragment:Sf,alphamap_pars_fragment:bf,alphatest_fragment:Ef,alphatest_pars_fragment:Tf,aomap_fragment:wf,aomap_pars_fragment:Af,batching_pars_vertex:Cf,batching_vertex:Rf,begin_vertex:Pf,beginnormal_vertex:Lf,bsdfs:Nf,iridescence_fragment:Df,bumpmap_pars_fragment:If,clipping_planes_fragment:Uf,clipping_planes_pars_fragment:Ff,clipping_planes_pars_vertex:Of,clipping_planes_vertex:kf,color_fragment:Bf,color_pars_fragment:zf,color_pars_vertex:Vf,color_vertex:Hf,common:Gf,cube_uv_reflection_fragment:Wf,defaultnormal_vertex:Xf,displacementmap_pars_vertex:qf,displacementmap_vertex:jf,emissivemap_fragment:Yf,emissivemap_pars_fragment:$f,colorspace_fragment:Kf,colorspace_pars_fragment:Jf,envmap_fragment:Zf,envmap_common_pars_fragment:Qf,envmap_pars_fragment:tp,envmap_pars_vertex:ep,envmap_physical_pars_fragment:dp,envmap_vertex:np,fog_vertex:ip,fog_pars_vertex:sp,fog_fragment:rp,fog_pars_fragment:ap,gradientmap_pars_fragment:op,lightmap_pars_fragment:lp,lights_lambert_fragment:cp,lights_lambert_pars_fragment:hp,lights_pars_begin:up,lights_toon_fragment:fp,lights_toon_pars_fragment:pp,lights_phong_fragment:mp,lights_phong_pars_fragment:gp,lights_physical_fragment:_p,lights_physical_pars_fragment:vp,lights_fragment_begin:xp,lights_fragment_maps:yp,lights_fragment_end:Mp,logdepthbuf_fragment:Sp,logdepthbuf_pars_fragment:bp,logdepthbuf_pars_vertex:Ep,logdepthbuf_vertex:Tp,map_fragment:wp,map_pars_fragment:Ap,map_particle_fragment:Cp,map_particle_pars_fragment:Rp,metalnessmap_fragment:Pp,metalnessmap_pars_fragment:Lp,morphinstance_vertex:Np,morphcolor_vertex:Dp,morphnormal_vertex:Ip,morphtarget_pars_vertex:Up,morphtarget_vertex:Fp,normal_fragment_begin:Op,normal_fragment_maps:kp,normal_pars_fragment:Bp,normal_pars_vertex:zp,normal_vertex:Vp,normalmap_pars_fragment:Hp,clearcoat_normal_fragment_begin:Gp,clearcoat_normal_fragment_maps:Wp,clearcoat_pars_fragment:Xp,iridescence_pars_fragment:qp,opaque_fragment:jp,packing:Yp,premultiplied_alpha_fragment:$p,project_vertex:Kp,dithering_fragment:Jp,dithering_pars_fragment:Zp,roughnessmap_fragment:Qp,roughnessmap_pars_fragment:tm,shadowmap_pars_fragment:em,shadowmap_pars_vertex:nm,shadowmap_vertex:im,shadowmask_pars_fragment:sm,skinbase_vertex:rm,skinning_pars_vertex:am,skinning_vertex:om,skinnormal_vertex:lm,specularmap_fragment:cm,specularmap_pars_fragment:hm,tonemapping_fragment:um,tonemapping_pars_fragment:dm,transmission_fragment:fm,transmission_pars_fragment:pm,uv_pars_fragment:mm,uv_pars_vertex:gm,uv_vertex:_m,worldpos_vertex:vm,background_vert:xm,background_frag:ym,backgroundCube_vert:Mm,backgroundCube_frag:Sm,cube_vert:bm,cube_frag:Em,depth_vert:Tm,depth_frag:wm,distanceRGBA_vert:Am,distanceRGBA_frag:Cm,equirect_vert:Rm,equirect_frag:Pm,linedashed_vert:Lm,linedashed_frag:Nm,meshbasic_vert:Dm,meshbasic_frag:Im,meshlambert_vert:Um,meshlambert_frag:Fm,meshmatcap_vert:Om,meshmatcap_frag:km,meshnormal_vert:Bm,meshnormal_frag:zm,meshphong_vert:Vm,meshphong_frag:Hm,meshphysical_vert:Gm,meshphysical_frag:Wm,meshtoon_vert:Xm,meshtoon_frag:qm,points_vert:jm,points_frag:Ym,shadow_vert:$m,shadow_frag:Km,sprite_vert:Jm,sprite_frag:Zm},gt={common:{diffuse:{value:new $t(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new ie},alphaMap:{value:null},alphaMapTransform:{value:new ie},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new ie}},envmap:{envMap:{value:null},envMapRotation:{value:new ie},flipEnvMap:{value:-1},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new ie}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new ie}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new ie},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new ie},normalScale:{value:new Lt(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new ie},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new ie}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new ie}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new ie}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new $t(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMap:{value:[]},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotShadowMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMap:{value:[]},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null}},points:{diffuse:{value:new $t(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new ie},alphaTest:{value:0},uvTransform:{value:new ie}},sprite:{diffuse:{value:new $t(16777215)},opacity:{value:1},center:{value:new Lt(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new ie},alphaMap:{value:null},alphaMapTransform:{value:new ie},alphaTest:{value:0}}},kn={basic:{uniforms:en([gt.common,gt.specularmap,gt.envmap,gt.aomap,gt.lightmap,gt.fog]),vertexShader:se.meshbasic_vert,fragmentShader:se.meshbasic_frag},lambert:{uniforms:en([gt.common,gt.specularmap,gt.envmap,gt.aomap,gt.lightmap,gt.emissivemap,gt.bumpmap,gt.normalmap,gt.displacementmap,gt.fog,gt.lights,{emissive:{value:new $t(0)}}]),vertexShader:se.meshlambert_vert,fragmentShader:se.meshlambert_frag},phong:{uniforms:en([gt.common,gt.specularmap,gt.envmap,gt.aomap,gt.lightmap,gt.emissivemap,gt.bumpmap,gt.normalmap,gt.displacementmap,gt.fog,gt.lights,{emissive:{value:new $t(0)},specular:{value:new $t(1118481)},shininess:{value:30}}]),vertexShader:se.meshphong_vert,fragmentShader:se.meshphong_frag},standard:{uniforms:en([gt.common,gt.envmap,gt.aomap,gt.lightmap,gt.emissivemap,gt.bumpmap,gt.normalmap,gt.displacementmap,gt.roughnessmap,gt.metalnessmap,gt.fog,gt.lights,{emissive:{value:new $t(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:se.meshphysical_vert,fragmentShader:se.meshphysical_frag},toon:{uniforms:en([gt.common,gt.aomap,gt.lightmap,gt.emissivemap,gt.bumpmap,gt.normalmap,gt.displacementmap,gt.gradientmap,gt.fog,gt.lights,{emissive:{value:new $t(0)}}]),vertexShader:se.meshtoon_vert,fragmentShader:se.meshtoon_frag},matcap:{uniforms:en([gt.common,gt.bumpmap,gt.normalmap,gt.displacementmap,gt.fog,{matcap:{value:null}}]),vertexShader:se.meshmatcap_vert,fragmentShader:se.meshmatcap_frag},points:{uniforms:en([gt.points,gt.fog]),vertexShader:se.points_vert,fragmentShader:se.points_frag},dashed:{uniforms:en([gt.common,gt.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:se.linedashed_vert,fragmentShader:se.linedashed_frag},depth:{uniforms:en([gt.common,gt.displacementmap]),vertexShader:se.depth_vert,fragmentShader:se.depth_frag},normal:{uniforms:en([gt.common,gt.bumpmap,gt.normalmap,gt.displacementmap,{opacity:{value:1}}]),vertexShader:se.meshnormal_vert,fragmentShader:se.meshnormal_frag},sprite:{uniforms:en([gt.sprite,gt.fog]),vertexShader:se.sprite_vert,fragmentShader:se.sprite_frag},background:{uniforms:{uvTransform:{value:new ie},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:se.background_vert,fragmentShader:se.background_frag},backgroundCube:{uniforms:{envMap:{value:null},flipEnvMap:{value:-1},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new ie}},vertexShader:se.backgroundCube_vert,fragmentShader:se.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:se.cube_vert,fragmentShader:se.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:se.equirect_vert,fragmentShader:se.equirect_frag},distanceRGBA:{uniforms:en([gt.common,gt.displacementmap,{referencePosition:{value:new F},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:se.distanceRGBA_vert,fragmentShader:se.distanceRGBA_frag},shadow:{uniforms:en([gt.lights,gt.fog,{color:{value:new $t(0)},opacity:{value:1}}]),vertexShader:se.shadow_vert,fragmentShader:se.shadow_frag}};kn.physical={uniforms:en([kn.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new ie},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new ie},clearcoatNormalScale:{value:new Lt(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new ie},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new ie},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new ie},sheen:{value:0},sheenColor:{value:new $t(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new ie},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new ie},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new ie},transmissionSamplerSize:{value:new Lt},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new ie},attenuationDistance:{value:0},attenuationColor:{value:new $t(0)},specularColor:{value:new $t(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new ie},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new ie},anisotropyVector:{value:new Lt},anisotropyMap:{value:null},anisotropyMapTransform:{value:new ie}}]),vertexShader:se.meshphysical_vert,fragmentShader:se.meshphysical_frag};const Vr={r:0,b:0,g:0},Fi=new cn,Qm=new Kt;function tg(i,t,e,n,s,r,a){const o=new $t(0);let c=r===!0?0:1,l,h,u=null,d=0,p=null;function g(R){let C=R.isScene===!0?R.background:null;return C&&C.isTexture&&(C=(R.backgroundBlurriness>0?e:t).get(C)),C}function _(R){let C=!1;const x=g(R);x===null?f(o,c):x&&x.isColor&&(f(x,1),C=!0);const N=i.xr.getEnvironmentBlendMode();N==="additive"?n.buffers.color.setClear(0,0,0,1,a):N==="alpha-blend"&&n.buffers.color.setClear(0,0,0,0,a),(i.autoClear||C)&&(n.buffers.depth.setTest(!0),n.buffers.depth.setMask(!0),n.buffers.color.setMask(!0),i.clear(i.autoClearColor,i.autoClearDepth,i.autoClearStencil))}function m(R,C){const x=g(C);x&&(x.isCubeTexture||x.mapping===Sa)?(h===void 0&&(h=new Ae(new Is(1,1,1),new Ti({name:"BackgroundCubeMaterial",uniforms:ws(kn.backgroundCube.uniforms),vertexShader:kn.backgroundCube.vertexShader,fragmentShader:kn.backgroundCube.fragmentShader,side:ln,depthTest:!1,depthWrite:!1,fog:!1})),h.geometry.deleteAttribute("normal"),h.geometry.deleteAttribute("uv"),h.onBeforeRender=function(N,P,L){this.matrixWorld.copyPosition(L.matrixWorld)},Object.defineProperty(h.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),s.update(h)),Fi.copy(C.backgroundRotation),Fi.x*=-1,Fi.y*=-1,Fi.z*=-1,x.isCubeTexture&&x.isRenderTargetTexture===!1&&(Fi.y*=-1,Fi.z*=-1),h.material.uniforms.envMap.value=x,h.material.uniforms.flipEnvMap.value=x.isCubeTexture&&x.isRenderTargetTexture===!1?-1:1,h.material.uniforms.backgroundBlurriness.value=C.backgroundBlurriness,h.material.uniforms.backgroundIntensity.value=C.backgroundIntensity,h.material.uniforms.backgroundRotation.value.setFromMatrix4(Qm.makeRotationFromEuler(Fi)),h.material.toneMapped=le.getTransfer(x.colorSpace)!==Ee,(u!==x||d!==x.version||p!==i.toneMapping)&&(h.material.needsUpdate=!0,u=x,d=x.version,p=i.toneMapping),h.layers.enableAll(),R.unshift(h,h.geometry,h.material,0,0,null)):x&&x.isTexture&&(l===void 0&&(l=new Ae(new Ta(2,2),new Ti({name:"BackgroundMaterial",uniforms:ws(kn.background.uniforms),vertexShader:kn.background.vertexShader,fragmentShader:kn.background.fragmentShader,side:ii,depthTest:!1,depthWrite:!1,fog:!1})),l.geometry.deleteAttribute("normal"),Object.defineProperty(l.material,"map",{get:function(){return this.uniforms.t2D.value}}),s.update(l)),l.material.uniforms.t2D.value=x,l.material.uniforms.backgroundIntensity.value=C.backgroundIntensity,l.material.toneMapped=le.getTransfer(x.colorSpace)!==Ee,x.matrixAutoUpdate===!0&&x.updateMatrix(),l.material.uniforms.uvTransform.value.copy(x.matrix),(u!==x||d!==x.version||p!==i.toneMapping)&&(l.material.needsUpdate=!0,u=x,d=x.version,p=i.toneMapping),l.layers.enableAll(),R.unshift(l,l.geometry,l.material,0,0,null))}function f(R,C){R.getRGB(Vr,Yh(i)),n.buffers.color.setClear(Vr.r,Vr.g,Vr.b,C,a)}return{getClearColor:function(){return o},setClearColor:function(R,C=1){o.set(R),c=C,f(o,c)},getClearAlpha:function(){return c},setClearAlpha:function(R){c=R,f(o,c)},render:_,addToRenderList:m}}function eg(i,t){const e=i.getParameter(i.MAX_VERTEX_ATTRIBS),n={},s=d(null);let r=s,a=!1;function o(S,U,k,H,J){let Q=!1;const O=u(H,k,U);r!==O&&(r=O,l(r.object)),Q=p(S,H,k,J),Q&&g(S,H,k,J),J!==null&&t.update(J,i.ELEMENT_ARRAY_BUFFER),(Q||a)&&(a=!1,x(S,U,k,H),J!==null&&i.bindBuffer(i.ELEMENT_ARRAY_BUFFER,t.get(J).buffer))}function c(){return i.createVertexArray()}function l(S){return i.bindVertexArray(S)}function h(S){return i.deleteVertexArray(S)}function u(S,U,k){const H=k.wireframe===!0;let J=n[S.id];J===void 0&&(J={},n[S.id]=J);let Q=J[U.id];Q===void 0&&(Q={},J[U.id]=Q);let O=Q[H];return O===void 0&&(O=d(c()),Q[H]=O),O}function d(S){const U=[],k=[],H=[];for(let J=0;J<e;J++)U[J]=0,k[J]=0,H[J]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:U,enabledAttributes:k,attributeDivisors:H,object:S,attributes:{},index:null}}function p(S,U,k,H){const J=r.attributes,Q=U.attributes;let O=0;const tt=k.getAttributes();for(const W in tt)if(tt[W].location>=0){const ft=J[W];let pt=Q[W];if(pt===void 0&&(W==="instanceMatrix"&&S.instanceMatrix&&(pt=S.instanceMatrix),W==="instanceColor"&&S.instanceColor&&(pt=S.instanceColor)),ft===void 0||ft.attribute!==pt||pt&&ft.data!==pt.data)return!0;O++}return r.attributesNum!==O||r.index!==H}function g(S,U,k,H){const J={},Q=U.attributes;let O=0;const tt=k.getAttributes();for(const W in tt)if(tt[W].location>=0){let ft=Q[W];ft===void 0&&(W==="instanceMatrix"&&S.instanceMatrix&&(ft=S.instanceMatrix),W==="instanceColor"&&S.instanceColor&&(ft=S.instanceColor));const pt={};pt.attribute=ft,ft&&ft.data&&(pt.data=ft.data),J[W]=pt,O++}r.attributes=J,r.attributesNum=O,r.index=H}function _(){const S=r.newAttributes;for(let U=0,k=S.length;U<k;U++)S[U]=0}function m(S){f(S,0)}function f(S,U){const k=r.newAttributes,H=r.enabledAttributes,J=r.attributeDivisors;k[S]=1,H[S]===0&&(i.enableVertexAttribArray(S),H[S]=1),J[S]!==U&&(i.vertexAttribDivisor(S,U),J[S]=U)}function R(){const S=r.newAttributes,U=r.enabledAttributes;for(let k=0,H=U.length;k<H;k++)U[k]!==S[k]&&(i.disableVertexAttribArray(k),U[k]=0)}function C(S,U,k,H,J,Q,O){O===!0?i.vertexAttribIPointer(S,U,k,J,Q):i.vertexAttribPointer(S,U,k,H,J,Q)}function x(S,U,k,H){_();const J=H.attributes,Q=k.getAttributes(),O=U.defaultAttributeValues;for(const tt in Q){const W=Q[tt];if(W.location>=0){let ot=J[tt];if(ot===void 0&&(tt==="instanceMatrix"&&S.instanceMatrix&&(ot=S.instanceMatrix),tt==="instanceColor"&&S.instanceColor&&(ot=S.instanceColor)),ot!==void 0){const ft=ot.normalized,pt=ot.itemSize,Ct=t.get(ot);if(Ct===void 0)continue;const Et=Ct.buffer,X=Ct.type,K=Ct.bytesPerElement,et=X===i.INT||X===i.UNSIGNED_INT||ot.gpuType===gl;if(ot.isInterleavedBufferAttribute){const nt=ot.data,_t=nt.stride,zt=ot.offset;if(nt.isInstancedInterleavedBuffer){for(let Zt=0;Zt<W.locationSize;Zt++)f(W.location+Zt,nt.meshPerAttribute);S.isInstancedMesh!==!0&&H._maxInstanceCount===void 0&&(H._maxInstanceCount=nt.meshPerAttribute*nt.count)}else for(let Zt=0;Zt<W.locationSize;Zt++)m(W.location+Zt);i.bindBuffer(i.ARRAY_BUFFER,Et);for(let Zt=0;Zt<W.locationSize;Zt++)C(W.location+Zt,pt/W.locationSize,X,ft,_t*K,(zt+pt/W.locationSize*Zt)*K,et)}else{if(ot.isInstancedBufferAttribute){for(let nt=0;nt<W.locationSize;nt++)f(W.location+nt,ot.meshPerAttribute);S.isInstancedMesh!==!0&&H._maxInstanceCount===void 0&&(H._maxInstanceCount=ot.meshPerAttribute*ot.count)}else for(let nt=0;nt<W.locationSize;nt++)m(W.location+nt);i.bindBuffer(i.ARRAY_BUFFER,Et);for(let nt=0;nt<W.locationSize;nt++)C(W.location+nt,pt/W.locationSize,X,ft,pt*K,pt/W.locationSize*nt*K,et)}}else if(O!==void 0){const ft=O[tt];if(ft!==void 0)switch(ft.length){case 2:i.vertexAttrib2fv(W.location,ft);break;case 3:i.vertexAttrib3fv(W.location,ft);break;case 4:i.vertexAttrib4fv(W.location,ft);break;default:i.vertexAttrib1fv(W.location,ft)}}}}R()}function N(){I();for(const S in n){const U=n[S];for(const k in U){const H=U[k];for(const J in H)h(H[J].object),delete H[J];delete U[k]}delete n[S]}}function P(S){if(n[S.id]===void 0)return;const U=n[S.id];for(const k in U){const H=U[k];for(const J in H)h(H[J].object),delete H[J];delete U[k]}delete n[S.id]}function L(S){for(const U in n){const k=n[U];if(k[S.id]===void 0)continue;const H=k[S.id];for(const J in H)h(H[J].object),delete H[J];delete k[S.id]}}function I(){M(),a=!0,r!==s&&(r=s,l(r.object))}function M(){s.geometry=null,s.program=null,s.wireframe=!1}return{setup:o,reset:I,resetDefaultState:M,dispose:N,releaseStatesOfGeometry:P,releaseStatesOfProgram:L,initAttributes:_,enableAttribute:m,disableUnusedAttributes:R}}function ng(i,t,e){let n;function s(l){n=l}function r(l,h){i.drawArrays(n,l,h),e.update(h,n,1)}function a(l,h,u){u!==0&&(i.drawArraysInstanced(n,l,h,u),e.update(h,n,u))}function o(l,h,u){if(u===0)return;t.get("WEBGL_multi_draw").multiDrawArraysWEBGL(n,l,0,h,0,u);let p=0;for(let g=0;g<u;g++)p+=h[g];e.update(p,n,1)}function c(l,h,u,d){if(u===0)return;const p=t.get("WEBGL_multi_draw");if(p===null)for(let g=0;g<l.length;g++)a(l[g],h[g],d[g]);else{p.multiDrawArraysInstancedWEBGL(n,l,0,h,0,d,0,u);let g=0;for(let _=0;_<u;_++)g+=h[_]*d[_];e.update(g,n,1)}}this.setMode=s,this.render=r,this.renderInstances=a,this.renderMultiDraw=o,this.renderMultiDrawInstances=c}function ig(i,t,e,n){let s;function r(){if(s!==void 0)return s;if(t.has("EXT_texture_filter_anisotropic")===!0){const L=t.get("EXT_texture_filter_anisotropic");s=i.getParameter(L.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else s=0;return s}function a(L){return!(L!==Sn&&n.convert(L)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_FORMAT))}function o(L){const I=L===hr&&(t.has("EXT_color_buffer_half_float")||t.has("EXT_color_buffer_float"));return!(L!==si&&n.convert(L)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_TYPE)&&L!==Bn&&!I)}function c(L){if(L==="highp"){if(i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.HIGH_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.HIGH_FLOAT).precision>0)return"highp";L="mediump"}return L==="mediump"&&i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.MEDIUM_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let l=e.precision!==void 0?e.precision:"highp";const h=c(l);h!==l&&(console.warn("THREE.WebGLRenderer:",l,"not supported, using",h,"instead."),l=h);const u=e.logarithmicDepthBuffer===!0,d=e.reverseDepthBuffer===!0&&t.has("EXT_clip_control"),p=i.getParameter(i.MAX_TEXTURE_IMAGE_UNITS),g=i.getParameter(i.MAX_VERTEX_TEXTURE_IMAGE_UNITS),_=i.getParameter(i.MAX_TEXTURE_SIZE),m=i.getParameter(i.MAX_CUBE_MAP_TEXTURE_SIZE),f=i.getParameter(i.MAX_VERTEX_ATTRIBS),R=i.getParameter(i.MAX_VERTEX_UNIFORM_VECTORS),C=i.getParameter(i.MAX_VARYING_VECTORS),x=i.getParameter(i.MAX_FRAGMENT_UNIFORM_VECTORS),N=g>0,P=i.getParameter(i.MAX_SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:r,getMaxPrecision:c,textureFormatReadable:a,textureTypeReadable:o,precision:l,logarithmicDepthBuffer:u,reverseDepthBuffer:d,maxTextures:p,maxVertexTextures:g,maxTextureSize:_,maxCubemapSize:m,maxAttributes:f,maxVertexUniforms:R,maxVaryings:C,maxFragmentUniforms:x,vertexTextures:N,maxSamples:P}}function sg(i){const t=this;let e=null,n=0,s=!1,r=!1;const a=new vi,o=new ie,c={value:null,needsUpdate:!1};this.uniform=c,this.numPlanes=0,this.numIntersection=0,this.init=function(u,d){const p=u.length!==0||d||n!==0||s;return s=d,n=u.length,p},this.beginShadows=function(){r=!0,h(null)},this.endShadows=function(){r=!1},this.setGlobalState=function(u,d){e=h(u,d,0)},this.setState=function(u,d,p){const g=u.clippingPlanes,_=u.clipIntersection,m=u.clipShadows,f=i.get(u);if(!s||g===null||g.length===0||r&&!m)r?h(null):l();else{const R=r?0:n,C=R*4;let x=f.clippingState||null;c.value=x,x=h(g,d,C,p);for(let N=0;N!==C;++N)x[N]=e[N];f.clippingState=x,this.numIntersection=_?this.numPlanes:0,this.numPlanes+=R}};function l(){c.value!==e&&(c.value=e,c.needsUpdate=n>0),t.numPlanes=n,t.numIntersection=0}function h(u,d,p,g){const _=u!==null?u.length:0;let m=null;if(_!==0){if(m=c.value,g!==!0||m===null){const f=p+_*4,R=d.matrixWorldInverse;o.getNormalMatrix(R),(m===null||m.length<f)&&(m=new Float32Array(f));for(let C=0,x=p;C!==_;++C,x+=4)a.copy(u[C]).applyMatrix4(R,o),a.normal.toArray(m,x),m[x+3]=a.constant}c.value=m,c.needsUpdate=!0}return t.numPlanes=_,t.numIntersection=0,m}}function rg(i){let t=new WeakMap;function e(a,o){return o===Io?a.mapping=Ms:o===Uo&&(a.mapping=Ss),a}function n(a){if(a&&a.isTexture){const o=a.mapping;if(o===Io||o===Uo)if(t.has(a)){const c=t.get(a).texture;return e(c,a.mapping)}else{const c=a.image;if(c&&c.height>0){const l=new gf(c.height);return l.fromEquirectangularTexture(i,a),t.set(a,l),a.addEventListener("dispose",s),e(l.texture,a.mapping)}else return null}}return a}function s(a){const o=a.target;o.removeEventListener("dispose",s);const c=t.get(o);c!==void 0&&(t.delete(o),c.dispose())}function r(){t=new WeakMap}return{get:n,dispose:r}}class El extends $h{constructor(t=-1,e=1,n=1,s=-1,r=.1,a=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=t,this.right=e,this.top=n,this.bottom=s,this.near=r,this.far=a,this.updateProjectionMatrix()}copy(t,e){return super.copy(t,e),this.left=t.left,this.right=t.right,this.top=t.top,this.bottom=t.bottom,this.near=t.near,this.far=t.far,this.zoom=t.zoom,this.view=t.view===null?null:Object.assign({},t.view),this}setViewOffset(t,e,n,s,r,a){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=t,this.view.fullHeight=e,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const t=(this.right-this.left)/(2*this.zoom),e=(this.top-this.bottom)/(2*this.zoom),n=(this.right+this.left)/2,s=(this.top+this.bottom)/2;let r=n-t,a=n+t,o=s+e,c=s-e;if(this.view!==null&&this.view.enabled){const l=(this.right-this.left)/this.view.fullWidth/this.zoom,h=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=l*this.view.offsetX,a=r+l*this.view.width,o-=h*this.view.offsetY,c=o-h*this.view.height}this.projectionMatrix.makeOrthographic(r,a,o,c,this.near,this.far,this.coordinateSystem),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(t){const e=super.toJSON(t);return e.object.zoom=this.zoom,e.object.left=this.left,e.object.right=this.right,e.object.top=this.top,e.object.bottom=this.bottom,e.object.near=this.near,e.object.far=this.far,this.view!==null&&(e.object.view=Object.assign({},this.view)),e}}const fs=4,Sc=[.125,.215,.35,.446,.526,.582],Vi=20,io=new El,bc=new $t;let so=null,ro=0,ao=0,oo=!1;const Bi=(1+Math.sqrt(5))/2,hs=1/Bi,Ec=[new F(-Bi,hs,0),new F(Bi,hs,0),new F(-hs,0,Bi),new F(hs,0,Bi),new F(0,Bi,-hs),new F(0,Bi,hs),new F(-1,1,-1),new F(1,1,-1),new F(-1,1,1),new F(1,1,1)];class Tc{constructor(t){this._renderer=t,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._lodPlanes=[],this._sizeLods=[],this._sigmas=[],this._blurMaterial=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._compileMaterial(this._blurMaterial)}fromScene(t,e=0,n=.1,s=100){so=this._renderer.getRenderTarget(),ro=this._renderer.getActiveCubeFace(),ao=this._renderer.getActiveMipmapLevel(),oo=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(256);const r=this._allocateTargets();return r.depthBuffer=!0,this._sceneToCubeUV(t,n,s,r),e>0&&this._blur(r,0,0,e),this._applyPMREM(r),this._cleanup(r),r}fromEquirectangular(t,e=null){return this._fromTexture(t,e)}fromCubemap(t,e=null){return this._fromTexture(t,e)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=Cc(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=Ac(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose()}_setSize(t){this._lodMax=Math.floor(Math.log2(t)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let t=0;t<this._lodPlanes.length;t++)this._lodPlanes[t].dispose()}_cleanup(t){this._renderer.setRenderTarget(so,ro,ao),this._renderer.xr.enabled=oo,t.scissorTest=!1,Hr(t,0,0,t.width,t.height)}_fromTexture(t,e){t.mapping===Ms||t.mapping===Ss?this._setSize(t.image.length===0?16:t.image[0].width||t.image[0].image.width):this._setSize(t.image.width/4),so=this._renderer.getRenderTarget(),ro=this._renderer.getActiveCubeFace(),ao=this._renderer.getActiveMipmapLevel(),oo=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;const n=e||this._allocateTargets();return this._textureToCubeUV(t,n),this._applyPMREM(n),this._cleanup(n),n}_allocateTargets(){const t=3*Math.max(this._cubeSize,112),e=4*this._cubeSize,n={magFilter:fn,minFilter:fn,generateMipmaps:!1,type:hr,format:Sn,colorSpace:Ds,depthBuffer:!1},s=wc(t,e,n);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==t||this._pingPongRenderTarget.height!==e){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=wc(t,e,n);const{_lodMax:r}=this;({sizeLods:this._sizeLods,lodPlanes:this._lodPlanes,sigmas:this._sigmas}=ag(r)),this._blurMaterial=og(r,t,e)}return s}_compileMaterial(t){const e=new Ae(this._lodPlanes[0],t);this._renderer.compile(e,io)}_sceneToCubeUV(t,e,n,s){const o=new Je(90,1,e,n),c=[1,-1,1,1,1,1],l=[1,1,1,-1,-1,-1],h=this._renderer,u=h.autoClear,d=h.toneMapping;h.getClearColor(bc),h.toneMapping=bi,h.autoClear=!1;const p=new Un({name:"PMREM.Background",side:ln,depthWrite:!1,depthTest:!1}),g=new Ae(new Is,p);let _=!1;const m=t.background;m?m.isColor&&(p.color.copy(m),t.background=null,_=!0):(p.color.copy(bc),_=!0);for(let f=0;f<6;f++){const R=f%3;R===0?(o.up.set(0,c[f],0),o.lookAt(l[f],0,0)):R===1?(o.up.set(0,0,c[f]),o.lookAt(0,l[f],0)):(o.up.set(0,c[f],0),o.lookAt(0,0,l[f]));const C=this._cubeSize;Hr(s,R*C,f>2?C:0,C,C),h.setRenderTarget(s),_&&h.render(g,o),h.render(t,o)}g.geometry.dispose(),g.material.dispose(),h.toneMapping=d,h.autoClear=u,t.background=m}_textureToCubeUV(t,e){const n=this._renderer,s=t.mapping===Ms||t.mapping===Ss;s?(this._cubemapMaterial===null&&(this._cubemapMaterial=Cc()),this._cubemapMaterial.uniforms.flipEnvMap.value=t.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=Ac());const r=s?this._cubemapMaterial:this._equirectMaterial,a=new Ae(this._lodPlanes[0],r),o=r.uniforms;o.envMap.value=t;const c=this._cubeSize;Hr(e,0,0,3*c,2*c),n.setRenderTarget(e),n.render(a,io)}_applyPMREM(t){const e=this._renderer,n=e.autoClear;e.autoClear=!1;const s=this._lodPlanes.length;for(let r=1;r<s;r++){const a=Math.sqrt(this._sigmas[r]*this._sigmas[r]-this._sigmas[r-1]*this._sigmas[r-1]),o=Ec[(s-r-1)%Ec.length];this._blur(t,r-1,r,a,o)}e.autoClear=n}_blur(t,e,n,s,r){const a=this._pingPongRenderTarget;this._halfBlur(t,a,e,n,s,"latitudinal",r),this._halfBlur(a,t,n,n,s,"longitudinal",r)}_halfBlur(t,e,n,s,r,a,o){const c=this._renderer,l=this._blurMaterial;a!=="latitudinal"&&a!=="longitudinal"&&console.error("blur direction must be either latitudinal or longitudinal!");const h=3,u=new Ae(this._lodPlanes[s],l),d=l.uniforms,p=this._sizeLods[n]-1,g=isFinite(r)?Math.PI/(2*p):2*Math.PI/(2*Vi-1),_=r/g,m=isFinite(r)?1+Math.floor(h*_):Vi;m>Vi&&console.warn(`sigmaRadians, ${r}, is too large and will clip, as it requested ${m} samples when the maximum is set to ${Vi}`);const f=[];let R=0;for(let L=0;L<Vi;++L){const I=L/_,M=Math.exp(-I*I/2);f.push(M),L===0?R+=M:L<m&&(R+=2*M)}for(let L=0;L<f.length;L++)f[L]=f[L]/R;d.envMap.value=t.texture,d.samples.value=m,d.weights.value=f,d.latitudinal.value=a==="latitudinal",o&&(d.poleAxis.value=o);const{_lodMax:C}=this;d.dTheta.value=g,d.mipInt.value=C-n;const x=this._sizeLods[s],N=3*x*(s>C-fs?s-C+fs:0),P=4*(this._cubeSize-x);Hr(e,N,P,3*x,2*x),c.setRenderTarget(e),c.render(u,io)}}function ag(i){const t=[],e=[],n=[];let s=i;const r=i-fs+1+Sc.length;for(let a=0;a<r;a++){const o=Math.pow(2,s);e.push(o);let c=1/o;a>i-fs?c=Sc[a-i+fs-1]:a===0&&(c=0),n.push(c);const l=1/(o-2),h=-l,u=1+l,d=[h,h,u,h,u,u,h,h,u,u,h,u],p=6,g=6,_=3,m=2,f=1,R=new Float32Array(_*g*p),C=new Float32Array(m*g*p),x=new Float32Array(f*g*p);for(let P=0;P<p;P++){const L=P%3*2/3-1,I=P>2?0:-1,M=[L,I,0,L+2/3,I,0,L+2/3,I+1,0,L,I,0,L+2/3,I+1,0,L,I+1,0];R.set(M,_*g*P),C.set(d,m*g*P);const S=[P,P,P,P,P,P];x.set(S,f*g*P)}const N=new Ie;N.setAttribute("position",new gn(R,_)),N.setAttribute("uv",new gn(C,m)),N.setAttribute("faceIndex",new gn(x,f)),t.push(N),s>fs&&s--}return{lodPlanes:t,sizeLods:e,sigmas:n}}function wc(i,t,e){const n=new Wi(i,t,e);return n.texture.mapping=Sa,n.texture.name="PMREM.cubeUv",n.scissorTest=!0,n}function Hr(i,t,e,n,s){i.viewport.set(t,e,n,s),i.scissor.set(t,e,n,s)}function og(i,t,e){const n=new Float32Array(Vi),s=new F(0,1,0);return new Ti({name:"SphericalGaussianBlur",defines:{n:Vi,CUBEUV_TEXEL_WIDTH:1/t,CUBEUV_TEXEL_HEIGHT:1/e,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:n},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:s}},vertexShader:Tl(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`,blending:Si,depthTest:!1,depthWrite:!1})}function Ac(){return new Ti({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:Tl(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`,blending:Si,depthTest:!1,depthWrite:!1})}function Cc(){return new Ti({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:Tl(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:Si,depthTest:!1,depthWrite:!1})}function Tl(){return`

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`}function lg(i){let t=new WeakMap,e=null;function n(o){if(o&&o.isTexture){const c=o.mapping,l=c===Io||c===Uo,h=c===Ms||c===Ss;if(l||h){let u=t.get(o);const d=u!==void 0?u.texture.pmremVersion:0;if(o.isRenderTargetTexture&&o.pmremVersion!==d)return e===null&&(e=new Tc(i)),u=l?e.fromEquirectangular(o,u):e.fromCubemap(o,u),u.texture.pmremVersion=o.pmremVersion,t.set(o,u),u.texture;if(u!==void 0)return u.texture;{const p=o.image;return l&&p&&p.height>0||h&&p&&s(p)?(e===null&&(e=new Tc(i)),u=l?e.fromEquirectangular(o):e.fromCubemap(o),u.texture.pmremVersion=o.pmremVersion,t.set(o,u),o.addEventListener("dispose",r),u.texture):null}}}return o}function s(o){let c=0;const l=6;for(let h=0;h<l;h++)o[h]!==void 0&&c++;return c===l}function r(o){const c=o.target;c.removeEventListener("dispose",r);const l=t.get(c);l!==void 0&&(t.delete(c),l.dispose())}function a(){t=new WeakMap,e!==null&&(e.dispose(),e=null)}return{get:n,dispose:a}}function cg(i){const t={};function e(n){if(t[n]!==void 0)return t[n];let s;switch(n){case"WEBGL_depth_texture":s=i.getExtension("WEBGL_depth_texture")||i.getExtension("MOZ_WEBGL_depth_texture")||i.getExtension("WEBKIT_WEBGL_depth_texture");break;case"EXT_texture_filter_anisotropic":s=i.getExtension("EXT_texture_filter_anisotropic")||i.getExtension("MOZ_EXT_texture_filter_anisotropic")||i.getExtension("WEBKIT_EXT_texture_filter_anisotropic");break;case"WEBGL_compressed_texture_s3tc":s=i.getExtension("WEBGL_compressed_texture_s3tc")||i.getExtension("MOZ_WEBGL_compressed_texture_s3tc")||i.getExtension("WEBKIT_WEBGL_compressed_texture_s3tc");break;case"WEBGL_compressed_texture_pvrtc":s=i.getExtension("WEBGL_compressed_texture_pvrtc")||i.getExtension("WEBKIT_WEBGL_compressed_texture_pvrtc");break;default:s=i.getExtension(n)}return t[n]=s,s}return{has:function(n){return e(n)!==null},init:function(){e("EXT_color_buffer_float"),e("WEBGL_clip_cull_distance"),e("OES_texture_float_linear"),e("EXT_color_buffer_half_float"),e("WEBGL_multisampled_render_to_texture"),e("WEBGL_render_shared_exponent")},get:function(n){const s=e(n);return s===null&&Ys("THREE.WebGLRenderer: "+n+" extension not supported."),s}}}function hg(i,t,e,n){const s={},r=new WeakMap;function a(u){const d=u.target;d.index!==null&&t.remove(d.index);for(const g in d.attributes)t.remove(d.attributes[g]);for(const g in d.morphAttributes){const _=d.morphAttributes[g];for(let m=0,f=_.length;m<f;m++)t.remove(_[m])}d.removeEventListener("dispose",a),delete s[d.id];const p=r.get(d);p&&(t.remove(p),r.delete(d)),n.releaseStatesOfGeometry(d),d.isInstancedBufferGeometry===!0&&delete d._maxInstanceCount,e.memory.geometries--}function o(u,d){return s[d.id]===!0||(d.addEventListener("dispose",a),s[d.id]=!0,e.memory.geometries++),d}function c(u){const d=u.attributes;for(const g in d)t.update(d[g],i.ARRAY_BUFFER);const p=u.morphAttributes;for(const g in p){const _=p[g];for(let m=0,f=_.length;m<f;m++)t.update(_[m],i.ARRAY_BUFFER)}}function l(u){const d=[],p=u.index,g=u.attributes.position;let _=0;if(p!==null){const R=p.array;_=p.version;for(let C=0,x=R.length;C<x;C+=3){const N=R[C+0],P=R[C+1],L=R[C+2];d.push(N,P,P,L,L,N)}}else if(g!==void 0){const R=g.array;_=g.version;for(let C=0,x=R.length/3-1;C<x;C+=3){const N=C+0,P=C+1,L=C+2;d.push(N,P,P,L,L,N)}}else return;const m=new(Vh(d)?jh:qh)(d,1);m.version=_;const f=r.get(u);f&&t.remove(f),r.set(u,m)}function h(u){const d=r.get(u);if(d){const p=u.index;p!==null&&d.version<p.version&&l(u)}else l(u);return r.get(u)}return{get:o,update:c,getWireframeAttribute:h}}function ug(i,t,e){let n;function s(d){n=d}let r,a;function o(d){r=d.type,a=d.bytesPerElement}function c(d,p){i.drawElements(n,p,r,d*a),e.update(p,n,1)}function l(d,p,g){g!==0&&(i.drawElementsInstanced(n,p,r,d*a,g),e.update(p,n,g))}function h(d,p,g){if(g===0)return;t.get("WEBGL_multi_draw").multiDrawElementsWEBGL(n,p,0,r,d,0,g);let m=0;for(let f=0;f<g;f++)m+=p[f];e.update(m,n,1)}function u(d,p,g,_){if(g===0)return;const m=t.get("WEBGL_multi_draw");if(m===null)for(let f=0;f<d.length;f++)l(d[f]/a,p[f],_[f]);else{m.multiDrawElementsInstancedWEBGL(n,p,0,r,d,0,_,0,g);let f=0;for(let R=0;R<g;R++)f+=p[R]*_[R];e.update(f,n,1)}}this.setMode=s,this.setIndex=o,this.render=c,this.renderInstances=l,this.renderMultiDraw=h,this.renderMultiDrawInstances=u}function dg(i){const t={geometries:0,textures:0},e={frame:0,calls:0,triangles:0,points:0,lines:0};function n(r,a,o){switch(e.calls++,a){case i.TRIANGLES:e.triangles+=o*(r/3);break;case i.LINES:e.lines+=o*(r/2);break;case i.LINE_STRIP:e.lines+=o*(r-1);break;case i.LINE_LOOP:e.lines+=o*r;break;case i.POINTS:e.points+=o*r;break;default:console.error("THREE.WebGLInfo: Unknown draw mode:",a);break}}function s(){e.calls=0,e.triangles=0,e.points=0,e.lines=0}return{memory:t,render:e,programs:null,autoReset:!0,reset:s,update:n}}function fg(i,t,e){const n=new WeakMap,s=new ve;function r(a,o,c){const l=a.morphTargetInfluences,h=o.morphAttributes.position||o.morphAttributes.normal||o.morphAttributes.color,u=h!==void 0?h.length:0;let d=n.get(o);if(d===void 0||d.count!==u){let S=function(){I.dispose(),n.delete(o),o.removeEventListener("dispose",S)};var p=S;d!==void 0&&d.texture.dispose();const g=o.morphAttributes.position!==void 0,_=o.morphAttributes.normal!==void 0,m=o.morphAttributes.color!==void 0,f=o.morphAttributes.position||[],R=o.morphAttributes.normal||[],C=o.morphAttributes.color||[];let x=0;g===!0&&(x=1),_===!0&&(x=2),m===!0&&(x=3);let N=o.attributes.position.count*x,P=1;N>t.maxTextureSize&&(P=Math.ceil(N/t.maxTextureSize),N=t.maxTextureSize);const L=new Float32Array(N*P*4*u),I=new Gh(L,N,P,u);I.type=Bn,I.needsUpdate=!0;const M=x*4;for(let U=0;U<u;U++){const k=f[U],H=R[U],J=C[U],Q=N*P*4*U;for(let O=0;O<k.count;O++){const tt=O*M;g===!0&&(s.fromBufferAttribute(k,O),L[Q+tt+0]=s.x,L[Q+tt+1]=s.y,L[Q+tt+2]=s.z,L[Q+tt+3]=0),_===!0&&(s.fromBufferAttribute(H,O),L[Q+tt+4]=s.x,L[Q+tt+5]=s.y,L[Q+tt+6]=s.z,L[Q+tt+7]=0),m===!0&&(s.fromBufferAttribute(J,O),L[Q+tt+8]=s.x,L[Q+tt+9]=s.y,L[Q+tt+10]=s.z,L[Q+tt+11]=J.itemSize===4?s.w:1)}}d={count:u,texture:I,size:new Lt(N,P)},n.set(o,d),o.addEventListener("dispose",S)}if(a.isInstancedMesh===!0&&a.morphTexture!==null)c.getUniforms().setValue(i,"morphTexture",a.morphTexture,e);else{let g=0;for(let m=0;m<l.length;m++)g+=l[m];const _=o.morphTargetsRelative?1:1-g;c.getUniforms().setValue(i,"morphTargetBaseInfluence",_),c.getUniforms().setValue(i,"morphTargetInfluences",l)}c.getUniforms().setValue(i,"morphTargetsTexture",d.texture,e),c.getUniforms().setValue(i,"morphTargetsTextureSize",d.size)}return{update:r}}function pg(i,t,e,n){let s=new WeakMap;function r(c){const l=n.render.frame,h=c.geometry,u=t.get(c,h);if(s.get(u)!==l&&(t.update(u),s.set(u,l)),c.isInstancedMesh&&(c.hasEventListener("dispose",o)===!1&&c.addEventListener("dispose",o),s.get(c)!==l&&(e.update(c.instanceMatrix,i.ARRAY_BUFFER),c.instanceColor!==null&&e.update(c.instanceColor,i.ARRAY_BUFFER),s.set(c,l))),c.isSkinnedMesh){const d=c.skeleton;s.get(d)!==l&&(d.update(),s.set(d,l))}return u}function a(){s=new WeakMap}function o(c){const l=c.target;l.removeEventListener("dispose",o),e.remove(l.instanceMatrix),l.instanceColor!==null&&e.remove(l.instanceColor)}return{update:r,dispose:a}}class Zh extends Ye{constructor(t,e,n,s,r,a,o,c,l,h=gs){if(h!==gs&&h!==Es)throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");n===void 0&&h===gs&&(n=Gi),n===void 0&&h===Es&&(n=bs),super(null,s,r,a,o,c,h,n,l),this.isDepthTexture=!0,this.image={width:t,height:e},this.magFilter=o!==void 0?o:mn,this.minFilter=c!==void 0?c:mn,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(t){return super.copy(t),this.compareFunction=t.compareFunction,this}toJSON(t){const e=super.toJSON(t);return this.compareFunction!==null&&(e.compareFunction=this.compareFunction),e}}const Qh=new Ye,Rc=new Zh(1,1),tu=new Gh,eu=new Qd,nu=new Kh,Pc=[],Lc=[],Nc=new Float32Array(16),Dc=new Float32Array(9),Ic=new Float32Array(4);function Us(i,t,e){const n=i[0];if(n<=0||n>0)return i;const s=t*e;let r=Pc[s];if(r===void 0&&(r=new Float32Array(s),Pc[s]=r),t!==0){n.toArray(r,0);for(let a=1,o=0;a!==t;++a)o+=e,i[a].toArray(r,o)}return r}function Ve(i,t){if(i.length!==t.length)return!1;for(let e=0,n=i.length;e<n;e++)if(i[e]!==t[e])return!1;return!0}function He(i,t){for(let e=0,n=t.length;e<n;e++)i[e]=t[e]}function wa(i,t){let e=Lc[t];e===void 0&&(e=new Int32Array(t),Lc[t]=e);for(let n=0;n!==t;++n)e[n]=i.allocateTextureUnit();return e}function mg(i,t){const e=this.cache;e[0]!==t&&(i.uniform1f(this.addr,t),e[0]=t)}function gg(i,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y)&&(i.uniform2f(this.addr,t.x,t.y),e[0]=t.x,e[1]=t.y);else{if(Ve(e,t))return;i.uniform2fv(this.addr,t),He(e,t)}}function _g(i,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z)&&(i.uniform3f(this.addr,t.x,t.y,t.z),e[0]=t.x,e[1]=t.y,e[2]=t.z);else if(t.r!==void 0)(e[0]!==t.r||e[1]!==t.g||e[2]!==t.b)&&(i.uniform3f(this.addr,t.r,t.g,t.b),e[0]=t.r,e[1]=t.g,e[2]=t.b);else{if(Ve(e,t))return;i.uniform3fv(this.addr,t),He(e,t)}}function vg(i,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z||e[3]!==t.w)&&(i.uniform4f(this.addr,t.x,t.y,t.z,t.w),e[0]=t.x,e[1]=t.y,e[2]=t.z,e[3]=t.w);else{if(Ve(e,t))return;i.uniform4fv(this.addr,t),He(e,t)}}function xg(i,t){const e=this.cache,n=t.elements;if(n===void 0){if(Ve(e,t))return;i.uniformMatrix2fv(this.addr,!1,t),He(e,t)}else{if(Ve(e,n))return;Ic.set(n),i.uniformMatrix2fv(this.addr,!1,Ic),He(e,n)}}function yg(i,t){const e=this.cache,n=t.elements;if(n===void 0){if(Ve(e,t))return;i.uniformMatrix3fv(this.addr,!1,t),He(e,t)}else{if(Ve(e,n))return;Dc.set(n),i.uniformMatrix3fv(this.addr,!1,Dc),He(e,n)}}function Mg(i,t){const e=this.cache,n=t.elements;if(n===void 0){if(Ve(e,t))return;i.uniformMatrix4fv(this.addr,!1,t),He(e,t)}else{if(Ve(e,n))return;Nc.set(n),i.uniformMatrix4fv(this.addr,!1,Nc),He(e,n)}}function Sg(i,t){const e=this.cache;e[0]!==t&&(i.uniform1i(this.addr,t),e[0]=t)}function bg(i,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y)&&(i.uniform2i(this.addr,t.x,t.y),e[0]=t.x,e[1]=t.y);else{if(Ve(e,t))return;i.uniform2iv(this.addr,t),He(e,t)}}function Eg(i,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z)&&(i.uniform3i(this.addr,t.x,t.y,t.z),e[0]=t.x,e[1]=t.y,e[2]=t.z);else{if(Ve(e,t))return;i.uniform3iv(this.addr,t),He(e,t)}}function Tg(i,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z||e[3]!==t.w)&&(i.uniform4i(this.addr,t.x,t.y,t.z,t.w),e[0]=t.x,e[1]=t.y,e[2]=t.z,e[3]=t.w);else{if(Ve(e,t))return;i.uniform4iv(this.addr,t),He(e,t)}}function wg(i,t){const e=this.cache;e[0]!==t&&(i.uniform1ui(this.addr,t),e[0]=t)}function Ag(i,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y)&&(i.uniform2ui(this.addr,t.x,t.y),e[0]=t.x,e[1]=t.y);else{if(Ve(e,t))return;i.uniform2uiv(this.addr,t),He(e,t)}}function Cg(i,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z)&&(i.uniform3ui(this.addr,t.x,t.y,t.z),e[0]=t.x,e[1]=t.y,e[2]=t.z);else{if(Ve(e,t))return;i.uniform3uiv(this.addr,t),He(e,t)}}function Rg(i,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z||e[3]!==t.w)&&(i.uniform4ui(this.addr,t.x,t.y,t.z,t.w),e[0]=t.x,e[1]=t.y,e[2]=t.z,e[3]=t.w);else{if(Ve(e,t))return;i.uniform4uiv(this.addr,t),He(e,t)}}function Pg(i,t,e){const n=this.cache,s=e.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s);let r;this.type===i.SAMPLER_2D_SHADOW?(Rc.compareFunction=zh,r=Rc):r=Qh,e.setTexture2D(t||r,s)}function Lg(i,t,e){const n=this.cache,s=e.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),e.setTexture3D(t||eu,s)}function Ng(i,t,e){const n=this.cache,s=e.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),e.setTextureCube(t||nu,s)}function Dg(i,t,e){const n=this.cache,s=e.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),e.setTexture2DArray(t||tu,s)}function Ig(i){switch(i){case 5126:return mg;case 35664:return gg;case 35665:return _g;case 35666:return vg;case 35674:return xg;case 35675:return yg;case 35676:return Mg;case 5124:case 35670:return Sg;case 35667:case 35671:return bg;case 35668:case 35672:return Eg;case 35669:case 35673:return Tg;case 5125:return wg;case 36294:return Ag;case 36295:return Cg;case 36296:return Rg;case 35678:case 36198:case 36298:case 36306:case 35682:return Pg;case 35679:case 36299:case 36307:return Lg;case 35680:case 36300:case 36308:case 36293:return Ng;case 36289:case 36303:case 36311:case 36292:return Dg}}function Ug(i,t){i.uniform1fv(this.addr,t)}function Fg(i,t){const e=Us(t,this.size,2);i.uniform2fv(this.addr,e)}function Og(i,t){const e=Us(t,this.size,3);i.uniform3fv(this.addr,e)}function kg(i,t){const e=Us(t,this.size,4);i.uniform4fv(this.addr,e)}function Bg(i,t){const e=Us(t,this.size,4);i.uniformMatrix2fv(this.addr,!1,e)}function zg(i,t){const e=Us(t,this.size,9);i.uniformMatrix3fv(this.addr,!1,e)}function Vg(i,t){const e=Us(t,this.size,16);i.uniformMatrix4fv(this.addr,!1,e)}function Hg(i,t){i.uniform1iv(this.addr,t)}function Gg(i,t){i.uniform2iv(this.addr,t)}function Wg(i,t){i.uniform3iv(this.addr,t)}function Xg(i,t){i.uniform4iv(this.addr,t)}function qg(i,t){i.uniform1uiv(this.addr,t)}function jg(i,t){i.uniform2uiv(this.addr,t)}function Yg(i,t){i.uniform3uiv(this.addr,t)}function $g(i,t){i.uniform4uiv(this.addr,t)}function Kg(i,t,e){const n=this.cache,s=t.length,r=wa(e,s);Ve(n,r)||(i.uniform1iv(this.addr,r),He(n,r));for(let a=0;a!==s;++a)e.setTexture2D(t[a]||Qh,r[a])}function Jg(i,t,e){const n=this.cache,s=t.length,r=wa(e,s);Ve(n,r)||(i.uniform1iv(this.addr,r),He(n,r));for(let a=0;a!==s;++a)e.setTexture3D(t[a]||eu,r[a])}function Zg(i,t,e){const n=this.cache,s=t.length,r=wa(e,s);Ve(n,r)||(i.uniform1iv(this.addr,r),He(n,r));for(let a=0;a!==s;++a)e.setTextureCube(t[a]||nu,r[a])}function Qg(i,t,e){const n=this.cache,s=t.length,r=wa(e,s);Ve(n,r)||(i.uniform1iv(this.addr,r),He(n,r));for(let a=0;a!==s;++a)e.setTexture2DArray(t[a]||tu,r[a])}function t_(i){switch(i){case 5126:return Ug;case 35664:return Fg;case 35665:return Og;case 35666:return kg;case 35674:return Bg;case 35675:return zg;case 35676:return Vg;case 5124:case 35670:return Hg;case 35667:case 35671:return Gg;case 35668:case 35672:return Wg;case 35669:case 35673:return Xg;case 5125:return qg;case 36294:return jg;case 36295:return Yg;case 36296:return $g;case 35678:case 36198:case 36298:case 36306:case 35682:return Kg;case 35679:case 36299:case 36307:return Jg;case 35680:case 36300:case 36308:case 36293:return Zg;case 36289:case 36303:case 36311:case 36292:return Qg}}class e_{constructor(t,e,n){this.id=t,this.addr=n,this.cache=[],this.type=e.type,this.setValue=Ig(e.type)}}class n_{constructor(t,e,n){this.id=t,this.addr=n,this.cache=[],this.type=e.type,this.size=e.size,this.setValue=t_(e.type)}}class i_{constructor(t){this.id=t,this.seq=[],this.map={}}setValue(t,e,n){const s=this.seq;for(let r=0,a=s.length;r!==a;++r){const o=s[r];o.setValue(t,e[o.id],n)}}}const lo=/(\w+)(\])?(\[|\.)?/g;function Uc(i,t){i.seq.push(t),i.map[t.id]=t}function s_(i,t,e){const n=i.name,s=n.length;for(lo.lastIndex=0;;){const r=lo.exec(n),a=lo.lastIndex;let o=r[1];const c=r[2]==="]",l=r[3];if(c&&(o=o|0),l===void 0||l==="["&&a+2===s){Uc(e,l===void 0?new e_(o,i,t):new n_(o,i,t));break}else{let u=e.map[o];u===void 0&&(u=new i_(o),Uc(e,u)),e=u}}}class ca{constructor(t,e){this.seq=[],this.map={};const n=t.getProgramParameter(e,t.ACTIVE_UNIFORMS);for(let s=0;s<n;++s){const r=t.getActiveUniform(e,s),a=t.getUniformLocation(e,r.name);s_(r,a,this)}}setValue(t,e,n,s){const r=this.map[e];r!==void 0&&r.setValue(t,n,s)}setOptional(t,e,n){const s=e[n];s!==void 0&&this.setValue(t,n,s)}static upload(t,e,n,s){for(let r=0,a=e.length;r!==a;++r){const o=e[r],c=n[o.id];c.needsUpdate!==!1&&o.setValue(t,c.value,s)}}static seqWithValue(t,e){const n=[];for(let s=0,r=t.length;s!==r;++s){const a=t[s];a.id in e&&n.push(a)}return n}}function Fc(i,t,e){const n=i.createShader(t);return i.shaderSource(n,e),i.compileShader(n),n}const r_=37297;let a_=0;function o_(i,t){const e=i.split(`
`),n=[],s=Math.max(t-6,0),r=Math.min(t+6,e.length);for(let a=s;a<r;a++){const o=a+1;n.push(`${o===t?">":" "} ${o}: ${e[a]}`)}return n.join(`
`)}const Oc=new ie;function l_(i){le._getMatrix(Oc,le.workingColorSpace,i);const t=`mat3( ${Oc.elements.map(e=>e.toFixed(4))} )`;switch(le.getTransfer(i)){case Ea:return[t,"LinearTransferOETF"];case Ee:return[t,"sRGBTransferOETF"];default:return console.warn("THREE.WebGLProgram: Unsupported color space: ",i),[t,"LinearTransferOETF"]}}function kc(i,t,e){const n=i.getShaderParameter(t,i.COMPILE_STATUS),s=i.getShaderInfoLog(t).trim();if(n&&s==="")return"";const r=/ERROR: 0:(\d+)/.exec(s);if(r){const a=parseInt(r[1]);return e.toUpperCase()+`

`+s+`

`+o_(i.getShaderSource(t),a)}else return s}function c_(i,t){const e=l_(t);return[`vec4 ${i}( vec4 value ) {`,`	return ${e[1]}( vec4( value.rgb * ${e[0]}, value.a ) );`,"}"].join(`
`)}function h_(i,t){let e;switch(t){case hd:e="Linear";break;case ud:e="Reinhard";break;case dd:e="Cineon";break;case Ch:e="ACESFilmic";break;case pd:e="AgX";break;case md:e="Neutral";break;case fd:e="Custom";break;default:console.warn("THREE.WebGLProgram: Unsupported toneMapping:",t),e="Linear"}return"vec3 "+i+"( vec3 color ) { return "+e+"ToneMapping( color ); }"}const Gr=new F;function u_(){le.getLuminanceCoefficients(Gr);const i=Gr.x.toFixed(4),t=Gr.y.toFixed(4),e=Gr.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${i}, ${t}, ${e} );`,"	return dot( weights, rgb );","}"].join(`
`)}function d_(i){return[i.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",i.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter($s).join(`
`)}function f_(i){const t=[];for(const e in i){const n=i[e];n!==!1&&t.push("#define "+e+" "+n)}return t.join(`
`)}function p_(i,t){const e={},n=i.getProgramParameter(t,i.ACTIVE_ATTRIBUTES);for(let s=0;s<n;s++){const r=i.getActiveAttrib(t,s),a=r.name;let o=1;r.type===i.FLOAT_MAT2&&(o=2),r.type===i.FLOAT_MAT3&&(o=3),r.type===i.FLOAT_MAT4&&(o=4),e[a]={type:r.type,location:i.getAttribLocation(t,a),locationSize:o}}return e}function $s(i){return i!==""}function Bc(i,t){const e=t.numSpotLightShadows+t.numSpotLightMaps-t.numSpotLightShadowsWithMaps;return i.replace(/NUM_DIR_LIGHTS/g,t.numDirLights).replace(/NUM_SPOT_LIGHTS/g,t.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,t.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,e).replace(/NUM_RECT_AREA_LIGHTS/g,t.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,t.numPointLights).replace(/NUM_HEMI_LIGHTS/g,t.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,t.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,t.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,t.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,t.numPointLightShadows)}function zc(i,t){return i.replace(/NUM_CLIPPING_PLANES/g,t.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,t.numClippingPlanes-t.numClipIntersection)}const m_=/^[ \t]*#include +<([\w\d./]+)>/gm;function hl(i){return i.replace(m_,__)}const g_=new Map;function __(i,t){let e=se[t];if(e===void 0){const n=g_.get(t);if(n!==void 0)e=se[n],console.warn('THREE.WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',t,n);else throw new Error("Can not resolve #include <"+t+">")}return hl(e)}const v_=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function Vc(i){return i.replace(v_,x_)}function x_(i,t,e,n){let s="";for(let r=parseInt(t);r<parseInt(e);r++)s+=n.replace(/\[\s*i\s*\]/g,"[ "+r+" ]").replace(/UNROLLED_LOOP_INDEX/g,r);return s}function Hc(i){let t=`precision ${i.precision} float;
	precision ${i.precision} int;
	precision ${i.precision} sampler2D;
	precision ${i.precision} samplerCube;
	precision ${i.precision} sampler3D;
	precision ${i.precision} sampler2DArray;
	precision ${i.precision} sampler2DShadow;
	precision ${i.precision} samplerCubeShadow;
	precision ${i.precision} sampler2DArrayShadow;
	precision ${i.precision} isampler2D;
	precision ${i.precision} isampler3D;
	precision ${i.precision} isamplerCube;
	precision ${i.precision} isampler2DArray;
	precision ${i.precision} usampler2D;
	precision ${i.precision} usampler3D;
	precision ${i.precision} usamplerCube;
	precision ${i.precision} usampler2DArray;
	`;return i.precision==="highp"?t+=`
#define HIGH_PRECISION`:i.precision==="mediump"?t+=`
#define MEDIUM_PRECISION`:i.precision==="lowp"&&(t+=`
#define LOW_PRECISION`),t}function y_(i){let t="SHADOWMAP_TYPE_BASIC";return i.shadowMapType===Ah?t="SHADOWMAP_TYPE_PCF":i.shadowMapType===Gu?t="SHADOWMAP_TYPE_PCF_SOFT":i.shadowMapType===Qn&&(t="SHADOWMAP_TYPE_VSM"),t}function M_(i){let t="ENVMAP_TYPE_CUBE";if(i.envMap)switch(i.envMapMode){case Ms:case Ss:t="ENVMAP_TYPE_CUBE";break;case Sa:t="ENVMAP_TYPE_CUBE_UV";break}return t}function S_(i){let t="ENVMAP_MODE_REFLECTION";if(i.envMap)switch(i.envMapMode){case Ss:t="ENVMAP_MODE_REFRACTION";break}return t}function b_(i){let t="ENVMAP_BLENDING_NONE";if(i.envMap)switch(i.combine){case Ma:t="ENVMAP_BLENDING_MULTIPLY";break;case ld:t="ENVMAP_BLENDING_MIX";break;case cd:t="ENVMAP_BLENDING_ADD";break}return t}function E_(i){const t=i.envMapCubeUVHeight;if(t===null)return null;const e=Math.log2(t)-2,n=1/t;return{texelWidth:1/(3*Math.max(Math.pow(2,e),112)),texelHeight:n,maxMip:e}}function T_(i,t,e,n){const s=i.getContext(),r=e.defines;let a=e.vertexShader,o=e.fragmentShader;const c=y_(e),l=M_(e),h=S_(e),u=b_(e),d=E_(e),p=d_(e),g=f_(r),_=s.createProgram();let m,f,R=e.glslVersion?"#version "+e.glslVersion+`
`:"";e.isRawShaderMaterial?(m=["#define SHADER_TYPE "+e.shaderType,"#define SHADER_NAME "+e.shaderName,g].filter($s).join(`
`),m.length>0&&(m+=`
`),f=["#define SHADER_TYPE "+e.shaderType,"#define SHADER_NAME "+e.shaderName,g].filter($s).join(`
`),f.length>0&&(f+=`
`)):(m=[Hc(e),"#define SHADER_TYPE "+e.shaderType,"#define SHADER_NAME "+e.shaderName,g,e.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",e.batching?"#define USE_BATCHING":"",e.batchingColor?"#define USE_BATCHING_COLOR":"",e.instancing?"#define USE_INSTANCING":"",e.instancingColor?"#define USE_INSTANCING_COLOR":"",e.instancingMorph?"#define USE_INSTANCING_MORPH":"",e.useFog&&e.fog?"#define USE_FOG":"",e.useFog&&e.fogExp2?"#define FOG_EXP2":"",e.map?"#define USE_MAP":"",e.envMap?"#define USE_ENVMAP":"",e.envMap?"#define "+h:"",e.lightMap?"#define USE_LIGHTMAP":"",e.aoMap?"#define USE_AOMAP":"",e.bumpMap?"#define USE_BUMPMAP":"",e.normalMap?"#define USE_NORMALMAP":"",e.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",e.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",e.displacementMap?"#define USE_DISPLACEMENTMAP":"",e.emissiveMap?"#define USE_EMISSIVEMAP":"",e.anisotropy?"#define USE_ANISOTROPY":"",e.anisotropyMap?"#define USE_ANISOTROPYMAP":"",e.clearcoatMap?"#define USE_CLEARCOATMAP":"",e.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",e.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",e.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",e.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",e.specularMap?"#define USE_SPECULARMAP":"",e.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",e.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",e.roughnessMap?"#define USE_ROUGHNESSMAP":"",e.metalnessMap?"#define USE_METALNESSMAP":"",e.alphaMap?"#define USE_ALPHAMAP":"",e.alphaHash?"#define USE_ALPHAHASH":"",e.transmission?"#define USE_TRANSMISSION":"",e.transmissionMap?"#define USE_TRANSMISSIONMAP":"",e.thicknessMap?"#define USE_THICKNESSMAP":"",e.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",e.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",e.mapUv?"#define MAP_UV "+e.mapUv:"",e.alphaMapUv?"#define ALPHAMAP_UV "+e.alphaMapUv:"",e.lightMapUv?"#define LIGHTMAP_UV "+e.lightMapUv:"",e.aoMapUv?"#define AOMAP_UV "+e.aoMapUv:"",e.emissiveMapUv?"#define EMISSIVEMAP_UV "+e.emissiveMapUv:"",e.bumpMapUv?"#define BUMPMAP_UV "+e.bumpMapUv:"",e.normalMapUv?"#define NORMALMAP_UV "+e.normalMapUv:"",e.displacementMapUv?"#define DISPLACEMENTMAP_UV "+e.displacementMapUv:"",e.metalnessMapUv?"#define METALNESSMAP_UV "+e.metalnessMapUv:"",e.roughnessMapUv?"#define ROUGHNESSMAP_UV "+e.roughnessMapUv:"",e.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+e.anisotropyMapUv:"",e.clearcoatMapUv?"#define CLEARCOATMAP_UV "+e.clearcoatMapUv:"",e.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+e.clearcoatNormalMapUv:"",e.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+e.clearcoatRoughnessMapUv:"",e.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+e.iridescenceMapUv:"",e.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+e.iridescenceThicknessMapUv:"",e.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+e.sheenColorMapUv:"",e.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+e.sheenRoughnessMapUv:"",e.specularMapUv?"#define SPECULARMAP_UV "+e.specularMapUv:"",e.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+e.specularColorMapUv:"",e.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+e.specularIntensityMapUv:"",e.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+e.transmissionMapUv:"",e.thicknessMapUv?"#define THICKNESSMAP_UV "+e.thicknessMapUv:"",e.vertexTangents&&e.flatShading===!1?"#define USE_TANGENT":"",e.vertexColors?"#define USE_COLOR":"",e.vertexAlphas?"#define USE_COLOR_ALPHA":"",e.vertexUv1s?"#define USE_UV1":"",e.vertexUv2s?"#define USE_UV2":"",e.vertexUv3s?"#define USE_UV3":"",e.pointsUvs?"#define USE_POINTS_UV":"",e.flatShading?"#define FLAT_SHADED":"",e.skinning?"#define USE_SKINNING":"",e.morphTargets?"#define USE_MORPHTARGETS":"",e.morphNormals&&e.flatShading===!1?"#define USE_MORPHNORMALS":"",e.morphColors?"#define USE_MORPHCOLORS":"",e.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+e.morphTextureStride:"",e.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+e.morphTargetsCount:"",e.doubleSided?"#define DOUBLE_SIDED":"",e.flipSided?"#define FLIP_SIDED":"",e.shadowMapEnabled?"#define USE_SHADOWMAP":"",e.shadowMapEnabled?"#define "+c:"",e.sizeAttenuation?"#define USE_SIZEATTENUATION":"",e.numLightProbes>0?"#define USE_LIGHT_PROBES":"",e.logarithmicDepthBuffer?"#define USE_LOGDEPTHBUF":"",e.reverseDepthBuffer?"#define USE_REVERSEDEPTHBUF":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter($s).join(`
`),f=[Hc(e),"#define SHADER_TYPE "+e.shaderType,"#define SHADER_NAME "+e.shaderName,g,e.useFog&&e.fog?"#define USE_FOG":"",e.useFog&&e.fogExp2?"#define FOG_EXP2":"",e.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",e.map?"#define USE_MAP":"",e.matcap?"#define USE_MATCAP":"",e.envMap?"#define USE_ENVMAP":"",e.envMap?"#define "+l:"",e.envMap?"#define "+h:"",e.envMap?"#define "+u:"",d?"#define CUBEUV_TEXEL_WIDTH "+d.texelWidth:"",d?"#define CUBEUV_TEXEL_HEIGHT "+d.texelHeight:"",d?"#define CUBEUV_MAX_MIP "+d.maxMip+".0":"",e.lightMap?"#define USE_LIGHTMAP":"",e.aoMap?"#define USE_AOMAP":"",e.bumpMap?"#define USE_BUMPMAP":"",e.normalMap?"#define USE_NORMALMAP":"",e.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",e.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",e.emissiveMap?"#define USE_EMISSIVEMAP":"",e.anisotropy?"#define USE_ANISOTROPY":"",e.anisotropyMap?"#define USE_ANISOTROPYMAP":"",e.clearcoat?"#define USE_CLEARCOAT":"",e.clearcoatMap?"#define USE_CLEARCOATMAP":"",e.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",e.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",e.dispersion?"#define USE_DISPERSION":"",e.iridescence?"#define USE_IRIDESCENCE":"",e.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",e.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",e.specularMap?"#define USE_SPECULARMAP":"",e.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",e.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",e.roughnessMap?"#define USE_ROUGHNESSMAP":"",e.metalnessMap?"#define USE_METALNESSMAP":"",e.alphaMap?"#define USE_ALPHAMAP":"",e.alphaTest?"#define USE_ALPHATEST":"",e.alphaHash?"#define USE_ALPHAHASH":"",e.sheen?"#define USE_SHEEN":"",e.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",e.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",e.transmission?"#define USE_TRANSMISSION":"",e.transmissionMap?"#define USE_TRANSMISSIONMAP":"",e.thicknessMap?"#define USE_THICKNESSMAP":"",e.vertexTangents&&e.flatShading===!1?"#define USE_TANGENT":"",e.vertexColors||e.instancingColor||e.batchingColor?"#define USE_COLOR":"",e.vertexAlphas?"#define USE_COLOR_ALPHA":"",e.vertexUv1s?"#define USE_UV1":"",e.vertexUv2s?"#define USE_UV2":"",e.vertexUv3s?"#define USE_UV3":"",e.pointsUvs?"#define USE_POINTS_UV":"",e.gradientMap?"#define USE_GRADIENTMAP":"",e.flatShading?"#define FLAT_SHADED":"",e.doubleSided?"#define DOUBLE_SIDED":"",e.flipSided?"#define FLIP_SIDED":"",e.shadowMapEnabled?"#define USE_SHADOWMAP":"",e.shadowMapEnabled?"#define "+c:"",e.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",e.numLightProbes>0?"#define USE_LIGHT_PROBES":"",e.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",e.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",e.logarithmicDepthBuffer?"#define USE_LOGDEPTHBUF":"",e.reverseDepthBuffer?"#define USE_REVERSEDEPTHBUF":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",e.toneMapping!==bi?"#define TONE_MAPPING":"",e.toneMapping!==bi?se.tonemapping_pars_fragment:"",e.toneMapping!==bi?h_("toneMapping",e.toneMapping):"",e.dithering?"#define DITHERING":"",e.opaque?"#define OPAQUE":"",se.colorspace_pars_fragment,c_("linearToOutputTexel",e.outputColorSpace),u_(),e.useDepthPacking?"#define DEPTH_PACKING "+e.depthPacking:"",`
`].filter($s).join(`
`)),a=hl(a),a=Bc(a,e),a=zc(a,e),o=hl(o),o=Bc(o,e),o=zc(o,e),a=Vc(a),o=Vc(o),e.isRawShaderMaterial!==!0&&(R=`#version 300 es
`,m=[p,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+m,f=["#define varying in",e.glslVersion===tc?"":"layout(location = 0) out highp vec4 pc_fragColor;",e.glslVersion===tc?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+f);const C=R+m+a,x=R+f+o,N=Fc(s,s.VERTEX_SHADER,C),P=Fc(s,s.FRAGMENT_SHADER,x);s.attachShader(_,N),s.attachShader(_,P),e.index0AttributeName!==void 0?s.bindAttribLocation(_,0,e.index0AttributeName):e.morphTargets===!0&&s.bindAttribLocation(_,0,"position"),s.linkProgram(_);function L(U){if(i.debug.checkShaderErrors){const k=s.getProgramInfoLog(_).trim(),H=s.getShaderInfoLog(N).trim(),J=s.getShaderInfoLog(P).trim();let Q=!0,O=!0;if(s.getProgramParameter(_,s.LINK_STATUS)===!1)if(Q=!1,typeof i.debug.onShaderError=="function")i.debug.onShaderError(s,_,N,P);else{const tt=kc(s,N,"vertex"),W=kc(s,P,"fragment");console.error("THREE.WebGLProgram: Shader Error "+s.getError()+" - VALIDATE_STATUS "+s.getProgramParameter(_,s.VALIDATE_STATUS)+`

Material Name: `+U.name+`
Material Type: `+U.type+`

Program Info Log: `+k+`
`+tt+`
`+W)}else k!==""?console.warn("THREE.WebGLProgram: Program Info Log:",k):(H===""||J==="")&&(O=!1);O&&(U.diagnostics={runnable:Q,programLog:k,vertexShader:{log:H,prefix:m},fragmentShader:{log:J,prefix:f}})}s.deleteShader(N),s.deleteShader(P),I=new ca(s,_),M=p_(s,_)}let I;this.getUniforms=function(){return I===void 0&&L(this),I};let M;this.getAttributes=function(){return M===void 0&&L(this),M};let S=e.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return S===!1&&(S=s.getProgramParameter(_,r_)),S},this.destroy=function(){n.releaseStatesOfProgram(this),s.deleteProgram(_),this.program=void 0},this.type=e.shaderType,this.name=e.shaderName,this.id=a_++,this.cacheKey=t,this.usedTimes=1,this.program=_,this.vertexShader=N,this.fragmentShader=P,this}let w_=0;class A_{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(t){const e=t.vertexShader,n=t.fragmentShader,s=this._getShaderStage(e),r=this._getShaderStage(n),a=this._getShaderCacheForMaterial(t);return a.has(s)===!1&&(a.add(s),s.usedTimes++),a.has(r)===!1&&(a.add(r),r.usedTimes++),this}remove(t){const e=this.materialCache.get(t);for(const n of e)n.usedTimes--,n.usedTimes===0&&this.shaderCache.delete(n.code);return this.materialCache.delete(t),this}getVertexShaderID(t){return this._getShaderStage(t.vertexShader).id}getFragmentShaderID(t){return this._getShaderStage(t.fragmentShader).id}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(t){const e=this.materialCache;let n=e.get(t);return n===void 0&&(n=new Set,e.set(t,n)),n}_getShaderStage(t){const e=this.shaderCache;let n=e.get(t);return n===void 0&&(n=new C_(t),e.set(t,n)),n}}class C_{constructor(t){this.id=w_++,this.code=t,this.usedTimes=0}}function R_(i,t,e,n,s,r,a){const o=new Wh,c=new A_,l=new Set,h=[],u=s.logarithmicDepthBuffer,d=s.vertexTextures;let p=s.precision;const g={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distanceRGBA",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function _(M){return l.add(M),M===0?"uv":`uv${M}`}function m(M,S,U,k,H){const J=k.fog,Q=H.geometry,O=M.isMeshStandardMaterial?k.environment:null,tt=(M.isMeshStandardMaterial?e:t).get(M.envMap||O),W=tt&&tt.mapping===Sa?tt.image.height:null,ot=g[M.type];M.precision!==null&&(p=s.getMaxPrecision(M.precision),p!==M.precision&&console.warn("THREE.WebGLProgram.getParameters:",M.precision,"not supported, using",p,"instead."));const ft=Q.morphAttributes.position||Q.morphAttributes.normal||Q.morphAttributes.color,pt=ft!==void 0?ft.length:0;let Ct=0;Q.morphAttributes.position!==void 0&&(Ct=1),Q.morphAttributes.normal!==void 0&&(Ct=2),Q.morphAttributes.color!==void 0&&(Ct=3);let Et,X,K,et;if(ot){const ne=kn[ot];Et=ne.vertexShader,X=ne.fragmentShader}else Et=M.vertexShader,X=M.fragmentShader,c.update(M),K=c.getVertexShaderID(M),et=c.getFragmentShaderID(M);const nt=i.getRenderTarget(),_t=i.state.buffers.depth.getReversed(),zt=H.isInstancedMesh===!0,Zt=H.isBatchedMesh===!0,Ce=!!M.map,ue=!!M.matcap,Ne=!!tt,G=!!M.aoMap,nn=!!M.lightMap,re=!!M.bumpMap,ae=!!M.normalMap,Wt=!!M.displacementMap,we=!!M.emissiveMap,Ht=!!M.metalnessMap,D=!!M.roughnessMap,b=M.anisotropy>0,q=M.clearcoat>0,rt=M.dispersion>0,lt=M.iridescence>0,st=M.sheen>0,Ft=M.transmission>0,yt=b&&!!M.anisotropyMap,At=q&&!!M.clearcoatMap,de=q&&!!M.clearcoatNormalMap,dt=q&&!!M.clearcoatRoughnessMap,Rt=lt&&!!M.iridescenceMap,Xt=lt&&!!M.iridescenceThicknessMap,jt=st&&!!M.sheenColorMap,Pt=st&&!!M.sheenRoughnessMap,ce=!!M.specularMap,ee=!!M.specularColorMap,ye=!!M.specularIntensityMap,z=Ft&&!!M.transmissionMap,vt=Ft&&!!M.thicknessMap,Z=!!M.gradientMap,at=!!M.alphaMap,bt=M.alphaTest>0,Mt=!!M.alphaHash,Qt=!!M.extensions;let Re=bi;M.toneMapped&&(nt===null||nt.isXRRenderTarget===!0)&&(Re=i.toneMapping);const Ge={shaderID:ot,shaderType:M.type,shaderName:M.name,vertexShader:Et,fragmentShader:X,defines:M.defines,customVertexShaderID:K,customFragmentShaderID:et,isRawShaderMaterial:M.isRawShaderMaterial===!0,glslVersion:M.glslVersion,precision:p,batching:Zt,batchingColor:Zt&&H._colorsTexture!==null,instancing:zt,instancingColor:zt&&H.instanceColor!==null,instancingMorph:zt&&H.morphTexture!==null,supportsVertexTextures:d,outputColorSpace:nt===null?i.outputColorSpace:nt.isXRRenderTarget===!0?nt.texture.colorSpace:Ds,alphaToCoverage:!!M.alphaToCoverage,map:Ce,matcap:ue,envMap:Ne,envMapMode:Ne&&tt.mapping,envMapCubeUVHeight:W,aoMap:G,lightMap:nn,bumpMap:re,normalMap:ae,displacementMap:d&&Wt,emissiveMap:we,normalMapObjectSpace:ae&&M.normalMapType===Md,normalMapTangentSpace:ae&&M.normalMapType===ba,metalnessMap:Ht,roughnessMap:D,anisotropy:b,anisotropyMap:yt,clearcoat:q,clearcoatMap:At,clearcoatNormalMap:de,clearcoatRoughnessMap:dt,dispersion:rt,iridescence:lt,iridescenceMap:Rt,iridescenceThicknessMap:Xt,sheen:st,sheenColorMap:jt,sheenRoughnessMap:Pt,specularMap:ce,specularColorMap:ee,specularIntensityMap:ye,transmission:Ft,transmissionMap:z,thicknessMap:vt,gradientMap:Z,opaque:M.transparent===!1&&M.blending===ms&&M.alphaToCoverage===!1,alphaMap:at,alphaTest:bt,alphaHash:Mt,combine:M.combine,mapUv:Ce&&_(M.map.channel),aoMapUv:G&&_(M.aoMap.channel),lightMapUv:nn&&_(M.lightMap.channel),bumpMapUv:re&&_(M.bumpMap.channel),normalMapUv:ae&&_(M.normalMap.channel),displacementMapUv:Wt&&_(M.displacementMap.channel),emissiveMapUv:we&&_(M.emissiveMap.channel),metalnessMapUv:Ht&&_(M.metalnessMap.channel),roughnessMapUv:D&&_(M.roughnessMap.channel),anisotropyMapUv:yt&&_(M.anisotropyMap.channel),clearcoatMapUv:At&&_(M.clearcoatMap.channel),clearcoatNormalMapUv:de&&_(M.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:dt&&_(M.clearcoatRoughnessMap.channel),iridescenceMapUv:Rt&&_(M.iridescenceMap.channel),iridescenceThicknessMapUv:Xt&&_(M.iridescenceThicknessMap.channel),sheenColorMapUv:jt&&_(M.sheenColorMap.channel),sheenRoughnessMapUv:Pt&&_(M.sheenRoughnessMap.channel),specularMapUv:ce&&_(M.specularMap.channel),specularColorMapUv:ee&&_(M.specularColorMap.channel),specularIntensityMapUv:ye&&_(M.specularIntensityMap.channel),transmissionMapUv:z&&_(M.transmissionMap.channel),thicknessMapUv:vt&&_(M.thicknessMap.channel),alphaMapUv:at&&_(M.alphaMap.channel),vertexTangents:!!Q.attributes.tangent&&(ae||b),vertexColors:M.vertexColors,vertexAlphas:M.vertexColors===!0&&!!Q.attributes.color&&Q.attributes.color.itemSize===4,pointsUvs:H.isPoints===!0&&!!Q.attributes.uv&&(Ce||at),fog:!!J,useFog:M.fog===!0,fogExp2:!!J&&J.isFogExp2,flatShading:M.flatShading===!0,sizeAttenuation:M.sizeAttenuation===!0,logarithmicDepthBuffer:u,reverseDepthBuffer:_t,skinning:H.isSkinnedMesh===!0,morphTargets:Q.morphAttributes.position!==void 0,morphNormals:Q.morphAttributes.normal!==void 0,morphColors:Q.morphAttributes.color!==void 0,morphTargetsCount:pt,morphTextureStride:Ct,numDirLights:S.directional.length,numPointLights:S.point.length,numSpotLights:S.spot.length,numSpotLightMaps:S.spotLightMap.length,numRectAreaLights:S.rectArea.length,numHemiLights:S.hemi.length,numDirLightShadows:S.directionalShadowMap.length,numPointLightShadows:S.pointShadowMap.length,numSpotLightShadows:S.spotShadowMap.length,numSpotLightShadowsWithMaps:S.numSpotLightShadowsWithMaps,numLightProbes:S.numLightProbes,numClippingPlanes:a.numPlanes,numClipIntersection:a.numIntersection,dithering:M.dithering,shadowMapEnabled:i.shadowMap.enabled&&U.length>0,shadowMapType:i.shadowMap.type,toneMapping:Re,decodeVideoTexture:Ce&&M.map.isVideoTexture===!0&&le.getTransfer(M.map.colorSpace)===Ee,decodeVideoTextureEmissive:we&&M.emissiveMap.isVideoTexture===!0&&le.getTransfer(M.emissiveMap.colorSpace)===Ee,premultipliedAlpha:M.premultipliedAlpha,doubleSided:M.side===Pn,flipSided:M.side===ln,useDepthPacking:M.depthPacking>=0,depthPacking:M.depthPacking||0,index0AttributeName:M.index0AttributeName,extensionClipCullDistance:Qt&&M.extensions.clipCullDistance===!0&&n.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(Qt&&M.extensions.multiDraw===!0||Zt)&&n.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:n.has("KHR_parallel_shader_compile"),customProgramCacheKey:M.customProgramCacheKey()};return Ge.vertexUv1s=l.has(1),Ge.vertexUv2s=l.has(2),Ge.vertexUv3s=l.has(3),l.clear(),Ge}function f(M){const S=[];if(M.shaderID?S.push(M.shaderID):(S.push(M.customVertexShaderID),S.push(M.customFragmentShaderID)),M.defines!==void 0)for(const U in M.defines)S.push(U),S.push(M.defines[U]);return M.isRawShaderMaterial===!1&&(R(S,M),C(S,M),S.push(i.outputColorSpace)),S.push(M.customProgramCacheKey),S.join()}function R(M,S){M.push(S.precision),M.push(S.outputColorSpace),M.push(S.envMapMode),M.push(S.envMapCubeUVHeight),M.push(S.mapUv),M.push(S.alphaMapUv),M.push(S.lightMapUv),M.push(S.aoMapUv),M.push(S.bumpMapUv),M.push(S.normalMapUv),M.push(S.displacementMapUv),M.push(S.emissiveMapUv),M.push(S.metalnessMapUv),M.push(S.roughnessMapUv),M.push(S.anisotropyMapUv),M.push(S.clearcoatMapUv),M.push(S.clearcoatNormalMapUv),M.push(S.clearcoatRoughnessMapUv),M.push(S.iridescenceMapUv),M.push(S.iridescenceThicknessMapUv),M.push(S.sheenColorMapUv),M.push(S.sheenRoughnessMapUv),M.push(S.specularMapUv),M.push(S.specularColorMapUv),M.push(S.specularIntensityMapUv),M.push(S.transmissionMapUv),M.push(S.thicknessMapUv),M.push(S.combine),M.push(S.fogExp2),M.push(S.sizeAttenuation),M.push(S.morphTargetsCount),M.push(S.morphAttributeCount),M.push(S.numDirLights),M.push(S.numPointLights),M.push(S.numSpotLights),M.push(S.numSpotLightMaps),M.push(S.numHemiLights),M.push(S.numRectAreaLights),M.push(S.numDirLightShadows),M.push(S.numPointLightShadows),M.push(S.numSpotLightShadows),M.push(S.numSpotLightShadowsWithMaps),M.push(S.numLightProbes),M.push(S.shadowMapType),M.push(S.toneMapping),M.push(S.numClippingPlanes),M.push(S.numClipIntersection),M.push(S.depthPacking)}function C(M,S){o.disableAll(),S.supportsVertexTextures&&o.enable(0),S.instancing&&o.enable(1),S.instancingColor&&o.enable(2),S.instancingMorph&&o.enable(3),S.matcap&&o.enable(4),S.envMap&&o.enable(5),S.normalMapObjectSpace&&o.enable(6),S.normalMapTangentSpace&&o.enable(7),S.clearcoat&&o.enable(8),S.iridescence&&o.enable(9),S.alphaTest&&o.enable(10),S.vertexColors&&o.enable(11),S.vertexAlphas&&o.enable(12),S.vertexUv1s&&o.enable(13),S.vertexUv2s&&o.enable(14),S.vertexUv3s&&o.enable(15),S.vertexTangents&&o.enable(16),S.anisotropy&&o.enable(17),S.alphaHash&&o.enable(18),S.batching&&o.enable(19),S.dispersion&&o.enable(20),S.batchingColor&&o.enable(21),M.push(o.mask),o.disableAll(),S.fog&&o.enable(0),S.useFog&&o.enable(1),S.flatShading&&o.enable(2),S.logarithmicDepthBuffer&&o.enable(3),S.reverseDepthBuffer&&o.enable(4),S.skinning&&o.enable(5),S.morphTargets&&o.enable(6),S.morphNormals&&o.enable(7),S.morphColors&&o.enable(8),S.premultipliedAlpha&&o.enable(9),S.shadowMapEnabled&&o.enable(10),S.doubleSided&&o.enable(11),S.flipSided&&o.enable(12),S.useDepthPacking&&o.enable(13),S.dithering&&o.enable(14),S.transmission&&o.enable(15),S.sheen&&o.enable(16),S.opaque&&o.enable(17),S.pointsUvs&&o.enable(18),S.decodeVideoTexture&&o.enable(19),S.decodeVideoTextureEmissive&&o.enable(20),S.alphaToCoverage&&o.enable(21),M.push(o.mask)}function x(M){const S=g[M.type];let U;if(S){const k=kn[S];U=df.clone(k.uniforms)}else U=M.uniforms;return U}function N(M,S){let U;for(let k=0,H=h.length;k<H;k++){const J=h[k];if(J.cacheKey===S){U=J,++U.usedTimes;break}}return U===void 0&&(U=new T_(i,S,M,r),h.push(U)),U}function P(M){if(--M.usedTimes===0){const S=h.indexOf(M);h[S]=h[h.length-1],h.pop(),M.destroy()}}function L(M){c.remove(M)}function I(){c.dispose()}return{getParameters:m,getProgramCacheKey:f,getUniforms:x,acquireProgram:N,releaseProgram:P,releaseShaderCache:L,programs:h,dispose:I}}function P_(){let i=new WeakMap;function t(a){return i.has(a)}function e(a){let o=i.get(a);return o===void 0&&(o={},i.set(a,o)),o}function n(a){i.delete(a)}function s(a,o,c){i.get(a)[o]=c}function r(){i=new WeakMap}return{has:t,get:e,remove:n,update:s,dispose:r}}function L_(i,t){return i.groupOrder!==t.groupOrder?i.groupOrder-t.groupOrder:i.renderOrder!==t.renderOrder?i.renderOrder-t.renderOrder:i.material.id!==t.material.id?i.material.id-t.material.id:i.z!==t.z?i.z-t.z:i.id-t.id}function Gc(i,t){return i.groupOrder!==t.groupOrder?i.groupOrder-t.groupOrder:i.renderOrder!==t.renderOrder?i.renderOrder-t.renderOrder:i.z!==t.z?t.z-i.z:i.id-t.id}function Wc(){const i=[];let t=0;const e=[],n=[],s=[];function r(){t=0,e.length=0,n.length=0,s.length=0}function a(u,d,p,g,_,m){let f=i[t];return f===void 0?(f={id:u.id,object:u,geometry:d,material:p,groupOrder:g,renderOrder:u.renderOrder,z:_,group:m},i[t]=f):(f.id=u.id,f.object=u,f.geometry=d,f.material=p,f.groupOrder=g,f.renderOrder=u.renderOrder,f.z=_,f.group=m),t++,f}function o(u,d,p,g,_,m){const f=a(u,d,p,g,_,m);p.transmission>0?n.push(f):p.transparent===!0?s.push(f):e.push(f)}function c(u,d,p,g,_,m){const f=a(u,d,p,g,_,m);p.transmission>0?n.unshift(f):p.transparent===!0?s.unshift(f):e.unshift(f)}function l(u,d){e.length>1&&e.sort(u||L_),n.length>1&&n.sort(d||Gc),s.length>1&&s.sort(d||Gc)}function h(){for(let u=t,d=i.length;u<d;u++){const p=i[u];if(p.id===null)break;p.id=null,p.object=null,p.geometry=null,p.material=null,p.group=null}}return{opaque:e,transmissive:n,transparent:s,init:r,push:o,unshift:c,finish:h,sort:l}}function N_(){let i=new WeakMap;function t(n,s){const r=i.get(n);let a;return r===void 0?(a=new Wc,i.set(n,[a])):s>=r.length?(a=new Wc,r.push(a)):a=r[s],a}function e(){i=new WeakMap}return{get:t,dispose:e}}function D_(){const i={};return{get:function(t){if(i[t.id]!==void 0)return i[t.id];let e;switch(t.type){case"DirectionalLight":e={direction:new F,color:new $t};break;case"SpotLight":e={position:new F,direction:new F,color:new $t,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":e={position:new F,color:new $t,distance:0,decay:0};break;case"HemisphereLight":e={direction:new F,skyColor:new $t,groundColor:new $t};break;case"RectAreaLight":e={color:new $t,position:new F,halfWidth:new F,halfHeight:new F};break}return i[t.id]=e,e}}}function I_(){const i={};return{get:function(t){if(i[t.id]!==void 0)return i[t.id];let e;switch(t.type){case"DirectionalLight":e={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Lt};break;case"SpotLight":e={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Lt};break;case"PointLight":e={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Lt,shadowCameraNear:1,shadowCameraFar:1e3};break}return i[t.id]=e,e}}}let U_=0;function F_(i,t){return(t.castShadow?2:0)-(i.castShadow?2:0)+(t.map?1:0)-(i.map?1:0)}function O_(i){const t=new D_,e=I_(),n={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let l=0;l<9;l++)n.probe.push(new F);const s=new F,r=new Kt,a=new Kt;function o(l){let h=0,u=0,d=0;for(let M=0;M<9;M++)n.probe[M].set(0,0,0);let p=0,g=0,_=0,m=0,f=0,R=0,C=0,x=0,N=0,P=0,L=0;l.sort(F_);for(let M=0,S=l.length;M<S;M++){const U=l[M],k=U.color,H=U.intensity,J=U.distance,Q=U.shadow&&U.shadow.map?U.shadow.map.texture:null;if(U.isAmbientLight)h+=k.r*H,u+=k.g*H,d+=k.b*H;else if(U.isLightProbe){for(let O=0;O<9;O++)n.probe[O].addScaledVector(U.sh.coefficients[O],H);L++}else if(U.isDirectionalLight){const O=t.get(U);if(O.color.copy(U.color).multiplyScalar(U.intensity),U.castShadow){const tt=U.shadow,W=e.get(U);W.shadowIntensity=tt.intensity,W.shadowBias=tt.bias,W.shadowNormalBias=tt.normalBias,W.shadowRadius=tt.radius,W.shadowMapSize=tt.mapSize,n.directionalShadow[p]=W,n.directionalShadowMap[p]=Q,n.directionalShadowMatrix[p]=U.shadow.matrix,R++}n.directional[p]=O,p++}else if(U.isSpotLight){const O=t.get(U);O.position.setFromMatrixPosition(U.matrixWorld),O.color.copy(k).multiplyScalar(H),O.distance=J,O.coneCos=Math.cos(U.angle),O.penumbraCos=Math.cos(U.angle*(1-U.penumbra)),O.decay=U.decay,n.spot[_]=O;const tt=U.shadow;if(U.map&&(n.spotLightMap[N]=U.map,N++,tt.updateMatrices(U),U.castShadow&&P++),n.spotLightMatrix[_]=tt.matrix,U.castShadow){const W=e.get(U);W.shadowIntensity=tt.intensity,W.shadowBias=tt.bias,W.shadowNormalBias=tt.normalBias,W.shadowRadius=tt.radius,W.shadowMapSize=tt.mapSize,n.spotShadow[_]=W,n.spotShadowMap[_]=Q,x++}_++}else if(U.isRectAreaLight){const O=t.get(U);O.color.copy(k).multiplyScalar(H),O.halfWidth.set(U.width*.5,0,0),O.halfHeight.set(0,U.height*.5,0),n.rectArea[m]=O,m++}else if(U.isPointLight){const O=t.get(U);if(O.color.copy(U.color).multiplyScalar(U.intensity),O.distance=U.distance,O.decay=U.decay,U.castShadow){const tt=U.shadow,W=e.get(U);W.shadowIntensity=tt.intensity,W.shadowBias=tt.bias,W.shadowNormalBias=tt.normalBias,W.shadowRadius=tt.radius,W.shadowMapSize=tt.mapSize,W.shadowCameraNear=tt.camera.near,W.shadowCameraFar=tt.camera.far,n.pointShadow[g]=W,n.pointShadowMap[g]=Q,n.pointShadowMatrix[g]=U.shadow.matrix,C++}n.point[g]=O,g++}else if(U.isHemisphereLight){const O=t.get(U);O.skyColor.copy(U.color).multiplyScalar(H),O.groundColor.copy(U.groundColor).multiplyScalar(H),n.hemi[f]=O,f++}}m>0&&(i.has("OES_texture_float_linear")===!0?(n.rectAreaLTC1=gt.LTC_FLOAT_1,n.rectAreaLTC2=gt.LTC_FLOAT_2):(n.rectAreaLTC1=gt.LTC_HALF_1,n.rectAreaLTC2=gt.LTC_HALF_2)),n.ambient[0]=h,n.ambient[1]=u,n.ambient[2]=d;const I=n.hash;(I.directionalLength!==p||I.pointLength!==g||I.spotLength!==_||I.rectAreaLength!==m||I.hemiLength!==f||I.numDirectionalShadows!==R||I.numPointShadows!==C||I.numSpotShadows!==x||I.numSpotMaps!==N||I.numLightProbes!==L)&&(n.directional.length=p,n.spot.length=_,n.rectArea.length=m,n.point.length=g,n.hemi.length=f,n.directionalShadow.length=R,n.directionalShadowMap.length=R,n.pointShadow.length=C,n.pointShadowMap.length=C,n.spotShadow.length=x,n.spotShadowMap.length=x,n.directionalShadowMatrix.length=R,n.pointShadowMatrix.length=C,n.spotLightMatrix.length=x+N-P,n.spotLightMap.length=N,n.numSpotLightShadowsWithMaps=P,n.numLightProbes=L,I.directionalLength=p,I.pointLength=g,I.spotLength=_,I.rectAreaLength=m,I.hemiLength=f,I.numDirectionalShadows=R,I.numPointShadows=C,I.numSpotShadows=x,I.numSpotMaps=N,I.numLightProbes=L,n.version=U_++)}function c(l,h){let u=0,d=0,p=0,g=0,_=0;const m=h.matrixWorldInverse;for(let f=0,R=l.length;f<R;f++){const C=l[f];if(C.isDirectionalLight){const x=n.directional[u];x.direction.setFromMatrixPosition(C.matrixWorld),s.setFromMatrixPosition(C.target.matrixWorld),x.direction.sub(s),x.direction.transformDirection(m),u++}else if(C.isSpotLight){const x=n.spot[p];x.position.setFromMatrixPosition(C.matrixWorld),x.position.applyMatrix4(m),x.direction.setFromMatrixPosition(C.matrixWorld),s.setFromMatrixPosition(C.target.matrixWorld),x.direction.sub(s),x.direction.transformDirection(m),p++}else if(C.isRectAreaLight){const x=n.rectArea[g];x.position.setFromMatrixPosition(C.matrixWorld),x.position.applyMatrix4(m),a.identity(),r.copy(C.matrixWorld),r.premultiply(m),a.extractRotation(r),x.halfWidth.set(C.width*.5,0,0),x.halfHeight.set(0,C.height*.5,0),x.halfWidth.applyMatrix4(a),x.halfHeight.applyMatrix4(a),g++}else if(C.isPointLight){const x=n.point[d];x.position.setFromMatrixPosition(C.matrixWorld),x.position.applyMatrix4(m),d++}else if(C.isHemisphereLight){const x=n.hemi[_];x.direction.setFromMatrixPosition(C.matrixWorld),x.direction.transformDirection(m),_++}}}return{setup:o,setupView:c,state:n}}function Xc(i){const t=new O_(i),e=[],n=[];function s(h){l.camera=h,e.length=0,n.length=0}function r(h){e.push(h)}function a(h){n.push(h)}function o(){t.setup(e)}function c(h){t.setupView(e,h)}const l={lightsArray:e,shadowsArray:n,camera:null,lights:t,transmissionRenderTarget:{}};return{init:s,state:l,setupLights:o,setupLightsView:c,pushLight:r,pushShadow:a}}function k_(i){let t=new WeakMap;function e(s,r=0){const a=t.get(s);let o;return a===void 0?(o=new Xc(i),t.set(s,[o])):r>=a.length?(o=new Xc(i),a.push(o)):o=a[r],o}function n(){t=new WeakMap}return{get:e,dispose:n}}class B_ extends In{static get type(){return"MeshDepthMaterial"}constructor(t){super(),this.isMeshDepthMaterial=!0,this.depthPacking=xd,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(t)}copy(t){return super.copy(t),this.depthPacking=t.depthPacking,this.map=t.map,this.alphaMap=t.alphaMap,this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this}}class z_ extends In{static get type(){return"MeshDistanceMaterial"}constructor(t){super(),this.isMeshDistanceMaterial=!0,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(t)}copy(t){return super.copy(t),this.map=t.map,this.alphaMap=t.alphaMap,this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this}}const V_=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,H_=`uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
#include <packing>
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = unpackRGBATo2Half( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ) );
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = unpackRGBAToDepth( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ) );
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( squared_mean - mean * mean );
	gl_FragColor = pack2HalfToRGBA( vec2( mean, std_dev ) );
}`;function G_(i,t,e){let n=new bl;const s=new Lt,r=new Lt,a=new ve,o=new B_({depthPacking:yd}),c=new z_,l={},h=e.maxTextureSize,u={[ii]:ln,[ln]:ii,[Pn]:Pn},d=new Ti({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new Lt},radius:{value:4}},vertexShader:V_,fragmentShader:H_}),p=d.clone();p.defines.HORIZONTAL_PASS=1;const g=new Ie;g.setAttribute("position",new gn(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));const _=new Ae(g,d),m=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=Ah;let f=this.type;this.render=function(P,L,I){if(m.enabled===!1||m.autoUpdate===!1&&m.needsUpdate===!1||P.length===0)return;const M=i.getRenderTarget(),S=i.getActiveCubeFace(),U=i.getActiveMipmapLevel(),k=i.state;k.setBlending(Si),k.buffers.color.setClear(1,1,1,1),k.buffers.depth.setTest(!0),k.setScissorTest(!1);const H=f!==Qn&&this.type===Qn,J=f===Qn&&this.type!==Qn;for(let Q=0,O=P.length;Q<O;Q++){const tt=P[Q],W=tt.shadow;if(W===void 0){console.warn("THREE.WebGLShadowMap:",tt,"has no shadow.");continue}if(W.autoUpdate===!1&&W.needsUpdate===!1)continue;s.copy(W.mapSize);const ot=W.getFrameExtents();if(s.multiply(ot),r.copy(W.mapSize),(s.x>h||s.y>h)&&(s.x>h&&(r.x=Math.floor(h/ot.x),s.x=r.x*ot.x,W.mapSize.x=r.x),s.y>h&&(r.y=Math.floor(h/ot.y),s.y=r.y*ot.y,W.mapSize.y=r.y)),W.map===null||H===!0||J===!0){const pt=this.type!==Qn?{minFilter:mn,magFilter:mn}:{};W.map!==null&&W.map.dispose(),W.map=new Wi(s.x,s.y,pt),W.map.texture.name=tt.name+".shadowMap",W.camera.updateProjectionMatrix()}i.setRenderTarget(W.map),i.clear();const ft=W.getViewportCount();for(let pt=0;pt<ft;pt++){const Ct=W.getViewport(pt);a.set(r.x*Ct.x,r.y*Ct.y,r.x*Ct.z,r.y*Ct.w),k.viewport(a),W.updateMatrices(tt,pt),n=W.getFrustum(),x(L,I,W.camera,tt,this.type)}W.isPointLightShadow!==!0&&this.type===Qn&&R(W,I),W.needsUpdate=!1}f=this.type,m.needsUpdate=!1,i.setRenderTarget(M,S,U)};function R(P,L){const I=t.update(_);d.defines.VSM_SAMPLES!==P.blurSamples&&(d.defines.VSM_SAMPLES=P.blurSamples,p.defines.VSM_SAMPLES=P.blurSamples,d.needsUpdate=!0,p.needsUpdate=!0),P.mapPass===null&&(P.mapPass=new Wi(s.x,s.y)),d.uniforms.shadow_pass.value=P.map.texture,d.uniforms.resolution.value=P.mapSize,d.uniforms.radius.value=P.radius,i.setRenderTarget(P.mapPass),i.clear(),i.renderBufferDirect(L,null,I,d,_,null),p.uniforms.shadow_pass.value=P.mapPass.texture,p.uniforms.resolution.value=P.mapSize,p.uniforms.radius.value=P.radius,i.setRenderTarget(P.map),i.clear(),i.renderBufferDirect(L,null,I,p,_,null)}function C(P,L,I,M){let S=null;const U=I.isPointLight===!0?P.customDistanceMaterial:P.customDepthMaterial;if(U!==void 0)S=U;else if(S=I.isPointLight===!0?c:o,i.localClippingEnabled&&L.clipShadows===!0&&Array.isArray(L.clippingPlanes)&&L.clippingPlanes.length!==0||L.displacementMap&&L.displacementScale!==0||L.alphaMap&&L.alphaTest>0||L.map&&L.alphaTest>0){const k=S.uuid,H=L.uuid;let J=l[k];J===void 0&&(J={},l[k]=J);let Q=J[H];Q===void 0&&(Q=S.clone(),J[H]=Q,L.addEventListener("dispose",N)),S=Q}if(S.visible=L.visible,S.wireframe=L.wireframe,M===Qn?S.side=L.shadowSide!==null?L.shadowSide:L.side:S.side=L.shadowSide!==null?L.shadowSide:u[L.side],S.alphaMap=L.alphaMap,S.alphaTest=L.alphaTest,S.map=L.map,S.clipShadows=L.clipShadows,S.clippingPlanes=L.clippingPlanes,S.clipIntersection=L.clipIntersection,S.displacementMap=L.displacementMap,S.displacementScale=L.displacementScale,S.displacementBias=L.displacementBias,S.wireframeLinewidth=L.wireframeLinewidth,S.linewidth=L.linewidth,I.isPointLight===!0&&S.isMeshDistanceMaterial===!0){const k=i.properties.get(S);k.light=I}return S}function x(P,L,I,M,S){if(P.visible===!1)return;if(P.layers.test(L.layers)&&(P.isMesh||P.isLine||P.isPoints)&&(P.castShadow||P.receiveShadow&&S===Qn)&&(!P.frustumCulled||n.intersectsObject(P))){P.modelViewMatrix.multiplyMatrices(I.matrixWorldInverse,P.matrixWorld);const H=t.update(P),J=P.material;if(Array.isArray(J)){const Q=H.groups;for(let O=0,tt=Q.length;O<tt;O++){const W=Q[O],ot=J[W.materialIndex];if(ot&&ot.visible){const ft=C(P,ot,M,S);P.onBeforeShadow(i,P,L,I,H,ft,W),i.renderBufferDirect(I,null,H,ft,P,W),P.onAfterShadow(i,P,L,I,H,ft,W)}}}else if(J.visible){const Q=C(P,J,M,S);P.onBeforeShadow(i,P,L,I,H,Q,null),i.renderBufferDirect(I,null,H,Q,P,null),P.onAfterShadow(i,P,L,I,H,Q,null)}}const k=P.children;for(let H=0,J=k.length;H<J;H++)x(k[H],L,I,M,S)}function N(P){P.target.removeEventListener("dispose",N);for(const I in l){const M=l[I],S=P.target.uuid;S in M&&(M[S].dispose(),delete M[S])}}}const W_={[Ao]:Co,[Ro]:No,[Po]:Do,[ys]:Lo,[Co]:Ao,[No]:Ro,[Do]:Po,[Lo]:ys};function X_(i,t){function e(){let z=!1;const vt=new ve;let Z=null;const at=new ve(0,0,0,0);return{setMask:function(bt){Z!==bt&&!z&&(i.colorMask(bt,bt,bt,bt),Z=bt)},setLocked:function(bt){z=bt},setClear:function(bt,Mt,Qt,Re,Ge){Ge===!0&&(bt*=Re,Mt*=Re,Qt*=Re),vt.set(bt,Mt,Qt,Re),at.equals(vt)===!1&&(i.clearColor(bt,Mt,Qt,Re),at.copy(vt))},reset:function(){z=!1,Z=null,at.set(-1,0,0,0)}}}function n(){let z=!1,vt=!1,Z=null,at=null,bt=null;return{setReversed:function(Mt){if(vt!==Mt){const Qt=t.get("EXT_clip_control");vt?Qt.clipControlEXT(Qt.LOWER_LEFT_EXT,Qt.ZERO_TO_ONE_EXT):Qt.clipControlEXT(Qt.LOWER_LEFT_EXT,Qt.NEGATIVE_ONE_TO_ONE_EXT);const Re=bt;bt=null,this.setClear(Re)}vt=Mt},getReversed:function(){return vt},setTest:function(Mt){Mt?nt(i.DEPTH_TEST):_t(i.DEPTH_TEST)},setMask:function(Mt){Z!==Mt&&!z&&(i.depthMask(Mt),Z=Mt)},setFunc:function(Mt){if(vt&&(Mt=W_[Mt]),at!==Mt){switch(Mt){case Ao:i.depthFunc(i.NEVER);break;case Co:i.depthFunc(i.ALWAYS);break;case Ro:i.depthFunc(i.LESS);break;case ys:i.depthFunc(i.LEQUAL);break;case Po:i.depthFunc(i.EQUAL);break;case Lo:i.depthFunc(i.GEQUAL);break;case No:i.depthFunc(i.GREATER);break;case Do:i.depthFunc(i.NOTEQUAL);break;default:i.depthFunc(i.LEQUAL)}at=Mt}},setLocked:function(Mt){z=Mt},setClear:function(Mt){bt!==Mt&&(vt&&(Mt=1-Mt),i.clearDepth(Mt),bt=Mt)},reset:function(){z=!1,Z=null,at=null,bt=null,vt=!1}}}function s(){let z=!1,vt=null,Z=null,at=null,bt=null,Mt=null,Qt=null,Re=null,Ge=null;return{setTest:function(ne){z||(ne?nt(i.STENCIL_TEST):_t(i.STENCIL_TEST))},setMask:function(ne){vt!==ne&&!z&&(i.stencilMask(ne),vt=ne)},setFunc:function(ne,sn,En){(Z!==ne||at!==sn||bt!==En)&&(i.stencilFunc(ne,sn,En),Z=ne,at=sn,bt=En)},setOp:function(ne,sn,En){(Mt!==ne||Qt!==sn||Re!==En)&&(i.stencilOp(ne,sn,En),Mt=ne,Qt=sn,Re=En)},setLocked:function(ne){z=ne},setClear:function(ne){Ge!==ne&&(i.clearStencil(ne),Ge=ne)},reset:function(){z=!1,vt=null,Z=null,at=null,bt=null,Mt=null,Qt=null,Re=null,Ge=null}}}const r=new e,a=new n,o=new s,c=new WeakMap,l=new WeakMap;let h={},u={},d=new WeakMap,p=[],g=null,_=!1,m=null,f=null,R=null,C=null,x=null,N=null,P=null,L=new $t(0,0,0),I=0,M=!1,S=null,U=null,k=null,H=null,J=null;const Q=i.getParameter(i.MAX_COMBINED_TEXTURE_IMAGE_UNITS);let O=!1,tt=0;const W=i.getParameter(i.VERSION);W.indexOf("WebGL")!==-1?(tt=parseFloat(/^WebGL (\d)/.exec(W)[1]),O=tt>=1):W.indexOf("OpenGL ES")!==-1&&(tt=parseFloat(/^OpenGL ES (\d)/.exec(W)[1]),O=tt>=2);let ot=null,ft={};const pt=i.getParameter(i.SCISSOR_BOX),Ct=i.getParameter(i.VIEWPORT),Et=new ve().fromArray(pt),X=new ve().fromArray(Ct);function K(z,vt,Z,at){const bt=new Uint8Array(4),Mt=i.createTexture();i.bindTexture(z,Mt),i.texParameteri(z,i.TEXTURE_MIN_FILTER,i.NEAREST),i.texParameteri(z,i.TEXTURE_MAG_FILTER,i.NEAREST);for(let Qt=0;Qt<Z;Qt++)z===i.TEXTURE_3D||z===i.TEXTURE_2D_ARRAY?i.texImage3D(vt,0,i.RGBA,1,1,at,0,i.RGBA,i.UNSIGNED_BYTE,bt):i.texImage2D(vt+Qt,0,i.RGBA,1,1,0,i.RGBA,i.UNSIGNED_BYTE,bt);return Mt}const et={};et[i.TEXTURE_2D]=K(i.TEXTURE_2D,i.TEXTURE_2D,1),et[i.TEXTURE_CUBE_MAP]=K(i.TEXTURE_CUBE_MAP,i.TEXTURE_CUBE_MAP_POSITIVE_X,6),et[i.TEXTURE_2D_ARRAY]=K(i.TEXTURE_2D_ARRAY,i.TEXTURE_2D_ARRAY,1,1),et[i.TEXTURE_3D]=K(i.TEXTURE_3D,i.TEXTURE_3D,1,1),r.setClear(0,0,0,1),a.setClear(1),o.setClear(0),nt(i.DEPTH_TEST),a.setFunc(ys),re(!1),ae(Wl),nt(i.CULL_FACE),G(Si);function nt(z){h[z]!==!0&&(i.enable(z),h[z]=!0)}function _t(z){h[z]!==!1&&(i.disable(z),h[z]=!1)}function zt(z,vt){return u[z]!==vt?(i.bindFramebuffer(z,vt),u[z]=vt,z===i.DRAW_FRAMEBUFFER&&(u[i.FRAMEBUFFER]=vt),z===i.FRAMEBUFFER&&(u[i.DRAW_FRAMEBUFFER]=vt),!0):!1}function Zt(z,vt){let Z=p,at=!1;if(z){Z=d.get(vt),Z===void 0&&(Z=[],d.set(vt,Z));const bt=z.textures;if(Z.length!==bt.length||Z[0]!==i.COLOR_ATTACHMENT0){for(let Mt=0,Qt=bt.length;Mt<Qt;Mt++)Z[Mt]=i.COLOR_ATTACHMENT0+Mt;Z.length=bt.length,at=!0}}else Z[0]!==i.BACK&&(Z[0]=i.BACK,at=!0);at&&i.drawBuffers(Z)}function Ce(z){return g!==z?(i.useProgram(z),g=z,!0):!1}const ue={[zi]:i.FUNC_ADD,[Xu]:i.FUNC_SUBTRACT,[qu]:i.FUNC_REVERSE_SUBTRACT};ue[ju]=i.MIN,ue[Yu]=i.MAX;const Ne={[$u]:i.ZERO,[Ku]:i.ONE,[Ju]:i.SRC_COLOR,[To]:i.SRC_ALPHA,[id]:i.SRC_ALPHA_SATURATE,[ed]:i.DST_COLOR,[Qu]:i.DST_ALPHA,[Zu]:i.ONE_MINUS_SRC_COLOR,[wo]:i.ONE_MINUS_SRC_ALPHA,[nd]:i.ONE_MINUS_DST_COLOR,[td]:i.ONE_MINUS_DST_ALPHA,[sd]:i.CONSTANT_COLOR,[rd]:i.ONE_MINUS_CONSTANT_COLOR,[ad]:i.CONSTANT_ALPHA,[od]:i.ONE_MINUS_CONSTANT_ALPHA};function G(z,vt,Z,at,bt,Mt,Qt,Re,Ge,ne){if(z===Si){_===!0&&(_t(i.BLEND),_=!1);return}if(_===!1&&(nt(i.BLEND),_=!0),z!==Wu){if(z!==m||ne!==M){if((f!==zi||x!==zi)&&(i.blendEquation(i.FUNC_ADD),f=zi,x=zi),ne)switch(z){case ms:i.blendFuncSeparate(i.ONE,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case Xl:i.blendFunc(i.ONE,i.ONE);break;case ql:i.blendFuncSeparate(i.ZERO,i.ONE_MINUS_SRC_COLOR,i.ZERO,i.ONE);break;case jl:i.blendFuncSeparate(i.ZERO,i.SRC_COLOR,i.ZERO,i.SRC_ALPHA);break;default:console.error("THREE.WebGLState: Invalid blending: ",z);break}else switch(z){case ms:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case Xl:i.blendFunc(i.SRC_ALPHA,i.ONE);break;case ql:i.blendFuncSeparate(i.ZERO,i.ONE_MINUS_SRC_COLOR,i.ZERO,i.ONE);break;case jl:i.blendFunc(i.ZERO,i.SRC_COLOR);break;default:console.error("THREE.WebGLState: Invalid blending: ",z);break}R=null,C=null,N=null,P=null,L.set(0,0,0),I=0,m=z,M=ne}return}bt=bt||vt,Mt=Mt||Z,Qt=Qt||at,(vt!==f||bt!==x)&&(i.blendEquationSeparate(ue[vt],ue[bt]),f=vt,x=bt),(Z!==R||at!==C||Mt!==N||Qt!==P)&&(i.blendFuncSeparate(Ne[Z],Ne[at],Ne[Mt],Ne[Qt]),R=Z,C=at,N=Mt,P=Qt),(Re.equals(L)===!1||Ge!==I)&&(i.blendColor(Re.r,Re.g,Re.b,Ge),L.copy(Re),I=Ge),m=z,M=!1}function nn(z,vt){z.side===Pn?_t(i.CULL_FACE):nt(i.CULL_FACE);let Z=z.side===ln;vt&&(Z=!Z),re(Z),z.blending===ms&&z.transparent===!1?G(Si):G(z.blending,z.blendEquation,z.blendSrc,z.blendDst,z.blendEquationAlpha,z.blendSrcAlpha,z.blendDstAlpha,z.blendColor,z.blendAlpha,z.premultipliedAlpha),a.setFunc(z.depthFunc),a.setTest(z.depthTest),a.setMask(z.depthWrite),r.setMask(z.colorWrite);const at=z.stencilWrite;o.setTest(at),at&&(o.setMask(z.stencilWriteMask),o.setFunc(z.stencilFunc,z.stencilRef,z.stencilFuncMask),o.setOp(z.stencilFail,z.stencilZFail,z.stencilZPass)),we(z.polygonOffset,z.polygonOffsetFactor,z.polygonOffsetUnits),z.alphaToCoverage===!0?nt(i.SAMPLE_ALPHA_TO_COVERAGE):_t(i.SAMPLE_ALPHA_TO_COVERAGE)}function re(z){S!==z&&(z?i.frontFace(i.CW):i.frontFace(i.CCW),S=z)}function ae(z){z!==Vu?(nt(i.CULL_FACE),z!==U&&(z===Wl?i.cullFace(i.BACK):z===Hu?i.cullFace(i.FRONT):i.cullFace(i.FRONT_AND_BACK))):_t(i.CULL_FACE),U=z}function Wt(z){z!==k&&(O&&i.lineWidth(z),k=z)}function we(z,vt,Z){z?(nt(i.POLYGON_OFFSET_FILL),(H!==vt||J!==Z)&&(i.polygonOffset(vt,Z),H=vt,J=Z)):_t(i.POLYGON_OFFSET_FILL)}function Ht(z){z?nt(i.SCISSOR_TEST):_t(i.SCISSOR_TEST)}function D(z){z===void 0&&(z=i.TEXTURE0+Q-1),ot!==z&&(i.activeTexture(z),ot=z)}function b(z,vt,Z){Z===void 0&&(ot===null?Z=i.TEXTURE0+Q-1:Z=ot);let at=ft[Z];at===void 0&&(at={type:void 0,texture:void 0},ft[Z]=at),(at.type!==z||at.texture!==vt)&&(ot!==Z&&(i.activeTexture(Z),ot=Z),i.bindTexture(z,vt||et[z]),at.type=z,at.texture=vt)}function q(){const z=ft[ot];z!==void 0&&z.type!==void 0&&(i.bindTexture(z.type,null),z.type=void 0,z.texture=void 0)}function rt(){try{i.compressedTexImage2D.apply(i,arguments)}catch(z){console.error("THREE.WebGLState:",z)}}function lt(){try{i.compressedTexImage3D.apply(i,arguments)}catch(z){console.error("THREE.WebGLState:",z)}}function st(){try{i.texSubImage2D.apply(i,arguments)}catch(z){console.error("THREE.WebGLState:",z)}}function Ft(){try{i.texSubImage3D.apply(i,arguments)}catch(z){console.error("THREE.WebGLState:",z)}}function yt(){try{i.compressedTexSubImage2D.apply(i,arguments)}catch(z){console.error("THREE.WebGLState:",z)}}function At(){try{i.compressedTexSubImage3D.apply(i,arguments)}catch(z){console.error("THREE.WebGLState:",z)}}function de(){try{i.texStorage2D.apply(i,arguments)}catch(z){console.error("THREE.WebGLState:",z)}}function dt(){try{i.texStorage3D.apply(i,arguments)}catch(z){console.error("THREE.WebGLState:",z)}}function Rt(){try{i.texImage2D.apply(i,arguments)}catch(z){console.error("THREE.WebGLState:",z)}}function Xt(){try{i.texImage3D.apply(i,arguments)}catch(z){console.error("THREE.WebGLState:",z)}}function jt(z){Et.equals(z)===!1&&(i.scissor(z.x,z.y,z.z,z.w),Et.copy(z))}function Pt(z){X.equals(z)===!1&&(i.viewport(z.x,z.y,z.z,z.w),X.copy(z))}function ce(z,vt){let Z=l.get(vt);Z===void 0&&(Z=new WeakMap,l.set(vt,Z));let at=Z.get(z);at===void 0&&(at=i.getUniformBlockIndex(vt,z.name),Z.set(z,at))}function ee(z,vt){const at=l.get(vt).get(z);c.get(vt)!==at&&(i.uniformBlockBinding(vt,at,z.__bindingPointIndex),c.set(vt,at))}function ye(){i.disable(i.BLEND),i.disable(i.CULL_FACE),i.disable(i.DEPTH_TEST),i.disable(i.POLYGON_OFFSET_FILL),i.disable(i.SCISSOR_TEST),i.disable(i.STENCIL_TEST),i.disable(i.SAMPLE_ALPHA_TO_COVERAGE),i.blendEquation(i.FUNC_ADD),i.blendFunc(i.ONE,i.ZERO),i.blendFuncSeparate(i.ONE,i.ZERO,i.ONE,i.ZERO),i.blendColor(0,0,0,0),i.colorMask(!0,!0,!0,!0),i.clearColor(0,0,0,0),i.depthMask(!0),i.depthFunc(i.LESS),a.setReversed(!1),i.clearDepth(1),i.stencilMask(4294967295),i.stencilFunc(i.ALWAYS,0,4294967295),i.stencilOp(i.KEEP,i.KEEP,i.KEEP),i.clearStencil(0),i.cullFace(i.BACK),i.frontFace(i.CCW),i.polygonOffset(0,0),i.activeTexture(i.TEXTURE0),i.bindFramebuffer(i.FRAMEBUFFER,null),i.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),i.bindFramebuffer(i.READ_FRAMEBUFFER,null),i.useProgram(null),i.lineWidth(1),i.scissor(0,0,i.canvas.width,i.canvas.height),i.viewport(0,0,i.canvas.width,i.canvas.height),h={},ot=null,ft={},u={},d=new WeakMap,p=[],g=null,_=!1,m=null,f=null,R=null,C=null,x=null,N=null,P=null,L=new $t(0,0,0),I=0,M=!1,S=null,U=null,k=null,H=null,J=null,Et.set(0,0,i.canvas.width,i.canvas.height),X.set(0,0,i.canvas.width,i.canvas.height),r.reset(),a.reset(),o.reset()}return{buffers:{color:r,depth:a,stencil:o},enable:nt,disable:_t,bindFramebuffer:zt,drawBuffers:Zt,useProgram:Ce,setBlending:G,setMaterial:nn,setFlipSided:re,setCullFace:ae,setLineWidth:Wt,setPolygonOffset:we,setScissorTest:Ht,activeTexture:D,bindTexture:b,unbindTexture:q,compressedTexImage2D:rt,compressedTexImage3D:lt,texImage2D:Rt,texImage3D:Xt,updateUBOMapping:ce,uniformBlockBinding:ee,texStorage2D:de,texStorage3D:dt,texSubImage2D:st,texSubImage3D:Ft,compressedTexSubImage2D:yt,compressedTexSubImage3D:At,scissor:jt,viewport:Pt,reset:ye}}function qc(i,t,e,n){const s=q_(n);switch(e){case Dh:return i*t;case Uh:return i*t;case Fh:return i*t*2;case Oh:return i*t/s.components*s.byteLength;case xl:return i*t/s.components*s.byteLength;case kh:return i*t*2/s.components*s.byteLength;case yl:return i*t*2/s.components*s.byteLength;case Ih:return i*t*3/s.components*s.byteLength;case Sn:return i*t*4/s.components*s.byteLength;case Ml:return i*t*4/s.components*s.byteLength;case sa:case ra:return Math.floor((i+3)/4)*Math.floor((t+3)/4)*8;case aa:case oa:return Math.floor((i+3)/4)*Math.floor((t+3)/4)*16;case ko:case zo:return Math.max(i,16)*Math.max(t,8)/4;case Oo:case Bo:return Math.max(i,8)*Math.max(t,8)/2;case Vo:case Ho:return Math.floor((i+3)/4)*Math.floor((t+3)/4)*8;case Go:return Math.floor((i+3)/4)*Math.floor((t+3)/4)*16;case Wo:return Math.floor((i+3)/4)*Math.floor((t+3)/4)*16;case Xo:return Math.floor((i+4)/5)*Math.floor((t+3)/4)*16;case qo:return Math.floor((i+4)/5)*Math.floor((t+4)/5)*16;case jo:return Math.floor((i+5)/6)*Math.floor((t+4)/5)*16;case Yo:return Math.floor((i+5)/6)*Math.floor((t+5)/6)*16;case $o:return Math.floor((i+7)/8)*Math.floor((t+4)/5)*16;case Ko:return Math.floor((i+7)/8)*Math.floor((t+5)/6)*16;case Jo:return Math.floor((i+7)/8)*Math.floor((t+7)/8)*16;case Zo:return Math.floor((i+9)/10)*Math.floor((t+4)/5)*16;case Qo:return Math.floor((i+9)/10)*Math.floor((t+5)/6)*16;case tl:return Math.floor((i+9)/10)*Math.floor((t+7)/8)*16;case el:return Math.floor((i+9)/10)*Math.floor((t+9)/10)*16;case nl:return Math.floor((i+11)/12)*Math.floor((t+9)/10)*16;case il:return Math.floor((i+11)/12)*Math.floor((t+11)/12)*16;case la:case sl:case rl:return Math.ceil(i/4)*Math.ceil(t/4)*16;case Bh:case al:return Math.ceil(i/4)*Math.ceil(t/4)*8;case ol:case ll:return Math.ceil(i/4)*Math.ceil(t/4)*16}throw new Error(`Unable to determine texture byte length for ${e} format.`)}function q_(i){switch(i){case si:case Ph:return{byteLength:1,components:1};case ir:case Lh:case hr:return{byteLength:2,components:1};case _l:case vl:return{byteLength:2,components:4};case Gi:case gl:case Bn:return{byteLength:4,components:1};case Nh:return{byteLength:4,components:3}}throw new Error(`Unknown texture type ${i}.`)}function j_(i,t,e,n,s,r,a){const o=t.has("WEBGL_multisampled_render_to_texture")?t.get("WEBGL_multisampled_render_to_texture"):null,c=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),l=new Lt,h=new WeakMap;let u;const d=new WeakMap;let p=!1;try{p=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function g(D,b){return p?new OffscreenCanvas(D,b):sr("canvas")}function _(D,b,q){let rt=1;const lt=Ht(D);if((lt.width>q||lt.height>q)&&(rt=q/Math.max(lt.width,lt.height)),rt<1)if(typeof HTMLImageElement<"u"&&D instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&D instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&D instanceof ImageBitmap||typeof VideoFrame<"u"&&D instanceof VideoFrame){const st=Math.floor(rt*lt.width),Ft=Math.floor(rt*lt.height);u===void 0&&(u=g(st,Ft));const yt=b?g(st,Ft):u;return yt.width=st,yt.height=Ft,yt.getContext("2d").drawImage(D,0,0,st,Ft),console.warn("THREE.WebGLRenderer: Texture has been resized from ("+lt.width+"x"+lt.height+") to ("+st+"x"+Ft+")."),yt}else return"data"in D&&console.warn("THREE.WebGLRenderer: Image in DataTexture is too big ("+lt.width+"x"+lt.height+")."),D;return D}function m(D){return D.generateMipmaps}function f(D){i.generateMipmap(D)}function R(D){return D.isWebGLCubeRenderTarget?i.TEXTURE_CUBE_MAP:D.isWebGL3DRenderTarget?i.TEXTURE_3D:D.isWebGLArrayRenderTarget||D.isCompressedArrayTexture?i.TEXTURE_2D_ARRAY:i.TEXTURE_2D}function C(D,b,q,rt,lt=!1){if(D!==null){if(i[D]!==void 0)return i[D];console.warn("THREE.WebGLRenderer: Attempt to use non-existing WebGL internal format '"+D+"'")}let st=b;if(b===i.RED&&(q===i.FLOAT&&(st=i.R32F),q===i.HALF_FLOAT&&(st=i.R16F),q===i.UNSIGNED_BYTE&&(st=i.R8)),b===i.RED_INTEGER&&(q===i.UNSIGNED_BYTE&&(st=i.R8UI),q===i.UNSIGNED_SHORT&&(st=i.R16UI),q===i.UNSIGNED_INT&&(st=i.R32UI),q===i.BYTE&&(st=i.R8I),q===i.SHORT&&(st=i.R16I),q===i.INT&&(st=i.R32I)),b===i.RG&&(q===i.FLOAT&&(st=i.RG32F),q===i.HALF_FLOAT&&(st=i.RG16F),q===i.UNSIGNED_BYTE&&(st=i.RG8)),b===i.RG_INTEGER&&(q===i.UNSIGNED_BYTE&&(st=i.RG8UI),q===i.UNSIGNED_SHORT&&(st=i.RG16UI),q===i.UNSIGNED_INT&&(st=i.RG32UI),q===i.BYTE&&(st=i.RG8I),q===i.SHORT&&(st=i.RG16I),q===i.INT&&(st=i.RG32I)),b===i.RGB_INTEGER&&(q===i.UNSIGNED_BYTE&&(st=i.RGB8UI),q===i.UNSIGNED_SHORT&&(st=i.RGB16UI),q===i.UNSIGNED_INT&&(st=i.RGB32UI),q===i.BYTE&&(st=i.RGB8I),q===i.SHORT&&(st=i.RGB16I),q===i.INT&&(st=i.RGB32I)),b===i.RGBA_INTEGER&&(q===i.UNSIGNED_BYTE&&(st=i.RGBA8UI),q===i.UNSIGNED_SHORT&&(st=i.RGBA16UI),q===i.UNSIGNED_INT&&(st=i.RGBA32UI),q===i.BYTE&&(st=i.RGBA8I),q===i.SHORT&&(st=i.RGBA16I),q===i.INT&&(st=i.RGBA32I)),b===i.RGB&&q===i.UNSIGNED_INT_5_9_9_9_REV&&(st=i.RGB9_E5),b===i.RGBA){const Ft=lt?Ea:le.getTransfer(rt);q===i.FLOAT&&(st=i.RGBA32F),q===i.HALF_FLOAT&&(st=i.RGBA16F),q===i.UNSIGNED_BYTE&&(st=Ft===Ee?i.SRGB8_ALPHA8:i.RGBA8),q===i.UNSIGNED_SHORT_4_4_4_4&&(st=i.RGBA4),q===i.UNSIGNED_SHORT_5_5_5_1&&(st=i.RGB5_A1)}return(st===i.R16F||st===i.R32F||st===i.RG16F||st===i.RG32F||st===i.RGBA16F||st===i.RGBA32F)&&t.get("EXT_color_buffer_float"),st}function x(D,b){let q;return D?b===null||b===Gi||b===bs?q=i.DEPTH24_STENCIL8:b===Bn?q=i.DEPTH32F_STENCIL8:b===ir&&(q=i.DEPTH24_STENCIL8,console.warn("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):b===null||b===Gi||b===bs?q=i.DEPTH_COMPONENT24:b===Bn?q=i.DEPTH_COMPONENT32F:b===ir&&(q=i.DEPTH_COMPONENT16),q}function N(D,b){return m(D)===!0||D.isFramebufferTexture&&D.minFilter!==mn&&D.minFilter!==fn?Math.log2(Math.max(b.width,b.height))+1:D.mipmaps!==void 0&&D.mipmaps.length>0?D.mipmaps.length:D.isCompressedTexture&&Array.isArray(D.image)?b.mipmaps.length:1}function P(D){const b=D.target;b.removeEventListener("dispose",P),I(b),b.isVideoTexture&&h.delete(b)}function L(D){const b=D.target;b.removeEventListener("dispose",L),S(b)}function I(D){const b=n.get(D);if(b.__webglInit===void 0)return;const q=D.source,rt=d.get(q);if(rt){const lt=rt[b.__cacheKey];lt.usedTimes--,lt.usedTimes===0&&M(D),Object.keys(rt).length===0&&d.delete(q)}n.remove(D)}function M(D){const b=n.get(D);i.deleteTexture(b.__webglTexture);const q=D.source,rt=d.get(q);delete rt[b.__cacheKey],a.memory.textures--}function S(D){const b=n.get(D);if(D.depthTexture&&(D.depthTexture.dispose(),n.remove(D.depthTexture)),D.isWebGLCubeRenderTarget)for(let rt=0;rt<6;rt++){if(Array.isArray(b.__webglFramebuffer[rt]))for(let lt=0;lt<b.__webglFramebuffer[rt].length;lt++)i.deleteFramebuffer(b.__webglFramebuffer[rt][lt]);else i.deleteFramebuffer(b.__webglFramebuffer[rt]);b.__webglDepthbuffer&&i.deleteRenderbuffer(b.__webglDepthbuffer[rt])}else{if(Array.isArray(b.__webglFramebuffer))for(let rt=0;rt<b.__webglFramebuffer.length;rt++)i.deleteFramebuffer(b.__webglFramebuffer[rt]);else i.deleteFramebuffer(b.__webglFramebuffer);if(b.__webglDepthbuffer&&i.deleteRenderbuffer(b.__webglDepthbuffer),b.__webglMultisampledFramebuffer&&i.deleteFramebuffer(b.__webglMultisampledFramebuffer),b.__webglColorRenderbuffer)for(let rt=0;rt<b.__webglColorRenderbuffer.length;rt++)b.__webglColorRenderbuffer[rt]&&i.deleteRenderbuffer(b.__webglColorRenderbuffer[rt]);b.__webglDepthRenderbuffer&&i.deleteRenderbuffer(b.__webglDepthRenderbuffer)}const q=D.textures;for(let rt=0,lt=q.length;rt<lt;rt++){const st=n.get(q[rt]);st.__webglTexture&&(i.deleteTexture(st.__webglTexture),a.memory.textures--),n.remove(q[rt])}n.remove(D)}let U=0;function k(){U=0}function H(){const D=U;return D>=s.maxTextures&&console.warn("THREE.WebGLTextures: Trying to use "+D+" texture units while this GPU supports only "+s.maxTextures),U+=1,D}function J(D){const b=[];return b.push(D.wrapS),b.push(D.wrapT),b.push(D.wrapR||0),b.push(D.magFilter),b.push(D.minFilter),b.push(D.anisotropy),b.push(D.internalFormat),b.push(D.format),b.push(D.type),b.push(D.generateMipmaps),b.push(D.premultiplyAlpha),b.push(D.flipY),b.push(D.unpackAlignment),b.push(D.colorSpace),b.join()}function Q(D,b){const q=n.get(D);if(D.isVideoTexture&&Wt(D),D.isRenderTargetTexture===!1&&D.version>0&&q.__version!==D.version){const rt=D.image;if(rt===null)console.warn("THREE.WebGLRenderer: Texture marked for update but no image data found.");else if(rt.complete===!1)console.warn("THREE.WebGLRenderer: Texture marked for update but image is incomplete");else{X(q,D,b);return}}e.bindTexture(i.TEXTURE_2D,q.__webglTexture,i.TEXTURE0+b)}function O(D,b){const q=n.get(D);if(D.version>0&&q.__version!==D.version){X(q,D,b);return}e.bindTexture(i.TEXTURE_2D_ARRAY,q.__webglTexture,i.TEXTURE0+b)}function tt(D,b){const q=n.get(D);if(D.version>0&&q.__version!==D.version){X(q,D,b);return}e.bindTexture(i.TEXTURE_3D,q.__webglTexture,i.TEXTURE0+b)}function W(D,b){const q=n.get(D);if(D.version>0&&q.__version!==D.version){K(q,D,b);return}e.bindTexture(i.TEXTURE_CUBE_MAP,q.__webglTexture,i.TEXTURE0+b)}const ot={[Hi]:i.REPEAT,[Nn]:i.CLAMP_TO_EDGE,[Fo]:i.MIRRORED_REPEAT},ft={[mn]:i.NEAREST,[_d]:i.NEAREST_MIPMAP_NEAREST,[br]:i.NEAREST_MIPMAP_LINEAR,[fn]:i.LINEAR,[Ua]:i.LINEAR_MIPMAP_NEAREST,[ti]:i.LINEAR_MIPMAP_LINEAR},pt={[Sd]:i.NEVER,[Cd]:i.ALWAYS,[bd]:i.LESS,[zh]:i.LEQUAL,[Ed]:i.EQUAL,[Ad]:i.GEQUAL,[Td]:i.GREATER,[wd]:i.NOTEQUAL};function Ct(D,b){if(b.type===Bn&&t.has("OES_texture_float_linear")===!1&&(b.magFilter===fn||b.magFilter===Ua||b.magFilter===br||b.magFilter===ti||b.minFilter===fn||b.minFilter===Ua||b.minFilter===br||b.minFilter===ti)&&console.warn("THREE.WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),i.texParameteri(D,i.TEXTURE_WRAP_S,ot[b.wrapS]),i.texParameteri(D,i.TEXTURE_WRAP_T,ot[b.wrapT]),(D===i.TEXTURE_3D||D===i.TEXTURE_2D_ARRAY)&&i.texParameteri(D,i.TEXTURE_WRAP_R,ot[b.wrapR]),i.texParameteri(D,i.TEXTURE_MAG_FILTER,ft[b.magFilter]),i.texParameteri(D,i.TEXTURE_MIN_FILTER,ft[b.minFilter]),b.compareFunction&&(i.texParameteri(D,i.TEXTURE_COMPARE_MODE,i.COMPARE_REF_TO_TEXTURE),i.texParameteri(D,i.TEXTURE_COMPARE_FUNC,pt[b.compareFunction])),t.has("EXT_texture_filter_anisotropic")===!0){if(b.magFilter===mn||b.minFilter!==br&&b.minFilter!==ti||b.type===Bn&&t.has("OES_texture_float_linear")===!1)return;if(b.anisotropy>1||n.get(b).__currentAnisotropy){const q=t.get("EXT_texture_filter_anisotropic");i.texParameterf(D,q.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(b.anisotropy,s.getMaxAnisotropy())),n.get(b).__currentAnisotropy=b.anisotropy}}}function Et(D,b){let q=!1;D.__webglInit===void 0&&(D.__webglInit=!0,b.addEventListener("dispose",P));const rt=b.source;let lt=d.get(rt);lt===void 0&&(lt={},d.set(rt,lt));const st=J(b);if(st!==D.__cacheKey){lt[st]===void 0&&(lt[st]={texture:i.createTexture(),usedTimes:0},a.memory.textures++,q=!0),lt[st].usedTimes++;const Ft=lt[D.__cacheKey];Ft!==void 0&&(lt[D.__cacheKey].usedTimes--,Ft.usedTimes===0&&M(b)),D.__cacheKey=st,D.__webglTexture=lt[st].texture}return q}function X(D,b,q){let rt=i.TEXTURE_2D;(b.isDataArrayTexture||b.isCompressedArrayTexture)&&(rt=i.TEXTURE_2D_ARRAY),b.isData3DTexture&&(rt=i.TEXTURE_3D);const lt=Et(D,b),st=b.source;e.bindTexture(rt,D.__webglTexture,i.TEXTURE0+q);const Ft=n.get(st);if(st.version!==Ft.__version||lt===!0){e.activeTexture(i.TEXTURE0+q);const yt=le.getPrimaries(le.workingColorSpace),At=b.colorSpace===yi?null:le.getPrimaries(b.colorSpace),de=b.colorSpace===yi||yt===At?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,b.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,b.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,b.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,de);let dt=_(b.image,!1,s.maxTextureSize);dt=we(b,dt);const Rt=r.convert(b.format,b.colorSpace),Xt=r.convert(b.type);let jt=C(b.internalFormat,Rt,Xt,b.colorSpace,b.isVideoTexture);Ct(rt,b);let Pt;const ce=b.mipmaps,ee=b.isVideoTexture!==!0,ye=Ft.__version===void 0||lt===!0,z=st.dataReady,vt=N(b,dt);if(b.isDepthTexture)jt=x(b.format===Es,b.type),ye&&(ee?e.texStorage2D(i.TEXTURE_2D,1,jt,dt.width,dt.height):e.texImage2D(i.TEXTURE_2D,0,jt,dt.width,dt.height,0,Rt,Xt,null));else if(b.isDataTexture)if(ce.length>0){ee&&ye&&e.texStorage2D(i.TEXTURE_2D,vt,jt,ce[0].width,ce[0].height);for(let Z=0,at=ce.length;Z<at;Z++)Pt=ce[Z],ee?z&&e.texSubImage2D(i.TEXTURE_2D,Z,0,0,Pt.width,Pt.height,Rt,Xt,Pt.data):e.texImage2D(i.TEXTURE_2D,Z,jt,Pt.width,Pt.height,0,Rt,Xt,Pt.data);b.generateMipmaps=!1}else ee?(ye&&e.texStorage2D(i.TEXTURE_2D,vt,jt,dt.width,dt.height),z&&e.texSubImage2D(i.TEXTURE_2D,0,0,0,dt.width,dt.height,Rt,Xt,dt.data)):e.texImage2D(i.TEXTURE_2D,0,jt,dt.width,dt.height,0,Rt,Xt,dt.data);else if(b.isCompressedTexture)if(b.isCompressedArrayTexture){ee&&ye&&e.texStorage3D(i.TEXTURE_2D_ARRAY,vt,jt,ce[0].width,ce[0].height,dt.depth);for(let Z=0,at=ce.length;Z<at;Z++)if(Pt=ce[Z],b.format!==Sn)if(Rt!==null)if(ee){if(z)if(b.layerUpdates.size>0){const bt=qc(Pt.width,Pt.height,b.format,b.type);for(const Mt of b.layerUpdates){const Qt=Pt.data.subarray(Mt*bt/Pt.data.BYTES_PER_ELEMENT,(Mt+1)*bt/Pt.data.BYTES_PER_ELEMENT);e.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,Z,0,0,Mt,Pt.width,Pt.height,1,Rt,Qt)}b.clearLayerUpdates()}else e.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,Z,0,0,0,Pt.width,Pt.height,dt.depth,Rt,Pt.data)}else e.compressedTexImage3D(i.TEXTURE_2D_ARRAY,Z,jt,Pt.width,Pt.height,dt.depth,0,Pt.data,0,0);else console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else ee?z&&e.texSubImage3D(i.TEXTURE_2D_ARRAY,Z,0,0,0,Pt.width,Pt.height,dt.depth,Rt,Xt,Pt.data):e.texImage3D(i.TEXTURE_2D_ARRAY,Z,jt,Pt.width,Pt.height,dt.depth,0,Rt,Xt,Pt.data)}else{ee&&ye&&e.texStorage2D(i.TEXTURE_2D,vt,jt,ce[0].width,ce[0].height);for(let Z=0,at=ce.length;Z<at;Z++)Pt=ce[Z],b.format!==Sn?Rt!==null?ee?z&&e.compressedTexSubImage2D(i.TEXTURE_2D,Z,0,0,Pt.width,Pt.height,Rt,Pt.data):e.compressedTexImage2D(i.TEXTURE_2D,Z,jt,Pt.width,Pt.height,0,Pt.data):console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):ee?z&&e.texSubImage2D(i.TEXTURE_2D,Z,0,0,Pt.width,Pt.height,Rt,Xt,Pt.data):e.texImage2D(i.TEXTURE_2D,Z,jt,Pt.width,Pt.height,0,Rt,Xt,Pt.data)}else if(b.isDataArrayTexture)if(ee){if(ye&&e.texStorage3D(i.TEXTURE_2D_ARRAY,vt,jt,dt.width,dt.height,dt.depth),z)if(b.layerUpdates.size>0){const Z=qc(dt.width,dt.height,b.format,b.type);for(const at of b.layerUpdates){const bt=dt.data.subarray(at*Z/dt.data.BYTES_PER_ELEMENT,(at+1)*Z/dt.data.BYTES_PER_ELEMENT);e.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,at,dt.width,dt.height,1,Rt,Xt,bt)}b.clearLayerUpdates()}else e.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,0,dt.width,dt.height,dt.depth,Rt,Xt,dt.data)}else e.texImage3D(i.TEXTURE_2D_ARRAY,0,jt,dt.width,dt.height,dt.depth,0,Rt,Xt,dt.data);else if(b.isData3DTexture)ee?(ye&&e.texStorage3D(i.TEXTURE_3D,vt,jt,dt.width,dt.height,dt.depth),z&&e.texSubImage3D(i.TEXTURE_3D,0,0,0,0,dt.width,dt.height,dt.depth,Rt,Xt,dt.data)):e.texImage3D(i.TEXTURE_3D,0,jt,dt.width,dt.height,dt.depth,0,Rt,Xt,dt.data);else if(b.isFramebufferTexture){if(ye)if(ee)e.texStorage2D(i.TEXTURE_2D,vt,jt,dt.width,dt.height);else{let Z=dt.width,at=dt.height;for(let bt=0;bt<vt;bt++)e.texImage2D(i.TEXTURE_2D,bt,jt,Z,at,0,Rt,Xt,null),Z>>=1,at>>=1}}else if(ce.length>0){if(ee&&ye){const Z=Ht(ce[0]);e.texStorage2D(i.TEXTURE_2D,vt,jt,Z.width,Z.height)}for(let Z=0,at=ce.length;Z<at;Z++)Pt=ce[Z],ee?z&&e.texSubImage2D(i.TEXTURE_2D,Z,0,0,Rt,Xt,Pt):e.texImage2D(i.TEXTURE_2D,Z,jt,Rt,Xt,Pt);b.generateMipmaps=!1}else if(ee){if(ye){const Z=Ht(dt);e.texStorage2D(i.TEXTURE_2D,vt,jt,Z.width,Z.height)}z&&e.texSubImage2D(i.TEXTURE_2D,0,0,0,Rt,Xt,dt)}else e.texImage2D(i.TEXTURE_2D,0,jt,Rt,Xt,dt);m(b)&&f(rt),Ft.__version=st.version,b.onUpdate&&b.onUpdate(b)}D.__version=b.version}function K(D,b,q){if(b.image.length!==6)return;const rt=Et(D,b),lt=b.source;e.bindTexture(i.TEXTURE_CUBE_MAP,D.__webglTexture,i.TEXTURE0+q);const st=n.get(lt);if(lt.version!==st.__version||rt===!0){e.activeTexture(i.TEXTURE0+q);const Ft=le.getPrimaries(le.workingColorSpace),yt=b.colorSpace===yi?null:le.getPrimaries(b.colorSpace),At=b.colorSpace===yi||Ft===yt?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,b.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,b.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,b.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,At);const de=b.isCompressedTexture||b.image[0].isCompressedTexture,dt=b.image[0]&&b.image[0].isDataTexture,Rt=[];for(let at=0;at<6;at++)!de&&!dt?Rt[at]=_(b.image[at],!0,s.maxCubemapSize):Rt[at]=dt?b.image[at].image:b.image[at],Rt[at]=we(b,Rt[at]);const Xt=Rt[0],jt=r.convert(b.format,b.colorSpace),Pt=r.convert(b.type),ce=C(b.internalFormat,jt,Pt,b.colorSpace),ee=b.isVideoTexture!==!0,ye=st.__version===void 0||rt===!0,z=lt.dataReady;let vt=N(b,Xt);Ct(i.TEXTURE_CUBE_MAP,b);let Z;if(de){ee&&ye&&e.texStorage2D(i.TEXTURE_CUBE_MAP,vt,ce,Xt.width,Xt.height);for(let at=0;at<6;at++){Z=Rt[at].mipmaps;for(let bt=0;bt<Z.length;bt++){const Mt=Z[bt];b.format!==Sn?jt!==null?ee?z&&e.compressedTexSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+at,bt,0,0,Mt.width,Mt.height,jt,Mt.data):e.compressedTexImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+at,bt,ce,Mt.width,Mt.height,0,Mt.data):console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):ee?z&&e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+at,bt,0,0,Mt.width,Mt.height,jt,Pt,Mt.data):e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+at,bt,ce,Mt.width,Mt.height,0,jt,Pt,Mt.data)}}}else{if(Z=b.mipmaps,ee&&ye){Z.length>0&&vt++;const at=Ht(Rt[0]);e.texStorage2D(i.TEXTURE_CUBE_MAP,vt,ce,at.width,at.height)}for(let at=0;at<6;at++)if(dt){ee?z&&e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+at,0,0,0,Rt[at].width,Rt[at].height,jt,Pt,Rt[at].data):e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+at,0,ce,Rt[at].width,Rt[at].height,0,jt,Pt,Rt[at].data);for(let bt=0;bt<Z.length;bt++){const Qt=Z[bt].image[at].image;ee?z&&e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+at,bt+1,0,0,Qt.width,Qt.height,jt,Pt,Qt.data):e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+at,bt+1,ce,Qt.width,Qt.height,0,jt,Pt,Qt.data)}}else{ee?z&&e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+at,0,0,0,jt,Pt,Rt[at]):e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+at,0,ce,jt,Pt,Rt[at]);for(let bt=0;bt<Z.length;bt++){const Mt=Z[bt];ee?z&&e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+at,bt+1,0,0,jt,Pt,Mt.image[at]):e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+at,bt+1,ce,jt,Pt,Mt.image[at])}}}m(b)&&f(i.TEXTURE_CUBE_MAP),st.__version=lt.version,b.onUpdate&&b.onUpdate(b)}D.__version=b.version}function et(D,b,q,rt,lt,st){const Ft=r.convert(q.format,q.colorSpace),yt=r.convert(q.type),At=C(q.internalFormat,Ft,yt,q.colorSpace),de=n.get(b),dt=n.get(q);if(dt.__renderTarget=b,!de.__hasExternalTextures){const Rt=Math.max(1,b.width>>st),Xt=Math.max(1,b.height>>st);lt===i.TEXTURE_3D||lt===i.TEXTURE_2D_ARRAY?e.texImage3D(lt,st,At,Rt,Xt,b.depth,0,Ft,yt,null):e.texImage2D(lt,st,At,Rt,Xt,0,Ft,yt,null)}e.bindFramebuffer(i.FRAMEBUFFER,D),ae(b)?o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,rt,lt,dt.__webglTexture,0,re(b)):(lt===i.TEXTURE_2D||lt>=i.TEXTURE_CUBE_MAP_POSITIVE_X&&lt<=i.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&i.framebufferTexture2D(i.FRAMEBUFFER,rt,lt,dt.__webglTexture,st),e.bindFramebuffer(i.FRAMEBUFFER,null)}function nt(D,b,q){if(i.bindRenderbuffer(i.RENDERBUFFER,D),b.depthBuffer){const rt=b.depthTexture,lt=rt&&rt.isDepthTexture?rt.type:null,st=x(b.stencilBuffer,lt),Ft=b.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,yt=re(b);ae(b)?o.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,yt,st,b.width,b.height):q?i.renderbufferStorageMultisample(i.RENDERBUFFER,yt,st,b.width,b.height):i.renderbufferStorage(i.RENDERBUFFER,st,b.width,b.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,Ft,i.RENDERBUFFER,D)}else{const rt=b.textures;for(let lt=0;lt<rt.length;lt++){const st=rt[lt],Ft=r.convert(st.format,st.colorSpace),yt=r.convert(st.type),At=C(st.internalFormat,Ft,yt,st.colorSpace),de=re(b);q&&ae(b)===!1?i.renderbufferStorageMultisample(i.RENDERBUFFER,de,At,b.width,b.height):ae(b)?o.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,de,At,b.width,b.height):i.renderbufferStorage(i.RENDERBUFFER,At,b.width,b.height)}}i.bindRenderbuffer(i.RENDERBUFFER,null)}function _t(D,b){if(b&&b.isWebGLCubeRenderTarget)throw new Error("Depth Texture with cube render targets is not supported");if(e.bindFramebuffer(i.FRAMEBUFFER,D),!(b.depthTexture&&b.depthTexture.isDepthTexture))throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");const rt=n.get(b.depthTexture);rt.__renderTarget=b,(!rt.__webglTexture||b.depthTexture.image.width!==b.width||b.depthTexture.image.height!==b.height)&&(b.depthTexture.image.width=b.width,b.depthTexture.image.height=b.height,b.depthTexture.needsUpdate=!0),Q(b.depthTexture,0);const lt=rt.__webglTexture,st=re(b);if(b.depthTexture.format===gs)ae(b)?o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,i.DEPTH_ATTACHMENT,i.TEXTURE_2D,lt,0,st):i.framebufferTexture2D(i.FRAMEBUFFER,i.DEPTH_ATTACHMENT,i.TEXTURE_2D,lt,0);else if(b.depthTexture.format===Es)ae(b)?o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,i.DEPTH_STENCIL_ATTACHMENT,i.TEXTURE_2D,lt,0,st):i.framebufferTexture2D(i.FRAMEBUFFER,i.DEPTH_STENCIL_ATTACHMENT,i.TEXTURE_2D,lt,0);else throw new Error("Unknown depthTexture format")}function zt(D){const b=n.get(D),q=D.isWebGLCubeRenderTarget===!0;if(b.__boundDepthTexture!==D.depthTexture){const rt=D.depthTexture;if(b.__depthDisposeCallback&&b.__depthDisposeCallback(),rt){const lt=()=>{delete b.__boundDepthTexture,delete b.__depthDisposeCallback,rt.removeEventListener("dispose",lt)};rt.addEventListener("dispose",lt),b.__depthDisposeCallback=lt}b.__boundDepthTexture=rt}if(D.depthTexture&&!b.__autoAllocateDepthBuffer){if(q)throw new Error("target.depthTexture not supported in Cube render targets");_t(b.__webglFramebuffer,D)}else if(q){b.__webglDepthbuffer=[];for(let rt=0;rt<6;rt++)if(e.bindFramebuffer(i.FRAMEBUFFER,b.__webglFramebuffer[rt]),b.__webglDepthbuffer[rt]===void 0)b.__webglDepthbuffer[rt]=i.createRenderbuffer(),nt(b.__webglDepthbuffer[rt],D,!1);else{const lt=D.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,st=b.__webglDepthbuffer[rt];i.bindRenderbuffer(i.RENDERBUFFER,st),i.framebufferRenderbuffer(i.FRAMEBUFFER,lt,i.RENDERBUFFER,st)}}else if(e.bindFramebuffer(i.FRAMEBUFFER,b.__webglFramebuffer),b.__webglDepthbuffer===void 0)b.__webglDepthbuffer=i.createRenderbuffer(),nt(b.__webglDepthbuffer,D,!1);else{const rt=D.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,lt=b.__webglDepthbuffer;i.bindRenderbuffer(i.RENDERBUFFER,lt),i.framebufferRenderbuffer(i.FRAMEBUFFER,rt,i.RENDERBUFFER,lt)}e.bindFramebuffer(i.FRAMEBUFFER,null)}function Zt(D,b,q){const rt=n.get(D);b!==void 0&&et(rt.__webglFramebuffer,D,D.texture,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,0),q!==void 0&&zt(D)}function Ce(D){const b=D.texture,q=n.get(D),rt=n.get(b);D.addEventListener("dispose",L);const lt=D.textures,st=D.isWebGLCubeRenderTarget===!0,Ft=lt.length>1;if(Ft||(rt.__webglTexture===void 0&&(rt.__webglTexture=i.createTexture()),rt.__version=b.version,a.memory.textures++),st){q.__webglFramebuffer=[];for(let yt=0;yt<6;yt++)if(b.mipmaps&&b.mipmaps.length>0){q.__webglFramebuffer[yt]=[];for(let At=0;At<b.mipmaps.length;At++)q.__webglFramebuffer[yt][At]=i.createFramebuffer()}else q.__webglFramebuffer[yt]=i.createFramebuffer()}else{if(b.mipmaps&&b.mipmaps.length>0){q.__webglFramebuffer=[];for(let yt=0;yt<b.mipmaps.length;yt++)q.__webglFramebuffer[yt]=i.createFramebuffer()}else q.__webglFramebuffer=i.createFramebuffer();if(Ft)for(let yt=0,At=lt.length;yt<At;yt++){const de=n.get(lt[yt]);de.__webglTexture===void 0&&(de.__webglTexture=i.createTexture(),a.memory.textures++)}if(D.samples>0&&ae(D)===!1){q.__webglMultisampledFramebuffer=i.createFramebuffer(),q.__webglColorRenderbuffer=[],e.bindFramebuffer(i.FRAMEBUFFER,q.__webglMultisampledFramebuffer);for(let yt=0;yt<lt.length;yt++){const At=lt[yt];q.__webglColorRenderbuffer[yt]=i.createRenderbuffer(),i.bindRenderbuffer(i.RENDERBUFFER,q.__webglColorRenderbuffer[yt]);const de=r.convert(At.format,At.colorSpace),dt=r.convert(At.type),Rt=C(At.internalFormat,de,dt,At.colorSpace,D.isXRRenderTarget===!0),Xt=re(D);i.renderbufferStorageMultisample(i.RENDERBUFFER,Xt,Rt,D.width,D.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+yt,i.RENDERBUFFER,q.__webglColorRenderbuffer[yt])}i.bindRenderbuffer(i.RENDERBUFFER,null),D.depthBuffer&&(q.__webglDepthRenderbuffer=i.createRenderbuffer(),nt(q.__webglDepthRenderbuffer,D,!0)),e.bindFramebuffer(i.FRAMEBUFFER,null)}}if(st){e.bindTexture(i.TEXTURE_CUBE_MAP,rt.__webglTexture),Ct(i.TEXTURE_CUBE_MAP,b);for(let yt=0;yt<6;yt++)if(b.mipmaps&&b.mipmaps.length>0)for(let At=0;At<b.mipmaps.length;At++)et(q.__webglFramebuffer[yt][At],D,b,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+yt,At);else et(q.__webglFramebuffer[yt],D,b,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+yt,0);m(b)&&f(i.TEXTURE_CUBE_MAP),e.unbindTexture()}else if(Ft){for(let yt=0,At=lt.length;yt<At;yt++){const de=lt[yt],dt=n.get(de);e.bindTexture(i.TEXTURE_2D,dt.__webglTexture),Ct(i.TEXTURE_2D,de),et(q.__webglFramebuffer,D,de,i.COLOR_ATTACHMENT0+yt,i.TEXTURE_2D,0),m(de)&&f(i.TEXTURE_2D)}e.unbindTexture()}else{let yt=i.TEXTURE_2D;if((D.isWebGL3DRenderTarget||D.isWebGLArrayRenderTarget)&&(yt=D.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),e.bindTexture(yt,rt.__webglTexture),Ct(yt,b),b.mipmaps&&b.mipmaps.length>0)for(let At=0;At<b.mipmaps.length;At++)et(q.__webglFramebuffer[At],D,b,i.COLOR_ATTACHMENT0,yt,At);else et(q.__webglFramebuffer,D,b,i.COLOR_ATTACHMENT0,yt,0);m(b)&&f(yt),e.unbindTexture()}D.depthBuffer&&zt(D)}function ue(D){const b=D.textures;for(let q=0,rt=b.length;q<rt;q++){const lt=b[q];if(m(lt)){const st=R(D),Ft=n.get(lt).__webglTexture;e.bindTexture(st,Ft),f(st),e.unbindTexture()}}}const Ne=[],G=[];function nn(D){if(D.samples>0){if(ae(D)===!1){const b=D.textures,q=D.width,rt=D.height;let lt=i.COLOR_BUFFER_BIT;const st=D.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,Ft=n.get(D),yt=b.length>1;if(yt)for(let At=0;At<b.length;At++)e.bindFramebuffer(i.FRAMEBUFFER,Ft.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+At,i.RENDERBUFFER,null),e.bindFramebuffer(i.FRAMEBUFFER,Ft.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+At,i.TEXTURE_2D,null,0);e.bindFramebuffer(i.READ_FRAMEBUFFER,Ft.__webglMultisampledFramebuffer),e.bindFramebuffer(i.DRAW_FRAMEBUFFER,Ft.__webglFramebuffer);for(let At=0;At<b.length;At++){if(D.resolveDepthBuffer&&(D.depthBuffer&&(lt|=i.DEPTH_BUFFER_BIT),D.stencilBuffer&&D.resolveStencilBuffer&&(lt|=i.STENCIL_BUFFER_BIT)),yt){i.framebufferRenderbuffer(i.READ_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.RENDERBUFFER,Ft.__webglColorRenderbuffer[At]);const de=n.get(b[At]).__webglTexture;i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,de,0)}i.blitFramebuffer(0,0,q,rt,0,0,q,rt,lt,i.NEAREST),c===!0&&(Ne.length=0,G.length=0,Ne.push(i.COLOR_ATTACHMENT0+At),D.depthBuffer&&D.resolveDepthBuffer===!1&&(Ne.push(st),G.push(st),i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,G)),i.invalidateFramebuffer(i.READ_FRAMEBUFFER,Ne))}if(e.bindFramebuffer(i.READ_FRAMEBUFFER,null),e.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),yt)for(let At=0;At<b.length;At++){e.bindFramebuffer(i.FRAMEBUFFER,Ft.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+At,i.RENDERBUFFER,Ft.__webglColorRenderbuffer[At]);const de=n.get(b[At]).__webglTexture;e.bindFramebuffer(i.FRAMEBUFFER,Ft.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+At,i.TEXTURE_2D,de,0)}e.bindFramebuffer(i.DRAW_FRAMEBUFFER,Ft.__webglMultisampledFramebuffer)}else if(D.depthBuffer&&D.resolveDepthBuffer===!1&&c){const b=D.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,[b])}}}function re(D){return Math.min(s.maxSamples,D.samples)}function ae(D){const b=n.get(D);return D.samples>0&&t.has("WEBGL_multisampled_render_to_texture")===!0&&b.__useRenderToTexture!==!1}function Wt(D){const b=a.render.frame;h.get(D)!==b&&(h.set(D,b),D.update())}function we(D,b){const q=D.colorSpace,rt=D.format,lt=D.type;return D.isCompressedTexture===!0||D.isVideoTexture===!0||q!==Ds&&q!==yi&&(le.getTransfer(q)===Ee?(rt!==Sn||lt!==si)&&console.warn("THREE.WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):console.error("THREE.WebGLTextures: Unsupported texture color space:",q)),b}function Ht(D){return typeof HTMLImageElement<"u"&&D instanceof HTMLImageElement?(l.width=D.naturalWidth||D.width,l.height=D.naturalHeight||D.height):typeof VideoFrame<"u"&&D instanceof VideoFrame?(l.width=D.displayWidth,l.height=D.displayHeight):(l.width=D.width,l.height=D.height),l}this.allocateTextureUnit=H,this.resetTextureUnits=k,this.setTexture2D=Q,this.setTexture2DArray=O,this.setTexture3D=tt,this.setTextureCube=W,this.rebindTextures=Zt,this.setupRenderTarget=Ce,this.updateRenderTargetMipmap=ue,this.updateMultisampleRenderTarget=nn,this.setupDepthRenderbuffer=zt,this.setupFrameBufferTexture=et,this.useMultisampledRTT=ae}function Y_(i,t){function e(n,s=yi){let r;const a=le.getTransfer(s);if(n===si)return i.UNSIGNED_BYTE;if(n===_l)return i.UNSIGNED_SHORT_4_4_4_4;if(n===vl)return i.UNSIGNED_SHORT_5_5_5_1;if(n===Nh)return i.UNSIGNED_INT_5_9_9_9_REV;if(n===Ph)return i.BYTE;if(n===Lh)return i.SHORT;if(n===ir)return i.UNSIGNED_SHORT;if(n===gl)return i.INT;if(n===Gi)return i.UNSIGNED_INT;if(n===Bn)return i.FLOAT;if(n===hr)return i.HALF_FLOAT;if(n===Dh)return i.ALPHA;if(n===Ih)return i.RGB;if(n===Sn)return i.RGBA;if(n===Uh)return i.LUMINANCE;if(n===Fh)return i.LUMINANCE_ALPHA;if(n===gs)return i.DEPTH_COMPONENT;if(n===Es)return i.DEPTH_STENCIL;if(n===Oh)return i.RED;if(n===xl)return i.RED_INTEGER;if(n===kh)return i.RG;if(n===yl)return i.RG_INTEGER;if(n===Ml)return i.RGBA_INTEGER;if(n===sa||n===ra||n===aa||n===oa)if(a===Ee)if(r=t.get("WEBGL_compressed_texture_s3tc_srgb"),r!==null){if(n===sa)return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(n===ra)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(n===aa)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(n===oa)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(r=t.get("WEBGL_compressed_texture_s3tc"),r!==null){if(n===sa)return r.COMPRESSED_RGB_S3TC_DXT1_EXT;if(n===ra)return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(n===aa)return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(n===oa)return r.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(n===Oo||n===ko||n===Bo||n===zo)if(r=t.get("WEBGL_compressed_texture_pvrtc"),r!==null){if(n===Oo)return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(n===ko)return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(n===Bo)return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(n===zo)return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(n===Vo||n===Ho||n===Go)if(r=t.get("WEBGL_compressed_texture_etc"),r!==null){if(n===Vo||n===Ho)return a===Ee?r.COMPRESSED_SRGB8_ETC2:r.COMPRESSED_RGB8_ETC2;if(n===Go)return a===Ee?r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:r.COMPRESSED_RGBA8_ETC2_EAC}else return null;if(n===Wo||n===Xo||n===qo||n===jo||n===Yo||n===$o||n===Ko||n===Jo||n===Zo||n===Qo||n===tl||n===el||n===nl||n===il)if(r=t.get("WEBGL_compressed_texture_astc"),r!==null){if(n===Wo)return a===Ee?r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:r.COMPRESSED_RGBA_ASTC_4x4_KHR;if(n===Xo)return a===Ee?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:r.COMPRESSED_RGBA_ASTC_5x4_KHR;if(n===qo)return a===Ee?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:r.COMPRESSED_RGBA_ASTC_5x5_KHR;if(n===jo)return a===Ee?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:r.COMPRESSED_RGBA_ASTC_6x5_KHR;if(n===Yo)return a===Ee?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:r.COMPRESSED_RGBA_ASTC_6x6_KHR;if(n===$o)return a===Ee?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:r.COMPRESSED_RGBA_ASTC_8x5_KHR;if(n===Ko)return a===Ee?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:r.COMPRESSED_RGBA_ASTC_8x6_KHR;if(n===Jo)return a===Ee?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:r.COMPRESSED_RGBA_ASTC_8x8_KHR;if(n===Zo)return a===Ee?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:r.COMPRESSED_RGBA_ASTC_10x5_KHR;if(n===Qo)return a===Ee?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:r.COMPRESSED_RGBA_ASTC_10x6_KHR;if(n===tl)return a===Ee?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:r.COMPRESSED_RGBA_ASTC_10x8_KHR;if(n===el)return a===Ee?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:r.COMPRESSED_RGBA_ASTC_10x10_KHR;if(n===nl)return a===Ee?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:r.COMPRESSED_RGBA_ASTC_12x10_KHR;if(n===il)return a===Ee?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:r.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(n===la||n===sl||n===rl)if(r=t.get("EXT_texture_compression_bptc"),r!==null){if(n===la)return a===Ee?r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:r.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(n===sl)return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(n===rl)return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(n===Bh||n===al||n===ol||n===ll)if(r=t.get("EXT_texture_compression_rgtc"),r!==null){if(n===la)return r.COMPRESSED_RED_RGTC1_EXT;if(n===al)return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(n===ol)return r.COMPRESSED_RED_GREEN_RGTC2_EXT;if(n===ll)return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return n===bs?i.UNSIGNED_INT_24_8:i[n]!==void 0?i[n]:null}return{convert:e}}class $_ extends Je{constructor(t=[]){super(),this.isArrayCamera=!0,this.cameras=t}}class je extends Le{constructor(){super(),this.isGroup=!0,this.type="Group"}}const K_={type:"move"};class co{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new je,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new je,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new F,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new F),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new je,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new F,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new F),this._grip}dispatchEvent(t){return this._targetRay!==null&&this._targetRay.dispatchEvent(t),this._grip!==null&&this._grip.dispatchEvent(t),this._hand!==null&&this._hand.dispatchEvent(t),this}connect(t){if(t&&t.hand){const e=this._hand;if(e)for(const n of t.hand.values())this._getHandJoint(e,n)}return this.dispatchEvent({type:"connected",data:t}),this}disconnect(t){return this.dispatchEvent({type:"disconnected",data:t}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(t,e,n){let s=null,r=null,a=null;const o=this._targetRay,c=this._grip,l=this._hand;if(t&&e.session.visibilityState!=="visible-blurred"){if(l&&t.hand){a=!0;for(const _ of t.hand.values()){const m=e.getJointPose(_,n),f=this._getHandJoint(l,_);m!==null&&(f.matrix.fromArray(m.transform.matrix),f.matrix.decompose(f.position,f.rotation,f.scale),f.matrixWorldNeedsUpdate=!0,f.jointRadius=m.radius),f.visible=m!==null}const h=l.joints["index-finger-tip"],u=l.joints["thumb-tip"],d=h.position.distanceTo(u.position),p=.02,g=.005;l.inputState.pinching&&d>p+g?(l.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:t.handedness,target:this})):!l.inputState.pinching&&d<=p-g&&(l.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:t.handedness,target:this}))}else c!==null&&t.gripSpace&&(r=e.getPose(t.gripSpace,n),r!==null&&(c.matrix.fromArray(r.transform.matrix),c.matrix.decompose(c.position,c.rotation,c.scale),c.matrixWorldNeedsUpdate=!0,r.linearVelocity?(c.hasLinearVelocity=!0,c.linearVelocity.copy(r.linearVelocity)):c.hasLinearVelocity=!1,r.angularVelocity?(c.hasAngularVelocity=!0,c.angularVelocity.copy(r.angularVelocity)):c.hasAngularVelocity=!1));o!==null&&(s=e.getPose(t.targetRaySpace,n),s===null&&r!==null&&(s=r),s!==null&&(o.matrix.fromArray(s.transform.matrix),o.matrix.decompose(o.position,o.rotation,o.scale),o.matrixWorldNeedsUpdate=!0,s.linearVelocity?(o.hasLinearVelocity=!0,o.linearVelocity.copy(s.linearVelocity)):o.hasLinearVelocity=!1,s.angularVelocity?(o.hasAngularVelocity=!0,o.angularVelocity.copy(s.angularVelocity)):o.hasAngularVelocity=!1,this.dispatchEvent(K_)))}return o!==null&&(o.visible=s!==null),c!==null&&(c.visible=r!==null),l!==null&&(l.visible=a!==null),this}_getHandJoint(t,e){if(t.joints[e.jointName]===void 0){const n=new je;n.matrixAutoUpdate=!1,n.visible=!1,t.joints[e.jointName]=n,t.add(n)}return t.joints[e.jointName]}}const J_=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,Z_=`
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

}`;class Q_{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(t,e,n){if(this.texture===null){const s=new Ye,r=t.properties.get(s);r.__webglTexture=e.texture,(e.depthNear!=n.depthNear||e.depthFar!=n.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=s}}getMesh(t){if(this.texture!==null&&this.mesh===null){const e=t.cameras[0].viewport,n=new Ti({vertexShader:J_,fragmentShader:Z_,uniforms:{depthColor:{value:this.texture},depthWidth:{value:e.z},depthHeight:{value:e.w}}});this.mesh=new Ae(new Ta(20,20),n)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}}class t0 extends qi{constructor(t,e){super();const n=this;let s=null,r=1,a=null,o="local-floor",c=1,l=null,h=null,u=null,d=null,p=null,g=null;const _=new Q_,m=e.getContextAttributes();let f=null,R=null;const C=[],x=[],N=new Lt;let P=null;const L=new Je;L.viewport=new ve;const I=new Je;I.viewport=new ve;const M=[L,I],S=new $_;let U=null,k=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(X){let K=C[X];return K===void 0&&(K=new co,C[X]=K),K.getTargetRaySpace()},this.getControllerGrip=function(X){let K=C[X];return K===void 0&&(K=new co,C[X]=K),K.getGripSpace()},this.getHand=function(X){let K=C[X];return K===void 0&&(K=new co,C[X]=K),K.getHandSpace()};function H(X){const K=x.indexOf(X.inputSource);if(K===-1)return;const et=C[K];et!==void 0&&(et.update(X.inputSource,X.frame,l||a),et.dispatchEvent({type:X.type,data:X.inputSource}))}function J(){s.removeEventListener("select",H),s.removeEventListener("selectstart",H),s.removeEventListener("selectend",H),s.removeEventListener("squeeze",H),s.removeEventListener("squeezestart",H),s.removeEventListener("squeezeend",H),s.removeEventListener("end",J),s.removeEventListener("inputsourceschange",Q);for(let X=0;X<C.length;X++){const K=x[X];K!==null&&(x[X]=null,C[X].disconnect(K))}U=null,k=null,_.reset(),t.setRenderTarget(f),p=null,d=null,u=null,s=null,R=null,Et.stop(),n.isPresenting=!1,t.setPixelRatio(P),t.setSize(N.width,N.height,!1),n.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(X){r=X,n.isPresenting===!0&&console.warn("THREE.WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(X){o=X,n.isPresenting===!0&&console.warn("THREE.WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return l||a},this.setReferenceSpace=function(X){l=X},this.getBaseLayer=function(){return d!==null?d:p},this.getBinding=function(){return u},this.getFrame=function(){return g},this.getSession=function(){return s},this.setSession=async function(X){if(s=X,s!==null){if(f=t.getRenderTarget(),s.addEventListener("select",H),s.addEventListener("selectstart",H),s.addEventListener("selectend",H),s.addEventListener("squeeze",H),s.addEventListener("squeezestart",H),s.addEventListener("squeezeend",H),s.addEventListener("end",J),s.addEventListener("inputsourceschange",Q),m.xrCompatible!==!0&&await e.makeXRCompatible(),P=t.getPixelRatio(),t.getSize(N),s.renderState.layers===void 0){const K={antialias:m.antialias,alpha:!0,depth:m.depth,stencil:m.stencil,framebufferScaleFactor:r};p=new XRWebGLLayer(s,e,K),s.updateRenderState({baseLayer:p}),t.setPixelRatio(1),t.setSize(p.framebufferWidth,p.framebufferHeight,!1),R=new Wi(p.framebufferWidth,p.framebufferHeight,{format:Sn,type:si,colorSpace:t.outputColorSpace,stencilBuffer:m.stencil})}else{let K=null,et=null,nt=null;m.depth&&(nt=m.stencil?e.DEPTH24_STENCIL8:e.DEPTH_COMPONENT24,K=m.stencil?Es:gs,et=m.stencil?bs:Gi);const _t={colorFormat:e.RGBA8,depthFormat:nt,scaleFactor:r};u=new XRWebGLBinding(s,e),d=u.createProjectionLayer(_t),s.updateRenderState({layers:[d]}),t.setPixelRatio(1),t.setSize(d.textureWidth,d.textureHeight,!1),R=new Wi(d.textureWidth,d.textureHeight,{format:Sn,type:si,depthTexture:new Zh(d.textureWidth,d.textureHeight,et,void 0,void 0,void 0,void 0,void 0,void 0,K),stencilBuffer:m.stencil,colorSpace:t.outputColorSpace,samples:m.antialias?4:0,resolveDepthBuffer:d.ignoreDepthValues===!1})}R.isXRRenderTarget=!0,this.setFoveation(c),l=null,a=await s.requestReferenceSpace(o),Et.setContext(s),Et.start(),n.isPresenting=!0,n.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(s!==null)return s.environmentBlendMode},this.getDepthTexture=function(){return _.getDepthTexture()};function Q(X){for(let K=0;K<X.removed.length;K++){const et=X.removed[K],nt=x.indexOf(et);nt>=0&&(x[nt]=null,C[nt].disconnect(et))}for(let K=0;K<X.added.length;K++){const et=X.added[K];let nt=x.indexOf(et);if(nt===-1){for(let zt=0;zt<C.length;zt++)if(zt>=x.length){x.push(et),nt=zt;break}else if(x[zt]===null){x[zt]=et,nt=zt;break}if(nt===-1)break}const _t=C[nt];_t&&_t.connect(et)}}const O=new F,tt=new F;function W(X,K,et){O.setFromMatrixPosition(K.matrixWorld),tt.setFromMatrixPosition(et.matrixWorld);const nt=O.distanceTo(tt),_t=K.projectionMatrix.elements,zt=et.projectionMatrix.elements,Zt=_t[14]/(_t[10]-1),Ce=_t[14]/(_t[10]+1),ue=(_t[9]+1)/_t[5],Ne=(_t[9]-1)/_t[5],G=(_t[8]-1)/_t[0],nn=(zt[8]+1)/zt[0],re=Zt*G,ae=Zt*nn,Wt=nt/(-G+nn),we=Wt*-G;if(K.matrixWorld.decompose(X.position,X.quaternion,X.scale),X.translateX(we),X.translateZ(Wt),X.matrixWorld.compose(X.position,X.quaternion,X.scale),X.matrixWorldInverse.copy(X.matrixWorld).invert(),_t[10]===-1)X.projectionMatrix.copy(K.projectionMatrix),X.projectionMatrixInverse.copy(K.projectionMatrixInverse);else{const Ht=Zt+Wt,D=Ce+Wt,b=re-we,q=ae+(nt-we),rt=ue*Ce/D*Ht,lt=Ne*Ce/D*Ht;X.projectionMatrix.makePerspective(b,q,rt,lt,Ht,D),X.projectionMatrixInverse.copy(X.projectionMatrix).invert()}}function ot(X,K){K===null?X.matrixWorld.copy(X.matrix):X.matrixWorld.multiplyMatrices(K.matrixWorld,X.matrix),X.matrixWorldInverse.copy(X.matrixWorld).invert()}this.updateCamera=function(X){if(s===null)return;let K=X.near,et=X.far;_.texture!==null&&(_.depthNear>0&&(K=_.depthNear),_.depthFar>0&&(et=_.depthFar)),S.near=I.near=L.near=K,S.far=I.far=L.far=et,(U!==S.near||k!==S.far)&&(s.updateRenderState({depthNear:S.near,depthFar:S.far}),U=S.near,k=S.far),L.layers.mask=X.layers.mask|2,I.layers.mask=X.layers.mask|4,S.layers.mask=L.layers.mask|I.layers.mask;const nt=X.parent,_t=S.cameras;ot(S,nt);for(let zt=0;zt<_t.length;zt++)ot(_t[zt],nt);_t.length===2?W(S,L,I):S.projectionMatrix.copy(L.projectionMatrix),ft(X,S,nt)};function ft(X,K,et){et===null?X.matrix.copy(K.matrixWorld):(X.matrix.copy(et.matrixWorld),X.matrix.invert(),X.matrix.multiply(K.matrixWorld)),X.matrix.decompose(X.position,X.quaternion,X.scale),X.updateMatrixWorld(!0),X.projectionMatrix.copy(K.projectionMatrix),X.projectionMatrixInverse.copy(K.projectionMatrixInverse),X.isPerspectiveCamera&&(X.fov=Ts*2*Math.atan(1/X.projectionMatrix.elements[5]),X.zoom=1)}this.getCamera=function(){return S},this.getFoveation=function(){if(!(d===null&&p===null))return c},this.setFoveation=function(X){c=X,d!==null&&(d.fixedFoveation=X),p!==null&&p.fixedFoveation!==void 0&&(p.fixedFoveation=X)},this.hasDepthSensing=function(){return _.texture!==null},this.getDepthSensingMesh=function(){return _.getMesh(S)};let pt=null;function Ct(X,K){if(h=K.getViewerPose(l||a),g=K,h!==null){const et=h.views;p!==null&&(t.setRenderTargetFramebuffer(R,p.framebuffer),t.setRenderTarget(R));let nt=!1;et.length!==S.cameras.length&&(S.cameras.length=0,nt=!0);for(let zt=0;zt<et.length;zt++){const Zt=et[zt];let Ce=null;if(p!==null)Ce=p.getViewport(Zt);else{const Ne=u.getViewSubImage(d,Zt);Ce=Ne.viewport,zt===0&&(t.setRenderTargetTextures(R,Ne.colorTexture,d.ignoreDepthValues?void 0:Ne.depthStencilTexture),t.setRenderTarget(R))}let ue=M[zt];ue===void 0&&(ue=new Je,ue.layers.enable(zt),ue.viewport=new ve,M[zt]=ue),ue.matrix.fromArray(Zt.transform.matrix),ue.matrix.decompose(ue.position,ue.quaternion,ue.scale),ue.projectionMatrix.fromArray(Zt.projectionMatrix),ue.projectionMatrixInverse.copy(ue.projectionMatrix).invert(),ue.viewport.set(Ce.x,Ce.y,Ce.width,Ce.height),zt===0&&(S.matrix.copy(ue.matrix),S.matrix.decompose(S.position,S.quaternion,S.scale)),nt===!0&&S.cameras.push(ue)}const _t=s.enabledFeatures;if(_t&&_t.includes("depth-sensing")){const zt=u.getDepthInformation(et[0]);zt&&zt.isValid&&zt.texture&&_.init(t,zt,s.renderState)}}for(let et=0;et<C.length;et++){const nt=x[et],_t=C[et];nt!==null&&_t!==void 0&&_t.update(nt,K,l||a)}pt&&pt(X,K),K.detectedPlanes&&n.dispatchEvent({type:"planesdetected",data:K}),g=null}const Et=new Jh;Et.setAnimationLoop(Ct),this.setAnimationLoop=function(X){pt=X},this.dispose=function(){}}}const Oi=new cn,e0=new Kt;function n0(i,t){function e(m,f){m.matrixAutoUpdate===!0&&m.updateMatrix(),f.value.copy(m.matrix)}function n(m,f){f.color.getRGB(m.fogColor.value,Yh(i)),f.isFog?(m.fogNear.value=f.near,m.fogFar.value=f.far):f.isFogExp2&&(m.fogDensity.value=f.density)}function s(m,f,R,C,x){f.isMeshBasicMaterial||f.isMeshLambertMaterial?r(m,f):f.isMeshToonMaterial?(r(m,f),u(m,f)):f.isMeshPhongMaterial?(r(m,f),h(m,f)):f.isMeshStandardMaterial?(r(m,f),d(m,f),f.isMeshPhysicalMaterial&&p(m,f,x)):f.isMeshMatcapMaterial?(r(m,f),g(m,f)):f.isMeshDepthMaterial?r(m,f):f.isMeshDistanceMaterial?(r(m,f),_(m,f)):f.isMeshNormalMaterial?r(m,f):f.isLineBasicMaterial?(a(m,f),f.isLineDashedMaterial&&o(m,f)):f.isPointsMaterial?c(m,f,R,C):f.isSpriteMaterial?l(m,f):f.isShadowMaterial?(m.color.value.copy(f.color),m.opacity.value=f.opacity):f.isShaderMaterial&&(f.uniformsNeedUpdate=!1)}function r(m,f){m.opacity.value=f.opacity,f.color&&m.diffuse.value.copy(f.color),f.emissive&&m.emissive.value.copy(f.emissive).multiplyScalar(f.emissiveIntensity),f.map&&(m.map.value=f.map,e(f.map,m.mapTransform)),f.alphaMap&&(m.alphaMap.value=f.alphaMap,e(f.alphaMap,m.alphaMapTransform)),f.bumpMap&&(m.bumpMap.value=f.bumpMap,e(f.bumpMap,m.bumpMapTransform),m.bumpScale.value=f.bumpScale,f.side===ln&&(m.bumpScale.value*=-1)),f.normalMap&&(m.normalMap.value=f.normalMap,e(f.normalMap,m.normalMapTransform),m.normalScale.value.copy(f.normalScale),f.side===ln&&m.normalScale.value.negate()),f.displacementMap&&(m.displacementMap.value=f.displacementMap,e(f.displacementMap,m.displacementMapTransform),m.displacementScale.value=f.displacementScale,m.displacementBias.value=f.displacementBias),f.emissiveMap&&(m.emissiveMap.value=f.emissiveMap,e(f.emissiveMap,m.emissiveMapTransform)),f.specularMap&&(m.specularMap.value=f.specularMap,e(f.specularMap,m.specularMapTransform)),f.alphaTest>0&&(m.alphaTest.value=f.alphaTest);const R=t.get(f),C=R.envMap,x=R.envMapRotation;C&&(m.envMap.value=C,Oi.copy(x),Oi.x*=-1,Oi.y*=-1,Oi.z*=-1,C.isCubeTexture&&C.isRenderTargetTexture===!1&&(Oi.y*=-1,Oi.z*=-1),m.envMapRotation.value.setFromMatrix4(e0.makeRotationFromEuler(Oi)),m.flipEnvMap.value=C.isCubeTexture&&C.isRenderTargetTexture===!1?-1:1,m.reflectivity.value=f.reflectivity,m.ior.value=f.ior,m.refractionRatio.value=f.refractionRatio),f.lightMap&&(m.lightMap.value=f.lightMap,m.lightMapIntensity.value=f.lightMapIntensity,e(f.lightMap,m.lightMapTransform)),f.aoMap&&(m.aoMap.value=f.aoMap,m.aoMapIntensity.value=f.aoMapIntensity,e(f.aoMap,m.aoMapTransform))}function a(m,f){m.diffuse.value.copy(f.color),m.opacity.value=f.opacity,f.map&&(m.map.value=f.map,e(f.map,m.mapTransform))}function o(m,f){m.dashSize.value=f.dashSize,m.totalSize.value=f.dashSize+f.gapSize,m.scale.value=f.scale}function c(m,f,R,C){m.diffuse.value.copy(f.color),m.opacity.value=f.opacity,m.size.value=f.size*R,m.scale.value=C*.5,f.map&&(m.map.value=f.map,e(f.map,m.uvTransform)),f.alphaMap&&(m.alphaMap.value=f.alphaMap,e(f.alphaMap,m.alphaMapTransform)),f.alphaTest>0&&(m.alphaTest.value=f.alphaTest)}function l(m,f){m.diffuse.value.copy(f.color),m.opacity.value=f.opacity,m.rotation.value=f.rotation,f.map&&(m.map.value=f.map,e(f.map,m.mapTransform)),f.alphaMap&&(m.alphaMap.value=f.alphaMap,e(f.alphaMap,m.alphaMapTransform)),f.alphaTest>0&&(m.alphaTest.value=f.alphaTest)}function h(m,f){m.specular.value.copy(f.specular),m.shininess.value=Math.max(f.shininess,1e-4)}function u(m,f){f.gradientMap&&(m.gradientMap.value=f.gradientMap)}function d(m,f){m.metalness.value=f.metalness,f.metalnessMap&&(m.metalnessMap.value=f.metalnessMap,e(f.metalnessMap,m.metalnessMapTransform)),m.roughness.value=f.roughness,f.roughnessMap&&(m.roughnessMap.value=f.roughnessMap,e(f.roughnessMap,m.roughnessMapTransform)),f.envMap&&(m.envMapIntensity.value=f.envMapIntensity)}function p(m,f,R){m.ior.value=f.ior,f.sheen>0&&(m.sheenColor.value.copy(f.sheenColor).multiplyScalar(f.sheen),m.sheenRoughness.value=f.sheenRoughness,f.sheenColorMap&&(m.sheenColorMap.value=f.sheenColorMap,e(f.sheenColorMap,m.sheenColorMapTransform)),f.sheenRoughnessMap&&(m.sheenRoughnessMap.value=f.sheenRoughnessMap,e(f.sheenRoughnessMap,m.sheenRoughnessMapTransform))),f.clearcoat>0&&(m.clearcoat.value=f.clearcoat,m.clearcoatRoughness.value=f.clearcoatRoughness,f.clearcoatMap&&(m.clearcoatMap.value=f.clearcoatMap,e(f.clearcoatMap,m.clearcoatMapTransform)),f.clearcoatRoughnessMap&&(m.clearcoatRoughnessMap.value=f.clearcoatRoughnessMap,e(f.clearcoatRoughnessMap,m.clearcoatRoughnessMapTransform)),f.clearcoatNormalMap&&(m.clearcoatNormalMap.value=f.clearcoatNormalMap,e(f.clearcoatNormalMap,m.clearcoatNormalMapTransform),m.clearcoatNormalScale.value.copy(f.clearcoatNormalScale),f.side===ln&&m.clearcoatNormalScale.value.negate())),f.dispersion>0&&(m.dispersion.value=f.dispersion),f.iridescence>0&&(m.iridescence.value=f.iridescence,m.iridescenceIOR.value=f.iridescenceIOR,m.iridescenceThicknessMinimum.value=f.iridescenceThicknessRange[0],m.iridescenceThicknessMaximum.value=f.iridescenceThicknessRange[1],f.iridescenceMap&&(m.iridescenceMap.value=f.iridescenceMap,e(f.iridescenceMap,m.iridescenceMapTransform)),f.iridescenceThicknessMap&&(m.iridescenceThicknessMap.value=f.iridescenceThicknessMap,e(f.iridescenceThicknessMap,m.iridescenceThicknessMapTransform))),f.transmission>0&&(m.transmission.value=f.transmission,m.transmissionSamplerMap.value=R.texture,m.transmissionSamplerSize.value.set(R.width,R.height),f.transmissionMap&&(m.transmissionMap.value=f.transmissionMap,e(f.transmissionMap,m.transmissionMapTransform)),m.thickness.value=f.thickness,f.thicknessMap&&(m.thicknessMap.value=f.thicknessMap,e(f.thicknessMap,m.thicknessMapTransform)),m.attenuationDistance.value=f.attenuationDistance,m.attenuationColor.value.copy(f.attenuationColor)),f.anisotropy>0&&(m.anisotropyVector.value.set(f.anisotropy*Math.cos(f.anisotropyRotation),f.anisotropy*Math.sin(f.anisotropyRotation)),f.anisotropyMap&&(m.anisotropyMap.value=f.anisotropyMap,e(f.anisotropyMap,m.anisotropyMapTransform))),m.specularIntensity.value=f.specularIntensity,m.specularColor.value.copy(f.specularColor),f.specularColorMap&&(m.specularColorMap.value=f.specularColorMap,e(f.specularColorMap,m.specularColorMapTransform)),f.specularIntensityMap&&(m.specularIntensityMap.value=f.specularIntensityMap,e(f.specularIntensityMap,m.specularIntensityMapTransform))}function g(m,f){f.matcap&&(m.matcap.value=f.matcap)}function _(m,f){const R=t.get(f).light;m.referencePosition.value.setFromMatrixPosition(R.matrixWorld),m.nearDistance.value=R.shadow.camera.near,m.farDistance.value=R.shadow.camera.far}return{refreshFogUniforms:n,refreshMaterialUniforms:s}}function i0(i,t,e,n){let s={},r={},a=[];const o=i.getParameter(i.MAX_UNIFORM_BUFFER_BINDINGS);function c(R,C){const x=C.program;n.uniformBlockBinding(R,x)}function l(R,C){let x=s[R.id];x===void 0&&(g(R),x=h(R),s[R.id]=x,R.addEventListener("dispose",m));const N=C.program;n.updateUBOMapping(R,N);const P=t.render.frame;r[R.id]!==P&&(d(R),r[R.id]=P)}function h(R){const C=u();R.__bindingPointIndex=C;const x=i.createBuffer(),N=R.__size,P=R.usage;return i.bindBuffer(i.UNIFORM_BUFFER,x),i.bufferData(i.UNIFORM_BUFFER,N,P),i.bindBuffer(i.UNIFORM_BUFFER,null),i.bindBufferBase(i.UNIFORM_BUFFER,C,x),x}function u(){for(let R=0;R<o;R++)if(a.indexOf(R)===-1)return a.push(R),R;return console.error("THREE.WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function d(R){const C=s[R.id],x=R.uniforms,N=R.__cache;i.bindBuffer(i.UNIFORM_BUFFER,C);for(let P=0,L=x.length;P<L;P++){const I=Array.isArray(x[P])?x[P]:[x[P]];for(let M=0,S=I.length;M<S;M++){const U=I[M];if(p(U,P,M,N)===!0){const k=U.__offset,H=Array.isArray(U.value)?U.value:[U.value];let J=0;for(let Q=0;Q<H.length;Q++){const O=H[Q],tt=_(O);typeof O=="number"||typeof O=="boolean"?(U.__data[0]=O,i.bufferSubData(i.UNIFORM_BUFFER,k+J,U.__data)):O.isMatrix3?(U.__data[0]=O.elements[0],U.__data[1]=O.elements[1],U.__data[2]=O.elements[2],U.__data[3]=0,U.__data[4]=O.elements[3],U.__data[5]=O.elements[4],U.__data[6]=O.elements[5],U.__data[7]=0,U.__data[8]=O.elements[6],U.__data[9]=O.elements[7],U.__data[10]=O.elements[8],U.__data[11]=0):(O.toArray(U.__data,J),J+=tt.storage/Float32Array.BYTES_PER_ELEMENT)}i.bufferSubData(i.UNIFORM_BUFFER,k,U.__data)}}}i.bindBuffer(i.UNIFORM_BUFFER,null)}function p(R,C,x,N){const P=R.value,L=C+"_"+x;if(N[L]===void 0)return typeof P=="number"||typeof P=="boolean"?N[L]=P:N[L]=P.clone(),!0;{const I=N[L];if(typeof P=="number"||typeof P=="boolean"){if(I!==P)return N[L]=P,!0}else if(I.equals(P)===!1)return I.copy(P),!0}return!1}function g(R){const C=R.uniforms;let x=0;const N=16;for(let L=0,I=C.length;L<I;L++){const M=Array.isArray(C[L])?C[L]:[C[L]];for(let S=0,U=M.length;S<U;S++){const k=M[S],H=Array.isArray(k.value)?k.value:[k.value];for(let J=0,Q=H.length;J<Q;J++){const O=H[J],tt=_(O),W=x%N,ot=W%tt.boundary,ft=W+ot;x+=ot,ft!==0&&N-ft<tt.storage&&(x+=N-ft),k.__data=new Float32Array(tt.storage/Float32Array.BYTES_PER_ELEMENT),k.__offset=x,x+=tt.storage}}}const P=x%N;return P>0&&(x+=N-P),R.__size=x,R.__cache={},this}function _(R){const C={boundary:0,storage:0};return typeof R=="number"||typeof R=="boolean"?(C.boundary=4,C.storage=4):R.isVector2?(C.boundary=8,C.storage=8):R.isVector3||R.isColor?(C.boundary=16,C.storage=12):R.isVector4?(C.boundary=16,C.storage=16):R.isMatrix3?(C.boundary=48,C.storage=48):R.isMatrix4?(C.boundary=64,C.storage=64):R.isTexture?console.warn("THREE.WebGLRenderer: Texture samplers can not be part of an uniforms group."):console.warn("THREE.WebGLRenderer: Unsupported uniform value type.",R),C}function m(R){const C=R.target;C.removeEventListener("dispose",m);const x=a.indexOf(C.__bindingPointIndex);a.splice(x,1),i.deleteBuffer(s[C.id]),delete s[C.id],delete r[C.id]}function f(){for(const R in s)i.deleteBuffer(s[R]);a=[],s={},r={}}return{bind:c,update:l,dispose:f}}class s0{constructor(t={}){const{canvas:e=Xd(),context:n=null,depth:s=!0,stencil:r=!1,alpha:a=!1,antialias:o=!1,premultipliedAlpha:c=!0,preserveDrawingBuffer:l=!1,powerPreference:h="default",failIfMajorPerformanceCaveat:u=!1,reverseDepthBuffer:d=!1}=t;this.isWebGLRenderer=!0;let p;if(n!==null){if(typeof WebGLRenderingContext<"u"&&n instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");p=n.getContextAttributes().alpha}else p=a;const g=new Uint32Array(4),_=new Int32Array(4);let m=null,f=null;const R=[],C=[];this.domElement=e,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this._outputColorSpace=Pe,this.toneMapping=bi,this.toneMappingExposure=1;const x=this;let N=!1,P=0,L=0,I=null,M=-1,S=null;const U=new ve,k=new ve;let H=null;const J=new $t(0);let Q=0,O=e.width,tt=e.height,W=1,ot=null,ft=null;const pt=new ve(0,0,O,tt),Ct=new ve(0,0,O,tt);let Et=!1;const X=new bl;let K=!1,et=!1;const nt=new Kt,_t=new Kt,zt=new F,Zt=new ve,Ce={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0};let ue=!1;function Ne(){return I===null?W:1}let G=n;function nn(w,V){return e.getContext(w,V)}try{const w={alpha:!0,depth:s,stencil:r,antialias:o,premultipliedAlpha:c,preserveDrawingBuffer:l,powerPreference:h,failIfMajorPerformanceCaveat:u};if("setAttribute"in e&&e.setAttribute("data-engine",`three.js r${ml}`),e.addEventListener("webglcontextlost",at,!1),e.addEventListener("webglcontextrestored",bt,!1),e.addEventListener("webglcontextcreationerror",Mt,!1),G===null){const V="webgl2";if(G=nn(V,w),G===null)throw nn(V)?new Error("Error creating WebGL context with your selected attributes."):new Error("Error creating WebGL context.")}}catch(w){throw console.error("THREE.WebGLRenderer: "+w.message),w}let re,ae,Wt,we,Ht,D,b,q,rt,lt,st,Ft,yt,At,de,dt,Rt,Xt,jt,Pt,ce,ee,ye,z;function vt(){re=new cg(G),re.init(),ee=new Y_(G,re),ae=new ig(G,re,t,ee),Wt=new X_(G,re),ae.reverseDepthBuffer&&d&&Wt.buffers.depth.setReversed(!0),we=new dg(G),Ht=new P_,D=new j_(G,re,Wt,Ht,ae,ee,we),b=new rg(x),q=new lg(x),rt=new xf(G),ye=new eg(G,rt),lt=new hg(G,rt,we,ye),st=new pg(G,lt,rt,we),jt=new fg(G,ae,D),dt=new sg(Ht),Ft=new R_(x,b,q,re,ae,ye,dt),yt=new n0(x,Ht),At=new N_,de=new k_(re),Xt=new tg(x,b,q,Wt,st,p,c),Rt=new G_(x,st,ae),z=new i0(G,we,ae,Wt),Pt=new ng(G,re,we),ce=new ug(G,re,we),we.programs=Ft.programs,x.capabilities=ae,x.extensions=re,x.properties=Ht,x.renderLists=At,x.shadowMap=Rt,x.state=Wt,x.info=we}vt();const Z=new t0(x,G);this.xr=Z,this.getContext=function(){return G},this.getContextAttributes=function(){return G.getContextAttributes()},this.forceContextLoss=function(){const w=re.get("WEBGL_lose_context");w&&w.loseContext()},this.forceContextRestore=function(){const w=re.get("WEBGL_lose_context");w&&w.restoreContext()},this.getPixelRatio=function(){return W},this.setPixelRatio=function(w){w!==void 0&&(W=w,this.setSize(O,tt,!1))},this.getSize=function(w){return w.set(O,tt)},this.setSize=function(w,V,j=!0){if(Z.isPresenting){console.warn("THREE.WebGLRenderer: Can't change size while VR device is presenting.");return}O=w,tt=V,e.width=Math.floor(w*W),e.height=Math.floor(V*W),j===!0&&(e.style.width=w+"px",e.style.height=V+"px"),this.setViewport(0,0,w,V)},this.getDrawingBufferSize=function(w){return w.set(O*W,tt*W).floor()},this.setDrawingBufferSize=function(w,V,j){O=w,tt=V,W=j,e.width=Math.floor(w*j),e.height=Math.floor(V*j),this.setViewport(0,0,w,V)},this.getCurrentViewport=function(w){return w.copy(U)},this.getViewport=function(w){return w.copy(pt)},this.setViewport=function(w,V,j,Y){w.isVector4?pt.set(w.x,w.y,w.z,w.w):pt.set(w,V,j,Y),Wt.viewport(U.copy(pt).multiplyScalar(W).round())},this.getScissor=function(w){return w.copy(Ct)},this.setScissor=function(w,V,j,Y){w.isVector4?Ct.set(w.x,w.y,w.z,w.w):Ct.set(w,V,j,Y),Wt.scissor(k.copy(Ct).multiplyScalar(W).round())},this.getScissorTest=function(){return Et},this.setScissorTest=function(w){Wt.setScissorTest(Et=w)},this.setOpaqueSort=function(w){ot=w},this.setTransparentSort=function(w){ft=w},this.getClearColor=function(w){return w.copy(Xt.getClearColor())},this.setClearColor=function(){Xt.setClearColor.apply(Xt,arguments)},this.getClearAlpha=function(){return Xt.getClearAlpha()},this.setClearAlpha=function(){Xt.setClearAlpha.apply(Xt,arguments)},this.clear=function(w=!0,V=!0,j=!0){let Y=0;if(w){let B=!1;if(I!==null){const ut=I.texture.format;B=ut===Ml||ut===yl||ut===xl}if(B){const ut=I.texture.type,xt=ut===si||ut===Gi||ut===ir||ut===bs||ut===_l||ut===vl,Dt=Xt.getClearColor(),It=Xt.getClearAlpha(),Yt=Dt.r,te=Dt.g,Ut=Dt.b;xt?(g[0]=Yt,g[1]=te,g[2]=Ut,g[3]=It,G.clearBufferuiv(G.COLOR,0,g)):(_[0]=Yt,_[1]=te,_[2]=Ut,_[3]=It,G.clearBufferiv(G.COLOR,0,_))}else Y|=G.COLOR_BUFFER_BIT}V&&(Y|=G.DEPTH_BUFFER_BIT),j&&(Y|=G.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),G.clear(Y)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.dispose=function(){e.removeEventListener("webglcontextlost",at,!1),e.removeEventListener("webglcontextrestored",bt,!1),e.removeEventListener("webglcontextcreationerror",Mt,!1),At.dispose(),de.dispose(),Ht.dispose(),b.dispose(),q.dispose(),st.dispose(),ye.dispose(),z.dispose(),Ft.dispose(),Z.dispose(),Z.removeEventListener("sessionstart",gr),Z.removeEventListener("sessionend",_r),Xn.stop()};function at(w){w.preventDefault(),console.log("THREE.WebGLRenderer: Context Lost."),N=!0}function bt(){console.log("THREE.WebGLRenderer: Context Restored."),N=!1;const w=we.autoReset,V=Rt.enabled,j=Rt.autoUpdate,Y=Rt.needsUpdate,B=Rt.type;vt(),we.autoReset=w,Rt.enabled=V,Rt.autoUpdate=j,Rt.needsUpdate=Y,Rt.type=B}function Mt(w){console.error("THREE.WebGLRenderer: A WebGL context could not be created. Reason: ",w.statusMessage)}function Qt(w){const V=w.target;V.removeEventListener("dispose",Qt),Re(V)}function Re(w){Ge(w),Ht.remove(w)}function Ge(w){const V=Ht.get(w).programs;V!==void 0&&(V.forEach(function(j){Ft.releaseProgram(j)}),w.isShaderMaterial&&Ft.releaseShaderCache(w))}this.renderBufferDirect=function(w,V,j,Y,B,ut){V===null&&(V=Ce);const xt=B.isMesh&&B.matrixWorld.determinant()<0,Dt=Na(w,V,j,Y,B);Wt.setMaterial(Y,xt);let It=j.index,Yt=1;if(Y.wireframe===!0){if(It=lt.getWireframeAttribute(j),It===void 0)return;Yt=2}const te=j.drawRange,Ut=j.attributes.position;let fe=te.start*Yt,Me=(te.start+te.count)*Yt;ut!==null&&(fe=Math.max(fe,ut.start*Yt),Me=Math.min(Me,(ut.start+ut.count)*Yt)),It!==null?(fe=Math.max(fe,0),Me=Math.min(Me,It.count)):Ut!=null&&(fe=Math.max(fe,0),Me=Math.min(Me,Ut.count));const be=Me-fe;if(be<0||be===1/0)return;ye.setup(B,Y,Dt,j,It);let Ue,me=Pt;if(It!==null&&(Ue=rt.get(It),me=ce,me.setIndex(Ue)),B.isMesh)Y.wireframe===!0?(Wt.setLineWidth(Y.wireframeLinewidth*Ne()),me.setMode(G.LINES)):me.setMode(G.TRIANGLES);else if(B.isLine){let Ot=Y.linewidth;Ot===void 0&&(Ot=1),Wt.setLineWidth(Ot*Ne()),B.isLineSegments?me.setMode(G.LINES):B.isLineLoop?me.setMode(G.LINE_LOOP):me.setMode(G.LINE_STRIP)}else B.isPoints?me.setMode(G.POINTS):B.isSprite&&me.setMode(G.TRIANGLES);if(B.isBatchedMesh)if(B._multiDrawInstances!==null)me.renderMultiDrawInstances(B._multiDrawStarts,B._multiDrawCounts,B._multiDrawCount,B._multiDrawInstances);else if(re.get("WEBGL_multi_draw"))me.renderMultiDraw(B._multiDrawStarts,B._multiDrawCounts,B._multiDrawCount);else{const Ot=B._multiDrawStarts,Tn=B._multiDrawCounts,pe=B._multiDrawCount,hn=It?rt.get(It).bytesPerElement:1,li=Ht.get(Y).currentProgram.getUniforms();for(let Ze=0;Ze<pe;Ze++)li.setValue(G,"_gl_DrawID",Ze),me.render(Ot[Ze]/hn,Tn[Ze])}else if(B.isInstancedMesh)me.renderInstances(fe,be,B.count);else if(j.isInstancedBufferGeometry){const Ot=j._maxInstanceCount!==void 0?j._maxInstanceCount:1/0,Tn=Math.min(j.instanceCount,Ot);me.renderInstances(fe,be,Tn)}else me.render(fe,be)};function ne(w,V,j){w.transparent===!0&&w.side===Pn&&w.forceSinglePass===!1?(w.side=ln,w.needsUpdate=!0,$i(w,V,j),w.side=ii,w.needsUpdate=!0,$i(w,V,j),w.side=Pn):$i(w,V,j)}this.compile=function(w,V,j=null){j===null&&(j=w),f=de.get(j),f.init(V),C.push(f),j.traverseVisible(function(B){B.isLight&&B.layers.test(V.layers)&&(f.pushLight(B),B.castShadow&&f.pushShadow(B))}),w!==j&&w.traverseVisible(function(B){B.isLight&&B.layers.test(V.layers)&&(f.pushLight(B),B.castShadow&&f.pushShadow(B))}),f.setupLights();const Y=new Set;return w.traverse(function(B){if(!(B.isMesh||B.isPoints||B.isLine||B.isSprite))return;const ut=B.material;if(ut)if(Array.isArray(ut))for(let xt=0;xt<ut.length;xt++){const Dt=ut[xt];ne(Dt,j,B),Y.add(Dt)}else ne(ut,j,B),Y.add(ut)}),C.pop(),f=null,Y},this.compileAsync=function(w,V,j=null){const Y=this.compile(w,V,j);return new Promise(B=>{function ut(){if(Y.forEach(function(xt){Ht.get(xt).currentProgram.isReady()&&Y.delete(xt)}),Y.size===0){B(w);return}setTimeout(ut,10)}re.get("KHR_parallel_shader_compile")!==null?ut():setTimeout(ut,10)})};let sn=null;function En(w){sn&&sn(w)}function gr(){Xn.stop()}function _r(){Xn.start()}const Xn=new Jh;Xn.setAnimationLoop(En),typeof self<"u"&&Xn.setContext(self),this.setAnimationLoop=function(w){sn=w,Z.setAnimationLoop(w),w===null?Xn.stop():Xn.start()},Z.addEventListener("sessionstart",gr),Z.addEventListener("sessionend",_r),this.render=function(w,V){if(V!==void 0&&V.isCamera!==!0){console.error("THREE.WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(N===!0)return;if(w.matrixWorldAutoUpdate===!0&&w.updateMatrixWorld(),V.parent===null&&V.matrixWorldAutoUpdate===!0&&V.updateMatrixWorld(),Z.enabled===!0&&Z.isPresenting===!0&&(Z.cameraAutoUpdate===!0&&Z.updateCamera(V),V=Z.getCamera()),w.isScene===!0&&w.onBeforeRender(x,w,V,I),f=de.get(w,C.length),f.init(V),C.push(f),_t.multiplyMatrices(V.projectionMatrix,V.matrixWorldInverse),X.setFromProjectionMatrix(_t),et=this.localClippingEnabled,K=dt.init(this.clippingPlanes,et),m=At.get(w,R.length),m.init(),R.push(m),Z.enabled===!0&&Z.isPresenting===!0){const ut=x.xr.getDepthSensingMesh();ut!==null&&Bs(ut,V,-1/0,x.sortObjects)}Bs(w,V,0,x.sortObjects),m.finish(),x.sortObjects===!0&&m.sort(ot,ft),ue=Z.enabled===!1||Z.isPresenting===!1||Z.hasDepthSensing()===!1,ue&&Xt.addToRenderList(m,w),this.info.render.frame++,K===!0&&dt.beginShadows();const j=f.state.shadowsArray;Rt.render(j,w,V),K===!0&&dt.endShadows(),this.info.autoReset===!0&&this.info.reset();const Y=m.opaque,B=m.transmissive;if(f.setupLights(),V.isArrayCamera){const ut=V.cameras;if(B.length>0)for(let xt=0,Dt=ut.length;xt<Dt;xt++){const It=ut[xt];zs(Y,B,w,It)}ue&&Xt.render(w);for(let xt=0,Dt=ut.length;xt<Dt;xt++){const It=ut[xt];vr(m,w,It,It.viewport)}}else B.length>0&&zs(Y,B,w,V),ue&&Xt.render(w),vr(m,w,V);I!==null&&(D.updateMultisampleRenderTarget(I),D.updateRenderTargetMipmap(I)),w.isScene===!0&&w.onAfterRender(x,w,V),ye.resetDefaultState(),M=-1,S=null,C.pop(),C.length>0?(f=C[C.length-1],K===!0&&dt.setGlobalState(x.clippingPlanes,f.state.camera)):f=null,R.pop(),R.length>0?m=R[R.length-1]:m=null};function Bs(w,V,j,Y){if(w.visible===!1)return;if(w.layers.test(V.layers)){if(w.isGroup)j=w.renderOrder;else if(w.isLOD)w.autoUpdate===!0&&w.update(V);else if(w.isLight)f.pushLight(w),w.castShadow&&f.pushShadow(w);else if(w.isSprite){if(!w.frustumCulled||X.intersectsSprite(w)){Y&&Zt.setFromMatrixPosition(w.matrixWorld).applyMatrix4(_t);const xt=st.update(w),Dt=w.material;Dt.visible&&m.push(w,xt,Dt,j,Zt.z,null)}}else if((w.isMesh||w.isLine||w.isPoints)&&(!w.frustumCulled||X.intersectsObject(w))){const xt=st.update(w),Dt=w.material;if(Y&&(w.boundingSphere!==void 0?(w.boundingSphere===null&&w.computeBoundingSphere(),Zt.copy(w.boundingSphere.center)):(xt.boundingSphere===null&&xt.computeBoundingSphere(),Zt.copy(xt.boundingSphere.center)),Zt.applyMatrix4(w.matrixWorld).applyMatrix4(_t)),Array.isArray(Dt)){const It=xt.groups;for(let Yt=0,te=It.length;Yt<te;Yt++){const Ut=It[Yt],fe=Dt[Ut.materialIndex];fe&&fe.visible&&m.push(w,xt,fe,j,Zt.z,Ut)}}else Dt.visible&&m.push(w,xt,Dt,j,Zt.z,null)}}const ut=w.children;for(let xt=0,Dt=ut.length;xt<Dt;xt++)Bs(ut[xt],V,j,Y)}function vr(w,V,j,Y){const B=w.opaque,ut=w.transmissive,xt=w.transparent;f.setupLightsView(j),K===!0&&dt.setGlobalState(x.clippingPlanes,j),Y&&Wt.viewport(U.copy(Y)),B.length>0&&Yi(B,V,j),ut.length>0&&Yi(ut,V,j),xt.length>0&&Yi(xt,V,j),Wt.buffers.depth.setTest(!0),Wt.buffers.depth.setMask(!0),Wt.buffers.color.setMask(!0),Wt.setPolygonOffset(!1)}function zs(w,V,j,Y){if((j.isScene===!0?j.overrideMaterial:null)!==null)return;f.state.transmissionRenderTarget[Y.id]===void 0&&(f.state.transmissionRenderTarget[Y.id]=new Wi(1,1,{generateMipmaps:!0,type:re.has("EXT_color_buffer_half_float")||re.has("EXT_color_buffer_float")?hr:si,minFilter:ti,samples:4,stencilBuffer:r,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:le.workingColorSpace}));const ut=f.state.transmissionRenderTarget[Y.id],xt=Y.viewport||U;ut.setSize(xt.z,xt.w);const Dt=x.getRenderTarget();x.setRenderTarget(ut),x.getClearColor(J),Q=x.getClearAlpha(),Q<1&&x.setClearColor(16777215,.5),x.clear(),ue&&Xt.render(j);const It=x.toneMapping;x.toneMapping=bi;const Yt=Y.viewport;if(Y.viewport!==void 0&&(Y.viewport=void 0),f.setupLightsView(Y),K===!0&&dt.setGlobalState(x.clippingPlanes,Y),Yi(w,j,Y),D.updateMultisampleRenderTarget(ut),D.updateRenderTargetMipmap(ut),re.has("WEBGL_multisampled_render_to_texture")===!1){let te=!1;for(let Ut=0,fe=V.length;Ut<fe;Ut++){const Me=V[Ut],be=Me.object,Ue=Me.geometry,me=Me.material,Ot=Me.group;if(me.side===Pn&&be.layers.test(Y.layers)){const Tn=me.side;me.side=ln,me.needsUpdate=!0,Vs(be,j,Y,Ue,me,Ot),me.side=Tn,me.needsUpdate=!0,te=!0}}te===!0&&(D.updateMultisampleRenderTarget(ut),D.updateRenderTargetMipmap(ut))}x.setRenderTarget(Dt),x.setClearColor(J,Q),Yt!==void 0&&(Y.viewport=Yt),x.toneMapping=It}function Yi(w,V,j){const Y=V.isScene===!0?V.overrideMaterial:null;for(let B=0,ut=w.length;B<ut;B++){const xt=w[B],Dt=xt.object,It=xt.geometry,Yt=Y===null?xt.material:Y,te=xt.group;Dt.layers.test(j.layers)&&Vs(Dt,V,j,It,Yt,te)}}function Vs(w,V,j,Y,B,ut){w.onBeforeRender(x,V,j,Y,B,ut),w.modelViewMatrix.multiplyMatrices(j.matrixWorldInverse,w.matrixWorld),w.normalMatrix.getNormalMatrix(w.modelViewMatrix),B.onBeforeRender(x,V,j,Y,w,ut),B.transparent===!0&&B.side===Pn&&B.forceSinglePass===!1?(B.side=ln,B.needsUpdate=!0,x.renderBufferDirect(j,V,Y,B,w,ut),B.side=ii,B.needsUpdate=!0,x.renderBufferDirect(j,V,Y,B,w,ut),B.side=Pn):x.renderBufferDirect(j,V,Y,B,w,ut),w.onAfterRender(x,V,j,Y,B,ut)}function $i(w,V,j){V.isScene!==!0&&(V=Ce);const Y=Ht.get(w),B=f.state.lights,ut=f.state.shadowsArray,xt=B.state.version,Dt=Ft.getParameters(w,B.state,ut,V,j),It=Ft.getProgramCacheKey(Dt);let Yt=Y.programs;Y.environment=w.isMeshStandardMaterial?V.environment:null,Y.fog=V.fog,Y.envMap=(w.isMeshStandardMaterial?q:b).get(w.envMap||Y.environment),Y.envMapRotation=Y.environment!==null&&w.envMap===null?V.environmentRotation:w.envMapRotation,Yt===void 0&&(w.addEventListener("dispose",Qt),Yt=new Map,Y.programs=Yt);let te=Yt.get(It);if(te!==void 0){if(Y.currentProgram===te&&Y.lightsStateVersion===xt)return yr(w,Dt),te}else Dt.uniforms=Ft.getUniforms(w),w.onBeforeCompile(Dt,x),te=Ft.acquireProgram(Dt,It),Yt.set(It,te),Y.uniforms=Dt.uniforms;const Ut=Y.uniforms;return(!w.isShaderMaterial&&!w.isRawShaderMaterial||w.clipping===!0)&&(Ut.clippingPlanes=dt.uniform),yr(w,Dt),Y.needsLights=Ia(w),Y.lightsStateVersion=xt,Y.needsLights&&(Ut.ambientLightColor.value=B.state.ambient,Ut.lightProbe.value=B.state.probe,Ut.directionalLights.value=B.state.directional,Ut.directionalLightShadows.value=B.state.directionalShadow,Ut.spotLights.value=B.state.spot,Ut.spotLightShadows.value=B.state.spotShadow,Ut.rectAreaLights.value=B.state.rectArea,Ut.ltc_1.value=B.state.rectAreaLTC1,Ut.ltc_2.value=B.state.rectAreaLTC2,Ut.pointLights.value=B.state.point,Ut.pointLightShadows.value=B.state.pointShadow,Ut.hemisphereLights.value=B.state.hemi,Ut.directionalShadowMap.value=B.state.directionalShadowMap,Ut.directionalShadowMatrix.value=B.state.directionalShadowMatrix,Ut.spotShadowMap.value=B.state.spotShadowMap,Ut.spotLightMatrix.value=B.state.spotLightMatrix,Ut.spotLightMap.value=B.state.spotLightMap,Ut.pointShadowMap.value=B.state.pointShadowMap,Ut.pointShadowMatrix.value=B.state.pointShadowMatrix),Y.currentProgram=te,Y.uniformsList=null,te}function xr(w){if(w.uniformsList===null){const V=w.currentProgram.getUniforms();w.uniformsList=ca.seqWithValue(V.seq,w.uniforms)}return w.uniformsList}function yr(w,V){const j=Ht.get(w);j.outputColorSpace=V.outputColorSpace,j.batching=V.batching,j.batchingColor=V.batchingColor,j.instancing=V.instancing,j.instancingColor=V.instancingColor,j.instancingMorph=V.instancingMorph,j.skinning=V.skinning,j.morphTargets=V.morphTargets,j.morphNormals=V.morphNormals,j.morphColors=V.morphColors,j.morphTargetsCount=V.morphTargetsCount,j.numClippingPlanes=V.numClippingPlanes,j.numIntersection=V.numClipIntersection,j.vertexAlphas=V.vertexAlphas,j.vertexTangents=V.vertexTangents,j.toneMapping=V.toneMapping}function Na(w,V,j,Y,B){V.isScene!==!0&&(V=Ce),D.resetTextureUnits();const ut=V.fog,xt=Y.isMeshStandardMaterial?V.environment:null,Dt=I===null?x.outputColorSpace:I.isXRRenderTarget===!0?I.texture.colorSpace:Ds,It=(Y.isMeshStandardMaterial?q:b).get(Y.envMap||xt),Yt=Y.vertexColors===!0&&!!j.attributes.color&&j.attributes.color.itemSize===4,te=!!j.attributes.tangent&&(!!Y.normalMap||Y.anisotropy>0),Ut=!!j.morphAttributes.position,fe=!!j.morphAttributes.normal,Me=!!j.morphAttributes.color;let be=bi;Y.toneMapped&&(I===null||I.isXRRenderTarget===!0)&&(be=x.toneMapping);const Ue=j.morphAttributes.position||j.morphAttributes.normal||j.morphAttributes.color,me=Ue!==void 0?Ue.length:0,Ot=Ht.get(Y),Tn=f.state.lights;if(K===!0&&(et===!0||w!==S)){const Qe=w===S&&Y.id===M;dt.setState(Y,w,Qe)}let pe=!1;Y.version===Ot.__version?(Ot.needsLights&&Ot.lightsStateVersion!==Tn.state.version||Ot.outputColorSpace!==Dt||B.isBatchedMesh&&Ot.batching===!1||!B.isBatchedMesh&&Ot.batching===!0||B.isBatchedMesh&&Ot.batchingColor===!0&&B.colorTexture===null||B.isBatchedMesh&&Ot.batchingColor===!1&&B.colorTexture!==null||B.isInstancedMesh&&Ot.instancing===!1||!B.isInstancedMesh&&Ot.instancing===!0||B.isSkinnedMesh&&Ot.skinning===!1||!B.isSkinnedMesh&&Ot.skinning===!0||B.isInstancedMesh&&Ot.instancingColor===!0&&B.instanceColor===null||B.isInstancedMesh&&Ot.instancingColor===!1&&B.instanceColor!==null||B.isInstancedMesh&&Ot.instancingMorph===!0&&B.morphTexture===null||B.isInstancedMesh&&Ot.instancingMorph===!1&&B.morphTexture!==null||Ot.envMap!==It||Y.fog===!0&&Ot.fog!==ut||Ot.numClippingPlanes!==void 0&&(Ot.numClippingPlanes!==dt.numPlanes||Ot.numIntersection!==dt.numIntersection)||Ot.vertexAlphas!==Yt||Ot.vertexTangents!==te||Ot.morphTargets!==Ut||Ot.morphNormals!==fe||Ot.morphColors!==Me||Ot.toneMapping!==be||Ot.morphTargetsCount!==me)&&(pe=!0):(pe=!0,Ot.__version=Y.version);let hn=Ot.currentProgram;pe===!0&&(hn=$i(Y,V,B));let li=!1,Ze=!1,ci=!1;const Jt=hn.getUniforms(),rn=Ot.uniforms;if(Wt.useProgram(hn.program)&&(li=!0,Ze=!0,ci=!0),Y.id!==M&&(M=Y.id,Ze=!0),li||S!==w){Wt.buffers.depth.getReversed()?(nt.copy(w.projectionMatrix),jd(nt),Yd(nt),Jt.setValue(G,"projectionMatrix",nt)):Jt.setValue(G,"projectionMatrix",w.projectionMatrix),Jt.setValue(G,"viewMatrix",w.matrixWorldInverse);const wn=Jt.map.cameraPosition;wn!==void 0&&wn.setValue(G,zt.setFromMatrixPosition(w.matrixWorld)),ae.logarithmicDepthBuffer&&Jt.setValue(G,"logDepthBufFC",2/(Math.log(w.far+1)/Math.LN2)),(Y.isMeshPhongMaterial||Y.isMeshToonMaterial||Y.isMeshLambertMaterial||Y.isMeshBasicMaterial||Y.isMeshStandardMaterial||Y.isShaderMaterial)&&Jt.setValue(G,"isOrthographic",w.isOrthographicCamera===!0),S!==w&&(S=w,Ze=!0,ci=!0)}if(B.isSkinnedMesh){Jt.setOptional(G,B,"bindMatrix"),Jt.setOptional(G,B,"bindMatrixInverse");const Qe=B.skeleton;Qe&&(Qe.boneTexture===null&&Qe.computeBoneTexture(),Jt.setValue(G,"boneTexture",Qe.boneTexture,D))}B.isBatchedMesh&&(Jt.setOptional(G,B,"batchingTexture"),Jt.setValue(G,"batchingTexture",B._matricesTexture,D),Jt.setOptional(G,B,"batchingIdTexture"),Jt.setValue(G,"batchingIdTexture",B._indirectTexture,D),Jt.setOptional(G,B,"batchingColorTexture"),B._colorsTexture!==null&&Jt.setValue(G,"batchingColorTexture",B._colorsTexture,D));const Pi=j.morphAttributes;if((Pi.position!==void 0||Pi.normal!==void 0||Pi.color!==void 0)&&jt.update(B,j,hn),(Ze||Ot.receiveShadow!==B.receiveShadow)&&(Ot.receiveShadow=B.receiveShadow,Jt.setValue(G,"receiveShadow",B.receiveShadow)),Y.isMeshGouraudMaterial&&Y.envMap!==null&&(rn.envMap.value=It,rn.flipEnvMap.value=It.isCubeTexture&&It.isRenderTargetTexture===!1?-1:1),Y.isMeshStandardMaterial&&Y.envMap===null&&V.environment!==null&&(rn.envMapIntensity.value=V.environmentIntensity),Ze&&(Jt.setValue(G,"toneMappingExposure",x.toneMappingExposure),Ot.needsLights&&Da(rn,ci),ut&&Y.fog===!0&&yt.refreshFogUniforms(rn,ut),yt.refreshMaterialUniforms(rn,Y,W,tt,f.state.transmissionRenderTarget[w.id]),ca.upload(G,xr(Ot),rn,D)),Y.isShaderMaterial&&Y.uniformsNeedUpdate===!0&&(ca.upload(G,xr(Ot),rn,D),Y.uniformsNeedUpdate=!1),Y.isSpriteMaterial&&Jt.setValue(G,"center",B.center),Jt.setValue(G,"modelViewMatrix",B.modelViewMatrix),Jt.setValue(G,"normalMatrix",B.normalMatrix),Jt.setValue(G,"modelMatrix",B.matrixWorld),Y.isShaderMaterial||Y.isRawShaderMaterial){const Qe=Y.uniformsGroups;for(let wn=0,xn=Qe.length;wn<xn;wn++){const Li=Qe[wn];z.update(Li,hn),z.bind(Li,hn)}}return hn}function Da(w,V){w.ambientLightColor.needsUpdate=V,w.lightProbe.needsUpdate=V,w.directionalLights.needsUpdate=V,w.directionalLightShadows.needsUpdate=V,w.pointLights.needsUpdate=V,w.pointLightShadows.needsUpdate=V,w.spotLights.needsUpdate=V,w.spotLightShadows.needsUpdate=V,w.rectAreaLights.needsUpdate=V,w.hemisphereLights.needsUpdate=V}function Ia(w){return w.isMeshLambertMaterial||w.isMeshToonMaterial||w.isMeshPhongMaterial||w.isMeshStandardMaterial||w.isShadowMaterial||w.isShaderMaterial&&w.lights===!0}this.getActiveCubeFace=function(){return P},this.getActiveMipmapLevel=function(){return L},this.getRenderTarget=function(){return I},this.setRenderTargetTextures=function(w,V,j){Ht.get(w.texture).__webglTexture=V,Ht.get(w.depthTexture).__webglTexture=j;const Y=Ht.get(w);Y.__hasExternalTextures=!0,Y.__autoAllocateDepthBuffer=j===void 0,Y.__autoAllocateDepthBuffer||re.has("WEBGL_multisampled_render_to_texture")===!0&&(console.warn("THREE.WebGLRenderer: Render-to-texture extension was disabled because an external texture was provided"),Y.__useRenderToTexture=!1)},this.setRenderTargetFramebuffer=function(w,V){const j=Ht.get(w);j.__webglFramebuffer=V,j.__useDefaultFramebuffer=V===void 0},this.setRenderTarget=function(w,V=0,j=0){I=w,P=V,L=j;let Y=!0,B=null,ut=!1,xt=!1;if(w){const It=Ht.get(w);if(It.__useDefaultFramebuffer!==void 0)Wt.bindFramebuffer(G.FRAMEBUFFER,null),Y=!1;else if(It.__webglFramebuffer===void 0)D.setupRenderTarget(w);else if(It.__hasExternalTextures)D.rebindTextures(w,Ht.get(w.texture).__webglTexture,Ht.get(w.depthTexture).__webglTexture);else if(w.depthBuffer){const Ut=w.depthTexture;if(It.__boundDepthTexture!==Ut){if(Ut!==null&&Ht.has(Ut)&&(w.width!==Ut.image.width||w.height!==Ut.image.height))throw new Error("WebGLRenderTarget: Attached DepthTexture is initialized to the incorrect size.");D.setupDepthRenderbuffer(w)}}const Yt=w.texture;(Yt.isData3DTexture||Yt.isDataArrayTexture||Yt.isCompressedArrayTexture)&&(xt=!0);const te=Ht.get(w).__webglFramebuffer;w.isWebGLCubeRenderTarget?(Array.isArray(te[V])?B=te[V][j]:B=te[V],ut=!0):w.samples>0&&D.useMultisampledRTT(w)===!1?B=Ht.get(w).__webglMultisampledFramebuffer:Array.isArray(te)?B=te[j]:B=te,U.copy(w.viewport),k.copy(w.scissor),H=w.scissorTest}else U.copy(pt).multiplyScalar(W).floor(),k.copy(Ct).multiplyScalar(W).floor(),H=Et;if(Wt.bindFramebuffer(G.FRAMEBUFFER,B)&&Y&&Wt.drawBuffers(w,B),Wt.viewport(U),Wt.scissor(k),Wt.setScissorTest(H),ut){const It=Ht.get(w.texture);G.framebufferTexture2D(G.FRAMEBUFFER,G.COLOR_ATTACHMENT0,G.TEXTURE_CUBE_MAP_POSITIVE_X+V,It.__webglTexture,j)}else if(xt){const It=Ht.get(w.texture),Yt=V||0;G.framebufferTextureLayer(G.FRAMEBUFFER,G.COLOR_ATTACHMENT0,It.__webglTexture,j||0,Yt)}M=-1},this.readRenderTargetPixels=function(w,V,j,Y,B,ut,xt){if(!(w&&w.isWebGLRenderTarget)){console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let Dt=Ht.get(w).__webglFramebuffer;if(w.isWebGLCubeRenderTarget&&xt!==void 0&&(Dt=Dt[xt]),Dt){Wt.bindFramebuffer(G.FRAMEBUFFER,Dt);try{const It=w.texture,Yt=It.format,te=It.type;if(!ae.textureFormatReadable(Yt)){console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(!ae.textureTypeReadable(te)){console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}V>=0&&V<=w.width-Y&&j>=0&&j<=w.height-B&&G.readPixels(V,j,Y,B,ee.convert(Yt),ee.convert(te),ut)}finally{const It=I!==null?Ht.get(I).__webglFramebuffer:null;Wt.bindFramebuffer(G.FRAMEBUFFER,It)}}},this.readRenderTargetPixelsAsync=async function(w,V,j,Y,B,ut,xt){if(!(w&&w.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let Dt=Ht.get(w).__webglFramebuffer;if(w.isWebGLCubeRenderTarget&&xt!==void 0&&(Dt=Dt[xt]),Dt){const It=w.texture,Yt=It.format,te=It.type;if(!ae.textureFormatReadable(Yt))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(!ae.textureTypeReadable(te))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");if(V>=0&&V<=w.width-Y&&j>=0&&j<=w.height-B){Wt.bindFramebuffer(G.FRAMEBUFFER,Dt);const Ut=G.createBuffer();G.bindBuffer(G.PIXEL_PACK_BUFFER,Ut),G.bufferData(G.PIXEL_PACK_BUFFER,ut.byteLength,G.STREAM_READ),G.readPixels(V,j,Y,B,ee.convert(Yt),ee.convert(te),0);const fe=I!==null?Ht.get(I).__webglFramebuffer:null;Wt.bindFramebuffer(G.FRAMEBUFFER,fe);const Me=G.fenceSync(G.SYNC_GPU_COMMANDS_COMPLETE,0);return G.flush(),await qd(G,Me,4),G.bindBuffer(G.PIXEL_PACK_BUFFER,Ut),G.getBufferSubData(G.PIXEL_PACK_BUFFER,0,ut),G.deleteBuffer(Ut),G.deleteSync(Me),ut}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")}},this.copyFramebufferToTexture=function(w,V=null,j=0){w.isTexture!==!0&&(Ys("WebGLRenderer: copyFramebufferToTexture function signature has changed."),V=arguments[0]||null,w=arguments[1]);const Y=Math.pow(2,-j),B=Math.floor(w.image.width*Y),ut=Math.floor(w.image.height*Y),xt=V!==null?V.x:0,Dt=V!==null?V.y:0;D.setTexture2D(w,0),G.copyTexSubImage2D(G.TEXTURE_2D,j,0,0,xt,Dt,B,ut),Wt.unbindTexture()},this.copyTextureToTexture=function(w,V,j=null,Y=null,B=0){w.isTexture!==!0&&(Ys("WebGLRenderer: copyTextureToTexture function signature has changed."),Y=arguments[0]||null,w=arguments[1],V=arguments[2],B=arguments[3]||0,j=null);let ut,xt,Dt,It,Yt,te,Ut,fe,Me;const be=w.isCompressedTexture?w.mipmaps[B]:w.image;j!==null?(ut=j.max.x-j.min.x,xt=j.max.y-j.min.y,Dt=j.isBox3?j.max.z-j.min.z:1,It=j.min.x,Yt=j.min.y,te=j.isBox3?j.min.z:0):(ut=be.width,xt=be.height,Dt=be.depth||1,It=0,Yt=0,te=0),Y!==null?(Ut=Y.x,fe=Y.y,Me=Y.z):(Ut=0,fe=0,Me=0);const Ue=ee.convert(V.format),me=ee.convert(V.type);let Ot;V.isData3DTexture?(D.setTexture3D(V,0),Ot=G.TEXTURE_3D):V.isDataArrayTexture||V.isCompressedArrayTexture?(D.setTexture2DArray(V,0),Ot=G.TEXTURE_2D_ARRAY):(D.setTexture2D(V,0),Ot=G.TEXTURE_2D),G.pixelStorei(G.UNPACK_FLIP_Y_WEBGL,V.flipY),G.pixelStorei(G.UNPACK_PREMULTIPLY_ALPHA_WEBGL,V.premultiplyAlpha),G.pixelStorei(G.UNPACK_ALIGNMENT,V.unpackAlignment);const Tn=G.getParameter(G.UNPACK_ROW_LENGTH),pe=G.getParameter(G.UNPACK_IMAGE_HEIGHT),hn=G.getParameter(G.UNPACK_SKIP_PIXELS),li=G.getParameter(G.UNPACK_SKIP_ROWS),Ze=G.getParameter(G.UNPACK_SKIP_IMAGES);G.pixelStorei(G.UNPACK_ROW_LENGTH,be.width),G.pixelStorei(G.UNPACK_IMAGE_HEIGHT,be.height),G.pixelStorei(G.UNPACK_SKIP_PIXELS,It),G.pixelStorei(G.UNPACK_SKIP_ROWS,Yt),G.pixelStorei(G.UNPACK_SKIP_IMAGES,te);const ci=w.isDataArrayTexture||w.isData3DTexture,Jt=V.isDataArrayTexture||V.isData3DTexture;if(w.isRenderTargetTexture||w.isDepthTexture){const rn=Ht.get(w),Pi=Ht.get(V),Qe=Ht.get(rn.__renderTarget),wn=Ht.get(Pi.__renderTarget);Wt.bindFramebuffer(G.READ_FRAMEBUFFER,Qe.__webglFramebuffer),Wt.bindFramebuffer(G.DRAW_FRAMEBUFFER,wn.__webglFramebuffer);for(let xn=0;xn<Dt;xn++)ci&&G.framebufferTextureLayer(G.READ_FRAMEBUFFER,G.COLOR_ATTACHMENT0,Ht.get(w).__webglTexture,B,te+xn),w.isDepthTexture?(Jt&&G.framebufferTextureLayer(G.DRAW_FRAMEBUFFER,G.COLOR_ATTACHMENT0,Ht.get(V).__webglTexture,B,Me+xn),G.blitFramebuffer(It,Yt,ut,xt,Ut,fe,ut,xt,G.DEPTH_BUFFER_BIT,G.NEAREST)):Jt?G.copyTexSubImage3D(Ot,B,Ut,fe,Me+xn,It,Yt,ut,xt):G.copyTexSubImage2D(Ot,B,Ut,fe,Me+xn,It,Yt,ut,xt);Wt.bindFramebuffer(G.READ_FRAMEBUFFER,null),Wt.bindFramebuffer(G.DRAW_FRAMEBUFFER,null)}else Jt?w.isDataTexture||w.isData3DTexture?G.texSubImage3D(Ot,B,Ut,fe,Me,ut,xt,Dt,Ue,me,be.data):V.isCompressedArrayTexture?G.compressedTexSubImage3D(Ot,B,Ut,fe,Me,ut,xt,Dt,Ue,be.data):G.texSubImage3D(Ot,B,Ut,fe,Me,ut,xt,Dt,Ue,me,be):w.isDataTexture?G.texSubImage2D(G.TEXTURE_2D,B,Ut,fe,ut,xt,Ue,me,be.data):w.isCompressedTexture?G.compressedTexSubImage2D(G.TEXTURE_2D,B,Ut,fe,be.width,be.height,Ue,be.data):G.texSubImage2D(G.TEXTURE_2D,B,Ut,fe,ut,xt,Ue,me,be);G.pixelStorei(G.UNPACK_ROW_LENGTH,Tn),G.pixelStorei(G.UNPACK_IMAGE_HEIGHT,pe),G.pixelStorei(G.UNPACK_SKIP_PIXELS,hn),G.pixelStorei(G.UNPACK_SKIP_ROWS,li),G.pixelStorei(G.UNPACK_SKIP_IMAGES,Ze),B===0&&V.generateMipmaps&&G.generateMipmap(Ot),Wt.unbindTexture()},this.copyTextureToTexture3D=function(w,V,j=null,Y=null,B=0){return w.isTexture!==!0&&(Ys("WebGLRenderer: copyTextureToTexture3D function signature has changed."),j=arguments[0]||null,Y=arguments[1]||null,w=arguments[2],V=arguments[3],B=arguments[4]||0),Ys('WebGLRenderer: copyTextureToTexture3D function has been deprecated. Use "copyTextureToTexture" instead.'),this.copyTextureToTexture(w,V,j,Y,B)},this.initRenderTarget=function(w){Ht.get(w).__webglFramebuffer===void 0&&D.setupRenderTarget(w)},this.initTexture=function(w){w.isCubeTexture?D.setTextureCube(w,0):w.isData3DTexture?D.setTexture3D(w,0):w.isDataArrayTexture||w.isCompressedArrayTexture?D.setTexture2DArray(w,0):D.setTexture2D(w,0),Wt.unbindTexture()},this.resetState=function(){P=0,L=0,I=null,Wt.reset(),ye.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return ei}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(t){this._outputColorSpace=t;const e=this.getContext();e.drawingBufferColorspace=le._getDrawingBufferColorSpace(t),e.unpackColorSpace=le._getUnpackColorSpace()}}class wl{constructor(t,e=1,n=1e3){this.isFog=!0,this.name="",this.color=new $t(t),this.near=e,this.far=n}clone(){return new wl(this.color,this.near,this.far)}toJSON(){return{type:"Fog",name:this.name,color:this.color.getHex(),near:this.near,far:this.far}}}class iu extends Le{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new cn,this.environmentIntensity=1,this.environmentRotation=new cn,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(t,e){return super.copy(t,e),t.background!==null&&(this.background=t.background.clone()),t.environment!==null&&(this.environment=t.environment.clone()),t.fog!==null&&(this.fog=t.fog.clone()),this.backgroundBlurriness=t.backgroundBlurriness,this.backgroundIntensity=t.backgroundIntensity,this.backgroundRotation.copy(t.backgroundRotation),this.environmentIntensity=t.environmentIntensity,this.environmentRotation.copy(t.environmentRotation),t.overrideMaterial!==null&&(this.overrideMaterial=t.overrideMaterial.clone()),this.matrixAutoUpdate=t.matrixAutoUpdate,this}toJSON(t){const e=super.toJSON(t);return this.fog!==null&&(e.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(e.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(e.object.backgroundIntensity=this.backgroundIntensity),e.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(e.object.environmentIntensity=this.environmentIntensity),e.object.environmentRotation=this.environmentRotation.toArray(),e}}const jc=new F,Yc=new ve,$c=new ve,r0=new F,Kc=new Kt,Wr=new F,ho=new oi,Jc=new Kt,uo=new ur;class a0 extends Ae{constructor(t,e){super(t,e),this.isSkinnedMesh=!0,this.type="SkinnedMesh",this.bindMode=Yl,this.bindMatrix=new Kt,this.bindMatrixInverse=new Kt,this.boundingBox=null,this.boundingSphere=null}computeBoundingBox(){const t=this.geometry;this.boundingBox===null&&(this.boundingBox=new Fn),this.boundingBox.makeEmpty();const e=t.getAttribute("position");for(let n=0;n<e.count;n++)this.getVertexPosition(n,Wr),this.boundingBox.expandByPoint(Wr)}computeBoundingSphere(){const t=this.geometry;this.boundingSphere===null&&(this.boundingSphere=new oi),this.boundingSphere.makeEmpty();const e=t.getAttribute("position");for(let n=0;n<e.count;n++)this.getVertexPosition(n,Wr),this.boundingSphere.expandByPoint(Wr)}copy(t,e){return super.copy(t,e),this.bindMode=t.bindMode,this.bindMatrix.copy(t.bindMatrix),this.bindMatrixInverse.copy(t.bindMatrixInverse),this.skeleton=t.skeleton,t.boundingBox!==null&&(this.boundingBox=t.boundingBox.clone()),t.boundingSphere!==null&&(this.boundingSphere=t.boundingSphere.clone()),this}raycast(t,e){const n=this.material,s=this.matrixWorld;n!==void 0&&(this.boundingSphere===null&&this.computeBoundingSphere(),ho.copy(this.boundingSphere),ho.applyMatrix4(s),t.ray.intersectsSphere(ho)!==!1&&(Jc.copy(s).invert(),uo.copy(t.ray).applyMatrix4(Jc),!(this.boundingBox!==null&&uo.intersectsBox(this.boundingBox)===!1)&&this._computeIntersections(t,e,uo)))}getVertexPosition(t,e){return super.getVertexPosition(t,e),this.applyBoneTransform(t,e),e}bind(t,e){this.skeleton=t,e===void 0&&(this.updateMatrixWorld(!0),this.skeleton.calculateInverses(),e=this.matrixWorld),this.bindMatrix.copy(e),this.bindMatrixInverse.copy(e).invert()}pose(){this.skeleton.pose()}normalizeSkinWeights(){const t=new ve,e=this.geometry.attributes.skinWeight;for(let n=0,s=e.count;n<s;n++){t.fromBufferAttribute(e,n);const r=1/t.manhattanLength();r!==1/0?t.multiplyScalar(r):t.set(1,0,0,0),e.setXYZW(n,t.x,t.y,t.z,t.w)}}updateMatrixWorld(t){super.updateMatrixWorld(t),this.bindMode===Yl?this.bindMatrixInverse.copy(this.matrixWorld).invert():this.bindMode===gd?this.bindMatrixInverse.copy(this.bindMatrix).invert():console.warn("THREE.SkinnedMesh: Unrecognized bindMode: "+this.bindMode)}applyBoneTransform(t,e){const n=this.skeleton,s=this.geometry;Yc.fromBufferAttribute(s.attributes.skinIndex,t),$c.fromBufferAttribute(s.attributes.skinWeight,t),jc.copy(e).applyMatrix4(this.bindMatrix),e.set(0,0,0);for(let r=0;r<4;r++){const a=$c.getComponent(r);if(a!==0){const o=Yc.getComponent(r);Kc.multiplyMatrices(n.bones[o].matrixWorld,n.boneInverses[o]),e.addScaledVector(r0.copy(jc).applyMatrix4(Kc),a)}}return e.applyMatrix4(this.bindMatrixInverse)}}class su extends Le{constructor(){super(),this.isBone=!0,this.type="Bone"}}class ru extends Ye{constructor(t=null,e=1,n=1,s,r,a,o,c,l=mn,h=mn,u,d){super(null,a,o,c,l,h,s,r,u,d),this.isDataTexture=!0,this.image={data:t,width:e,height:n},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}const Zc=new Kt,o0=new Kt;class Al{constructor(t=[],e=[]){this.uuid=Ci(),this.bones=t.slice(0),this.boneInverses=e,this.boneMatrices=null,this.boneTexture=null,this.init()}init(){const t=this.bones,e=this.boneInverses;if(this.boneMatrices=new Float32Array(t.length*16),e.length===0)this.calculateInverses();else if(t.length!==e.length){console.warn("THREE.Skeleton: Number of inverse bone matrices does not match amount of bones."),this.boneInverses=[];for(let n=0,s=this.bones.length;n<s;n++)this.boneInverses.push(new Kt)}}calculateInverses(){this.boneInverses.length=0;for(let t=0,e=this.bones.length;t<e;t++){const n=new Kt;this.bones[t]&&n.copy(this.bones[t].matrixWorld).invert(),this.boneInverses.push(n)}}pose(){for(let t=0,e=this.bones.length;t<e;t++){const n=this.bones[t];n&&n.matrixWorld.copy(this.boneInverses[t]).invert()}for(let t=0,e=this.bones.length;t<e;t++){const n=this.bones[t];n&&(n.parent&&n.parent.isBone?(n.matrix.copy(n.parent.matrixWorld).invert(),n.matrix.multiply(n.matrixWorld)):n.matrix.copy(n.matrixWorld),n.matrix.decompose(n.position,n.quaternion,n.scale))}}update(){const t=this.bones,e=this.boneInverses,n=this.boneMatrices,s=this.boneTexture;for(let r=0,a=t.length;r<a;r++){const o=t[r]?t[r].matrixWorld:o0;Zc.multiplyMatrices(o,e[r]),Zc.toArray(n,r*16)}s!==null&&(s.needsUpdate=!0)}clone(){return new Al(this.bones,this.boneInverses)}computeBoneTexture(){let t=Math.sqrt(this.bones.length*4);t=Math.ceil(t/4)*4,t=Math.max(t,4);const e=new Float32Array(t*t*4);e.set(this.boneMatrices);const n=new ru(e,t,t,Sn,Bn);return n.needsUpdate=!0,this.boneMatrices=e,this.boneTexture=n,this}getBoneByName(t){for(let e=0,n=this.bones.length;e<n;e++){const s=this.bones[e];if(s.name===t)return s}}dispose(){this.boneTexture!==null&&(this.boneTexture.dispose(),this.boneTexture=null)}fromJSON(t,e){this.uuid=t.uuid;for(let n=0,s=t.bones.length;n<s;n++){const r=t.bones[n];let a=e[r];a===void 0&&(console.warn("THREE.Skeleton: No bone found with UUID:",r),a=new su),this.bones.push(a),this.boneInverses.push(new Kt().fromArray(t.boneInverses[n]))}return this.init(),this}toJSON(){const t={metadata:{version:4.6,type:"Skeleton",generator:"Skeleton.toJSON"},bones:[],boneInverses:[]};t.uuid=this.uuid;const e=this.bones,n=this.boneInverses;for(let s=0,r=e.length;s<r;s++){const a=e[s];t.bones.push(a.uuid);const o=n[s];t.boneInverses.push(o.toArray())}return t}}class zn extends In{static get type(){return"LineBasicMaterial"}constructor(t){super(),this.isLineBasicMaterial=!0,this.color=new $t(16777215),this.map=null,this.linewidth=1,this.linecap="round",this.linejoin="round",this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.map=t.map,this.linewidth=t.linewidth,this.linecap=t.linecap,this.linejoin=t.linejoin,this.fog=t.fog,this}}const pa=new F,ma=new F,Qc=new Kt,qs=new ur,Xr=new oi,fo=new F,th=new F;class dr extends Le{constructor(t=new Ie,e=new zn){super(),this.isLine=!0,this.type="Line",this.geometry=t,this.material=e,this.updateMorphTargets()}copy(t,e){return super.copy(t,e),this.material=Array.isArray(t.material)?t.material.slice():t.material,this.geometry=t.geometry,this}computeLineDistances(){const t=this.geometry;if(t.index===null){const e=t.attributes.position,n=[0];for(let s=1,r=e.count;s<r;s++)pa.fromBufferAttribute(e,s-1),ma.fromBufferAttribute(e,s),n[s]=n[s-1],n[s]+=pa.distanceTo(ma);t.setAttribute("lineDistance",new he(n,1))}else console.warn("THREE.Line.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");return this}raycast(t,e){const n=this.geometry,s=this.matrixWorld,r=t.params.Line.threshold,a=n.drawRange;if(n.boundingSphere===null&&n.computeBoundingSphere(),Xr.copy(n.boundingSphere),Xr.applyMatrix4(s),Xr.radius+=r,t.ray.intersectsSphere(Xr)===!1)return;Qc.copy(s).invert(),qs.copy(t.ray).applyMatrix4(Qc);const o=r/((this.scale.x+this.scale.y+this.scale.z)/3),c=o*o,l=this.isLineSegments?2:1,h=n.index,d=n.attributes.position;if(h!==null){const p=Math.max(0,a.start),g=Math.min(h.count,a.start+a.count);for(let _=p,m=g-1;_<m;_+=l){const f=h.getX(_),R=h.getX(_+1),C=qr(this,t,qs,c,f,R);C&&e.push(C)}if(this.isLineLoop){const _=h.getX(g-1),m=h.getX(p),f=qr(this,t,qs,c,_,m);f&&e.push(f)}}else{const p=Math.max(0,a.start),g=Math.min(d.count,a.start+a.count);for(let _=p,m=g-1;_<m;_+=l){const f=qr(this,t,qs,c,_,_+1);f&&e.push(f)}if(this.isLineLoop){const _=qr(this,t,qs,c,g-1,p);_&&e.push(_)}}}updateMorphTargets(){const e=this.geometry.morphAttributes,n=Object.keys(e);if(n.length>0){const s=e[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,a=s.length;r<a;r++){const o=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=r}}}}}function qr(i,t,e,n,s,r){const a=i.geometry.attributes.position;if(pa.fromBufferAttribute(a,s),ma.fromBufferAttribute(a,r),e.distanceSqToSegment(pa,ma,fo,th)>n)return;fo.applyMatrix4(i.matrixWorld);const c=t.ray.origin.distanceTo(fo);if(!(c<t.near||c>t.far))return{distance:c,point:th.clone().applyMatrix4(i.matrixWorld),index:s,face:null,faceIndex:null,barycoord:null,object:i}}const eh=new F,nh=new F;class ga extends dr{constructor(t,e){super(t,e),this.isLineSegments=!0,this.type="LineSegments"}computeLineDistances(){const t=this.geometry;if(t.index===null){const e=t.attributes.position,n=[];for(let s=0,r=e.count;s<r;s+=2)eh.fromBufferAttribute(e,s),nh.fromBufferAttribute(e,s+1),n[s]=s===0?0:n[s-1],n[s+1]=n[s]+eh.distanceTo(nh);t.setAttribute("lineDistance",new he(n,1))}else console.warn("THREE.LineSegments.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");return this}}class Ks extends In{static get type(){return"PointsMaterial"}constructor(t){super(),this.isPointsMaterial=!0,this.color=new $t(16777215),this.map=null,this.alphaMap=null,this.size=1,this.sizeAttenuation=!0,this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.map=t.map,this.alphaMap=t.alphaMap,this.size=t.size,this.sizeAttenuation=t.sizeAttenuation,this.fog=t.fog,this}}const ih=new Kt,ul=new ur,jr=new oi,Yr=new F;class po extends Le{constructor(t=new Ie,e=new Ks){super(),this.isPoints=!0,this.type="Points",this.geometry=t,this.material=e,this.updateMorphTargets()}copy(t,e){return super.copy(t,e),this.material=Array.isArray(t.material)?t.material.slice():t.material,this.geometry=t.geometry,this}raycast(t,e){const n=this.geometry,s=this.matrixWorld,r=t.params.Points.threshold,a=n.drawRange;if(n.boundingSphere===null&&n.computeBoundingSphere(),jr.copy(n.boundingSphere),jr.applyMatrix4(s),jr.radius+=r,t.ray.intersectsSphere(jr)===!1)return;ih.copy(s).invert(),ul.copy(t.ray).applyMatrix4(ih);const o=r/((this.scale.x+this.scale.y+this.scale.z)/3),c=o*o,l=n.index,u=n.attributes.position;if(l!==null){const d=Math.max(0,a.start),p=Math.min(l.count,a.start+a.count);for(let g=d,_=p;g<_;g++){const m=l.getX(g);Yr.fromBufferAttribute(u,m),sh(Yr,m,c,s,t,e,this)}}else{const d=Math.max(0,a.start),p=Math.min(u.count,a.start+a.count);for(let g=d,_=p;g<_;g++)Yr.fromBufferAttribute(u,g),sh(Yr,g,c,s,t,e,this)}}updateMorphTargets(){const e=this.geometry.morphAttributes,n=Object.keys(e);if(n.length>0){const s=e[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,a=s.length;r<a;r++){const o=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=r}}}}}function sh(i,t,e,n,s,r,a){const o=ul.distanceSqToPoint(i);if(o<e){const c=new F;ul.closestPointToPoint(i,c),c.applyMatrix4(n);const l=s.ray.origin.distanceTo(c);if(l<s.near||l>s.far)return;r.push({distance:l,distanceToRay:Math.sqrt(o),point:c,index:t,face:null,faceIndex:null,barycoord:null,object:a})}}class Gn{constructor(){this.type="Curve",this.arcLengthDivisions=200}getPoint(){return console.warn("THREE.Curve: .getPoint() not implemented."),null}getPointAt(t,e){const n=this.getUtoTmapping(t);return this.getPoint(n,e)}getPoints(t=5){const e=[];for(let n=0;n<=t;n++)e.push(this.getPoint(n/t));return e}getSpacedPoints(t=5){const e=[];for(let n=0;n<=t;n++)e.push(this.getPointAt(n/t));return e}getLength(){const t=this.getLengths();return t[t.length-1]}getLengths(t=this.arcLengthDivisions){if(this.cacheArcLengths&&this.cacheArcLengths.length===t+1&&!this.needsUpdate)return this.cacheArcLengths;this.needsUpdate=!1;const e=[];let n,s=this.getPoint(0),r=0;e.push(0);for(let a=1;a<=t;a++)n=this.getPoint(a/t),r+=n.distanceTo(s),e.push(r),s=n;return this.cacheArcLengths=e,e}updateArcLengths(){this.needsUpdate=!0,this.getLengths()}getUtoTmapping(t,e){const n=this.getLengths();let s=0;const r=n.length;let a;e?a=e:a=t*n[r-1];let o=0,c=r-1,l;for(;o<=c;)if(s=Math.floor(o+(c-o)/2),l=n[s]-a,l<0)o=s+1;else if(l>0)c=s-1;else{c=s;break}if(s=c,n[s]===a)return s/(r-1);const h=n[s],d=n[s+1]-h,p=(a-h)/d;return(s+p)/(r-1)}getTangent(t,e){let s=t-1e-4,r=t+1e-4;s<0&&(s=0),r>1&&(r=1);const a=this.getPoint(s),o=this.getPoint(r),c=e||(a.isVector2?new Lt:new F);return c.copy(o).sub(a).normalize(),c}getTangentAt(t,e){const n=this.getUtoTmapping(t);return this.getTangent(n,e)}computeFrenetFrames(t,e){const n=new F,s=[],r=[],a=[],o=new F,c=new Kt;for(let p=0;p<=t;p++){const g=p/t;s[p]=this.getTangentAt(g,new F)}r[0]=new F,a[0]=new F;let l=Number.MAX_VALUE;const h=Math.abs(s[0].x),u=Math.abs(s[0].y),d=Math.abs(s[0].z);h<=l&&(l=h,n.set(1,0,0)),u<=l&&(l=u,n.set(0,1,0)),d<=l&&n.set(0,0,1),o.crossVectors(s[0],n).normalize(),r[0].crossVectors(s[0],o),a[0].crossVectors(s[0],r[0]);for(let p=1;p<=t;p++){if(r[p]=r[p-1].clone(),a[p]=a[p-1].clone(),o.crossVectors(s[p-1],s[p]),o.length()>Number.EPSILON){o.normalize();const g=Math.acos(Xe(s[p-1].dot(s[p]),-1,1));r[p].applyMatrix4(c.makeRotationAxis(o,g))}a[p].crossVectors(s[p],r[p])}if(e===!0){let p=Math.acos(Xe(r[0].dot(r[t]),-1,1));p/=t,s[0].dot(o.crossVectors(r[0],r[t]))>0&&(p=-p);for(let g=1;g<=t;g++)r[g].applyMatrix4(c.makeRotationAxis(s[g],p*g)),a[g].crossVectors(s[g],r[g])}return{tangents:s,normals:r,binormals:a}}clone(){return new this.constructor().copy(this)}copy(t){return this.arcLengthDivisions=t.arcLengthDivisions,this}toJSON(){const t={metadata:{version:4.6,type:"Curve",generator:"Curve.toJSON"}};return t.arcLengthDivisions=this.arcLengthDivisions,t.type=this.type,t}fromJSON(t){return this.arcLengthDivisions=t.arcLengthDivisions,this}}class au extends Gn{constructor(t=0,e=0,n=1,s=1,r=0,a=Math.PI*2,o=!1,c=0){super(),this.isEllipseCurve=!0,this.type="EllipseCurve",this.aX=t,this.aY=e,this.xRadius=n,this.yRadius=s,this.aStartAngle=r,this.aEndAngle=a,this.aClockwise=o,this.aRotation=c}getPoint(t,e=new Lt){const n=e,s=Math.PI*2;let r=this.aEndAngle-this.aStartAngle;const a=Math.abs(r)<Number.EPSILON;for(;r<0;)r+=s;for(;r>s;)r-=s;r<Number.EPSILON&&(a?r=0:r=s),this.aClockwise===!0&&!a&&(r===s?r=-s:r=r-s);const o=this.aStartAngle+t*r;let c=this.aX+this.xRadius*Math.cos(o),l=this.aY+this.yRadius*Math.sin(o);if(this.aRotation!==0){const h=Math.cos(this.aRotation),u=Math.sin(this.aRotation),d=c-this.aX,p=l-this.aY;c=d*h-p*u+this.aX,l=d*u+p*h+this.aY}return n.set(c,l)}copy(t){return super.copy(t),this.aX=t.aX,this.aY=t.aY,this.xRadius=t.xRadius,this.yRadius=t.yRadius,this.aStartAngle=t.aStartAngle,this.aEndAngle=t.aEndAngle,this.aClockwise=t.aClockwise,this.aRotation=t.aRotation,this}toJSON(){const t=super.toJSON();return t.aX=this.aX,t.aY=this.aY,t.xRadius=this.xRadius,t.yRadius=this.yRadius,t.aStartAngle=this.aStartAngle,t.aEndAngle=this.aEndAngle,t.aClockwise=this.aClockwise,t.aRotation=this.aRotation,t}fromJSON(t){return super.fromJSON(t),this.aX=t.aX,this.aY=t.aY,this.xRadius=t.xRadius,this.yRadius=t.yRadius,this.aStartAngle=t.aStartAngle,this.aEndAngle=t.aEndAngle,this.aClockwise=t.aClockwise,this.aRotation=t.aRotation,this}}class l0 extends au{constructor(t,e,n,s,r,a){super(t,e,n,n,s,r,a),this.isArcCurve=!0,this.type="ArcCurve"}}function Cl(){let i=0,t=0,e=0,n=0;function s(r,a,o,c){i=r,t=o,e=-3*r+3*a-2*o-c,n=2*r-2*a+o+c}return{initCatmullRom:function(r,a,o,c,l){s(a,o,l*(o-r),l*(c-a))},initNonuniformCatmullRom:function(r,a,o,c,l,h,u){let d=(a-r)/l-(o-r)/(l+h)+(o-a)/h,p=(o-a)/h-(c-a)/(h+u)+(c-o)/u;d*=h,p*=h,s(a,o,d,p)},calc:function(r){const a=r*r,o=a*r;return i+t*r+e*a+n*o}}}const $r=new F,mo=new Cl,go=new Cl,_o=new Cl;class c0 extends Gn{constructor(t=[],e=!1,n="centripetal",s=.5){super(),this.isCatmullRomCurve3=!0,this.type="CatmullRomCurve3",this.points=t,this.closed=e,this.curveType=n,this.tension=s}getPoint(t,e=new F){const n=e,s=this.points,r=s.length,a=(r-(this.closed?0:1))*t;let o=Math.floor(a),c=a-o;this.closed?o+=o>0?0:(Math.floor(Math.abs(o)/r)+1)*r:c===0&&o===r-1&&(o=r-2,c=1);let l,h;this.closed||o>0?l=s[(o-1)%r]:($r.subVectors(s[0],s[1]).add(s[0]),l=$r);const u=s[o%r],d=s[(o+1)%r];if(this.closed||o+2<r?h=s[(o+2)%r]:($r.subVectors(s[r-1],s[r-2]).add(s[r-1]),h=$r),this.curveType==="centripetal"||this.curveType==="chordal"){const p=this.curveType==="chordal"?.5:.25;let g=Math.pow(l.distanceToSquared(u),p),_=Math.pow(u.distanceToSquared(d),p),m=Math.pow(d.distanceToSquared(h),p);_<1e-4&&(_=1),g<1e-4&&(g=_),m<1e-4&&(m=_),mo.initNonuniformCatmullRom(l.x,u.x,d.x,h.x,g,_,m),go.initNonuniformCatmullRom(l.y,u.y,d.y,h.y,g,_,m),_o.initNonuniformCatmullRom(l.z,u.z,d.z,h.z,g,_,m)}else this.curveType==="catmullrom"&&(mo.initCatmullRom(l.x,u.x,d.x,h.x,this.tension),go.initCatmullRom(l.y,u.y,d.y,h.y,this.tension),_o.initCatmullRom(l.z,u.z,d.z,h.z,this.tension));return n.set(mo.calc(c),go.calc(c),_o.calc(c)),n}copy(t){super.copy(t),this.points=[];for(let e=0,n=t.points.length;e<n;e++){const s=t.points[e];this.points.push(s.clone())}return this.closed=t.closed,this.curveType=t.curveType,this.tension=t.tension,this}toJSON(){const t=super.toJSON();t.points=[];for(let e=0,n=this.points.length;e<n;e++){const s=this.points[e];t.points.push(s.toArray())}return t.closed=this.closed,t.curveType=this.curveType,t.tension=this.tension,t}fromJSON(t){super.fromJSON(t),this.points=[];for(let e=0,n=t.points.length;e<n;e++){const s=t.points[e];this.points.push(new F().fromArray(s))}return this.closed=t.closed,this.curveType=t.curveType,this.tension=t.tension,this}}function rh(i,t,e,n,s){const r=(n-t)*.5,a=(s-e)*.5,o=i*i,c=i*o;return(2*e-2*n+r+a)*c+(-3*e+3*n-2*r-a)*o+r*i+e}function h0(i,t){const e=1-i;return e*e*t}function u0(i,t){return 2*(1-i)*i*t}function d0(i,t){return i*i*t}function tr(i,t,e,n){return h0(i,t)+u0(i,e)+d0(i,n)}function f0(i,t){const e=1-i;return e*e*e*t}function p0(i,t){const e=1-i;return 3*e*e*i*t}function m0(i,t){return 3*(1-i)*i*i*t}function g0(i,t){return i*i*i*t}function er(i,t,e,n,s){return f0(i,t)+p0(i,e)+m0(i,n)+g0(i,s)}class _0 extends Gn{constructor(t=new Lt,e=new Lt,n=new Lt,s=new Lt){super(),this.isCubicBezierCurve=!0,this.type="CubicBezierCurve",this.v0=t,this.v1=e,this.v2=n,this.v3=s}getPoint(t,e=new Lt){const n=e,s=this.v0,r=this.v1,a=this.v2,o=this.v3;return n.set(er(t,s.x,r.x,a.x,o.x),er(t,s.y,r.y,a.y,o.y)),n}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this.v3.copy(t.v3),this}toJSON(){const t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t.v3=this.v3.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this.v3.fromArray(t.v3),this}}class v0 extends Gn{constructor(t=new F,e=new F,n=new F,s=new F){super(),this.isCubicBezierCurve3=!0,this.type="CubicBezierCurve3",this.v0=t,this.v1=e,this.v2=n,this.v3=s}getPoint(t,e=new F){const n=e,s=this.v0,r=this.v1,a=this.v2,o=this.v3;return n.set(er(t,s.x,r.x,a.x,o.x),er(t,s.y,r.y,a.y,o.y),er(t,s.z,r.z,a.z,o.z)),n}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this.v3.copy(t.v3),this}toJSON(){const t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t.v3=this.v3.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this.v3.fromArray(t.v3),this}}class x0 extends Gn{constructor(t=new Lt,e=new Lt){super(),this.isLineCurve=!0,this.type="LineCurve",this.v1=t,this.v2=e}getPoint(t,e=new Lt){const n=e;return t===1?n.copy(this.v2):(n.copy(this.v2).sub(this.v1),n.multiplyScalar(t).add(this.v1)),n}getPointAt(t,e){return this.getPoint(t,e)}getTangent(t,e=new Lt){return e.subVectors(this.v2,this.v1).normalize()}getTangentAt(t,e){return this.getTangent(t,e)}copy(t){return super.copy(t),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){const t=super.toJSON();return t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}}class ou extends Gn{constructor(t=new F,e=new F){super(),this.isLineCurve3=!0,this.type="LineCurve3",this.v1=t,this.v2=e}getPoint(t,e=new F){const n=e;return t===1?n.copy(this.v2):(n.copy(this.v2).sub(this.v1),n.multiplyScalar(t).add(this.v1)),n}getPointAt(t,e){return this.getPoint(t,e)}getTangent(t,e=new F){return e.subVectors(this.v2,this.v1).normalize()}getTangentAt(t,e){return this.getTangent(t,e)}copy(t){return super.copy(t),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){const t=super.toJSON();return t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}}class y0 extends Gn{constructor(t=new Lt,e=new Lt,n=new Lt){super(),this.isQuadraticBezierCurve=!0,this.type="QuadraticBezierCurve",this.v0=t,this.v1=e,this.v2=n}getPoint(t,e=new Lt){const n=e,s=this.v0,r=this.v1,a=this.v2;return n.set(tr(t,s.x,r.x,a.x),tr(t,s.y,r.y,a.y)),n}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){const t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}}class lu extends Gn{constructor(t=new F,e=new F,n=new F){super(),this.isQuadraticBezierCurve3=!0,this.type="QuadraticBezierCurve3",this.v0=t,this.v1=e,this.v2=n}getPoint(t,e=new F){const n=e,s=this.v0,r=this.v1,a=this.v2;return n.set(tr(t,s.x,r.x,a.x),tr(t,s.y,r.y,a.y),tr(t,s.z,r.z,a.z)),n}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){const t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}}class M0 extends Gn{constructor(t=[]){super(),this.isSplineCurve=!0,this.type="SplineCurve",this.points=t}getPoint(t,e=new Lt){const n=e,s=this.points,r=(s.length-1)*t,a=Math.floor(r),o=r-a,c=s[a===0?a:a-1],l=s[a],h=s[a>s.length-2?s.length-1:a+1],u=s[a>s.length-3?s.length-1:a+2];return n.set(rh(o,c.x,l.x,h.x,u.x),rh(o,c.y,l.y,h.y,u.y)),n}copy(t){super.copy(t),this.points=[];for(let e=0,n=t.points.length;e<n;e++){const s=t.points[e];this.points.push(s.clone())}return this}toJSON(){const t=super.toJSON();t.points=[];for(let e=0,n=this.points.length;e<n;e++){const s=this.points[e];t.points.push(s.toArray())}return t}fromJSON(t){super.fromJSON(t),this.points=[];for(let e=0,n=t.points.length;e<n;e++){const s=t.points[e];this.points.push(new Lt().fromArray(s))}return this}}var dl=Object.freeze({__proto__:null,ArcCurve:l0,CatmullRomCurve3:c0,CubicBezierCurve:_0,CubicBezierCurve3:v0,EllipseCurve:au,LineCurve:x0,LineCurve3:ou,QuadraticBezierCurve:y0,QuadraticBezierCurve3:lu,SplineCurve:M0});class S0 extends Gn{constructor(){super(),this.type="CurvePath",this.curves=[],this.autoClose=!1}add(t){this.curves.push(t)}closePath(){const t=this.curves[0].getPoint(0),e=this.curves[this.curves.length-1].getPoint(1);if(!t.equals(e)){const n=t.isVector2===!0?"LineCurve":"LineCurve3";this.curves.push(new dl[n](e,t))}return this}getPoint(t,e){const n=t*this.getLength(),s=this.getCurveLengths();let r=0;for(;r<s.length;){if(s[r]>=n){const a=s[r]-n,o=this.curves[r],c=o.getLength(),l=c===0?0:1-a/c;return o.getPointAt(l,e)}r++}return null}getLength(){const t=this.getCurveLengths();return t[t.length-1]}updateArcLengths(){this.needsUpdate=!0,this.cacheLengths=null,this.getCurveLengths()}getCurveLengths(){if(this.cacheLengths&&this.cacheLengths.length===this.curves.length)return this.cacheLengths;const t=[];let e=0;for(let n=0,s=this.curves.length;n<s;n++)e+=this.curves[n].getLength(),t.push(e);return this.cacheLengths=t,t}getSpacedPoints(t=40){const e=[];for(let n=0;n<=t;n++)e.push(this.getPoint(n/t));return this.autoClose&&e.push(e[0]),e}getPoints(t=12){const e=[];let n;for(let s=0,r=this.curves;s<r.length;s++){const a=r[s],o=a.isEllipseCurve?t*2:a.isLineCurve||a.isLineCurve3?1:a.isSplineCurve?t*a.points.length:t,c=a.getPoints(o);for(let l=0;l<c.length;l++){const h=c[l];n&&n.equals(h)||(e.push(h),n=h)}}return this.autoClose&&e.length>1&&!e[e.length-1].equals(e[0])&&e.push(e[0]),e}copy(t){super.copy(t),this.curves=[];for(let e=0,n=t.curves.length;e<n;e++){const s=t.curves[e];this.curves.push(s.clone())}return this.autoClose=t.autoClose,this}toJSON(){const t=super.toJSON();t.autoClose=this.autoClose,t.curves=[];for(let e=0,n=this.curves.length;e<n;e++){const s=this.curves[e];t.curves.push(s.toJSON())}return t}fromJSON(t){super.fromJSON(t),this.autoClose=t.autoClose,this.curves=[];for(let e=0,n=t.curves.length;e<n;e++){const s=t.curves[e];this.curves.push(new dl[s.type]().fromJSON(s))}return this}}class Fs extends Ie{constructor(t=1,e=1,n=1,s=32,r=1,a=!1,o=0,c=Math.PI*2){super(),this.type="CylinderGeometry",this.parameters={radiusTop:t,radiusBottom:e,height:n,radialSegments:s,heightSegments:r,openEnded:a,thetaStart:o,thetaLength:c};const l=this;s=Math.floor(s),r=Math.floor(r);const h=[],u=[],d=[],p=[];let g=0;const _=[],m=n/2;let f=0;R(),a===!1&&(t>0&&C(!0),e>0&&C(!1)),this.setIndex(h),this.setAttribute("position",new he(u,3)),this.setAttribute("normal",new he(d,3)),this.setAttribute("uv",new he(p,2));function R(){const x=new F,N=new F;let P=0;const L=(e-t)/n;for(let I=0;I<=r;I++){const M=[],S=I/r,U=S*(e-t)+t;for(let k=0;k<=s;k++){const H=k/s,J=H*c+o,Q=Math.sin(J),O=Math.cos(J);N.x=U*Q,N.y=-S*n+m,N.z=U*O,u.push(N.x,N.y,N.z),x.set(Q,L,O).normalize(),d.push(x.x,x.y,x.z),p.push(H,1-S),M.push(g++)}_.push(M)}for(let I=0;I<s;I++)for(let M=0;M<r;M++){const S=_[M][I],U=_[M+1][I],k=_[M+1][I+1],H=_[M][I+1];(t>0||M!==0)&&(h.push(S,U,H),P+=3),(e>0||M!==r-1)&&(h.push(U,k,H),P+=3)}l.addGroup(f,P,0),f+=P}function C(x){const N=g,P=new Lt,L=new F;let I=0;const M=x===!0?t:e,S=x===!0?1:-1;for(let k=1;k<=s;k++)u.push(0,m*S,0),d.push(0,S,0),p.push(.5,.5),g++;const U=g;for(let k=0;k<=s;k++){const J=k/s*c+o,Q=Math.cos(J),O=Math.sin(J);L.x=M*O,L.y=m*S,L.z=M*Q,u.push(L.x,L.y,L.z),d.push(0,S,0),P.x=Q*.5+.5,P.y=O*.5*S+.5,p.push(P.x,P.y),g++}for(let k=0;k<s;k++){const H=N+k,J=U+k;x===!0?h.push(J,J+1,H):h.push(J+1,J,H),I+=3}l.addGroup(f,I,x===!0?1:2),f+=I}}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new Fs(t.radiusTop,t.radiusBottom,t.height,t.radialSegments,t.heightSegments,t.openEnded,t.thetaStart,t.thetaLength)}}class Rl extends Fs{constructor(t=1,e=1,n=32,s=1,r=!1,a=0,o=Math.PI*2){super(0,t,e,n,s,r,a,o),this.type="ConeGeometry",this.parameters={radius:t,height:e,radialSegments:n,heightSegments:s,openEnded:r,thetaStart:a,thetaLength:o}}static fromJSON(t){return new Rl(t.radius,t.height,t.radialSegments,t.heightSegments,t.openEnded,t.thetaStart,t.thetaLength)}}class ji extends Ie{constructor(t=1,e=32,n=16,s=0,r=Math.PI*2,a=0,o=Math.PI){super(),this.type="SphereGeometry",this.parameters={radius:t,widthSegments:e,heightSegments:n,phiStart:s,phiLength:r,thetaStart:a,thetaLength:o},e=Math.max(3,Math.floor(e)),n=Math.max(2,Math.floor(n));const c=Math.min(a+o,Math.PI);let l=0;const h=[],u=new F,d=new F,p=[],g=[],_=[],m=[];for(let f=0;f<=n;f++){const R=[],C=f/n;let x=0;f===0&&a===0?x=.5/e:f===n&&c===Math.PI&&(x=-.5/e);for(let N=0;N<=e;N++){const P=N/e;u.x=-t*Math.cos(s+P*r)*Math.sin(a+C*o),u.y=t*Math.cos(a+C*o),u.z=t*Math.sin(s+P*r)*Math.sin(a+C*o),g.push(u.x,u.y,u.z),d.copy(u).normalize(),_.push(d.x,d.y,d.z),m.push(P+x,1-C),R.push(l++)}h.push(R)}for(let f=0;f<n;f++)for(let R=0;R<e;R++){const C=h[f][R+1],x=h[f][R],N=h[f+1][R],P=h[f+1][R+1];(f!==0||a>0)&&p.push(C,x,P),(f!==n-1||c<Math.PI)&&p.push(x,N,P)}this.setIndex(p),this.setAttribute("position",new he(g,3)),this.setAttribute("normal",new he(_,3)),this.setAttribute("uv",new he(m,2))}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new ji(t.radius,t.widthSegments,t.heightSegments,t.phiStart,t.phiLength,t.thetaStart,t.thetaLength)}}class Pl extends Ie{constructor(t=new lu(new F(-1,-1,0),new F(-1,1,0),new F(1,1,0)),e=64,n=1,s=8,r=!1){super(),this.type="TubeGeometry",this.parameters={path:t,tubularSegments:e,radius:n,radialSegments:s,closed:r};const a=t.computeFrenetFrames(e,r);this.tangents=a.tangents,this.normals=a.normals,this.binormals=a.binormals;const o=new F,c=new F,l=new Lt;let h=new F;const u=[],d=[],p=[],g=[];_(),this.setIndex(g),this.setAttribute("position",new he(u,3)),this.setAttribute("normal",new he(d,3)),this.setAttribute("uv",new he(p,2));function _(){for(let C=0;C<e;C++)m(C);m(r===!1?e:0),R(),f()}function m(C){h=t.getPointAt(C/e,h);const x=a.normals[C],N=a.binormals[C];for(let P=0;P<=s;P++){const L=P/s*Math.PI*2,I=Math.sin(L),M=-Math.cos(L);c.x=M*x.x+I*N.x,c.y=M*x.y+I*N.y,c.z=M*x.z+I*N.z,c.normalize(),d.push(c.x,c.y,c.z),o.x=h.x+n*c.x,o.y=h.y+n*c.y,o.z=h.z+n*c.z,u.push(o.x,o.y,o.z)}}function f(){for(let C=1;C<=e;C++)for(let x=1;x<=s;x++){const N=(s+1)*(C-1)+(x-1),P=(s+1)*C+(x-1),L=(s+1)*C+x,I=(s+1)*(C-1)+x;g.push(N,P,I),g.push(P,L,I)}}function R(){for(let C=0;C<=e;C++)for(let x=0;x<=s;x++)l.x=C/e,l.y=x/s,p.push(l.x,l.y)}}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}toJSON(){const t=super.toJSON();return t.path=this.parameters.path.toJSON(),t}static fromJSON(t){return new Pl(new dl[t.path.type]().fromJSON(t.path),t.tubularSegments,t.radius,t.radialSegments,t.closed)}}class b0 extends In{static get type(){return"MeshStandardMaterial"}constructor(t){super(),this.isMeshStandardMaterial=!0,this.defines={STANDARD:""},this.color=new $t(16777215),this.roughness=1,this.metalness=0,this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new $t(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=ba,this.normalScale=new Lt(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.roughnessMap=null,this.metalnessMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new cn,this.envMapIntensity=1,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.flatShading=!1,this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.defines={STANDARD:""},this.color.copy(t.color),this.roughness=t.roughness,this.metalness=t.metalness,this.map=t.map,this.lightMap=t.lightMap,this.lightMapIntensity=t.lightMapIntensity,this.aoMap=t.aoMap,this.aoMapIntensity=t.aoMapIntensity,this.emissive.copy(t.emissive),this.emissiveMap=t.emissiveMap,this.emissiveIntensity=t.emissiveIntensity,this.bumpMap=t.bumpMap,this.bumpScale=t.bumpScale,this.normalMap=t.normalMap,this.normalMapType=t.normalMapType,this.normalScale.copy(t.normalScale),this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this.roughnessMap=t.roughnessMap,this.metalnessMap=t.metalnessMap,this.alphaMap=t.alphaMap,this.envMap=t.envMap,this.envMapRotation.copy(t.envMapRotation),this.envMapIntensity=t.envMapIntensity,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.wireframeLinecap=t.wireframeLinecap,this.wireframeLinejoin=t.wireframeLinejoin,this.flatShading=t.flatShading,this.fog=t.fog,this}}class vs extends In{static get type(){return"MeshPhongMaterial"}constructor(t){super(),this.isMeshPhongMaterial=!0,this.color=new $t(16777215),this.specular=new $t(1118481),this.shininess=30,this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new $t(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=ba,this.normalScale=new Lt(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new cn,this.combine=Ma,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.flatShading=!1,this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.specular.copy(t.specular),this.shininess=t.shininess,this.map=t.map,this.lightMap=t.lightMap,this.lightMapIntensity=t.lightMapIntensity,this.aoMap=t.aoMap,this.aoMapIntensity=t.aoMapIntensity,this.emissive.copy(t.emissive),this.emissiveMap=t.emissiveMap,this.emissiveIntensity=t.emissiveIntensity,this.bumpMap=t.bumpMap,this.bumpScale=t.bumpScale,this.normalMap=t.normalMap,this.normalMapType=t.normalMapType,this.normalScale.copy(t.normalScale),this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this.specularMap=t.specularMap,this.alphaMap=t.alphaMap,this.envMap=t.envMap,this.envMapRotation.copy(t.envMapRotation),this.combine=t.combine,this.reflectivity=t.reflectivity,this.refractionRatio=t.refractionRatio,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.wireframeLinecap=t.wireframeLinecap,this.wireframeLinejoin=t.wireframeLinejoin,this.flatShading=t.flatShading,this.fog=t.fog,this}}class E0 extends In{static get type(){return"MeshLambertMaterial"}constructor(t){super(),this.isMeshLambertMaterial=!0,this.color=new $t(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new $t(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=ba,this.normalScale=new Lt(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new cn,this.combine=Ma,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.flatShading=!1,this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.map=t.map,this.lightMap=t.lightMap,this.lightMapIntensity=t.lightMapIntensity,this.aoMap=t.aoMap,this.aoMapIntensity=t.aoMapIntensity,this.emissive.copy(t.emissive),this.emissiveMap=t.emissiveMap,this.emissiveIntensity=t.emissiveIntensity,this.bumpMap=t.bumpMap,this.bumpScale=t.bumpScale,this.normalMap=t.normalMap,this.normalMapType=t.normalMapType,this.normalScale.copy(t.normalScale),this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this.specularMap=t.specularMap,this.alphaMap=t.alphaMap,this.envMap=t.envMap,this.envMapRotation.copy(t.envMapRotation),this.combine=t.combine,this.reflectivity=t.reflectivity,this.refractionRatio=t.refractionRatio,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.wireframeLinecap=t.wireframeLinecap,this.wireframeLinejoin=t.wireframeLinejoin,this.flatShading=t.flatShading,this.fog=t.fog,this}}function Kr(i,t,e){return!i||!e&&i.constructor===t?i:typeof t.BYTES_PER_ELEMENT=="number"?new t(i):Array.prototype.slice.call(i)}function T0(i){return ArrayBuffer.isView(i)&&!(i instanceof DataView)}function w0(i){function t(s,r){return i[s]-i[r]}const e=i.length,n=new Array(e);for(let s=0;s!==e;++s)n[s]=s;return n.sort(t),n}function ah(i,t,e){const n=i.length,s=new i.constructor(n);for(let r=0,a=0;a!==n;++r){const o=e[r]*t;for(let c=0;c!==t;++c)s[a++]=i[o+c]}return s}function cu(i,t,e,n){let s=1,r=i[0];for(;r!==void 0&&r[n]===void 0;)r=i[s++];if(r===void 0)return;let a=r[n];if(a!==void 0)if(Array.isArray(a))do a=r[n],a!==void 0&&(t.push(r.time),e.push.apply(e,a)),r=i[s++];while(r!==void 0);else if(a.toArray!==void 0)do a=r[n],a!==void 0&&(t.push(r.time),a.toArray(e,e.length)),r=i[s++];while(r!==void 0);else do a=r[n],a!==void 0&&(t.push(r.time),e.push(a)),r=i[s++];while(r!==void 0)}class Aa{constructor(t,e,n,s){this.parameterPositions=t,this._cachedIndex=0,this.resultBuffer=s!==void 0?s:new e.constructor(n),this.sampleValues=e,this.valueSize=n,this.settings=null,this.DefaultSettings_={}}evaluate(t){const e=this.parameterPositions;let n=this._cachedIndex,s=e[n],r=e[n-1];n:{t:{let a;e:{i:if(!(t<s)){for(let o=n+2;;){if(s===void 0){if(t<r)break i;return n=e.length,this._cachedIndex=n,this.copySampleValue_(n-1)}if(n===o)break;if(r=s,s=e[++n],t<s)break t}a=e.length;break e}if(!(t>=r)){const o=e[1];t<o&&(n=2,r=o);for(let c=n-2;;){if(r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(n===c)break;if(s=r,r=e[--n-1],t>=r)break t}a=n,n=0;break e}break n}for(;n<a;){const o=n+a>>>1;t<e[o]?a=o:n=o+1}if(s=e[n],r=e[n-1],r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(s===void 0)return n=e.length,this._cachedIndex=n,this.copySampleValue_(n-1)}this._cachedIndex=n,this.intervalChanged_(n,r,s)}return this.interpolate_(n,r,t,s)}getSettings_(){return this.settings||this.DefaultSettings_}copySampleValue_(t){const e=this.resultBuffer,n=this.sampleValues,s=this.valueSize,r=t*s;for(let a=0;a!==s;++a)e[a]=n[r+a];return e}interpolate_(){throw new Error("call to abstract method")}intervalChanged_(){}}class A0 extends Aa{constructor(t,e,n,s){super(t,e,n,s),this._weightPrev=-0,this._offsetPrev=-0,this._weightNext=-0,this._offsetNext=-0,this.DefaultSettings_={endingStart:$l,endingEnd:$l}}intervalChanged_(t,e,n){const s=this.parameterPositions;let r=t-2,a=t+1,o=s[r],c=s[a];if(o===void 0)switch(this.getSettings_().endingStart){case Kl:r=t,o=2*e-n;break;case Jl:r=s.length-2,o=e+s[r]-s[r+1];break;default:r=t,o=n}if(c===void 0)switch(this.getSettings_().endingEnd){case Kl:a=t,c=2*n-e;break;case Jl:a=1,c=n+s[1]-s[0];break;default:a=t-1,c=e}const l=(n-e)*.5,h=this.valueSize;this._weightPrev=l/(e-o),this._weightNext=l/(c-n),this._offsetPrev=r*h,this._offsetNext=a*h}interpolate_(t,e,n,s){const r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,c=t*o,l=c-o,h=this._offsetPrev,u=this._offsetNext,d=this._weightPrev,p=this._weightNext,g=(n-e)/(s-e),_=g*g,m=_*g,f=-d*m+2*d*_-d*g,R=(1+d)*m+(-1.5-2*d)*_+(-.5+d)*g+1,C=(-1-p)*m+(1.5+p)*_+.5*g,x=p*m-p*_;for(let N=0;N!==o;++N)r[N]=f*a[h+N]+R*a[l+N]+C*a[c+N]+x*a[u+N];return r}}class C0 extends Aa{constructor(t,e,n,s){super(t,e,n,s)}interpolate_(t,e,n,s){const r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,c=t*o,l=c-o,h=(n-e)/(s-e),u=1-h;for(let d=0;d!==o;++d)r[d]=a[l+d]*u+a[c+d]*h;return r}}class R0 extends Aa{constructor(t,e,n,s){super(t,e,n,s)}interpolate_(t){return this.copySampleValue_(t-1)}}class Wn{constructor(t,e,n,s){if(t===void 0)throw new Error("THREE.KeyframeTrack: track name is undefined");if(e===void 0||e.length===0)throw new Error("THREE.KeyframeTrack: no keyframes in track named "+t);this.name=t,this.times=Kr(e,this.TimeBufferType),this.values=Kr(n,this.ValueBufferType),this.setInterpolation(s||this.DefaultInterpolation)}static toJSON(t){const e=t.constructor;let n;if(e.toJSON!==this.toJSON)n=e.toJSON(t);else{n={name:t.name,times:Kr(t.times,Array),values:Kr(t.values,Array)};const s=t.getInterpolation();s!==t.DefaultInterpolation&&(n.interpolation=s)}return n.type=t.ValueTypeName,n}InterpolantFactoryMethodDiscrete(t){return new R0(this.times,this.values,this.getValueSize(),t)}InterpolantFactoryMethodLinear(t){return new C0(this.times,this.values,this.getValueSize(),t)}InterpolantFactoryMethodSmooth(t){return new A0(this.times,this.values,this.getValueSize(),t)}setInterpolation(t){let e;switch(t){case da:e=this.InterpolantFactoryMethodDiscrete;break;case cl:e=this.InterpolantFactoryMethodLinear;break;case Fa:e=this.InterpolantFactoryMethodSmooth;break}if(e===void 0){const n="unsupported interpolation for "+this.ValueTypeName+" keyframe track named "+this.name;if(this.createInterpolant===void 0)if(t!==this.DefaultInterpolation)this.setInterpolation(this.DefaultInterpolation);else throw new Error(n);return console.warn("THREE.KeyframeTrack:",n),this}return this.createInterpolant=e,this}getInterpolation(){switch(this.createInterpolant){case this.InterpolantFactoryMethodDiscrete:return da;case this.InterpolantFactoryMethodLinear:return cl;case this.InterpolantFactoryMethodSmooth:return Fa}}getValueSize(){return this.values.length/this.times.length}shift(t){if(t!==0){const e=this.times;for(let n=0,s=e.length;n!==s;++n)e[n]+=t}return this}scale(t){if(t!==1){const e=this.times;for(let n=0,s=e.length;n!==s;++n)e[n]*=t}return this}trim(t,e){const n=this.times,s=n.length;let r=0,a=s-1;for(;r!==s&&n[r]<t;)++r;for(;a!==-1&&n[a]>e;)--a;if(++a,r!==0||a!==s){r>=a&&(a=Math.max(a,1),r=a-1);const o=this.getValueSize();this.times=n.slice(r,a),this.values=this.values.slice(r*o,a*o)}return this}validate(){let t=!0;const e=this.getValueSize();e-Math.floor(e)!==0&&(console.error("THREE.KeyframeTrack: Invalid value size in track.",this),t=!1);const n=this.times,s=this.values,r=n.length;r===0&&(console.error("THREE.KeyframeTrack: Track is empty.",this),t=!1);let a=null;for(let o=0;o!==r;o++){const c=n[o];if(typeof c=="number"&&isNaN(c)){console.error("THREE.KeyframeTrack: Time is not a valid number.",this,o,c),t=!1;break}if(a!==null&&a>c){console.error("THREE.KeyframeTrack: Out of order keys.",this,o,c,a),t=!1;break}a=c}if(s!==void 0&&T0(s))for(let o=0,c=s.length;o!==c;++o){const l=s[o];if(isNaN(l)){console.error("THREE.KeyframeTrack: Value is not a valid number.",this,o,l),t=!1;break}}return t}optimize(){const t=this.times.slice(),e=this.values.slice(),n=this.getValueSize(),s=this.getInterpolation()===Fa,r=t.length-1;let a=1;for(let o=1;o<r;++o){let c=!1;const l=t[o],h=t[o+1];if(l!==h&&(o!==1||l!==t[0]))if(s)c=!0;else{const u=o*n,d=u-n,p=u+n;for(let g=0;g!==n;++g){const _=e[u+g];if(_!==e[d+g]||_!==e[p+g]){c=!0;break}}}if(c){if(o!==a){t[a]=t[o];const u=o*n,d=a*n;for(let p=0;p!==n;++p)e[d+p]=e[u+p]}++a}}if(r>0){t[a]=t[r];for(let o=r*n,c=a*n,l=0;l!==n;++l)e[c+l]=e[o+l];++a}return a!==t.length?(this.times=t.slice(0,a),this.values=e.slice(0,a*n)):(this.times=t,this.values=e),this}clone(){const t=this.times.slice(),e=this.values.slice(),n=this.constructor,s=new n(this.name,t,e);return s.createInterpolant=this.createInterpolant,s}}Wn.prototype.TimeBufferType=Float32Array;Wn.prototype.ValueBufferType=Float32Array;Wn.prototype.DefaultInterpolation=cl;class Os extends Wn{constructor(t,e,n){super(t,e,n)}}Os.prototype.ValueTypeName="bool";Os.prototype.ValueBufferType=Array;Os.prototype.DefaultInterpolation=da;Os.prototype.InterpolantFactoryMethodLinear=void 0;Os.prototype.InterpolantFactoryMethodSmooth=void 0;class hu extends Wn{}hu.prototype.ValueTypeName="color";class _a extends Wn{}_a.prototype.ValueTypeName="number";class P0 extends Aa{constructor(t,e,n,s){super(t,e,n,s)}interpolate_(t,e,n,s){const r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,c=(n-e)/(s-e);let l=t*o;for(let h=l+o;l!==h;l+=4)Dn.slerpFlat(r,0,a,l-o,a,l,c);return r}}class fr extends Wn{InterpolantFactoryMethodLinear(t){return new P0(this.times,this.values,this.getValueSize(),t)}}fr.prototype.ValueTypeName="quaternion";fr.prototype.InterpolantFactoryMethodSmooth=void 0;class ks extends Wn{constructor(t,e,n){super(t,e,n)}}ks.prototype.ValueTypeName="string";ks.prototype.ValueBufferType=Array;ks.prototype.DefaultInterpolation=da;ks.prototype.InterpolantFactoryMethodLinear=void 0;ks.prototype.InterpolantFactoryMethodSmooth=void 0;class As extends Wn{}As.prototype.ValueTypeName="vector";class oh{constructor(t="",e=-1,n=[],s=vd){this.name=t,this.tracks=n,this.duration=e,this.blendMode=s,this.uuid=Ci(),this.duration<0&&this.resetDuration()}static parse(t){const e=[],n=t.tracks,s=1/(t.fps||1);for(let a=0,o=n.length;a!==o;++a)e.push(N0(n[a]).scale(s));const r=new this(t.name,t.duration,e,t.blendMode);return r.uuid=t.uuid,r}static toJSON(t){const e=[],n=t.tracks,s={name:t.name,duration:t.duration,tracks:e,uuid:t.uuid,blendMode:t.blendMode};for(let r=0,a=n.length;r!==a;++r)e.push(Wn.toJSON(n[r]));return s}static CreateFromMorphTargetSequence(t,e,n,s){const r=e.length,a=[];for(let o=0;o<r;o++){let c=[],l=[];c.push((o+r-1)%r,o,(o+1)%r),l.push(0,1,0);const h=w0(c);c=ah(c,1,h),l=ah(l,1,h),!s&&c[0]===0&&(c.push(r),l.push(l[0])),a.push(new _a(".morphTargetInfluences["+e[o].name+"]",c,l).scale(1/n))}return new this(t,-1,a)}static findByName(t,e){let n=t;if(!Array.isArray(t)){const s=t;n=s.geometry&&s.geometry.animations||s.animations}for(let s=0;s<n.length;s++)if(n[s].name===e)return n[s];return null}static CreateClipsFromMorphTargetSequences(t,e,n){const s={},r=/^([\w-]*?)([\d]+)$/;for(let o=0,c=t.length;o<c;o++){const l=t[o],h=l.name.match(r);if(h&&h.length>1){const u=h[1];let d=s[u];d||(s[u]=d=[]),d.push(l)}}const a=[];for(const o in s)a.push(this.CreateFromMorphTargetSequence(o,s[o],e,n));return a}static parseAnimation(t,e){if(!t)return console.error("THREE.AnimationClip: No animation in JSONLoader data."),null;const n=function(u,d,p,g,_){if(p.length!==0){const m=[],f=[];cu(p,m,f,g),m.length!==0&&_.push(new u(d,m,f))}},s=[],r=t.name||"default",a=t.fps||30,o=t.blendMode;let c=t.length||-1;const l=t.hierarchy||[];for(let u=0;u<l.length;u++){const d=l[u].keys;if(!(!d||d.length===0))if(d[0].morphTargets){const p={};let g;for(g=0;g<d.length;g++)if(d[g].morphTargets)for(let _=0;_<d[g].morphTargets.length;_++)p[d[g].morphTargets[_]]=-1;for(const _ in p){const m=[],f=[];for(let R=0;R!==d[g].morphTargets.length;++R){const C=d[g];m.push(C.time),f.push(C.morphTarget===_?1:0)}s.push(new _a(".morphTargetInfluence["+_+"]",m,f))}c=p.length*a}else{const p=".bones["+e[u].name+"]";n(As,p+".position",d,"pos",s),n(fr,p+".quaternion",d,"rot",s),n(As,p+".scale",d,"scl",s)}}return s.length===0?null:new this(r,c,s,o)}resetDuration(){const t=this.tracks;let e=0;for(let n=0,s=t.length;n!==s;++n){const r=this.tracks[n];e=Math.max(e,r.times[r.times.length-1])}return this.duration=e,this}trim(){for(let t=0;t<this.tracks.length;t++)this.tracks[t].trim(0,this.duration);return this}validate(){let t=!0;for(let e=0;e<this.tracks.length;e++)t=t&&this.tracks[e].validate();return t}optimize(){for(let t=0;t<this.tracks.length;t++)this.tracks[t].optimize();return this}clone(){const t=[];for(let e=0;e<this.tracks.length;e++)t.push(this.tracks[e].clone());return new this.constructor(this.name,this.duration,t,this.blendMode)}toJSON(){return this.constructor.toJSON(this)}}function L0(i){switch(i.toLowerCase()){case"scalar":case"double":case"float":case"number":case"integer":return _a;case"vector":case"vector2":case"vector3":case"vector4":return As;case"color":return hu;case"quaternion":return fr;case"bool":case"boolean":return Os;case"string":return ks}throw new Error("THREE.KeyframeTrack: Unsupported typeName: "+i)}function N0(i){if(i.type===void 0)throw new Error("THREE.KeyframeTrack: track type undefined, can not parse");const t=L0(i.type);if(i.times===void 0){const e=[],n=[];cu(i.keys,e,n,"value"),i.times=e,i.values=n}return t.parse!==void 0?t.parse(i):new t(i.name,i.times,i.values,i.interpolation)}const va={enabled:!1,files:{},add:function(i,t){this.enabled!==!1&&(this.files[i]=t)},get:function(i){if(this.enabled!==!1)return this.files[i]},remove:function(i){delete this.files[i]},clear:function(){this.files={}}};class uu{constructor(t,e,n){const s=this;let r=!1,a=0,o=0,c;const l=[];this.onStart=void 0,this.onLoad=t,this.onProgress=e,this.onError=n,this.itemStart=function(h){o++,r===!1&&s.onStart!==void 0&&s.onStart(h,a,o),r=!0},this.itemEnd=function(h){a++,s.onProgress!==void 0&&s.onProgress(h,a,o),a===o&&(r=!1,s.onLoad!==void 0&&s.onLoad())},this.itemError=function(h){s.onError!==void 0&&s.onError(h)},this.resolveURL=function(h){return c?c(h):h},this.setURLModifier=function(h){return c=h,this},this.addHandler=function(h,u){return l.push(h,u),this},this.removeHandler=function(h){const u=l.indexOf(h);return u!==-1&&l.splice(u,2),this},this.getHandler=function(h){for(let u=0,d=l.length;u<d;u+=2){const p=l[u],g=l[u+1];if(p.global&&(p.lastIndex=0),p.test(h))return g}return null}}}const du=new uu;class ri{constructor(t){this.manager=t!==void 0?t:du,this.crossOrigin="anonymous",this.withCredentials=!1,this.path="",this.resourcePath="",this.requestHeader={}}load(){}loadAsync(t,e){const n=this;return new Promise(function(s,r){n.load(t,s,e,r)})}parse(){}setCrossOrigin(t){return this.crossOrigin=t,this}setWithCredentials(t){return this.withCredentials=t,this}setPath(t){return this.path=t,this}setResourcePath(t){return this.resourcePath=t,this}setRequestHeader(t){return this.requestHeader=t,this}}ri.DEFAULT_MATERIAL_NAME="__DEFAULT";const Zn={};class D0 extends Error{constructor(t,e){super(t),this.response=e}}class Ca extends ri{constructor(t){super(t)}load(t,e,n,s){t===void 0&&(t=""),this.path!==void 0&&(t=this.path+t),t=this.manager.resolveURL(t);const r=va.get(t);if(r!==void 0)return this.manager.itemStart(t),setTimeout(()=>{e&&e(r),this.manager.itemEnd(t)},0),r;if(Zn[t]!==void 0){Zn[t].push({onLoad:e,onProgress:n,onError:s});return}Zn[t]=[],Zn[t].push({onLoad:e,onProgress:n,onError:s});const a=new Request(t,{headers:new Headers(this.requestHeader),credentials:this.withCredentials?"include":"same-origin"}),o=this.mimeType,c=this.responseType;fetch(a).then(l=>{if(l.status===200||l.status===0){if(l.status===0&&console.warn("THREE.FileLoader: HTTP Status 0 received."),typeof ReadableStream>"u"||l.body===void 0||l.body.getReader===void 0)return l;const h=Zn[t],u=l.body.getReader(),d=l.headers.get("X-File-Size")||l.headers.get("Content-Length"),p=d?parseInt(d):0,g=p!==0;let _=0;const m=new ReadableStream({start(f){R();function R(){u.read().then(({done:C,value:x})=>{if(C)f.close();else{_+=x.byteLength;const N=new ProgressEvent("progress",{lengthComputable:g,loaded:_,total:p});for(let P=0,L=h.length;P<L;P++){const I=h[P];I.onProgress&&I.onProgress(N)}f.enqueue(x),R()}},C=>{f.error(C)})}}});return new Response(m)}else throw new D0(`fetch for "${l.url}" responded with ${l.status}: ${l.statusText}`,l)}).then(l=>{switch(c){case"arraybuffer":return l.arrayBuffer();case"blob":return l.blob();case"document":return l.text().then(h=>new DOMParser().parseFromString(h,o));case"json":return l.json();default:if(o===void 0)return l.text();{const u=/charset="?([^;"\s]*)"?/i.exec(o),d=u&&u[1]?u[1].toLowerCase():void 0,p=new TextDecoder(d);return l.arrayBuffer().then(g=>p.decode(g))}}}).then(l=>{va.add(t,l);const h=Zn[t];delete Zn[t];for(let u=0,d=h.length;u<d;u++){const p=h[u];p.onLoad&&p.onLoad(l)}}).catch(l=>{const h=Zn[t];if(h===void 0)throw this.manager.itemError(t),l;delete Zn[t];for(let u=0,d=h.length;u<d;u++){const p=h[u];p.onError&&p.onError(l)}this.manager.itemError(t)}).finally(()=>{this.manager.itemEnd(t)}),this.manager.itemStart(t)}setResponseType(t){return this.responseType=t,this}setMimeType(t){return this.mimeType=t,this}}class I0 extends ri{constructor(t){super(t)}load(t,e,n,s){this.path!==void 0&&(t=this.path+t),t=this.manager.resolveURL(t);const r=this,a=va.get(t);if(a!==void 0)return r.manager.itemStart(t),setTimeout(function(){e&&e(a),r.manager.itemEnd(t)},0),a;const o=sr("img");function c(){h(),va.add(t,this),e&&e(this),r.manager.itemEnd(t)}function l(u){h(),s&&s(u),r.manager.itemError(t),r.manager.itemEnd(t)}function h(){o.removeEventListener("load",c,!1),o.removeEventListener("error",l,!1)}return o.addEventListener("load",c,!1),o.addEventListener("error",l,!1),t.slice(0,5)!=="data:"&&this.crossOrigin!==void 0&&(o.crossOrigin=this.crossOrigin),r.manager.itemStart(t),o.src=t,o}}class U0 extends ri{constructor(t){super(t)}load(t,e,n,s){const r=this,a=new ru,o=new Ca(this.manager);return o.setResponseType("arraybuffer"),o.setRequestHeader(this.requestHeader),o.setPath(this.path),o.setWithCredentials(r.withCredentials),o.load(t,function(c){let l;try{l=r.parse(c)}catch(h){if(s!==void 0)s(h);else{console.error(h);return}}l.image!==void 0?a.image=l.image:l.data!==void 0&&(a.image.width=l.width,a.image.height=l.height,a.image.data=l.data),a.wrapS=l.wrapS!==void 0?l.wrapS:Nn,a.wrapT=l.wrapT!==void 0?l.wrapT:Nn,a.magFilter=l.magFilter!==void 0?l.magFilter:fn,a.minFilter=l.minFilter!==void 0?l.minFilter:fn,a.anisotropy=l.anisotropy!==void 0?l.anisotropy:1,l.colorSpace!==void 0&&(a.colorSpace=l.colorSpace),l.flipY!==void 0&&(a.flipY=l.flipY),l.format!==void 0&&(a.format=l.format),l.type!==void 0&&(a.type=l.type),l.mipmaps!==void 0&&(a.mipmaps=l.mipmaps,a.minFilter=ti),l.mipmapCount===1&&(a.minFilter=fn),l.generateMipmaps!==void 0&&(a.generateMipmaps=l.generateMipmaps),a.needsUpdate=!0,e&&e(a,l)},n,s),a}}class fu extends ri{constructor(t){super(t)}load(t,e,n,s){const r=new Ye,a=new I0(this.manager);return a.setCrossOrigin(this.crossOrigin),a.setPath(this.path),a.load(t,function(o){r.image=o,r.needsUpdate=!0,e!==void 0&&e(r)},n,s),r}}class pr extends Le{constructor(t,e=1){super(),this.isLight=!0,this.type="Light",this.color=new $t(t),this.intensity=e}dispose(){}copy(t,e){return super.copy(t,e),this.color.copy(t.color),this.intensity=t.intensity,this}toJSON(t){const e=super.toJSON(t);return e.object.color=this.color.getHex(),e.object.intensity=this.intensity,this.groundColor!==void 0&&(e.object.groundColor=this.groundColor.getHex()),this.distance!==void 0&&(e.object.distance=this.distance),this.angle!==void 0&&(e.object.angle=this.angle),this.decay!==void 0&&(e.object.decay=this.decay),this.penumbra!==void 0&&(e.object.penumbra=this.penumbra),this.shadow!==void 0&&(e.object.shadow=this.shadow.toJSON()),this.target!==void 0&&(e.object.target=this.target.uuid),e}}class F0 extends pr{constructor(t,e,n){super(t,n),this.isHemisphereLight=!0,this.type="HemisphereLight",this.position.copy(Le.DEFAULT_UP),this.updateMatrix(),this.groundColor=new $t(e)}copy(t,e){return super.copy(t,e),this.groundColor.copy(t.groundColor),this}}const vo=new Kt,lh=new F,ch=new F;class Ll{constructor(t){this.camera=t,this.intensity=1,this.bias=0,this.normalBias=0,this.radius=1,this.blurSamples=8,this.mapSize=new Lt(512,512),this.map=null,this.mapPass=null,this.matrix=new Kt,this.autoUpdate=!0,this.needsUpdate=!1,this._frustum=new bl,this._frameExtents=new Lt(1,1),this._viewportCount=1,this._viewports=[new ve(0,0,1,1)]}getViewportCount(){return this._viewportCount}getFrustum(){return this._frustum}updateMatrices(t){const e=this.camera,n=this.matrix;lh.setFromMatrixPosition(t.matrixWorld),e.position.copy(lh),ch.setFromMatrixPosition(t.target.matrixWorld),e.lookAt(ch),e.updateMatrixWorld(),vo.multiplyMatrices(e.projectionMatrix,e.matrixWorldInverse),this._frustum.setFromProjectionMatrix(vo),n.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),n.multiply(vo)}getViewport(t){return this._viewports[t]}getFrameExtents(){return this._frameExtents}dispose(){this.map&&this.map.dispose(),this.mapPass&&this.mapPass.dispose()}copy(t){return this.camera=t.camera.clone(),this.intensity=t.intensity,this.bias=t.bias,this.radius=t.radius,this.mapSize.copy(t.mapSize),this}clone(){return new this.constructor().copy(this)}toJSON(){const t={};return this.intensity!==1&&(t.intensity=this.intensity),this.bias!==0&&(t.bias=this.bias),this.normalBias!==0&&(t.normalBias=this.normalBias),this.radius!==1&&(t.radius=this.radius),(this.mapSize.x!==512||this.mapSize.y!==512)&&(t.mapSize=this.mapSize.toArray()),t.camera=this.camera.toJSON(!1).object,delete t.camera.matrix,t}}class O0 extends Ll{constructor(){super(new Je(50,1,.5,500)),this.isSpotLightShadow=!0,this.focus=1}updateMatrices(t){const e=this.camera,n=Ts*2*t.angle*this.focus,s=this.mapSize.width/this.mapSize.height,r=t.distance||e.far;(n!==e.fov||s!==e.aspect||r!==e.far)&&(e.fov=n,e.aspect=s,e.far=r,e.updateProjectionMatrix()),super.updateMatrices(t)}copy(t){return super.copy(t),this.focus=t.focus,this}}class k0 extends pr{constructor(t,e,n=0,s=Math.PI/3,r=0,a=2){super(t,e),this.isSpotLight=!0,this.type="SpotLight",this.position.copy(Le.DEFAULT_UP),this.updateMatrix(),this.target=new Le,this.distance=n,this.angle=s,this.penumbra=r,this.decay=a,this.map=null,this.shadow=new O0}get power(){return this.intensity*Math.PI}set power(t){this.intensity=t/Math.PI}dispose(){this.shadow.dispose()}copy(t,e){return super.copy(t,e),this.distance=t.distance,this.angle=t.angle,this.penumbra=t.penumbra,this.decay=t.decay,this.target=t.target.clone(),this.shadow=t.shadow.clone(),this}}const hh=new Kt,js=new F,xo=new F;class B0 extends Ll{constructor(){super(new Je(90,1,.5,500)),this.isPointLightShadow=!0,this._frameExtents=new Lt(4,2),this._viewportCount=6,this._viewports=[new ve(2,1,1,1),new ve(0,1,1,1),new ve(3,1,1,1),new ve(1,1,1,1),new ve(3,0,1,1),new ve(1,0,1,1)],this._cubeDirections=[new F(1,0,0),new F(-1,0,0),new F(0,0,1),new F(0,0,-1),new F(0,1,0),new F(0,-1,0)],this._cubeUps=[new F(0,1,0),new F(0,1,0),new F(0,1,0),new F(0,1,0),new F(0,0,1),new F(0,0,-1)]}updateMatrices(t,e=0){const n=this.camera,s=this.matrix,r=t.distance||n.far;r!==n.far&&(n.far=r,n.updateProjectionMatrix()),js.setFromMatrixPosition(t.matrixWorld),n.position.copy(js),xo.copy(n.position),xo.add(this._cubeDirections[e]),n.up.copy(this._cubeUps[e]),n.lookAt(xo),n.updateMatrixWorld(),s.makeTranslation(-js.x,-js.y,-js.z),hh.multiplyMatrices(n.projectionMatrix,n.matrixWorldInverse),this._frustum.setFromProjectionMatrix(hh)}}class z0 extends pr{constructor(t,e,n=0,s=2){super(t,e),this.isPointLight=!0,this.type="PointLight",this.distance=n,this.decay=s,this.shadow=new B0}get power(){return this.intensity*4*Math.PI}set power(t){this.intensity=t/(4*Math.PI)}dispose(){this.shadow.dispose()}copy(t,e){return super.copy(t,e),this.distance=t.distance,this.decay=t.decay,this.shadow=t.shadow.clone(),this}}class V0 extends Ll{constructor(){super(new El(-5,5,5,-5,.5,500)),this.isDirectionalLightShadow=!0}}class pu extends pr{constructor(t,e){super(t,e),this.isDirectionalLight=!0,this.type="DirectionalLight",this.position.copy(Le.DEFAULT_UP),this.updateMatrix(),this.target=new Le,this.shadow=new V0}dispose(){this.shadow.dispose()}copy(t){return super.copy(t),this.target=t.target.clone(),this.shadow=t.shadow.clone(),this}}class H0 extends pr{constructor(t,e){super(t,e),this.isAmbientLight=!0,this.type="AmbientLight"}}class mu{static decodeText(t){if(console.warn("THREE.LoaderUtils: decodeText() has been deprecated with r165 and will be removed with r175. Use TextDecoder instead."),typeof TextDecoder<"u")return new TextDecoder().decode(t);let e="";for(let n=0,s=t.length;n<s;n++)e+=String.fromCharCode(t[n]);try{return decodeURIComponent(escape(e))}catch{return e}}static extractUrlBase(t){const e=t.lastIndexOf("/");return e===-1?"./":t.slice(0,e+1)}static resolveURL(t,e){return typeof t!="string"||t===""?"":(/^https?:\/\//i.test(e)&&/^\//.test(t)&&(e=e.replace(/(^https?:\/\/[^\/]+).*/i,"$1")),/^(https?:)?\/\//i.test(t)||/^data:.*,.*$/i.test(t)||/^blob:.*$/i.test(t)?t:e+t)}}class uh{constructor(t=1,e=0,n=0){return this.radius=t,this.phi=e,this.theta=n,this}set(t,e,n){return this.radius=t,this.phi=e,this.theta=n,this}copy(t){return this.radius=t.radius,this.phi=t.phi,this.theta=t.theta,this}makeSafe(){return this.phi=Math.max(1e-6,Math.min(Math.PI-1e-6,this.phi)),this}setFromVector3(t){return this.setFromCartesianCoords(t.x,t.y,t.z)}setFromCartesianCoords(t,e,n){return this.radius=Math.sqrt(t*t+e*e+n*n),this.radius===0?(this.theta=0,this.phi=0):(this.theta=Math.atan2(t,n),this.phi=Math.acos(Xe(e/this.radius,-1,1))),this}clone(){return new this.constructor().copy(this)}}class G0 extends ga{constructor(t=10,e=10,n=4473924,s=8947848){n=new $t(n),s=new $t(s);const r=e/2,a=t/e,o=t/2,c=[],l=[];for(let d=0,p=0,g=-o;d<=e;d++,g+=a){c.push(-o,0,g,o,0,g),c.push(g,0,-o,g,0,o);const _=d===r?n:s;_.toArray(l,p),p+=3,_.toArray(l,p),p+=3,_.toArray(l,p),p+=3,_.toArray(l,p),p+=3}const h=new Ie;h.setAttribute("position",new he(c,3)),h.setAttribute("color",new he(l,3));const u=new zn({vertexColors:!0,toneMapped:!1});super(h,u),this.type="GridHelper"}dispose(){this.geometry.dispose(),this.material.dispose()}}const dh=new F;let Jr,yo;class rr extends Le{constructor(t=new F(0,0,1),e=new F(0,0,0),n=1,s=16776960,r=n*.2,a=r*.2){super(),this.type="ArrowHelper",Jr===void 0&&(Jr=new Ie,Jr.setAttribute("position",new he([0,0,0,0,1,0],3)),yo=new Fs(0,.5,1,5,1),yo.translate(0,-.5,0)),this.position.copy(e),this.line=new dr(Jr,new zn({color:s,toneMapped:!1})),this.line.matrixAutoUpdate=!1,this.add(this.line),this.cone=new Ae(yo,new Un({color:s,toneMapped:!1})),this.cone.matrixAutoUpdate=!1,this.add(this.cone),this.setDirection(t),this.setLength(n,r,a)}setDirection(t){if(t.y>.99999)this.quaternion.set(0,0,0,1);else if(t.y<-.99999)this.quaternion.set(1,0,0,0);else{dh.set(t.z,0,-t.x).normalize();const e=Math.acos(t.y);this.quaternion.setFromAxisAngle(dh,e)}}setLength(t,e=t*.2,n=e*.2){this.line.scale.set(1,Math.max(1e-4,t-e),1),this.line.updateMatrix(),this.cone.scale.set(n,e,n),this.cone.position.y=t,this.cone.updateMatrix()}setColor(t){this.line.material.color.set(t),this.cone.material.color.set(t)}copy(t){return super.copy(t,!1),this.line.copy(t.line),this.cone.copy(t.cone),this}dispose(){this.line.geometry.dispose(),this.line.material.dispose(),this.cone.geometry.dispose(),this.cone.material.dispose()}}class W0 extends qi{constructor(t,e=null){super(),this.object=t,this.domElement=e,this.enabled=!0,this.state=-1,this.keys={},this.mouseButtons={LEFT:null,MIDDLE:null,RIGHT:null},this.touches={ONE:null,TWO:null}}connect(){}disconnect(){}dispose(){}update(){}}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:ml}}));typeof window<"u"&&(window.__THREE__?console.warn("WARNING: Multiple instances of Three.js being imported."):window.__THREE__=ml);const fh={type:"change"},Nl={type:"start"},gu={type:"end"},Zr=new ur,ph=new vi,X0=Math.cos(70*Mi.DEG2RAD),Be=new F,on=2*Math.PI,Te={NONE:-1,ROTATE:0,DOLLY:1,PAN:2,TOUCH_ROTATE:3,TOUCH_PAN:4,TOUCH_DOLLY_PAN:5,TOUCH_DOLLY_ROTATE:6},Mo=1e-6;class q0 extends W0{constructor(t,e=null){super(t,e),this.state=Te.NONE,this.enabled=!0,this.target=new F,this.cursor=new F,this.minDistance=0,this.maxDistance=1/0,this.minZoom=0,this.maxZoom=1/0,this.minTargetRadius=0,this.maxTargetRadius=1/0,this.minPolarAngle=0,this.maxPolarAngle=Math.PI,this.minAzimuthAngle=-1/0,this.maxAzimuthAngle=1/0,this.enableDamping=!1,this.dampingFactor=.05,this.enableZoom=!0,this.zoomSpeed=1,this.enableRotate=!0,this.rotateSpeed=1,this.enablePan=!0,this.panSpeed=1,this.screenSpacePanning=!0,this.keyPanSpeed=7,this.zoomToCursor=!1,this.autoRotate=!1,this.autoRotateSpeed=2,this.keys={LEFT:"ArrowLeft",UP:"ArrowUp",RIGHT:"ArrowRight",BOTTOM:"ArrowDown"},this.mouseButtons={LEFT:ps.ROTATE,MIDDLE:ps.DOLLY,RIGHT:ps.PAN},this.touches={ONE:ds.ROTATE,TWO:ds.DOLLY_PAN},this.target0=this.target.clone(),this.position0=this.object.position.clone(),this.zoom0=this.object.zoom,this._domElementKeyEvents=null,this._lastPosition=new F,this._lastQuaternion=new Dn,this._lastTargetPosition=new F,this._quat=new Dn().setFromUnitVectors(t.up,new F(0,1,0)),this._quatInverse=this._quat.clone().invert(),this._spherical=new uh,this._sphericalDelta=new uh,this._scale=1,this._panOffset=new F,this._rotateStart=new Lt,this._rotateEnd=new Lt,this._rotateDelta=new Lt,this._panStart=new Lt,this._panEnd=new Lt,this._panDelta=new Lt,this._dollyStart=new Lt,this._dollyEnd=new Lt,this._dollyDelta=new Lt,this._dollyDirection=new F,this._mouse=new Lt,this._performCursorZoom=!1,this._pointers=[],this._pointerPositions={},this._controlActive=!1,this._onPointerMove=Y0.bind(this),this._onPointerDown=j0.bind(this),this._onPointerUp=$0.bind(this),this._onContextMenu=nv.bind(this),this._onMouseWheel=Z0.bind(this),this._onKeyDown=Q0.bind(this),this._onTouchStart=tv.bind(this),this._onTouchMove=ev.bind(this),this._onMouseDown=K0.bind(this),this._onMouseMove=J0.bind(this),this._interceptControlDown=iv.bind(this),this._interceptControlUp=sv.bind(this),this.domElement!==null&&this.connect(),this.update()}connect(){this.domElement.addEventListener("pointerdown",this._onPointerDown),this.domElement.addEventListener("pointercancel",this._onPointerUp),this.domElement.addEventListener("contextmenu",this._onContextMenu),this.domElement.addEventListener("wheel",this._onMouseWheel,{passive:!1}),this.domElement.getRootNode().addEventListener("keydown",this._interceptControlDown,{passive:!0,capture:!0}),this.domElement.style.touchAction="none"}disconnect(){this.domElement.removeEventListener("pointerdown",this._onPointerDown),this.domElement.removeEventListener("pointermove",this._onPointerMove),this.domElement.removeEventListener("pointerup",this._onPointerUp),this.domElement.removeEventListener("pointercancel",this._onPointerUp),this.domElement.removeEventListener("wheel",this._onMouseWheel),this.domElement.removeEventListener("contextmenu",this._onContextMenu),this.stopListenToKeyEvents(),this.domElement.getRootNode().removeEventListener("keydown",this._interceptControlDown,{capture:!0}),this.domElement.style.touchAction="auto"}dispose(){this.disconnect()}getPolarAngle(){return this._spherical.phi}getAzimuthalAngle(){return this._spherical.theta}getDistance(){return this.object.position.distanceTo(this.target)}listenToKeyEvents(t){t.addEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=t}stopListenToKeyEvents(){this._domElementKeyEvents!==null&&(this._domElementKeyEvents.removeEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=null)}saveState(){this.target0.copy(this.target),this.position0.copy(this.object.position),this.zoom0=this.object.zoom}reset(){this.target.copy(this.target0),this.object.position.copy(this.position0),this.object.zoom=this.zoom0,this.object.updateProjectionMatrix(),this.dispatchEvent(fh),this.update(),this.state=Te.NONE}update(t=null){const e=this.object.position;Be.copy(e).sub(this.target),Be.applyQuaternion(this._quat),this._spherical.setFromVector3(Be),this.autoRotate&&this.state===Te.NONE&&this._rotateLeft(this._getAutoRotationAngle(t)),this.enableDamping?(this._spherical.theta+=this._sphericalDelta.theta*this.dampingFactor,this._spherical.phi+=this._sphericalDelta.phi*this.dampingFactor):(this._spherical.theta+=this._sphericalDelta.theta,this._spherical.phi+=this._sphericalDelta.phi);let n=this.minAzimuthAngle,s=this.maxAzimuthAngle;isFinite(n)&&isFinite(s)&&(n<-Math.PI?n+=on:n>Math.PI&&(n-=on),s<-Math.PI?s+=on:s>Math.PI&&(s-=on),n<=s?this._spherical.theta=Math.max(n,Math.min(s,this._spherical.theta)):this._spherical.theta=this._spherical.theta>(n+s)/2?Math.max(n,this._spherical.theta):Math.min(s,this._spherical.theta)),this._spherical.phi=Math.max(this.minPolarAngle,Math.min(this.maxPolarAngle,this._spherical.phi)),this._spherical.makeSafe(),this.enableDamping===!0?this.target.addScaledVector(this._panOffset,this.dampingFactor):this.target.add(this._panOffset),this.target.sub(this.cursor),this.target.clampLength(this.minTargetRadius,this.maxTargetRadius),this.target.add(this.cursor);let r=!1;if(this.zoomToCursor&&this._performCursorZoom||this.object.isOrthographicCamera)this._spherical.radius=this._clampDistance(this._spherical.radius);else{const a=this._spherical.radius;this._spherical.radius=this._clampDistance(this._spherical.radius*this._scale),r=a!=this._spherical.radius}if(Be.setFromSpherical(this._spherical),Be.applyQuaternion(this._quatInverse),e.copy(this.target).add(Be),this.object.lookAt(this.target),this.enableDamping===!0?(this._sphericalDelta.theta*=1-this.dampingFactor,this._sphericalDelta.phi*=1-this.dampingFactor,this._panOffset.multiplyScalar(1-this.dampingFactor)):(this._sphericalDelta.set(0,0,0),this._panOffset.set(0,0,0)),this.zoomToCursor&&this._performCursorZoom){let a=null;if(this.object.isPerspectiveCamera){const o=Be.length();a=this._clampDistance(o*this._scale);const c=o-a;this.object.position.addScaledVector(this._dollyDirection,c),this.object.updateMatrixWorld(),r=!!c}else if(this.object.isOrthographicCamera){const o=new F(this._mouse.x,this._mouse.y,0);o.unproject(this.object);const c=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),this.object.updateProjectionMatrix(),r=c!==this.object.zoom;const l=new F(this._mouse.x,this._mouse.y,0);l.unproject(this.object),this.object.position.sub(l).add(o),this.object.updateMatrixWorld(),a=Be.length()}else console.warn("WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled."),this.zoomToCursor=!1;a!==null&&(this.screenSpacePanning?this.target.set(0,0,-1).transformDirection(this.object.matrix).multiplyScalar(a).add(this.object.position):(Zr.origin.copy(this.object.position),Zr.direction.set(0,0,-1).transformDirection(this.object.matrix),Math.abs(this.object.up.dot(Zr.direction))<X0?this.object.lookAt(this.target):(ph.setFromNormalAndCoplanarPoint(this.object.up,this.target),Zr.intersectPlane(ph,this.target))))}else if(this.object.isOrthographicCamera){const a=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),a!==this.object.zoom&&(this.object.updateProjectionMatrix(),r=!0)}return this._scale=1,this._performCursorZoom=!1,r||this._lastPosition.distanceToSquared(this.object.position)>Mo||8*(1-this._lastQuaternion.dot(this.object.quaternion))>Mo||this._lastTargetPosition.distanceToSquared(this.target)>Mo?(this.dispatchEvent(fh),this._lastPosition.copy(this.object.position),this._lastQuaternion.copy(this.object.quaternion),this._lastTargetPosition.copy(this.target),!0):!1}_getAutoRotationAngle(t){return t!==null?on/60*this.autoRotateSpeed*t:on/60/60*this.autoRotateSpeed}_getZoomScale(t){const e=Math.abs(t*.01);return Math.pow(.95,this.zoomSpeed*e)}_rotateLeft(t){this._sphericalDelta.theta-=t}_rotateUp(t){this._sphericalDelta.phi-=t}_panLeft(t,e){Be.setFromMatrixColumn(e,0),Be.multiplyScalar(-t),this._panOffset.add(Be)}_panUp(t,e){this.screenSpacePanning===!0?Be.setFromMatrixColumn(e,1):(Be.setFromMatrixColumn(e,0),Be.crossVectors(this.object.up,Be)),Be.multiplyScalar(t),this._panOffset.add(Be)}_pan(t,e){const n=this.domElement;if(this.object.isPerspectiveCamera){const s=this.object.position;Be.copy(s).sub(this.target);let r=Be.length();r*=Math.tan(this.object.fov/2*Math.PI/180),this._panLeft(2*t*r/n.clientHeight,this.object.matrix),this._panUp(2*e*r/n.clientHeight,this.object.matrix)}else this.object.isOrthographicCamera?(this._panLeft(t*(this.object.right-this.object.left)/this.object.zoom/n.clientWidth,this.object.matrix),this._panUp(e*(this.object.top-this.object.bottom)/this.object.zoom/n.clientHeight,this.object.matrix)):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."),this.enablePan=!1)}_dollyOut(t){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale/=t:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_dollyIn(t){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale*=t:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_updateZoomParameters(t,e){if(!this.zoomToCursor)return;this._performCursorZoom=!0;const n=this.domElement.getBoundingClientRect(),s=t-n.left,r=e-n.top,a=n.width,o=n.height;this._mouse.x=s/a*2-1,this._mouse.y=-(r/o)*2+1,this._dollyDirection.set(this._mouse.x,this._mouse.y,1).unproject(this.object).sub(this.object.position).normalize()}_clampDistance(t){return Math.max(this.minDistance,Math.min(this.maxDistance,t))}_handleMouseDownRotate(t){this._rotateStart.set(t.clientX,t.clientY)}_handleMouseDownDolly(t){this._updateZoomParameters(t.clientX,t.clientX),this._dollyStart.set(t.clientX,t.clientY)}_handleMouseDownPan(t){this._panStart.set(t.clientX,t.clientY)}_handleMouseMoveRotate(t){this._rotateEnd.set(t.clientX,t.clientY),this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);const e=this.domElement;this._rotateLeft(on*this._rotateDelta.x/e.clientHeight),this._rotateUp(on*this._rotateDelta.y/e.clientHeight),this._rotateStart.copy(this._rotateEnd),this.update()}_handleMouseMoveDolly(t){this._dollyEnd.set(t.clientX,t.clientY),this._dollyDelta.subVectors(this._dollyEnd,this._dollyStart),this._dollyDelta.y>0?this._dollyOut(this._getZoomScale(this._dollyDelta.y)):this._dollyDelta.y<0&&this._dollyIn(this._getZoomScale(this._dollyDelta.y)),this._dollyStart.copy(this._dollyEnd),this.update()}_handleMouseMovePan(t){this._panEnd.set(t.clientX,t.clientY),this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd),this.update()}_handleMouseWheel(t){this._updateZoomParameters(t.clientX,t.clientY),t.deltaY<0?this._dollyIn(this._getZoomScale(t.deltaY)):t.deltaY>0&&this._dollyOut(this._getZoomScale(t.deltaY)),this.update()}_handleKeyDown(t){let e=!1;switch(t.code){case this.keys.UP:t.ctrlKey||t.metaKey||t.shiftKey?this._rotateUp(on*this.rotateSpeed/this.domElement.clientHeight):this._pan(0,this.keyPanSpeed),e=!0;break;case this.keys.BOTTOM:t.ctrlKey||t.metaKey||t.shiftKey?this._rotateUp(-on*this.rotateSpeed/this.domElement.clientHeight):this._pan(0,-this.keyPanSpeed),e=!0;break;case this.keys.LEFT:t.ctrlKey||t.metaKey||t.shiftKey?this._rotateLeft(on*this.rotateSpeed/this.domElement.clientHeight):this._pan(this.keyPanSpeed,0),e=!0;break;case this.keys.RIGHT:t.ctrlKey||t.metaKey||t.shiftKey?this._rotateLeft(-on*this.rotateSpeed/this.domElement.clientHeight):this._pan(-this.keyPanSpeed,0),e=!0;break}e&&(t.preventDefault(),this.update())}_handleTouchStartRotate(t){if(this._pointers.length===1)this._rotateStart.set(t.pageX,t.pageY);else{const e=this._getSecondPointerPosition(t),n=.5*(t.pageX+e.x),s=.5*(t.pageY+e.y);this._rotateStart.set(n,s)}}_handleTouchStartPan(t){if(this._pointers.length===1)this._panStart.set(t.pageX,t.pageY);else{const e=this._getSecondPointerPosition(t),n=.5*(t.pageX+e.x),s=.5*(t.pageY+e.y);this._panStart.set(n,s)}}_handleTouchStartDolly(t){const e=this._getSecondPointerPosition(t),n=t.pageX-e.x,s=t.pageY-e.y,r=Math.sqrt(n*n+s*s);this._dollyStart.set(0,r)}_handleTouchStartDollyPan(t){this.enableZoom&&this._handleTouchStartDolly(t),this.enablePan&&this._handleTouchStartPan(t)}_handleTouchStartDollyRotate(t){this.enableZoom&&this._handleTouchStartDolly(t),this.enableRotate&&this._handleTouchStartRotate(t)}_handleTouchMoveRotate(t){if(this._pointers.length==1)this._rotateEnd.set(t.pageX,t.pageY);else{const n=this._getSecondPointerPosition(t),s=.5*(t.pageX+n.x),r=.5*(t.pageY+n.y);this._rotateEnd.set(s,r)}this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);const e=this.domElement;this._rotateLeft(on*this._rotateDelta.x/e.clientHeight),this._rotateUp(on*this._rotateDelta.y/e.clientHeight),this._rotateStart.copy(this._rotateEnd)}_handleTouchMovePan(t){if(this._pointers.length===1)this._panEnd.set(t.pageX,t.pageY);else{const e=this._getSecondPointerPosition(t),n=.5*(t.pageX+e.x),s=.5*(t.pageY+e.y);this._panEnd.set(n,s)}this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd)}_handleTouchMoveDolly(t){const e=this._getSecondPointerPosition(t),n=t.pageX-e.x,s=t.pageY-e.y,r=Math.sqrt(n*n+s*s);this._dollyEnd.set(0,r),this._dollyDelta.set(0,Math.pow(this._dollyEnd.y/this._dollyStart.y,this.zoomSpeed)),this._dollyOut(this._dollyDelta.y),this._dollyStart.copy(this._dollyEnd);const a=(t.pageX+e.x)*.5,o=(t.pageY+e.y)*.5;this._updateZoomParameters(a,o)}_handleTouchMoveDollyPan(t){this.enableZoom&&this._handleTouchMoveDolly(t),this.enablePan&&this._handleTouchMovePan(t)}_handleTouchMoveDollyRotate(t){this.enableZoom&&this._handleTouchMoveDolly(t),this.enableRotate&&this._handleTouchMoveRotate(t)}_addPointer(t){this._pointers.push(t.pointerId)}_removePointer(t){delete this._pointerPositions[t.pointerId];for(let e=0;e<this._pointers.length;e++)if(this._pointers[e]==t.pointerId){this._pointers.splice(e,1);return}}_isTrackingPointer(t){for(let e=0;e<this._pointers.length;e++)if(this._pointers[e]==t.pointerId)return!0;return!1}_trackPointer(t){let e=this._pointerPositions[t.pointerId];e===void 0&&(e=new Lt,this._pointerPositions[t.pointerId]=e),e.set(t.pageX,t.pageY)}_getSecondPointerPosition(t){const e=t.pointerId===this._pointers[0]?this._pointers[1]:this._pointers[0];return this._pointerPositions[e]}_customWheelEvent(t){const e=t.deltaMode,n={clientX:t.clientX,clientY:t.clientY,deltaY:t.deltaY};switch(e){case 1:n.deltaY*=16;break;case 2:n.deltaY*=100;break}return t.ctrlKey&&!this._controlActive&&(n.deltaY*=10),n}}function j0(i){this.enabled!==!1&&(this._pointers.length===0&&(this.domElement.setPointerCapture(i.pointerId),this.domElement.addEventListener("pointermove",this._onPointerMove),this.domElement.addEventListener("pointerup",this._onPointerUp)),!this._isTrackingPointer(i)&&(this._addPointer(i),i.pointerType==="touch"?this._onTouchStart(i):this._onMouseDown(i)))}function Y0(i){this.enabled!==!1&&(i.pointerType==="touch"?this._onTouchMove(i):this._onMouseMove(i))}function $0(i){switch(this._removePointer(i),this._pointers.length){case 0:this.domElement.releasePointerCapture(i.pointerId),this.domElement.removeEventListener("pointermove",this._onPointerMove),this.domElement.removeEventListener("pointerup",this._onPointerUp),this.dispatchEvent(gu),this.state=Te.NONE;break;case 1:const t=this._pointers[0],e=this._pointerPositions[t];this._onTouchStart({pointerId:t,pageX:e.x,pageY:e.y});break}}function K0(i){let t;switch(i.button){case 0:t=this.mouseButtons.LEFT;break;case 1:t=this.mouseButtons.MIDDLE;break;case 2:t=this.mouseButtons.RIGHT;break;default:t=-1}switch(t){case ps.DOLLY:if(this.enableZoom===!1)return;this._handleMouseDownDolly(i),this.state=Te.DOLLY;break;case ps.ROTATE:if(i.ctrlKey||i.metaKey||i.shiftKey){if(this.enablePan===!1)return;this._handleMouseDownPan(i),this.state=Te.PAN}else{if(this.enableRotate===!1)return;this._handleMouseDownRotate(i),this.state=Te.ROTATE}break;case ps.PAN:if(i.ctrlKey||i.metaKey||i.shiftKey){if(this.enableRotate===!1)return;this._handleMouseDownRotate(i),this.state=Te.ROTATE}else{if(this.enablePan===!1)return;this._handleMouseDownPan(i),this.state=Te.PAN}break;default:this.state=Te.NONE}this.state!==Te.NONE&&this.dispatchEvent(Nl)}function J0(i){switch(this.state){case Te.ROTATE:if(this.enableRotate===!1)return;this._handleMouseMoveRotate(i);break;case Te.DOLLY:if(this.enableZoom===!1)return;this._handleMouseMoveDolly(i);break;case Te.PAN:if(this.enablePan===!1)return;this._handleMouseMovePan(i);break}}function Z0(i){this.enabled===!1||this.enableZoom===!1||this.state!==Te.NONE||(i.preventDefault(),this.dispatchEvent(Nl),this._handleMouseWheel(this._customWheelEvent(i)),this.dispatchEvent(gu))}function Q0(i){this.enabled===!1||this.enablePan===!1||this._handleKeyDown(i)}function tv(i){switch(this._trackPointer(i),this._pointers.length){case 1:switch(this.touches.ONE){case ds.ROTATE:if(this.enableRotate===!1)return;this._handleTouchStartRotate(i),this.state=Te.TOUCH_ROTATE;break;case ds.PAN:if(this.enablePan===!1)return;this._handleTouchStartPan(i),this.state=Te.TOUCH_PAN;break;default:this.state=Te.NONE}break;case 2:switch(this.touches.TWO){case ds.DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchStartDollyPan(i),this.state=Te.TOUCH_DOLLY_PAN;break;case ds.DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchStartDollyRotate(i),this.state=Te.TOUCH_DOLLY_ROTATE;break;default:this.state=Te.NONE}break;default:this.state=Te.NONE}this.state!==Te.NONE&&this.dispatchEvent(Nl)}function ev(i){switch(this._trackPointer(i),this.state){case Te.TOUCH_ROTATE:if(this.enableRotate===!1)return;this._handleTouchMoveRotate(i),this.update();break;case Te.TOUCH_PAN:if(this.enablePan===!1)return;this._handleTouchMovePan(i),this.update();break;case Te.TOUCH_DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchMoveDollyPan(i),this.update();break;case Te.TOUCH_DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchMoveDollyRotate(i),this.update();break;default:this.state=Te.NONE}}function nv(i){this.enabled!==!1&&i.preventDefault()}function iv(i){i.key==="Control"&&(this._controlActive=!0,this.domElement.getRootNode().addEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}function sv(i){i.key==="Control"&&(this._controlActive=!1,this.domElement.getRootNode().removeEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}const rv=/^[og]\s*(.+)?/,av=/^mtllib /,ov=/^usemtl /,lv=/^usemap /,mh=/\s+/,gh=new F,So=new F,_h=new F,vh=new F,Mn=new F,Qr=new $t;function cv(){const i={objects:[],object:{},vertices:[],normals:[],colors:[],uvs:[],materials:{},materialLibraries:[],startObject:function(t,e){if(this.object&&this.object.fromDeclaration===!1){this.object.name=t,this.object.fromDeclaration=e!==!1;return}const n=this.object&&typeof this.object.currentMaterial=="function"?this.object.currentMaterial():void 0;if(this.object&&typeof this.object._finalize=="function"&&this.object._finalize(!0),this.object={name:t||"",fromDeclaration:e!==!1,geometry:{vertices:[],normals:[],colors:[],uvs:[],hasUVIndices:!1},materials:[],smooth:!0,startMaterial:function(s,r){const a=this._finalize(!1);a&&(a.inherited||a.groupCount<=0)&&this.materials.splice(a.index,1);const o={index:this.materials.length,name:s||"",mtllib:Array.isArray(r)&&r.length>0?r[r.length-1]:"",smooth:a!==void 0?a.smooth:this.smooth,groupStart:a!==void 0?a.groupEnd:0,groupEnd:-1,groupCount:-1,inherited:!1,clone:function(c){const l={index:typeof c=="number"?c:this.index,name:this.name,mtllib:this.mtllib,smooth:this.smooth,groupStart:0,groupEnd:-1,groupCount:-1,inherited:!1};return l.clone=this.clone.bind(l),l}};return this.materials.push(o),o},currentMaterial:function(){if(this.materials.length>0)return this.materials[this.materials.length-1]},_finalize:function(s){const r=this.currentMaterial();if(r&&r.groupEnd===-1&&(r.groupEnd=this.geometry.vertices.length/3,r.groupCount=r.groupEnd-r.groupStart,r.inherited=!1),s&&this.materials.length>1)for(let a=this.materials.length-1;a>=0;a--)this.materials[a].groupCount<=0&&this.materials.splice(a,1);return s&&this.materials.length===0&&this.materials.push({name:"",smooth:this.smooth}),r}},n&&n.name&&typeof n.clone=="function"){const s=n.clone(0);s.inherited=!0,this.object.materials.push(s)}this.objects.push(this.object)},finalize:function(){this.object&&typeof this.object._finalize=="function"&&this.object._finalize(!0)},parseVertexIndex:function(t,e){const n=parseInt(t,10);return(n>=0?n-1:n+e/3)*3},parseNormalIndex:function(t,e){const n=parseInt(t,10);return(n>=0?n-1:n+e/3)*3},parseUVIndex:function(t,e){const n=parseInt(t,10);return(n>=0?n-1:n+e/2)*2},addVertex:function(t,e,n){const s=this.vertices,r=this.object.geometry.vertices;r.push(s[t+0],s[t+1],s[t+2]),r.push(s[e+0],s[e+1],s[e+2]),r.push(s[n+0],s[n+1],s[n+2])},addVertexPoint:function(t){const e=this.vertices;this.object.geometry.vertices.push(e[t+0],e[t+1],e[t+2])},addVertexLine:function(t){const e=this.vertices;this.object.geometry.vertices.push(e[t+0],e[t+1],e[t+2])},addNormal:function(t,e,n){const s=this.normals,r=this.object.geometry.normals;r.push(s[t+0],s[t+1],s[t+2]),r.push(s[e+0],s[e+1],s[e+2]),r.push(s[n+0],s[n+1],s[n+2])},addFaceNormal:function(t,e,n){const s=this.vertices,r=this.object.geometry.normals;gh.fromArray(s,t),So.fromArray(s,e),_h.fromArray(s,n),Mn.subVectors(_h,So),vh.subVectors(gh,So),Mn.cross(vh),Mn.normalize(),r.push(Mn.x,Mn.y,Mn.z),r.push(Mn.x,Mn.y,Mn.z),r.push(Mn.x,Mn.y,Mn.z)},addColor:function(t,e,n){const s=this.colors,r=this.object.geometry.colors;s[t]!==void 0&&r.push(s[t+0],s[t+1],s[t+2]),s[e]!==void 0&&r.push(s[e+0],s[e+1],s[e+2]),s[n]!==void 0&&r.push(s[n+0],s[n+1],s[n+2])},addUV:function(t,e,n){const s=this.uvs,r=this.object.geometry.uvs;r.push(s[t+0],s[t+1]),r.push(s[e+0],s[e+1]),r.push(s[n+0],s[n+1])},addDefaultUV:function(){const t=this.object.geometry.uvs;t.push(0,0),t.push(0,0),t.push(0,0)},addUVLine:function(t){const e=this.uvs;this.object.geometry.uvs.push(e[t+0],e[t+1])},addFace:function(t,e,n,s,r,a,o,c,l){const h=this.vertices.length;let u=this.parseVertexIndex(t,h),d=this.parseVertexIndex(e,h),p=this.parseVertexIndex(n,h);if(this.addVertex(u,d,p),this.addColor(u,d,p),o!==void 0&&o!==""){const g=this.normals.length;u=this.parseNormalIndex(o,g),d=this.parseNormalIndex(c,g),p=this.parseNormalIndex(l,g),this.addNormal(u,d,p)}else this.addFaceNormal(u,d,p);if(s!==void 0&&s!==""){const g=this.uvs.length;u=this.parseUVIndex(s,g),d=this.parseUVIndex(r,g),p=this.parseUVIndex(a,g),this.addUV(u,d,p),this.object.geometry.hasUVIndices=!0}else this.addDefaultUV()},addPointGeometry:function(t){this.object.geometry.type="Points";const e=this.vertices.length;for(let n=0,s=t.length;n<s;n++){const r=this.parseVertexIndex(t[n],e);this.addVertexPoint(r),this.addColor(r)}},addLineGeometry:function(t,e){this.object.geometry.type="Line";const n=this.vertices.length,s=this.uvs.length;for(let r=0,a=t.length;r<a;r++)this.addVertexLine(this.parseVertexIndex(t[r],n));for(let r=0,a=e.length;r<a;r++)this.addUVLine(this.parseUVIndex(e[r],s))}};return i.startObject("",!1),i}class bo extends ri{constructor(t){super(t),this.materials=null}load(t,e,n,s){const r=this,a=new Ca(this.manager);a.setPath(this.path),a.setRequestHeader(this.requestHeader),a.setWithCredentials(this.withCredentials),a.load(t,function(o){try{e(r.parse(o))}catch(c){s?s(c):console.error(c),r.manager.itemError(t)}},n,s)}setMaterials(t){return this.materials=t,this}parse(t){const e=new cv;t.indexOf(`\r
`)!==-1&&(t=t.replace(/\r\n/g,`
`)),t.indexOf(`\\
`)!==-1&&(t=t.replace(/\\\n/g,""));const n=t.split(`
`);let s=[];for(let o=0,c=n.length;o<c;o++){const l=n[o].trimStart();if(l.length===0)continue;const h=l.charAt(0);if(h!=="#")if(h==="v"){const u=l.split(mh);switch(u[0]){case"v":e.vertices.push(parseFloat(u[1]),parseFloat(u[2]),parseFloat(u[3])),u.length>=7?(Qr.setRGB(parseFloat(u[4]),parseFloat(u[5]),parseFloat(u[6]),Pe),e.colors.push(Qr.r,Qr.g,Qr.b)):e.colors.push(void 0,void 0,void 0);break;case"vn":e.normals.push(parseFloat(u[1]),parseFloat(u[2]),parseFloat(u[3]));break;case"vt":e.uvs.push(parseFloat(u[1]),parseFloat(u[2]));break}}else if(h==="f"){const d=l.slice(1).trim().split(mh),p=[];for(let _=0,m=d.length;_<m;_++){const f=d[_];if(f.length>0){const R=f.split("/");p.push(R)}}const g=p[0];for(let _=1,m=p.length-1;_<m;_++){const f=p[_],R=p[_+1];e.addFace(g[0],f[0],R[0],g[1],f[1],R[1],g[2],f[2],R[2])}}else if(h==="l"){const u=l.substring(1).trim().split(" ");let d=[];const p=[];if(l.indexOf("/")===-1)d=u;else for(let g=0,_=u.length;g<_;g++){const m=u[g].split("/");m[0]!==""&&d.push(m[0]),m[1]!==""&&p.push(m[1])}e.addLineGeometry(d,p)}else if(h==="p"){const d=l.slice(1).trim().split(" ");e.addPointGeometry(d)}else if((s=rv.exec(l))!==null){const u=(" "+s[0].slice(1).trim()).slice(1);e.startObject(u)}else if(ov.test(l))e.object.startMaterial(l.substring(7).trim(),e.materialLibraries);else if(av.test(l))e.materialLibraries.push(l.substring(7).trim());else if(lv.test(l))console.warn('THREE.OBJLoader: Rendering identifier "usemap" not supported. Textures must be defined in MTL files.');else if(h==="s"){if(s=l.split(" "),s.length>1){const d=s[1].trim().toLowerCase();e.object.smooth=d!=="0"&&d!=="off"}else e.object.smooth=!0;const u=e.object.currentMaterial();u&&(u.smooth=e.object.smooth)}else{if(l==="\0")continue;console.warn('THREE.OBJLoader: Unexpected line: "'+l+'"')}}e.finalize();const r=new je;if(r.materialLibraries=[].concat(e.materialLibraries),!(e.objects.length===1&&e.objects[0].geometry.vertices.length===0)===!0)for(let o=0,c=e.objects.length;o<c;o++){const l=e.objects[o],h=l.geometry,u=l.materials,d=h.type==="Line",p=h.type==="Points";let g=!1;if(h.vertices.length===0)continue;const _=new Ie;_.setAttribute("position",new he(h.vertices,3)),h.normals.length>0&&_.setAttribute("normal",new he(h.normals,3)),h.colors.length>0&&(g=!0,_.setAttribute("color",new he(h.colors,3))),h.hasUVIndices===!0&&_.setAttribute("uv",new he(h.uvs,2));const m=[];for(let R=0,C=u.length;R<C;R++){const x=u[R],N=x.name+"_"+x.smooth+"_"+g;let P=e.materials[N];if(this.materials!==null){if(P=this.materials.create(x.name),d&&P&&!(P instanceof zn)){const L=new zn;In.prototype.copy.call(L,P),L.color.copy(P.color),P=L}else if(p&&P&&!(P instanceof Ks)){const L=new Ks({size:10,sizeAttenuation:!1});In.prototype.copy.call(L,P),L.color.copy(P.color),L.map=P.map,P=L}}P===void 0&&(d?P=new zn:p?P=new Ks({size:1,sizeAttenuation:!1}):P=new vs,P.name=x.name,P.flatShading=!x.smooth,P.vertexColors=g,e.materials[N]=P),m.push(P)}let f;if(m.length>1){for(let R=0,C=u.length;R<C;R++){const x=u[R];_.addGroup(x.groupStart,x.groupCount,R)}d?f=new ga(_,m):p?f=new po(_,m):f=new Ae(_,m)}else d?f=new ga(_,m[0]):p?f=new po(_,m[0]):f=new Ae(_,m[0]);f.name=l.name,r.add(f)}else if(e.vertices.length>0){const o=new Ks({size:1,sizeAttenuation:!1}),c=new Ie;c.setAttribute("position",new he(e.vertices,3)),e.colors.length>0&&e.colors[0]!==void 0&&(c.setAttribute("color",new he(e.colors,3)),o.vertexColors=!0);const l=new po(c,o);r.add(l)}return r}}class hv extends ri{constructor(t){super(t)}load(t,e,n,s){const r=this,a=new Ca(this.manager);a.setPath(this.path),a.setResponseType("arraybuffer"),a.setRequestHeader(this.requestHeader),a.setWithCredentials(this.withCredentials),a.load(t,function(o){try{e(r.parse(o))}catch(c){s?s(c):console.error(c),r.manager.itemError(t)}},n,s)}parse(t){function e(l){const h=new DataView(l),u=32/8*3+32/8*3*3+16/8,d=h.getUint32(80,!0);if(80+32/8+d*u===h.byteLength)return!0;const g=[115,111,108,105,100];for(let _=0;_<5;_++)if(n(g,h,_))return!1;return!0}function n(l,h,u){for(let d=0,p=l.length;d<p;d++)if(l[d]!==h.getUint8(u+d))return!1;return!0}function s(l){const h=new DataView(l),u=h.getUint32(80,!0);let d,p,g,_=!1,m,f,R,C,x;for(let U=0;U<70;U++)h.getUint32(U,!1)==1129270351&&h.getUint8(U+4)==82&&h.getUint8(U+5)==61&&(_=!0,m=new Float32Array(u*3*3),f=h.getUint8(U+6)/255,R=h.getUint8(U+7)/255,C=h.getUint8(U+8)/255,x=h.getUint8(U+9)/255);const N=84,P=50,L=new Ie,I=new Float32Array(u*3*3),M=new Float32Array(u*3*3),S=new $t;for(let U=0;U<u;U++){const k=N+U*P,H=h.getFloat32(k,!0),J=h.getFloat32(k+4,!0),Q=h.getFloat32(k+8,!0);if(_){const O=h.getUint16(k+48,!0);(O&32768)===0?(d=(O&31)/31,p=(O>>5&31)/31,g=(O>>10&31)/31):(d=f,p=R,g=C)}for(let O=1;O<=3;O++){const tt=k+O*12,W=U*3*3+(O-1)*3;I[W]=h.getFloat32(tt,!0),I[W+1]=h.getFloat32(tt+4,!0),I[W+2]=h.getFloat32(tt+8,!0),M[W]=H,M[W+1]=J,M[W+2]=Q,_&&(S.setRGB(d,p,g,Pe),m[W]=S.r,m[W+1]=S.g,m[W+2]=S.b)}}return L.setAttribute("position",new gn(I,3)),L.setAttribute("normal",new gn(M,3)),_&&(L.setAttribute("color",new gn(m,3)),L.hasColors=!0,L.alpha=x),L}function r(l){const h=new Ie,u=/solid([\s\S]*?)endsolid/g,d=/facet([\s\S]*?)endfacet/g,p=/solid\s(.+)/;let g=0;const _=/[\s]+([+-]?(?:\d*)(?:\.\d*)?(?:[eE][+-]?\d+)?)/.source,m=new RegExp("vertex"+_+_+_,"g"),f=new RegExp("normal"+_+_+_,"g"),R=[],C=[],x=[],N=new F;let P,L=0,I=0,M=0;for(;(P=u.exec(l))!==null;){I=M;const S=P[0],U=(P=p.exec(S))!==null?P[1]:"";for(x.push(U);(P=d.exec(S))!==null;){let J=0,Q=0;const O=P[0];for(;(P=f.exec(O))!==null;)N.x=parseFloat(P[1]),N.y=parseFloat(P[2]),N.z=parseFloat(P[3]),Q++;for(;(P=m.exec(O))!==null;)R.push(parseFloat(P[1]),parseFloat(P[2]),parseFloat(P[3])),C.push(N.x,N.y,N.z),J++,M++;Q!==1&&console.error("THREE.STLLoader: Something isn't right with the normal of face number "+g),J!==3&&console.error("THREE.STLLoader: Something isn't right with the vertices of face number "+g),g++}const k=I,H=M-I;h.userData.groupNames=x,h.addGroup(k,H,L),L++}return h.setAttribute("position",new he(R,3)),h.setAttribute("normal",new he(C,3)),h}function a(l){return typeof l!="string"?new TextDecoder().decode(l):l}function o(l){if(typeof l=="string"){const h=new Uint8Array(l.length);for(let u=0;u<l.length;u++)h[u]=l.charCodeAt(u)&255;return h.buffer||h}else return l}const c=o(t);return e(c)?s(c):r(a(t))}}class xh extends U0{constructor(t){super(t)}parse(t){function e(O){switch(O.image_type){case d:case _:if(O.colormap_length>256||O.colormap_size!==24||O.colormap_type!==1)throw new Error("THREE.TGALoader: Invalid type colormap data for indexed type.");break;case p:case g:case m:case f:if(O.colormap_type)throw new Error("THREE.TGALoader: Invalid type colormap data for colormap type.");break;case u:throw new Error("THREE.TGALoader: No data.");default:throw new Error("THREE.TGALoader: Invalid type "+O.image_type)}if(O.width<=0||O.height<=0)throw new Error("THREE.TGALoader: Invalid image size.");if(O.pixel_size!==8&&O.pixel_size!==16&&O.pixel_size!==24&&O.pixel_size!==32)throw new Error("THREE.TGALoader: Invalid pixel size "+O.pixel_size)}function n(O,tt,W,ot,ft){let pt,Ct;const Et=W.pixel_size>>3,X=W.width*W.height*Et;if(tt&&(Ct=ft.subarray(ot,ot+=W.colormap_length*(W.colormap_size>>3))),O){pt=new Uint8Array(X);let K,et,nt,_t=0;const zt=new Uint8Array(Et);for(;_t<X;)if(K=ft[ot++],et=(K&127)+1,K&128){for(nt=0;nt<Et;++nt)zt[nt]=ft[ot++];for(nt=0;nt<et;++nt)pt.set(zt,_t+nt*Et);_t+=Et*et}else{for(et*=Et,nt=0;nt<et;++nt)pt[_t+nt]=ft[ot++];_t+=et}}else pt=ft.subarray(ot,ot+=tt?W.width*W.height:X);return{pixel_data:pt,palettes:Ct}}function s(O,tt,W,ot,ft,pt,Ct,Et,X){const K=X;let et,nt=0,_t,zt;const Zt=S.width;for(zt=tt;zt!==ot;zt+=W)for(_t=ft;_t!==Ct;_t+=pt,nt++)et=Et[nt],O[(_t+Zt*zt)*4+3]=255,O[(_t+Zt*zt)*4+2]=K[et*3+0],O[(_t+Zt*zt)*4+1]=K[et*3+1],O[(_t+Zt*zt)*4+0]=K[et*3+2];return O}function r(O,tt,W,ot,ft,pt,Ct,Et){let X,K=0,et,nt;const _t=S.width;for(nt=tt;nt!==ot;nt+=W)for(et=ft;et!==Ct;et+=pt,K+=2)X=Et[K+0]+(Et[K+1]<<8),O[(et+_t*nt)*4+0]=(X&31744)>>7,O[(et+_t*nt)*4+1]=(X&992)>>2,O[(et+_t*nt)*4+2]=(X&31)<<3,O[(et+_t*nt)*4+3]=X&32768?0:255;return O}function a(O,tt,W,ot,ft,pt,Ct,Et){let X=0,K,et;const nt=S.width;for(et=tt;et!==ot;et+=W)for(K=ft;K!==Ct;K+=pt,X+=3)O[(K+nt*et)*4+3]=255,O[(K+nt*et)*4+2]=Et[X+0],O[(K+nt*et)*4+1]=Et[X+1],O[(K+nt*et)*4+0]=Et[X+2];return O}function o(O,tt,W,ot,ft,pt,Ct,Et){let X=0,K,et;const nt=S.width;for(et=tt;et!==ot;et+=W)for(K=ft;K!==Ct;K+=pt,X+=4)O[(K+nt*et)*4+2]=Et[X+0],O[(K+nt*et)*4+1]=Et[X+1],O[(K+nt*et)*4+0]=Et[X+2],O[(K+nt*et)*4+3]=Et[X+3];return O}function c(O,tt,W,ot,ft,pt,Ct,Et){let X,K=0,et,nt;const _t=S.width;for(nt=tt;nt!==ot;nt+=W)for(et=ft;et!==Ct;et+=pt,K++)X=Et[K],O[(et+_t*nt)*4+0]=X,O[(et+_t*nt)*4+1]=X,O[(et+_t*nt)*4+2]=X,O[(et+_t*nt)*4+3]=255;return O}function l(O,tt,W,ot,ft,pt,Ct,Et){let X=0,K,et;const nt=S.width;for(et=tt;et!==ot;et+=W)for(K=ft;K!==Ct;K+=pt,X+=2)O[(K+nt*et)*4+0]=Et[X+0],O[(K+nt*et)*4+1]=Et[X+0],O[(K+nt*et)*4+2]=Et[X+0],O[(K+nt*et)*4+3]=Et[X+1];return O}function h(O,tt,W,ot,ft){let pt,Ct,Et,X,K,et;switch((S.flags&R)>>C){default:case P:pt=0,Et=1,K=tt,Ct=0,X=1,et=W;break;case x:pt=0,Et=1,K=tt,Ct=W-1,X=-1,et=-1;break;case L:pt=tt-1,Et=-1,K=-1,Ct=0,X=1,et=W;break;case N:pt=tt-1,Et=-1,K=-1,Ct=W-1,X=-1,et=-1;break}if(H)switch(S.pixel_size){case 8:c(O,Ct,X,et,pt,Et,K,ot);break;case 16:l(O,Ct,X,et,pt,Et,K,ot);break;default:throw new Error("THREE.TGALoader: Format not supported.")}else switch(S.pixel_size){case 8:s(O,Ct,X,et,pt,Et,K,ot,ft);break;case 16:r(O,Ct,X,et,pt,Et,K,ot);break;case 24:a(O,Ct,X,et,pt,Et,K,ot);break;case 32:o(O,Ct,X,et,pt,Et,K,ot);break;default:throw new Error("THREE.TGALoader: Format not supported.")}return O}const u=0,d=1,p=2,g=3,_=9,m=10,f=11,R=48,C=4,x=0,N=1,P=2,L=3;if(t.length<19)throw new Error("THREE.TGALoader: Not enough data to contain header.");let I=0;const M=new Uint8Array(t),S={id_length:M[I++],colormap_type:M[I++],image_type:M[I++],colormap_index:M[I++]|M[I++]<<8,colormap_length:M[I++]|M[I++]<<8,colormap_size:M[I++],origin:[M[I++]|M[I++]<<8,M[I++]|M[I++]<<8],width:M[I++]|M[I++]<<8,height:M[I++]|M[I++]<<8,pixel_size:M[I++],flags:M[I++]};if(e(S),S.id_length+I>t.length)throw new Error("THREE.TGALoader: No data.");I+=S.id_length;let U=!1,k=!1,H=!1;switch(S.image_type){case _:U=!0,k=!0;break;case d:k=!0;break;case m:U=!0;break;case p:break;case f:U=!0,H=!0;break;case g:H=!0;break}const J=new Uint8Array(S.width*S.height*4),Q=n(U,k,S,I,M);return h(J,S.width,S.height,Q.pixel_data,Q.palettes),{data:J,width:S.width,height:S.height,flipY:!0,generateMipmaps:!0,minFilter:ti}}}class uv extends ri{load(t,e,n,s){const r=this,a=r.path===""?mu.extractUrlBase(t):r.path,o=new Ca(r.manager);o.setPath(r.path),o.setRequestHeader(r.requestHeader),o.setWithCredentials(r.withCredentials),o.load(t,function(c){try{e(r.parse(c,a))}catch(l){s?s(l):console.error(l),r.manager.itemError(t)}},n,s)}parse(t,e){function n(y,v){const T=[],E=y.childNodes;for(let A=0,$=E.length;A<$;A++){const it=E[A];it.nodeName===v&&T.push(it)}return T}function s(y){if(y.length===0)return[];const v=y.trim().split(/\s+/),T=new Array(v.length);for(let E=0,A=v.length;E<A;E++)T[E]=v[E];return T}function r(y){if(y.length===0)return[];const v=y.trim().split(/\s+/),T=new Array(v.length);for(let E=0,A=v.length;E<A;E++)T[E]=parseFloat(v[E]);return T}function a(y){if(y.length===0)return[];const v=y.trim().split(/\s+/),T=new Array(v.length);for(let E=0,A=v.length;E<A;E++)T[E]=parseInt(v[E]);return T}function o(y){return y.substring(1)}function c(){return"three_default_"+Ou++}function l(y){return Object.keys(y).length===0}function h(y){return{unit:u(n(y,"unit")[0]),upAxis:d(n(y,"up_axis")[0])}}function u(y){return y!==void 0&&y.hasAttribute("meter")===!0?parseFloat(y.getAttribute("meter")):1}function d(y){return y!==void 0?y.textContent:"Y_UP"}function p(y,v,T,E){const A=n(y,v)[0];if(A!==void 0){const $=n(A,T);for(let it=0;it<$.length;it++)E($[it])}}function g(y,v){for(const T in y){const E=y[T];E.build=v(y[T])}}function _(y,v){return y.build!==void 0||(y.build=v(y)),y.build}function m(y){const v={sources:{},samplers:{},channels:{}};let T=!1;for(let E=0,A=y.childNodes.length;E<A;E++){const $=y.childNodes[E];if($.nodeType!==1)continue;let it;switch($.nodeName){case"source":it=$.getAttribute("id"),v.sources[it]=Z($);break;case"sampler":it=$.getAttribute("id"),v.samplers[it]=f($);break;case"channel":it=$.getAttribute("target"),v.channels[it]=R($);break;case"animation":m($),T=!0;break;default:console.log($)}}T===!1&&(qt.animations[y.getAttribute("id")||Mi.generateUUID()]=v)}function f(y){const v={inputs:{}};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"input":const $=o(A.getAttribute("source")),it=A.getAttribute("semantic");v.inputs[it]=$;break}}return v}function R(y){const v={};let E=y.getAttribute("target").split("/");const A=E.shift();let $=E.shift();const it=$.indexOf("(")!==-1,wt=$.indexOf(".")!==-1;if(wt)E=$.split("."),$=E.shift(),v.member=E.shift();else if(it){const mt=$.split("(");$=mt.shift();for(let St=0;St<mt.length;St++)mt[St]=parseInt(mt[St].replace(/\)/,""));v.indices=mt}return v.id=A,v.sid=$,v.arraySyntax=it,v.memberSyntax=wt,v.sampler=o(y.getAttribute("source")),v}function C(y){const v=[],T=y.channels,E=y.samplers,A=y.sources;for(const $ in T)if(T.hasOwnProperty($)){const it=T[$],wt=E[it.sampler],mt=wt.inputs.INPUT,St=wt.inputs.OUTPUT,Bt=A[mt],ht=A[St],kt=N(it,Bt,ht);S(kt,v)}return v}function x(y){return _(qt.animations[y],C)}function N(y,v,T){const E=qt.nodes[y.id],A=Ue(E.id),$=E.transforms[y.sid],it=E.matrix.clone().transpose();let wt,mt,St,Bt,ht,kt;const Nt={};switch($){case"matrix":for(St=0,Bt=v.array.length;St<Bt;St++)if(wt=v.array[St],mt=St*T.stride,Nt[wt]===void 0&&(Nt[wt]={}),y.arraySyntax===!0){const De=T.array[mt],Se=y.indices[0]+4*y.indices[1];Nt[wt][Se]=De}else for(ht=0,kt=T.stride;ht<kt;ht++)Nt[wt][ht]=T.array[mt+ht];break;case"translate":console.warn('THREE.ColladaLoader: Animation transform type "%s" not yet implemented.',$);break;case"rotate":console.warn('THREE.ColladaLoader: Animation transform type "%s" not yet implemented.',$);break;case"scale":console.warn('THREE.ColladaLoader: Animation transform type "%s" not yet implemented.',$);break}const Gt=P(Nt,it);return{name:A.uuid,keyframes:Gt}}function P(y,v){const T=[];for(const A in y)T.push({time:parseFloat(A),value:y[A]});T.sort(E);for(let A=0;A<16;A++)U(T,A,v.elements[A]);return T;function E(A,$){return A.time-$.time}}const L=new F,I=new F,M=new Dn;function S(y,v){const T=y.keyframes,E=y.name,A=[],$=[],it=[],wt=[];for(let mt=0,St=T.length;mt<St;mt++){const Bt=T[mt],ht=Bt.time,kt=Bt.value;B.fromArray(kt).transpose(),B.decompose(L,M,I),A.push(ht),$.push(L.x,L.y,L.z),it.push(M.x,M.y,M.z,M.w),wt.push(I.x,I.y,I.z)}return $.length>0&&v.push(new As(E+".position",A,$)),it.length>0&&v.push(new fr(E+".quaternion",A,it)),wt.length>0&&v.push(new As(E+".scale",A,wt)),v}function U(y,v,T){let E,A=!0,$,it;for($=0,it=y.length;$<it;$++)E=y[$],E.value[v]===void 0?E.value[v]=null:A=!1;if(A===!0)for($=0,it=y.length;$<it;$++)E=y[$],E.value[v]=T;else k(y,v)}function k(y,v){let T,E;for(let A=0,$=y.length;A<$;A++){const it=y[A];if(it.value[v]===null){if(T=H(y,A,v),E=J(y,A,v),T===null){it.value[v]=E.value[v];continue}if(E===null){it.value[v]=T.value[v];continue}Q(it,T,E,v)}}}function H(y,v,T){for(;v>=0;){const E=y[v];if(E.value[T]!==null)return E;v--}return null}function J(y,v,T){for(;v<y.length;){const E=y[v];if(E.value[T]!==null)return E;v++}return null}function Q(y,v,T,E){if(T.time-v.time===0){y.value[E]=v.value[E];return}y.value[E]=(y.time-v.time)*(T.value[E]-v.value[E])/(T.time-v.time)+v.value[E]}function O(y){const v={name:y.getAttribute("id")||"default",start:parseFloat(y.getAttribute("start")||0),end:parseFloat(y.getAttribute("end")||0),animations:[]};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"instance_animation":v.animations.push(o(A.getAttribute("url")));break}}qt.clips[y.getAttribute("id")]=v}function tt(y){const v=[],T=y.name,E=y.end-y.start||-1,A=y.animations;for(let $=0,it=A.length;$<it;$++){const wt=x(A[$]);for(let mt=0,St=wt.length;mt<St;mt++)v.push(wt[mt])}return new oh(T,E,v)}function W(y){return _(qt.clips[y],tt)}function ot(y){const v={};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"skin":v.id=o(A.getAttribute("source")),v.skin=ft(A);break;case"morph":v.id=o(A.getAttribute("source")),console.warn("THREE.ColladaLoader: Morph target animation not supported yet.");break}}qt.controllers[y.getAttribute("id")]=v}function ft(y){const v={sources:{}};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"bind_shape_matrix":v.bindShapeMatrix=r(A.textContent);break;case"source":const $=A.getAttribute("id");v.sources[$]=Z(A);break;case"joints":v.joints=pt(A);break;case"vertex_weights":v.vertexWeights=Ct(A);break}}return v}function pt(y){const v={inputs:{}};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"input":const $=A.getAttribute("semantic"),it=o(A.getAttribute("source"));v.inputs[$]=it;break}}return v}function Ct(y){const v={inputs:{}};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"input":const $=A.getAttribute("semantic"),it=o(A.getAttribute("source")),wt=parseInt(A.getAttribute("offset"));v.inputs[$]={id:it,offset:wt};break;case"vcount":v.vcount=a(A.textContent);break;case"v":v.v=a(A.textContent);break}}return v}function Et(y){const v={id:y.id},T=qt.geometries[v.id];return y.skin!==void 0&&(v.skin=X(y.skin),T.sources.skinIndices=v.skin.indices,T.sources.skinWeights=v.skin.weights),v}function X(y){const T={joints:[],indices:{array:[],stride:4},weights:{array:[],stride:4}},E=y.sources,A=y.vertexWeights,$=A.vcount,it=A.v,wt=A.inputs.JOINT.offset,mt=A.inputs.WEIGHT.offset,St=y.sources[y.joints.inputs.JOINT],Bt=y.sources[y.joints.inputs.INV_BIND_MATRIX],ht=E[A.inputs.WEIGHT.id].array;let kt=0,Nt,Gt,Vt;for(Nt=0,Vt=$.length;Nt<Vt;Nt++){const Se=$[Nt],ge=[];for(Gt=0;Gt<Se;Gt++){const _e=it[kt+wt],qn=it[kt+mt],an=ht[qn];ge.push({index:_e,weight:an}),kt+=2}for(ge.sort(De),Gt=0;Gt<4;Gt++){const _e=ge[Gt];_e!==void 0?(T.indices.array.push(_e.index),T.weights.array.push(_e.weight)):(T.indices.array.push(0),T.weights.array.push(0))}}for(y.bindShapeMatrix?T.bindMatrix=new Kt().fromArray(y.bindShapeMatrix).transpose():T.bindMatrix=new Kt().identity(),Nt=0,Vt=St.array.length;Nt<Vt;Nt++){const Se=St.array[Nt],ge=new Kt().fromArray(Bt.array,Nt*Bt.stride).transpose();T.joints.push({name:Se,boneInverse:ge})}return T;function De(Se,ge){return ge.weight-Se.weight}}function K(y){return _(qt.controllers[y],Et)}function et(y){const v={init_from:n(y,"init_from")[0].textContent};qt.images[y.getAttribute("id")]=v}function nt(y){return y.build!==void 0?y.build:y.init_from}function _t(y){const v=qt.images[y];return v!==void 0?_(v,nt):(console.warn("THREE.ColladaLoader: Couldn't find image with ID:",y),null)}function zt(y){const v={};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"profile_COMMON":v.profile=Zt(A);break}}qt.effects[y.getAttribute("id")]=v}function Zt(y){const v={surfaces:{},samplers:{}};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"newparam":Ce(A,v);break;case"technique":v.technique=G(A);break;case"extra":v.extra=Ht(A);break}}return v}function Ce(y,v){const T=y.getAttribute("sid");for(let E=0,A=y.childNodes.length;E<A;E++){const $=y.childNodes[E];if($.nodeType===1)switch($.nodeName){case"surface":v.surfaces[T]=ue($);break;case"sampler2D":v.samplers[T]=Ne($);break}}}function ue(y){const v={};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"init_from":v.init_from=A.textContent;break}}return v}function Ne(y){const v={};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"source":v.source=A.textContent;break}}return v}function G(y){const v={};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"constant":case"lambert":case"blinn":case"phong":v.type=A.nodeName,v.parameters=nn(A);break;case"extra":v.extra=Ht(A);break}}return v}function nn(y){const v={};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"emission":case"diffuse":case"specular":case"bump":case"ambient":case"shininess":case"transparency":v[A.nodeName]=re(A);break;case"transparent":v[A.nodeName]={opaque:A.hasAttribute("opaque")?A.getAttribute("opaque"):"A_ONE",data:re(A)};break}}return v}function re(y){const v={};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"color":v[A.nodeName]=r(A.textContent);break;case"float":v[A.nodeName]=parseFloat(A.textContent);break;case"texture":v[A.nodeName]={id:A.getAttribute("texture"),extra:ae(A)};break}}return v}function ae(y){const v={technique:{}};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"extra":Wt(A,v);break}}return v}function Wt(y,v){for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"technique":we(A,v);break}}}function we(y,v){for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"repeatU":case"repeatV":case"offsetU":case"offsetV":v.technique[A.nodeName]=parseFloat(A.textContent);break;case"wrapU":case"wrapV":A.textContent.toUpperCase()==="TRUE"?v.technique[A.nodeName]=1:A.textContent.toUpperCase()==="FALSE"?v.technique[A.nodeName]=0:v.technique[A.nodeName]=parseInt(A.textContent);break;case"bump":v[A.nodeName]=b(A);break}}}function Ht(y){const v={};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"technique":v.technique=D(A);break}}return v}function D(y){const v={};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"double_sided":v[A.nodeName]=parseInt(A.textContent);break;case"bump":v[A.nodeName]=b(A);break}}return v}function b(y){const v={};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"texture":v[A.nodeName]={id:A.getAttribute("texture"),texcoord:A.getAttribute("texcoord"),extra:ae(A)};break}}return v}function q(y){return y}function rt(y){return _(qt.effects[y],q)}function lt(y){const v={name:y.getAttribute("name")};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"instance_effect":v.url=o(A.getAttribute("url"));break}}qt.materials[y.getAttribute("id")]=v}function st(y){let v,T=y.slice((y.lastIndexOf(".")-1>>>0)+2);switch(T=T.toLowerCase(),T){case"tga":v=xn;break;default:v=wn}return v}function Ft(y){const v=rt(y.url),T=v.profile.technique;let E;switch(T.type){case"phong":case"blinn":E=new vs;break;case"lambert":E=new E0;break;default:E=new Un;break}E.name=y.name||"";function A(mt,St=null){const Bt=v.profile.samplers[mt.id];let ht=null;if(Bt!==void 0){const kt=v.profile.surfaces[Bt.source];ht=_t(kt.init_from)}else console.warn("THREE.ColladaLoader: Undefined sampler. Access image directly (see #12530)."),ht=_t(mt.id);if(ht!==null){const kt=st(ht);if(kt!==void 0){const Nt=kt.load(ht),Gt=mt.extra;if(Gt!==void 0&&Gt.technique!==void 0&&l(Gt.technique)===!1){const Vt=Gt.technique;Nt.wrapS=Vt.wrapU?Hi:Nn,Nt.wrapT=Vt.wrapV?Hi:Nn,Nt.offset.set(Vt.offsetU||0,Vt.offsetV||0),Nt.repeat.set(Vt.repeatU||1,Vt.repeatV||1)}else Nt.wrapS=Hi,Nt.wrapT=Hi;return St!==null&&(Nt.colorSpace=St),Nt}else return console.warn("THREE.ColladaLoader: Loader for texture %s not found.",ht),null}else return console.warn("THREE.ColladaLoader: Couldn't create texture with ID:",mt.id),null}const $=T.parameters;for(const mt in $){const St=$[mt];switch(mt){case"diffuse":St.color&&E.color.fromArray(St.color),St.texture&&(E.map=A(St.texture,Pe));break;case"specular":St.color&&E.specular&&E.specular.fromArray(St.color),St.texture&&(E.specularMap=A(St.texture));break;case"bump":St.texture&&(E.normalMap=A(St.texture));break;case"ambient":St.texture&&(E.lightMap=A(St.texture,Pe));break;case"shininess":St.float&&E.shininess&&(E.shininess=St.float);break;case"emission":St.color&&E.emissive&&E.emissive.fromArray(St.color),St.texture&&(E.emissiveMap=A(St.texture,Pe));break}}le.toWorkingColorSpace(E.color,Pe),E.specular&&le.toWorkingColorSpace(E.specular,Pe),E.emissive&&le.toWorkingColorSpace(E.emissive,Pe);let it=$.transparent,wt=$.transparency;if(wt===void 0&&it&&(wt={float:1}),it===void 0&&wt&&(it={opaque:"A_ONE",data:{color:[1,1,1,1]}}),it&&wt)if(it.data.texture)E.transparent=!0;else{const mt=it.data.color;switch(it.opaque){case"A_ONE":E.opacity=mt[3]*wt.float;break;case"RGB_ZERO":E.opacity=1-mt[0]*wt.float;break;case"A_ZERO":E.opacity=1-mt[3]*wt.float;break;case"RGB_ONE":E.opacity=mt[0]*wt.float;break;default:console.warn('THREE.ColladaLoader: Invalid opaque type "%s" of transparent tag.',it.opaque)}E.opacity<1&&(E.transparent=!0)}if(T.extra!==void 0&&T.extra.technique!==void 0){const mt=T.extra.technique;for(const St in mt){const Bt=mt[St];switch(St){case"double_sided":E.side=Bt===1?Pn:ii;break;case"bump":E.normalMap=A(Bt.texture),E.normalScale=new Lt(1,1);break}}}return E}function yt(y){return _(qt.materials[y],Ft)}function At(y){const v={name:y.getAttribute("name")};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"optics":v.optics=de(A);break}}qt.cameras[y.getAttribute("id")]=v}function de(y){for(let v=0;v<y.childNodes.length;v++){const T=y.childNodes[v];switch(T.nodeName){case"technique_common":return dt(T)}}return{}}function dt(y){const v={};for(let T=0;T<y.childNodes.length;T++){const E=y.childNodes[T];switch(E.nodeName){case"perspective":case"orthographic":v.technique=E.nodeName,v.parameters=Rt(E);break}}return v}function Rt(y){const v={};for(let T=0;T<y.childNodes.length;T++){const E=y.childNodes[T];switch(E.nodeName){case"xfov":case"yfov":case"xmag":case"ymag":case"znear":case"zfar":case"aspect_ratio":v[E.nodeName]=parseFloat(E.textContent);break}}return v}function Xt(y){let v;switch(y.optics.technique){case"perspective":v=new Je(y.optics.parameters.yfov,y.optics.parameters.aspect_ratio,y.optics.parameters.znear,y.optics.parameters.zfar);break;case"orthographic":let T=y.optics.parameters.ymag,E=y.optics.parameters.xmag;const A=y.optics.parameters.aspect_ratio;E=E===void 0?T*A:E,T=T===void 0?E/A:T,E*=.5,T*=.5,v=new El(-E,E,T,-T,y.optics.parameters.znear,y.optics.parameters.zfar);break;default:v=new Je;break}return v.name=y.name||"",v}function jt(y){const v=qt.cameras[y];return v!==void 0?_(v,Xt):(console.warn("THREE.ColladaLoader: Couldn't find camera with ID:",y),null)}function Pt(y){let v={};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"technique_common":v=ce(A);break}}qt.lights[y.getAttribute("id")]=v}function ce(y){const v={};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"directional":case"point":case"spot":case"ambient":v.technique=A.nodeName,v.parameters=ee(A)}}return v}function ee(y){const v={};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"color":const $=r(A.textContent);v.color=new $t().fromArray($),le.toWorkingColorSpace(v.color,Pe);break;case"falloff_angle":v.falloffAngle=parseFloat(A.textContent);break;case"quadratic_attenuation":const it=parseFloat(A.textContent);v.distance=it?Math.sqrt(1/it):0;break}}return v}function ye(y){let v;switch(y.technique){case"directional":v=new pu;break;case"point":v=new z0;break;case"spot":v=new k0;break;case"ambient":v=new H0;break}return y.parameters.color&&v.color.copy(y.parameters.color),y.parameters.distance&&(v.distance=y.parameters.distance),v}function z(y){const v=qt.lights[y];return v!==void 0?_(v,ye):(console.warn("THREE.ColladaLoader: Couldn't find light with ID:",y),null)}function vt(y){const v={name:y.getAttribute("name"),sources:{},vertices:{},primitives:[]},T=n(y,"mesh")[0];if(T!==void 0){for(let E=0;E<T.childNodes.length;E++){const A=T.childNodes[E];if(A.nodeType!==1)continue;const $=A.getAttribute("id");switch(A.nodeName){case"source":v.sources[$]=Z(A);break;case"vertices":v.vertices=at(A);break;case"polygons":console.warn("THREE.ColladaLoader: Unsupported primitive type: ",A.nodeName);break;case"lines":case"linestrips":case"polylist":case"triangles":v.primitives.push(bt(A));break;default:console.log(A)}}qt.geometries[y.getAttribute("id")]=v}}function Z(y){const v={array:[],stride:3};for(let T=0;T<y.childNodes.length;T++){const E=y.childNodes[T];if(E.nodeType===1)switch(E.nodeName){case"float_array":v.array=r(E.textContent);break;case"Name_array":v.array=s(E.textContent);break;case"technique_common":const A=n(E,"accessor")[0];A!==void 0&&(v.stride=parseInt(A.getAttribute("stride")));break}}return v}function at(y){const v={};for(let T=0;T<y.childNodes.length;T++){const E=y.childNodes[T];E.nodeType===1&&(v[E.getAttribute("semantic")]=o(E.getAttribute("source")))}return v}function bt(y){const v={type:y.nodeName,material:y.getAttribute("material"),count:parseInt(y.getAttribute("count")),inputs:{},stride:0,hasUV:!1};for(let T=0,E=y.childNodes.length;T<E;T++){const A=y.childNodes[T];if(A.nodeType===1)switch(A.nodeName){case"input":const $=o(A.getAttribute("source")),it=A.getAttribute("semantic"),wt=parseInt(A.getAttribute("offset")),mt=parseInt(A.getAttribute("set")),St=mt>0?it+mt:it;v.inputs[St]={id:$,offset:wt},v.stride=Math.max(v.stride,wt+1),it==="TEXCOORD"&&(v.hasUV=!0);break;case"vcount":v.vcount=a(A.textContent);break;case"p":v.p=a(A.textContent);break}}return v}function Mt(y){const v={};for(let T=0;T<y.length;T++){const E=y[T];v[E.type]===void 0&&(v[E.type]=[]),v[E.type].push(E)}return v}function Qt(y){let v=0;for(let T=0,E=y.length;T<E;T++)y[T].hasUV===!0&&v++;v>0&&v<y.length&&(y.uvsNeedsFix=!0)}function Re(y){const v={},T=y.sources,E=y.vertices,A=y.primitives;if(A.length===0)return{};const $=Mt(A);for(const it in $){const wt=$[it];Qt(wt),v[it]=Ge(wt,T,E)}return v}function Ge(y,v,T){const E={},A={array:[],stride:0},$={array:[],stride:0},it={array:[],stride:0},wt={array:[],stride:0},mt={array:[],stride:0},St={array:[],stride:4},Bt={array:[],stride:4},ht=new Ie,kt=[];let Nt=0;for(let Gt=0;Gt<y.length;Gt++){const Vt=y[Gt],De=Vt.inputs;let Se=0;switch(Vt.type){case"lines":case"linestrips":Se=Vt.count*2;break;case"triangles":Se=Vt.count*3;break;case"polylist":for(let ge=0;ge<Vt.count;ge++){const _e=Vt.vcount[ge];switch(_e){case 3:Se+=3;break;case 4:Se+=6;break;default:Se+=(_e-2)*3;break}}break;default:console.warn("THREE.ColladaLoader: Unknow primitive type:",Vt.type)}ht.addGroup(Nt,Se,Gt),Nt+=Se,Vt.material&&kt.push(Vt.material);for(const ge in De){const _e=De[ge];switch(ge){case"VERTEX":for(const qn in T){const an=T[qn];switch(qn){case"POSITION":const Ki=A.array.length;if(ne(Vt,v[an],_e.offset,A.array),A.stride=v[an].stride,v.skinWeights&&v.skinIndices&&(ne(Vt,v.skinIndices,_e.offset,St.array),ne(Vt,v.skinWeights,_e.offset,Bt.array)),Vt.hasUV===!1&&y.uvsNeedsFix===!0){const ku=(A.array.length-Ki)/A.stride;for(let Gl=0;Gl<ku;Gl++)it.array.push(0,0)}break;case"NORMAL":ne(Vt,v[an],_e.offset,$.array),$.stride=v[an].stride;break;case"COLOR":ne(Vt,v[an],_e.offset,mt.array),mt.stride=v[an].stride;break;case"TEXCOORD":ne(Vt,v[an],_e.offset,it.array),it.stride=v[an].stride;break;case"TEXCOORD1":ne(Vt,v[an],_e.offset,wt.array),it.stride=v[an].stride;break;default:console.warn('THREE.ColladaLoader: Semantic "%s" not handled in geometry build process.',qn)}}break;case"NORMAL":ne(Vt,v[_e.id],_e.offset,$.array),$.stride=v[_e.id].stride;break;case"COLOR":ne(Vt,v[_e.id],_e.offset,mt.array,!0),mt.stride=v[_e.id].stride;break;case"TEXCOORD":ne(Vt,v[_e.id],_e.offset,it.array),it.stride=v[_e.id].stride;break;case"TEXCOORD1":ne(Vt,v[_e.id],_e.offset,wt.array),wt.stride=v[_e.id].stride;break}}}return A.array.length>0&&ht.setAttribute("position",new he(A.array,A.stride)),$.array.length>0&&ht.setAttribute("normal",new he($.array,$.stride)),mt.array.length>0&&ht.setAttribute("color",new he(mt.array,mt.stride)),it.array.length>0&&ht.setAttribute("uv",new he(it.array,it.stride)),wt.array.length>0&&ht.setAttribute("uv1",new he(wt.array,wt.stride)),St.array.length>0&&ht.setAttribute("skinIndex",new he(St.array,St.stride)),Bt.array.length>0&&ht.setAttribute("skinWeight",new he(Bt.array,Bt.stride)),E.data=ht,E.type=y[0].type,E.materialKeys=kt,E}function ne(y,v,T,E,A=!1){const $=y.p,it=y.stride,wt=y.vcount;function mt(ht){let kt=$[ht+T]*Bt;const Nt=kt+Bt;for(;kt<Nt;kt++)E.push(St[kt]);if(A){const Gt=E.length-Bt-1;Li.setRGB(E[Gt+0],E[Gt+1],E[Gt+2],Pe),E[Gt+0]=Li.r,E[Gt+1]=Li.g,E[Gt+2]=Li.b}}const St=v.array,Bt=v.stride;if(y.vcount!==void 0){let ht=0;for(let kt=0,Nt=wt.length;kt<Nt;kt++){const Gt=wt[kt];if(Gt===4){const Vt=ht+it*0,De=ht+it*1,Se=ht+it*2,ge=ht+it*3;mt(Vt),mt(De),mt(ge),mt(De),mt(Se),mt(ge)}else if(Gt===3){const Vt=ht+it*0,De=ht+it*1,Se=ht+it*2;mt(Vt),mt(De),mt(Se)}else if(Gt>4)for(let Vt=1,De=Gt-2;Vt<=De;Vt++){const Se=ht+it*0,ge=ht+it*Vt,_e=ht+it*(Vt+1);mt(Se),mt(ge),mt(_e)}ht+=it*Gt}}else for(let ht=0,kt=$.length;ht<kt;ht+=it)mt(ht)}function sn(y){return _(qt.geometries[y],Re)}function En(y){const v={name:y.getAttribute("name")||"",joints:{},links:[]};for(let T=0;T<y.childNodes.length;T++){const E=y.childNodes[T];if(E.nodeType===1)switch(E.nodeName){case"technique_common":Xn(E,v);break}}qt.kinematicsModels[y.getAttribute("id")]=v}function gr(y){return y.build!==void 0?y.build:y}function _r(y){return _(qt.kinematicsModels[y],gr)}function Xn(y,v){for(let T=0;T<y.childNodes.length;T++){const E=y.childNodes[T];if(E.nodeType===1)switch(E.nodeName){case"joint":v.joints[E.getAttribute("sid")]=Bs(E);break;case"link":v.links.push(zs(E));break}}}function Bs(y){let v;for(let T=0;T<y.childNodes.length;T++){const E=y.childNodes[T];if(E.nodeType===1)switch(E.nodeName){case"prismatic":case"revolute":v=vr(E);break}}return v}function vr(y){const v={sid:y.getAttribute("sid"),name:y.getAttribute("name")||"",axis:new F,limits:{min:0,max:0},type:y.nodeName,static:!1,zeroPosition:0,middlePosition:0};for(let T=0;T<y.childNodes.length;T++){const E=y.childNodes[T];if(E.nodeType===1)switch(E.nodeName){case"axis":const A=r(E.textContent);v.axis.fromArray(A);break;case"limits":const $=E.getElementsByTagName("max")[0],it=E.getElementsByTagName("min")[0];v.limits.max=parseFloat($.textContent),v.limits.min=parseFloat(it.textContent);break}}return v.limits.min>=v.limits.max&&(v.static=!0),v.middlePosition=(v.limits.min+v.limits.max)/2,v}function zs(y){const v={sid:y.getAttribute("sid"),name:y.getAttribute("name")||"",attachments:[],transforms:[]};for(let T=0;T<y.childNodes.length;T++){const E=y.childNodes[T];if(E.nodeType===1)switch(E.nodeName){case"attachment_full":v.attachments.push(Yi(E));break;case"matrix":case"translate":case"rotate":v.transforms.push(Vs(E));break}}return v}function Yi(y){const v={joint:y.getAttribute("joint").split("/").pop(),transforms:[],links:[]};for(let T=0;T<y.childNodes.length;T++){const E=y.childNodes[T];if(E.nodeType===1)switch(E.nodeName){case"link":v.links.push(zs(E));break;case"matrix":case"translate":case"rotate":v.transforms.push(Vs(E));break}}return v}function Vs(y){const v={type:y.nodeName},T=r(y.textContent);switch(v.type){case"matrix":v.obj=new Kt,v.obj.fromArray(T).transpose();break;case"translate":v.obj=new F,v.obj.fromArray(T);break;case"rotate":v.obj=new F,v.obj.fromArray(T),v.angle=Mi.degToRad(T[3]);break}return v}function $i(y){const v={name:y.getAttribute("name")||"",rigidBodies:{}};for(let T=0;T<y.childNodes.length;T++){const E=y.childNodes[T];if(E.nodeType===1)switch(E.nodeName){case"rigid_body":v.rigidBodies[E.getAttribute("name")]={},xr(E,v.rigidBodies[E.getAttribute("name")]);break}}qt.physicsModels[y.getAttribute("id")]=v}function xr(y,v){for(let T=0;T<y.childNodes.length;T++){const E=y.childNodes[T];if(E.nodeType===1)switch(E.nodeName){case"technique_common":yr(E,v);break}}}function yr(y,v){for(let T=0;T<y.childNodes.length;T++){const E=y.childNodes[T];if(E.nodeType===1)switch(E.nodeName){case"inertia":v.inertia=r(E.textContent);break;case"mass":v.mass=r(E.textContent)[0];break}}}function Na(y){const v={bindJointAxis:[]};for(let T=0;T<y.childNodes.length;T++){const E=y.childNodes[T];if(E.nodeType===1)switch(E.nodeName){case"bind_joint_axis":v.bindJointAxis.push(Da(E));break}}qt.kinematicsScenes[o(y.getAttribute("url"))]=v}function Da(y){const v={target:y.getAttribute("target").split("/").pop()};for(let T=0;T<y.childNodes.length;T++){const E=y.childNodes[T];if(E.nodeType===1)switch(E.nodeName){case"axis":const A=E.getElementsByTagName("param")[0];v.axis=A.textContent;const $=v.axis.split("inst_").pop().split("axis")[0];v.jointIndex=$.substring(0,$.length-1);break}}return v}function Ia(y){return y.build!==void 0?y.build:y}function w(y){return _(qt.kinematicsScenes[y],Ia)}function V(){const y=Object.keys(qt.kinematicsModels)[0],v=Object.keys(qt.kinematicsScenes)[0],T=Object.keys(qt.visualScenes)[0];if(y===void 0||v===void 0)return;const E=_r(y),A=w(v),$=pe(T),it=A.bindJointAxis,wt={};for(let Bt=0,ht=it.length;Bt<ht;Bt++){const kt=it[Bt],Nt=Jt.querySelector('[sid="'+kt.target+'"]');if(Nt){const Gt=Nt.parentElement;mt(kt.jointIndex,Gt)}}function mt(Bt,ht){const kt=ht.getAttribute("name"),Nt=E.joints[Bt];$.traverse(function(Gt){Gt.name===kt&&(wt[Bt]={object:Gt,transforms:j(ht),joint:Nt,position:Nt.zeroPosition})})}const St=new Kt;Hl={joints:E&&E.joints,getJointValue:function(Bt){const ht=wt[Bt];if(ht)return ht.position;console.warn("THREE.ColladaLoader: Joint "+Bt+" doesn't exist.")},setJointValue:function(Bt,ht){const kt=wt[Bt];if(kt){const Nt=kt.joint;if(ht>Nt.limits.max||ht<Nt.limits.min)console.warn("THREE.ColladaLoader: Joint "+Bt+" value "+ht+" outside of limits (min: "+Nt.limits.min+", max: "+Nt.limits.max+").");else if(Nt.static)console.warn("THREE.ColladaLoader: Joint "+Bt+" is static.");else{const Gt=kt.object,Vt=Nt.axis,De=kt.transforms;B.identity();for(let Se=0;Se<De.length;Se++){const ge=De[Se];if(ge.sid&&ge.sid.indexOf(Bt)!==-1)switch(Nt.type){case"revolute":B.multiply(St.makeRotationAxis(Vt,Mi.degToRad(ht)));break;case"prismatic":B.multiply(St.makeTranslation(Vt.x*ht,Vt.y*ht,Vt.z*ht));break;default:console.warn("THREE.ColladaLoader: Unknown joint type: "+Nt.type);break}else switch(ge.type){case"matrix":B.multiply(ge.obj);break;case"translate":B.multiply(St.makeTranslation(ge.obj.x,ge.obj.y,ge.obj.z));break;case"scale":B.scale(ge.obj);break;case"rotate":B.multiply(St.makeRotationAxis(ge.obj,ge.angle));break}}Gt.matrix.copy(B),Gt.matrix.decompose(Gt.position,Gt.quaternion,Gt.scale),wt[Bt].position=ht}}else console.log("THREE.ColladaLoader: "+Bt+" does not exist.")}}}function j(y){const v=[],T=Jt.querySelector('[id="'+y.id+'"]');for(let E=0;E<T.childNodes.length;E++){const A=T.childNodes[E];if(A.nodeType!==1)continue;let $,it;switch(A.nodeName){case"matrix":$=r(A.textContent);const wt=new Kt().fromArray($).transpose();v.push({sid:A.getAttribute("sid"),type:A.nodeName,obj:wt});break;case"translate":case"scale":$=r(A.textContent),it=new F().fromArray($),v.push({sid:A.getAttribute("sid"),type:A.nodeName,obj:it});break;case"rotate":$=r(A.textContent),it=new F().fromArray($);const mt=Mi.degToRad($[3]);v.push({sid:A.getAttribute("sid"),type:A.nodeName,obj:it,angle:mt});break}}return v}function Y(y){const v=y.getElementsByTagName("node");for(let T=0;T<v.length;T++){const E=v[T];E.hasAttribute("id")===!1&&E.setAttribute("id",c())}}const B=new Kt,ut=new F;function xt(y){const v={name:y.getAttribute("name")||"",type:y.getAttribute("type"),id:y.getAttribute("id"),sid:y.getAttribute("sid"),matrix:new Kt,nodes:[],instanceCameras:[],instanceControllers:[],instanceLights:[],instanceGeometries:[],instanceNodes:[],transforms:{}};for(let T=0;T<y.childNodes.length;T++){const E=y.childNodes[T];if(E.nodeType!==1)continue;let A;switch(E.nodeName){case"node":v.nodes.push(E.getAttribute("id")),xt(E);break;case"instance_camera":v.instanceCameras.push(o(E.getAttribute("url")));break;case"instance_controller":v.instanceControllers.push(Dt(E));break;case"instance_light":v.instanceLights.push(o(E.getAttribute("url")));break;case"instance_geometry":v.instanceGeometries.push(Dt(E));break;case"instance_node":v.instanceNodes.push(o(E.getAttribute("url")));break;case"matrix":A=r(E.textContent),v.matrix.multiply(B.fromArray(A).transpose()),v.transforms[E.getAttribute("sid")]=E.nodeName;break;case"translate":A=r(E.textContent),ut.fromArray(A),v.matrix.multiply(B.makeTranslation(ut.x,ut.y,ut.z)),v.transforms[E.getAttribute("sid")]=E.nodeName;break;case"rotate":A=r(E.textContent);const $=Mi.degToRad(A[3]);v.matrix.multiply(B.makeRotationAxis(ut.fromArray(A),$)),v.transforms[E.getAttribute("sid")]=E.nodeName;break;case"scale":A=r(E.textContent),v.matrix.scale(ut.fromArray(A)),v.transforms[E.getAttribute("sid")]=E.nodeName;break;case"extra":break;default:console.log(E)}}return be(v.id)?console.warn("THREE.ColladaLoader: There is already a node with ID %s. Exclude current node from further processing.",v.id):qt.nodes[v.id]=v,v}function Dt(y){const v={id:o(y.getAttribute("url")),materials:{},skeletons:[]};for(let T=0;T<y.childNodes.length;T++){const E=y.childNodes[T];switch(E.nodeName){case"bind_material":const A=E.getElementsByTagName("instance_material");for(let $=0;$<A.length;$++){const it=A[$],wt=it.getAttribute("symbol"),mt=it.getAttribute("target");v.materials[wt]=o(mt)}break;case"skeleton":v.skeletons.push(o(E.textContent));break}}return v}function It(y,v){const T=[],E=[];let A,$,it;for(A=0;A<y.length;A++){const St=y[A];let Bt;if(be(St))Bt=Ue(St),Yt(Bt,v,T);else if(Tn(St)){const kt=qt.visualScenes[St].children;for(let Nt=0;Nt<kt.length;Nt++){const Gt=kt[Nt];if(Gt.type==="JOINT"){const Vt=Ue(Gt.id);Yt(Vt,v,T)}}}else console.error("THREE.ColladaLoader: Unable to find root bone of skeleton with ID:",St)}for(A=0;A<v.length;A++)for($=0;$<T.length;$++)if(it=T[$],it.bone.name===v[A].name){E[A]=it,it.processed=!0;break}for(A=0;A<T.length;A++)it=T[A],it.processed===!1&&(E.push(it),it.processed=!0);const wt=[],mt=[];for(A=0;A<E.length;A++)it=E[A],wt.push(it.bone),mt.push(it.boneInverse);return new Al(wt,mt)}function Yt(y,v,T){y.traverse(function(E){if(E.isBone===!0){let A;for(let $=0;$<v.length;$++){const it=v[$];if(it.name===E.name){A=it.boneInverse;break}}A===void 0&&(A=new Kt),T.push({bone:E,boneInverse:A,processed:!1})}})}function te(y){const v=[],T=y.matrix,E=y.nodes,A=y.type,$=y.instanceCameras,it=y.instanceControllers,wt=y.instanceLights,mt=y.instanceGeometries,St=y.instanceNodes;for(let ht=0,kt=E.length;ht<kt;ht++)v.push(Ue(E[ht]));for(let ht=0,kt=$.length;ht<kt;ht++){const Nt=jt($[ht]);Nt!==null&&v.push(Nt.clone())}for(let ht=0,kt=it.length;ht<kt;ht++){const Nt=it[ht],Gt=K(Nt.id),Vt=sn(Gt.id),De=Me(Vt,Nt.materials),Se=Nt.skeletons,ge=Gt.skin.joints,_e=It(Se,ge);for(let qn=0,an=De.length;qn<an;qn++){const Ki=De[qn];Ki.isSkinnedMesh&&(Ki.bind(_e,Gt.skin.bindMatrix),Ki.normalizeSkinWeights()),v.push(Ki)}}for(let ht=0,kt=wt.length;ht<kt;ht++){const Nt=z(wt[ht]);Nt!==null&&v.push(Nt.clone())}for(let ht=0,kt=mt.length;ht<kt;ht++){const Nt=mt[ht],Gt=sn(Nt.id),Vt=Me(Gt,Nt.materials);for(let De=0,Se=Vt.length;De<Se;De++)v.push(Vt[De])}for(let ht=0,kt=St.length;ht<kt;ht++)v.push(Ue(St[ht]).clone());let Bt;if(E.length===0&&v.length===1)Bt=v[0];else{Bt=A==="JOINT"?new su:new je;for(let ht=0;ht<v.length;ht++)Bt.add(v[ht])}return Bt.name=A==="JOINT"?y.sid:y.name,Bt.matrix.copy(T),Bt.matrix.decompose(Bt.position,Bt.quaternion,Bt.scale),Bt}const Ut=new Un({name:ri.DEFAULT_MATERIAL_NAME,color:16711935});function fe(y,v){const T=[];for(let E=0,A=y.length;E<A;E++){const $=v[y[E]];$===void 0?(console.warn("THREE.ColladaLoader: Material with key %s not found. Apply fallback material.",y[E]),T.push(Ut)):T.push(yt($))}return T}function Me(y,v){const T=[];for(const E in y){const A=y[E],$=fe(A.materialKeys,v);if($.length===0&&(E==="lines"||E==="linestrips"?$.push(new zn):$.push(new vs)),E==="lines"||E==="linestrips")for(let St=0,Bt=$.length;St<Bt;St++){const ht=$[St];if(ht.isMeshPhongMaterial===!0||ht.isMeshLambertMaterial===!0){const kt=new zn;kt.color.copy(ht.color),kt.opacity=ht.opacity,kt.transparent=ht.transparent,$[St]=kt}}const it=A.data.attributes.skinIndex!==void 0,wt=$.length===1?$[0]:$;let mt;switch(E){case"lines":mt=new ga(A.data,wt);break;case"linestrips":mt=new dr(A.data,wt);break;case"triangles":case"polylist":it?mt=new a0(A.data,wt):mt=new Ae(A.data,wt);break}T.push(mt)}return T}function be(y){return qt.nodes[y]!==void 0}function Ue(y){return _(qt.nodes[y],te)}function me(y){const v={name:y.getAttribute("name"),children:[]};Y(y);const T=n(y,"node");for(let E=0;E<T.length;E++)v.children.push(xt(T[E]));qt.visualScenes[y.getAttribute("id")]=v}function Ot(y){const v=new je;v.name=y.name;const T=y.children;for(let E=0;E<T.length;E++){const A=T[E];v.add(Ue(A.id))}return v}function Tn(y){return qt.visualScenes[y]!==void 0}function pe(y){return _(qt.visualScenes[y],Ot)}function hn(y){const v=n(y,"instance_visual_scene")[0];return pe(o(v.getAttribute("url")))}function li(){const y=qt.clips;if(l(y)===!0){if(l(qt.animations)===!1){const v=[];for(const T in qt.animations){const E=x(T);for(let A=0,$=E.length;A<$;A++)v.push(E[A])}Mr.push(new oh("default",-1,v))}}else for(const v in y)Mr.push(W(v))}function Ze(y){let v="";const T=[y];for(;T.length;){const E=T.shift();E.nodeType===Node.TEXT_NODE?v+=E.textContent:(v+=`
`,T.push.apply(T,E.childNodes))}return v.trim()}if(t.length===0)return{scene:new iu};const ci=new DOMParser().parseFromString(t,"application/xml"),Jt=n(ci,"COLLADA")[0],rn=ci.getElementsByTagName("parsererror")[0];if(rn!==void 0){const y=n(rn,"div")[0];let v;return y?v=y.textContent:v=Ze(rn),console.error(`THREE.ColladaLoader: Failed to parse collada file.
`,v),null}const Pi=Jt.getAttribute("version");console.debug("THREE.ColladaLoader: File version",Pi);const Qe=h(n(Jt,"asset")[0]),wn=new fu(this.manager);wn.setPath(this.resourcePath||e).setCrossOrigin(this.crossOrigin);let xn;xh&&(xn=new xh(this.manager),xn.setPath(this.resourcePath||e));const Li=new $t,Mr=[];let Hl={},Ou=0;const qt={animations:{},clips:{},controllers:{},images:{},effects:{},materials:{},cameras:{},lights:{},geometries:{},nodes:{},visualScenes:{},kinematicsModels:{},physicsModels:{},kinematicsScenes:{}};p(Jt,"library_animations","animation",m),p(Jt,"library_animation_clips","animation_clip",O),p(Jt,"library_controllers","controller",ot),p(Jt,"library_images","image",et),p(Jt,"library_effects","effect",zt),p(Jt,"library_materials","material",lt),p(Jt,"library_cameras","camera",At),p(Jt,"library_lights","light",Pt),p(Jt,"library_geometries","geometry",vt),p(Jt,"library_nodes","node",xt),p(Jt,"library_visual_scenes","visual_scene",me),p(Jt,"library_kinematics_models","kinematics_model",En),p(Jt,"library_physics_models","physics_model",$i),p(Jt,"scene","instance_kinematics_scene",Na),g(qt.animations,C),g(qt.clips,tt),g(qt.controllers,Et),g(qt.images,nt),g(qt.effects,q),g(qt.materials,Ft),g(qt.cameras,Xt),g(qt.lights,ye),g(qt.geometries,Re),g(qt.visualScenes,Ot),li(),V();const Sr=hn(n(Jt,"scene")[0]);return Sr.animations=Mr,Qe.upAxis==="Z_UP"&&(console.warn("THREE.ColladaLoader: You are loading an asset with a Z-UP coordinate system. The loader just rotates the asset to transform it into Y-UP. The vertex data are not converted, see #24289."),Sr.rotation.set(-Math.PI/2,0,0)),Sr.scale.multiplyScalar(Qe.unit),{get animations(){return console.warn("THREE.ColladaLoader: Please access animations over scene.animations now."),Mr},kinematics:Hl,library:qt,scene:Sr}}}const yh=new F,dv=new cn,ta=new Kt,gi=new Kt,ea=new Dn,na=new F(1,1,1),ia=new F;class Ra extends Le{constructor(...t){super(...t),this.urdfNode=null,this.urdfName=""}copy(t,e){return super.copy(t,e),this.urdfNode=t.urdfNode,this.urdfName=t.urdfName,this}}class fv extends Ra{constructor(...t){super(...t),this.isURDFCollider=!0,this.type="URDFCollider"}}class pv extends Ra{constructor(...t){super(...t),this.isURDFVisual=!0,this.type="URDFVisual"}}class _u extends Ra{constructor(...t){super(...t),this.isURDFLink=!0,this.type="URDFLink",this.inertial={mass:0,origin:{xyz:[0,0,0],rpy:[0,0,0]},inertia:{ixx:0,ixy:0,ixz:0,iyy:0,iyz:0,izz:0}}}copy(t,e){return super.copy(t,e),this.inertial={mass:t.inertial.mass,origin:{xyz:[...t.inertial.origin.xyz],rpy:[...t.inertial.origin.rpy]},inertia:{...t.inertial.inertia}},this}}class vu extends Ra{get jointType(){return this._jointType}set jointType(t){if(this.jointType!==t)switch(this._jointType=t,this.matrixWorldNeedsUpdate=!0,t){case"fixed":this.jointValue=[];break;case"continuous":case"revolute":case"prismatic":this.jointValue=new Array(1).fill(0);break;case"planar":this.jointValue=new Array(3).fill(0),this.axis=new F(0,0,1);break;case"floating":this.jointValue=new Array(6).fill(0);break}}get angle(){return this.jointValue[0]}constructor(...t){super(...t),this.isURDFJoint=!0,this.type="URDFJoint",this.jointValue=null,this.jointType="fixed",this.axis=new F(1,0,0),this.limit={lower:0,upper:0,effort:0,velocity:0},this.ignoreLimits=!1,this.origPosition=null,this.origQuaternion=null,this.mimicJoints=[]}copy(t,e){return super.copy(t,e),this.jointType=t.jointType,this.axis=t.axis.clone(),this.limit.lower=t.limit.lower,this.limit.upper=t.limit.upper,this.limit.effort=t.limit.effort,this.limit.velocity=t.limit.velocity,this.ignoreLimits=!1,this.jointValue=[...t.jointValue],this.origPosition=t.origPosition?t.origPosition.clone():null,this.origQuaternion=t.origQuaternion?t.origQuaternion.clone():null,this.mimicJoints=[...t.mimicJoints],this}setJointValue(...t){t=t.map(n=>n===null?null:parseFloat(n)),(!this.origPosition||!this.origQuaternion)&&(this.origPosition=this.position.clone(),this.origQuaternion=this.quaternion.clone());let e=!1;switch(this.mimicJoints.forEach(n=>{e=n.updateFromMimickedJoint(...t)||e}),this.jointType){case"fixed":return e;case"continuous":case"revolute":{let n=t[0];return n==null||n===this.jointValue[0]?e:(!this.ignoreLimits&&this.jointType==="revolute"&&(n=Math.min(this.limit.upper,n),n=Math.max(this.limit.lower,n)),this.quaternion.setFromAxisAngle(this.axis,n).premultiply(this.origQuaternion),this.jointValue[0]!==n?(this.jointValue[0]=n,this.matrixWorldNeedsUpdate=!0,!0):e)}case"prismatic":{let n=t[0];return n==null||n===this.jointValue[0]?e:(this.ignoreLimits||(n=Math.min(this.limit.upper,n),n=Math.max(this.limit.lower,n)),this.position.copy(this.origPosition),yh.copy(this.axis).applyEuler(this.rotation),this.position.addScaledVector(yh,n),this.jointValue[0]!==n?(this.jointValue[0]=n,this.matrixWorldNeedsUpdate=!0,!0):e)}case"floating":return this.jointValue.every((n,s)=>t[s]===n||t[s]===null)?e:(this.jointValue[0]=t[0]!==null?t[0]:this.jointValue[0],this.jointValue[1]=t[1]!==null?t[1]:this.jointValue[1],this.jointValue[2]=t[2]!==null?t[2]:this.jointValue[2],this.jointValue[3]=t[3]!==null?t[3]:this.jointValue[3],this.jointValue[4]=t[4]!==null?t[4]:this.jointValue[4],this.jointValue[5]=t[5]!==null?t[5]:this.jointValue[5],gi.compose(this.origPosition,this.origQuaternion,na),ea.setFromEuler(dv.set(this.jointValue[3],this.jointValue[4],this.jointValue[5],"XYZ")),ia.set(this.jointValue[0],this.jointValue[1],this.jointValue[2]),ta.compose(ia,ea,na),gi.premultiply(ta),this.position.setFromMatrixPosition(gi),this.rotation.setFromRotationMatrix(gi),this.matrixWorldNeedsUpdate=!0,!0);case"planar":return this.jointValue.every((n,s)=>t[s]===n||t[s]===null)?e:(this.jointValue[0]=t[0]!==null?t[0]:this.jointValue[0],this.jointValue[1]=t[1]!==null?t[1]:this.jointValue[1],this.jointValue[2]=t[2]!==null?t[2]:this.jointValue[2],gi.compose(this.origPosition,this.origQuaternion,na),ea.setFromAxisAngle(this.axis,this.jointValue[2]),ia.set(this.jointValue[0],this.jointValue[1],0),ta.compose(ia,ea,na),gi.premultiply(ta),this.position.setFromMatrixPosition(gi),this.rotation.setFromRotationMatrix(gi),this.matrixWorldNeedsUpdate=!0,!0)}return e}}class Mh extends vu{constructor(...t){super(...t),this.type="URDFMimicJoint",this.mimicJoint=null,this.offset=0,this.multiplier=1}updateFromMimickedJoint(...t){const e=t.map(n=>n===null?null:n*this.multiplier+this.offset);return super.setJointValue(...e)}copy(t,e){return super.copy(t,e),this.mimicJoint=t.mimicJoint,this.offset=t.offset,this.multiplier=t.multiplier,this}}class mv extends _u{constructor(...t){super(...t),this.isURDFRobot=!0,this.urdfNode=null,this.urdfRobotNode=null,this.robotName=null,this.links=null,this.joints=null,this.colliders=null,this.visual=null,this.frames=null}copy(t,e){super.copy(t,e),this.urdfRobotNode=t.urdfRobotNode,this.robotName=t.robotName,this.links={},this.joints={},this.colliders={},this.visual={},this.traverse(n=>{n.isURDFJoint&&n.urdfName in t.joints&&(this.joints[n.urdfName]=n),n.isURDFLink&&n.urdfName in t.links&&(this.links[n.urdfName]=n),n.isURDFCollider&&n.urdfName in t.colliders&&(this.colliders[n.urdfName]=n),n.isURDFVisual&&n.urdfName in t.visual&&(this.visual[n.urdfName]=n)});for(const n in this.joints)this.joints[n].mimicJoints=this.joints[n].mimicJoints.map(s=>this.joints[s.name]);return this.frames={...this.colliders,...this.visual,...this.links,...this.joints},this}getFrame(t){return this.frames[t]}setJointValue(t,...e){const n=this.joints[t];return n?n.setJointValue(...e):!1}setJointValues(t){let e=!1;for(const n in t){const s=t[n];Array.isArray(s)?e=this.setJointValue(n,...s)||e:e=this.setJointValue(n,s)||e}return e}}const Eo=new Dn,Sh=new cn;function _i(i){return i?i.trim().split(/\s+/g).map(t=>parseFloat(t)):[0,0,0]}function bh(i,t,e=!1){e||i.rotation.set(0,0,0),Sh.set(t[0],t[1],t[2],"ZYX"),Eo.setFromEuler(Sh),Eo.multiply(i.quaternion),i.quaternion.copy(Eo)}class gv{constructor(t){this.manager=t||du,this.loadMeshCb=this.defaultMeshLoader.bind(this),this.parseVisual=!0,this.parseCollision=!1,this.packages="",this.workingPath="",this.fetchOptions={}}loadAsync(t){return new Promise((e,n)=>{this.load(t,e,null,n)})}load(t,e,n,s){const r=this.manager,a=mu.extractUrlBase(t),o=this.manager.resolveURL(t);r.itemStart(o),fetch(o,this.fetchOptions).then(c=>{if(c.ok)return n&&n(null),c.text();throw new Error(`URDFLoader: Failed to load url '${o}' with error code ${c.status} : ${c.statusText}.`)}).then(c=>{const l=this.parse(c,this.workingPath||a);e(l),r.itemEnd(o)}).catch(c=>{s?s(c):console.error("URDFLoader: Error loading file.",c),r.itemError(o),r.itemEnd(o)})}parse(t,e=this.workingPath){const n=this.packages,s=this.loadMeshCb,r=this.parseVisual,a=this.parseCollision,o=this.manager,c={},l={},h={};function u(R){if(!/^package:\/\//.test(R))return e?e+R:R;const[C,x]=R.replace(/^package:\/\//,"").split(/\/(.+)/);if(typeof n=="string")return n.endsWith(C)?n+"/"+x:n+"/"+C+"/"+x;if(typeof n=="function")return n(C)+"/"+x;if(typeof n=="object")return C in n?n[C]+"/"+x:(console.error(`URDFLoader : ${C} not found in provided package list.`),null)}function d(R){let C;R instanceof Document?C=[...R.children]:R instanceof Element?C=[R]:C=[...new DOMParser().parseFromString(R,"text/xml").children];const x=C.filter(N=>N.nodeName==="robot").pop();return p(x)}function p(R){const C=[...R.children],x=C.filter(U=>U.nodeName.toLowerCase()==="link"),N=C.filter(U=>U.nodeName.toLowerCase()==="joint"),P=C.filter(U=>U.nodeName.toLowerCase()==="material"),L=new mv;L.robotName=R.getAttribute("name"),L.urdfRobotNode=R,P.forEach(U=>{const k=U.getAttribute("name");h[k]=m(U)});const I={},M={};x.forEach(U=>{const k=U.getAttribute("name"),H=R.querySelector(`child[link="${k}"]`)===null;c[k]=_(U,I,M,H?L:null)}),N.forEach(U=>{const k=U.getAttribute("name");l[k]=g(U)}),L.joints=l,L.links=c,L.colliders=M,L.visual=I;const S=Object.values(l);return S.forEach(U=>{U instanceof Mh&&l[U.mimicJoint].mimicJoints.push(U)}),S.forEach(U=>{const k=new Set,H=J=>{if(k.has(J))throw new Error("URDFLoader: Detected an infinite loop of mimic joints.");k.add(J),J.mimicJoints.forEach(Q=>{H(Q)})};H(U)}),L.frames={...M,...I,...c,...l},L}function g(R){const C=[...R.children],x=R.getAttribute("type");let N;const P=C.find(k=>k.nodeName.toLowerCase()==="mimic");P?(N=new Mh,N.mimicJoint=P.getAttribute("joint"),N.multiplier=parseFloat(P.getAttribute("multiplier")||1),N.offset=parseFloat(P.getAttribute("offset")||0)):N=new vu,N.urdfNode=R,N.name=R.getAttribute("name"),N.urdfName=N.name,N.jointType=x;let L=null,I=null,M=[0,0,0],S=[0,0,0];C.forEach(k=>{const H=k.nodeName.toLowerCase();H==="origin"?(M=_i(k.getAttribute("xyz")),S=_i(k.getAttribute("rpy"))):H==="child"?I=c[k.getAttribute("link")]:H==="parent"?L=c[k.getAttribute("link")]:H==="limit"&&(N.limit.lower=parseFloat(k.getAttribute("lower")||N.limit.lower),N.limit.upper=parseFloat(k.getAttribute("upper")||N.limit.upper),N.limit.effort=parseFloat(k.getAttribute("effort")||N.limit.effort),N.limit.velocity=parseFloat(k.getAttribute("velocity")||N.limit.velocity))}),L.add(N),N.add(I),bh(N,S),N.position.set(M[0],M[1],M[2]);const U=C.filter(k=>k.nodeName.toLowerCase()==="axis")[0];if(U){const k=U.getAttribute("xyz").split(/\s+/g).map(H=>parseFloat(H));N.axis=new F(k[0],k[1],k[2]),N.axis.normalize()}return N}function _(R,C,x,N=null){N===null&&(N=new _u);const P=[...R.children];N.name=R.getAttribute("name"),N.urdfName=N.name,N.urdfNode=R;const L=P.find(I=>I.nodeName.toLowerCase()==="inertial");return L&&[...L.children].forEach(I=>{const M=I.nodeName.toLowerCase();M==="origin"?(N.inertial.origin.xyz=_i(I.getAttribute("xyz")),N.inertial.origin.rpy=_i(I.getAttribute("rpy"))):M==="mass"?N.inertial.mass=parseFloat(I.getAttribute("value"))||0:M==="inertia"&&(N.inertial.inertia.ixx=parseFloat(I.getAttribute("ixx"))||0,N.inertial.inertia.ixy=parseFloat(I.getAttribute("ixy"))||0,N.inertial.inertia.ixz=parseFloat(I.getAttribute("ixz"))||0,N.inertial.inertia.iyy=parseFloat(I.getAttribute("iyy"))||0,N.inertial.inertia.iyz=parseFloat(I.getAttribute("iyz"))||0,N.inertial.inertia.izz=parseFloat(I.getAttribute("izz"))||0)}),r&&P.filter(M=>M.nodeName.toLowerCase()==="visual").forEach(M=>{const S=f(M,h);if(N.add(S),M.hasAttribute("name")){const U=M.getAttribute("name");S.name=U,S.urdfName=U,C[U]=S}}),a&&P.filter(M=>M.nodeName.toLowerCase()==="collision").forEach(M=>{const S=f(M);if(N.add(S),M.hasAttribute("name")){const U=M.getAttribute("name");S.name=U,S.urdfName=U,x[U]=S}}),N}function m(R){const C=[...R.children],x=new vs;return x.name=R.getAttribute("name")||"",C.forEach(N=>{const P=N.nodeName.toLowerCase();if(P==="color"){const L=N.getAttribute("rgba").split(/\s/g).map(I=>parseFloat(I));x.color.setRGB(L[0],L[1],L[2]),x.opacity=L[3],x.transparent=L[3]<1,x.depthWrite=!x.transparent}else if(P==="texture"){const L=N.getAttribute("filename");if(L){const I=new fu(o),M=u(L);x.map=I.load(M),x.map.colorSpace=Pe}}}),x}function f(R,C={}){const x=R.nodeName.toLowerCase()==="collision",N=[...R.children];let P=null;const L=N.filter(M=>M.nodeName.toLowerCase()==="material")[0];if(L){const M=L.getAttribute("name");M&&M in C?P=C[M]:P=m(L)}else P=new vs;const I=x?new fv:new pv;return I.urdfNode=R,N.forEach(M=>{const S=M.nodeName.toLowerCase();if(S==="geometry"){const U=M.children[0].nodeName.toLowerCase();if(U==="mesh"){const k=M.children[0].getAttribute("filename"),H=u(k);if(H!==null){const J=M.children[0].getAttribute("scale");if(J){const Q=_i(J);I.scale.set(Q[0],Q[1],Q[2])}s(H,o,(Q,O)=>{O?console.error("URDFLoader: Error loading mesh.",O):Q&&(Q instanceof Ae&&(Q.material=P),Q.position.set(0,0,0),Q.quaternion.identity(),I.add(Q))})}}else if(U==="box"){const k=new Ae;k.geometry=new Is(1,1,1),k.material=P;const H=_i(M.children[0].getAttribute("size"));k.scale.set(H[0],H[1],H[2]),I.add(k)}else if(U==="sphere"){const k=new Ae;k.geometry=new ji(1,30,30),k.material=P;const H=parseFloat(M.children[0].getAttribute("radius"))||0;k.scale.set(H,H,H),I.add(k)}else if(U==="cylinder"){const k=new Ae;k.geometry=new Fs(1,1,1,30),k.material=P;const H=parseFloat(M.children[0].getAttribute("radius"))||0,J=parseFloat(M.children[0].getAttribute("length"))||0;k.scale.set(H,J,H),k.rotation.set(Math.PI/2,0,0),I.add(k)}}else if(S==="origin"){const U=_i(M.getAttribute("xyz")),k=_i(M.getAttribute("rpy"));I.position.set(U[0],U[1],U[2]),I.rotation.set(0,0,0),bh(I,k)}}),I}return d(t)}defaultMeshLoader(t,e,n){/\.stl$/i.test(t)?new hv(e).load(t,r=>{const a=new Ae(r,new vs);n(a)},null,r=>n(null,r)):/\.dae$/i.test(t)?new uv(e).load(t,r=>n(r.scene),null,r=>n(null,r)):console.warn(`URDFLoader: Could not load model at ${t}.
No loader available`)}}class xu extends Error{constructor(t,e){super(e),this.status=t}}async function Cs(i,t){const e=await fetch(i,{headers:{"Content-Type":"application/json"},...t});if(!e.ok){const n=await e.json().catch(()=>({})),s=Array.isArray(n.detail)?n.detail.map(r=>{var a;return`${((a=r.loc)==null?void 0:a.join("."))||"Request"}: ${r.msg||"Invalid value"}`}).join(`
`):typeof n.detail=="string"?n.detail:e.statusText;throw new xu(e.status,s||`HTTP ${e.status}`)}return e.json()}function xi(i){const t=new Set,e=new Set,n=new Set;i.traverse(s=>{const r=s;r.geometry&&!(s.parent instanceof rr)&&t.add(r.geometry),(Array.isArray(r.material)?r.material:r.material?[r.material]:[]).forEach(o=>e.add(o))}),e.forEach(s=>{Object.values(s).forEach(r=>{r instanceof Ye&&n.add(r)}),s.dispose()}),n.forEach(s=>s.dispose()),t.forEach(s=>s.dispose())}function Rs(i){xi(i),i.clear()}const Eh=new WeakMap;function _v(i){const t=Eh.get(i);if(t)return t;const e=i.contact_force.part_order.map((n,s)=>{var o;const r=[];let a=-1;for(let c=0;c<=i.qpos.length;c++){const l=c<i.qpos.length&&!!((o=i.contact_force.masks[c])!=null&&o[s]);l&&a<0&&(a=c),!l&&a>=0&&(r.push([a,c]),a=-1)}return r});return Eh.set(i,e),e}function yu(i,t=8){const e=i.getBoundingClientRect(),n=Math.max(1,t),s=e.width<560?54:68,r=28;return{width:e.width,height:e.height,labelW:s,rulerH:r,rowH:Math.max(1,(e.height-r)/n),plotW:Math.max(1,e.width-s),rows:n}}function ki(i,t,e){return t.labelW+i/Math.max(1,e-1)*t.plotW}function vv(i,t,e){const n=Math.max(0,Math.min(1,(i-t.labelW)/t.plotW));return Math.round(n*Math.max(0,e-1))}function xv(i){const t=10**Math.floor(Math.log10(Math.max(i,1e-6))),e=i/t;return(e<=1?1:e<=2?2:e<=5?5:10)*t}function Mu(i){const t=Math.floor(i/60),e=i-t*60;return t?`${t}:${e.toFixed(e<10?1:0).padStart(e<10?4:2,"0")}`:`${e.toFixed(e<10?1:0)}s`}function yv(i){var C;const{canvas:t,session:e,frame:n,stableCuts:s,partColors:r}=i,a=yu(t,(e==null?void 0:e.contact_force.part_order.length)??8),o=t.getContext("2d");if(!o)return;const c=Math.min(window.devicePixelRatio,2),l=Math.max(1,Math.round(a.width*c)),h=Math.max(1,Math.round(a.height*c));if(t.width!==l&&(t.width=l),t.height!==h&&(t.height=h),o.setTransform(c,0,0,c,0,0),o.clearRect(0,0,a.width,a.height),o.fillStyle="#0b1117",o.fillRect(0,0,a.width,a.height),!e)return;const u=e.qpos.length;o.fillStyle="#0d141b",o.fillRect(0,0,a.labelW,a.height),o.fillStyle="#111923",o.fillRect(a.labelW,0,a.plotW,a.rulerH);for(let x=0;x<a.rows;x++){const N=a.rulerH+x*a.rowH;o.fillStyle=x%2===0?"#121a22":"#0f171f",o.fillRect(a.labelW,N,a.plotW,a.rowH),o.strokeStyle="#22303c",o.beginPath(),o.moveTo(0,Math.round(N)+.5),o.lineTo(a.width,Math.round(N)+.5),o.stroke(),o.fillStyle=`#${r[x%r.length].toString(16).padStart(6,"0")}`,o.fillRect(8,N+a.rowH/2-3,6,6),o.fillStyle="#aebbc7",o.font="600 10px ui-monospace,monospace",o.textBaseline="middle",o.fillText(e.contact_force.part_order[x]||"",20,N+a.rowH/2)}const d=Math.max(1,u-1)/e.fps,p=xv(d/Math.max(2,Math.floor(a.plotW/92))),g=Math.max(1,Math.round(p*e.fps)),_=Math.max(1,Math.round(g/5)),m=new Set;for(let x=0;x<u;x+=_)m.add(x);for(let x=0;x<u;x+=g)m.add(x);for(const x of[...m].sort((N,P)=>N-P)){const N=Math.round(ki(x,a,u))+.5,P=x%g===0;o.strokeStyle=P?"#344451":"#1d2a34",o.globalAlpha=P?.9:.65,o.beginPath(),o.moveTo(N,P?a.rulerH-7:a.rulerH-4),o.lineTo(N,a.height),o.stroke(),o.globalAlpha=1,P&&(o.fillStyle="#aeb5bd",o.font="10px ui-monospace,monospace",o.fillText(Mu(x/e.fps),Math.min(N+4,a.width-38),10),o.fillStyle="#6f7780",o.font="9px ui-monospace,monospace",o.fillText(String(x),Math.min(N+4,a.width-32),21))}o.fillStyle="#7e8790",o.font="9px ui-monospace,monospace",o.fillText("TIME",8,9),o.fillText("FRAME",8,21);const f=_v(e);for(let x=0;x<f.length;x++)for(const[N,P]of f[x]){const L=ki(N,a,u),I=ki(Math.max(N,P-1),a,u)+Math.max(1,a.plotW/u);o.fillStyle=`#${r[x%r.length].toString(16).padStart(6,"0")}`,o.globalAlpha=.78,o.beginPath(),o.roundRect(L,a.rulerH+x*a.rowH+3,Math.max(2,I-L),Math.max(2,a.rowH-6),2),o.fill(),o.globalAlpha=1}if(!e.contact_force.part_order.length||e.contact_force.mask_status&&e.contact_force.mask_status!=="verified_newton_primary_faces"){o.fillStyle="#8c9aaa",o.font="12px sans-serif",o.fillText("接触状态未知 · 时间轴用于回放定位",a.labelW+12,a.rulerH+24);for(const x of((C=e.review)==null?void 0:C.segments)??[]){const N=ki(x.start,a,u),P=ki(x.end,a,u);o.fillStyle="#27505e",o.fillRect(N,a.rulerH+42,Math.max(2,P-N),18),P-N>36&&(o.fillStyle="#dce7ed",o.fillText(x.label,N+3,a.rulerH+55))}}o.setLineDash([3,3]);for(const x of s){const N=Math.round(ki(x,a,u))+.5;o.strokeStyle="#d9b96e",o.globalAlpha=.38,o.beginPath(),o.moveTo(N,a.rulerH),o.lineTo(N,a.height),o.stroke()}o.setLineDash([]),o.globalAlpha=1;const R=Math.round(ki(n,a,u))+.5;o.strokeStyle="#72ead3",o.lineWidth=1.5,o.beginPath(),o.moveTo(R,0),o.lineTo(R,a.height),o.stroke(),o.fillStyle="#72ead3",o.beginPath(),o.moveTo(R-5,0),o.lineTo(R+5,0),o.lineTo(R,7),o.closePath(),o.fill(),o.lineWidth=1}/**
 * @license lucide v0.468.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Su=(i,t,e=[])=>{const n=document.createElementNS("http://www.w3.org/2000/svg",i);return Object.keys(t).forEach(s=>{n.setAttribute(s,String(t[s]))}),e.length&&e.forEach(s=>{const r=Su(...s);n.appendChild(r)}),n};var Mv=([i,t,e])=>Su(i,t,e);/**
 * @license lucide v0.468.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Sv=i=>Array.from(i.attributes).reduce((t,e)=>(t[e.name]=e.value,t),{}),bv=i=>typeof i=="string"?i:!i||!i.class?"":i.class&&typeof i.class=="string"?i.class.split(" "):i.class&&Array.isArray(i.class)?i.class:"",Ev=i=>i.flatMap(bv).map(e=>e.trim()).filter(Boolean).filter((e,n,s)=>s.indexOf(e)===n).join(" "),Tv=i=>i.replace(/(\w)(\w*)(_|-|\s*)/g,(t,e,n)=>e.toUpperCase()+n.toLowerCase()),Th=(i,{nameAttr:t,icons:e,attrs:n})=>{var g;const s=i.getAttribute(t);if(s==null)return;const r=Tv(s),a=e[r];if(!a)return console.warn(`${i.outerHTML} icon name was not found in the provided icons object.`);const o=Sv(i),[c,l,h]=a,u={...l,"data-lucide":s,...n,...o},d=Ev(["lucide",`lucide-${s}`,o,n]);d&&Object.assign(u,{class:d});const p=Mv([c,u,h]);return(g=i.parentNode)==null?void 0:g.replaceChild(p,i)};/**
 * @license lucide v0.468.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Ri={xmlns:"http://www.w3.org/2000/svg",width:24,height:24,viewBox:"0 0 24 24",fill:"none",stroke:"currentColor","stroke-width":2,"stroke-linecap":"round","stroke-linejoin":"round"};/**
 * @license lucide v0.468.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const wv=["svg",Ri,[["path",{d:"m11 17-5-5 5-5"}],["path",{d:"m18 17-5-5 5-5"}]]];/**
 * @license lucide v0.468.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Av=["svg",Ri,[["path",{d:"m6 17 5-5-5-5"}],["path",{d:"m13 17 5-5-5-5"}]]];/**
 * @license lucide v0.468.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Cv=["svg",Ri,[["circle",{cx:"12",cy:"12",r:"3"}],["path",{d:"M3 7V5a2 2 0 0 1 2-2h2"}],["path",{d:"M17 3h2a2 2 0 0 1 2 2v2"}],["path",{d:"M21 17v2a2 2 0 0 1-2 2h-2"}],["path",{d:"M7 21H5a2 2 0 0 1-2-2v-2"}]]];/**
 * @license lucide v0.468.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Rv=["svg",Ri,[["path",{d:"m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2"}]]];/**
 * @license lucide v0.468.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Pv=["svg",Ri,[["rect",{x:"14",y:"4",width:"4",height:"16",rx:"1"}],["rect",{x:"6",y:"4",width:"4",height:"16",rx:"1"}]]];/**
 * @license lucide v0.468.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Lv=["svg",Ri,[["polygon",{points:"6 3 20 12 6 21 6 3"}]]];/**
 * @license lucide v0.468.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Nv=["svg",Ri,[["line",{x1:"18",x2:"18",y1:"20",y2:"4"}],["polygon",{points:"14,20 4,12 14,4"}]]];/**
 * @license lucide v0.468.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Dv=["svg",Ri,[["line",{x1:"6",x2:"6",y1:"4",y2:"20"}],["polygon",{points:"10,4 20,12 10,20"}]]];/**
 * @license lucide v0.468.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Iv=({icons:i={},nameAttr:t="data-lucide",attrs:e={}}={})=>{if(!Object.values(i).length)throw new Error(`Please provide an icons object.
If you want to use all the icons you can import it like:
 \`import { createIcons, icons } from 'lucide';
lucide.createIcons({icons});\``);if(typeof document>"u")throw new Error("`createIcons()` only works in a browser environment.");const n=document.querySelectorAll(`[${t}]`);if(Array.from(n).forEach(s=>Th(s,{nameAttr:t,icons:i,attrs:e})),t==="data-lucide"){const s=document.querySelectorAll("[icon-name]");s.length>0&&(console.warn("[Lucide] Some icons were found with the now deprecated icon-name attribute. These will still be replaced for backwards compatibility, but will no longer be supported in v1.0 and you should switch to data-lucide"),Array.from(s).forEach(r=>Th(r,{nameAttr:"icon-name",icons:i,attrs:e})))}},Uv=()=>`
  <header class="topbar">
    <div class="brand"><div><strong>Motion Review</strong><span id="motionLabel">No motion</span></div></div>
    <div class="top-actions">
      <button class="icon" id="openMotionBtn" title="打开轨迹／回放／报告" aria-label="打开轨迹／回放／报告"><i data-lucide="folder-open"></i></button>
      <button class="icon" id="focusBtn" title="Frame scene"><i data-lucide="focus"></i></button>
    </div>
  </header>`,Fv=()=>`
  <section class="viewport">
    <canvas id="scene"></canvas>
    <div class="viewport-hud"><strong id="viewportMotion">No motion loaded</strong><span id="viewportFrame">Frame 0</span><span id="viewportTime">0.00 s</span></div>
    <div class="shortcut-hint"><span>Space</span> Play <span>← →</span> Frame <span><span>[ ]</span> Cut</span></div>
    <div id="sceneStatus" class="scene-status">Select a motion asset</div>
    <button id="retryLoad" class="retry-load" hidden>Retry loading</button>
  </section>`,Ov=()=>`
  <div class="tab-page active" data-page="contact">
    <div class="page-heading"><strong id="contactHeading">Playback</strong></div>
    <section id="collectionPanel" hidden>
      <label class="stack-field"><span>案例</span><select id="reviewCase" aria-label="案例"></select></label>
      <label class="stack-field"><span>参考轨迹</span><select id="reviewReference" aria-label="参考轨迹"></select></label>
      <label class="stack-field"><span>对比方式</span><select id="reviewLayout" aria-label="对比方式"><option value="beside">并排：参考在 −Y，当前在 +Y</option><option value="overlay">原坐标叠加：蓝色为参考</option></select></label>
      <p id="comparisonClock"></p>
      <label class="stack-field"><span>片段</span><select id="reviewSegment" aria-label="片段"></select></label>
      <p id="reviewMetrics"></p><button id="reviewWorst">定位最大穿透帧</button>
    </section>
    <section id="reviewPanel" hidden></section>
    <p id="playbackNotice" class="playback-notice" hidden>只读轨迹：接触状态以已验证的 Newton 数据为准。</p>
  </div>`,kv=()=>`
  <div class="tab-page" data-page="display">
    <div class="page-heading"><strong>Display</strong></div>
    <section class="display-section"><div class="section-heading">Scene</div><div class="layer-list">
      <label class="toggle"><input id="terrainToggle" type="checkbox" checked /><span>Terrain mesh</span></label><label class="toggle"><input id="surfaceToggle" type="checkbox" /><span>Contact surfaces</span></label><label class="toggle"><input id="contactSampleToggle" type="checkbox" checked /><span>Contact point + force</span></label><label class="toggle"><input id="rootPathToggle" type="checkbox" /><span>Root path</span></label>
    </div></section>
    <section class="display-section"><div class="section-heading">Camera / Terrain</div>
      <button id="focusRobot">聚焦机器人</button>
      <label class="toggle"><input id="followRobot" type="checkbox" /><span>跟随当前机器人</span></label>
      <label class="slider-field"><span>地形透明度</span><input id="terrainOpacity" aria-label="地形透明度" type="range" min="0.05" max="1" step="0.05" value="1" /></label>
    </section>
    <details id="displayOptions" class="compact-options"><summary>显示细节</summary>
      <label class="slider-field"><span>Force scale</span><input id="forceScale" type="range" min="25" max="300" value="100" /></label>
      <div id="partLegend" class="part-legend"></div>
    </details>
  </div>`,Bv=()=>`
  <aside class="inspector">
    <div class="inspector-tabs"><button data-tab="contact" class="active">Contact</button><button data-tab="display">Display</button></div>
    <div class="inspector-pages">${Ov()}${kv()}</div>
    <details class="recent-motion-dock" aria-label="Motions opened this session"><summary class="recent-motion-head"><span>最近打开</span><small id="recentMotionCount">0</small></summary><div id="recentMotionList" class="recent-motion-list"></div></details>
  </aside>`,zv=()=>`
  <section class="timeline-panel">
    <div id="timelineResizer" class="timeline-resizer" title="Drag to resize timeline"><span></span></div>
    <div class="transport"><div class="toolbar-group cut-nav" role="group" aria-label="切分点导航"><button class="icon" id="prevCut" title="上一切分点（到开头后循环）" aria-label="上一切分点"><i data-lucide="chevrons-left"></i></button><span>切分点</span><button class="icon" id="nextCut" title="下一切分点（到末尾后循环）" aria-label="下一切分点"><i data-lucide="chevrons-right"></i></button></div><div class="transport-main"><div class="toolbar-group playback-nav"><button class="icon" id="prevBtn" title="Previous frame"><i data-lucide="step-back"></i></button><button class="icon play" id="playBtn" title="Play or pause"><i data-lucide="play"></i></button><button class="icon" id="nextBtn" title="Next frame"><i data-lucide="step-forward"></i></button></div><div class="frame-readout"><input id="frameInput" class="frame-input" type="number" min="0" value="0" /><span id="frameTotal">/ 0</span></div></div><label class="fps">FPS <input id="fpsInput" type="number" min="1" max="240" value="50" /></label></div>
    <div class="timeline-wrap"><canvas id="timeline" aria-label="Contact timeline"></canvas><div id="timelineHover" class="timeline-hover hidden"></div></div>
  </section>`;function Vv(i){const t=document.createElement("template");t.innerHTML=`${Uv()}<main class="workspace">${Fv()}${Bv()}${zv()}</main>
    <dialog id="motionPicker" aria-labelledby="motionPickerTitle"><div class="motion-picker-head"><strong id="motionPickerTitle">打开轨迹／回放／报告</strong><button id="closeMotionPicker" aria-label="Close motion picker">关闭</button></div>
      <form id="reviewOpenForm" class="review-open-form"><label for="reviewPath">文件路径</label><input id="reviewPath" required aria-label="回放或案例清单路径" placeholder="NPZ / 报告 / manifest.json / cases.json" /><input id="reviewTerrain" aria-label="地形路径" placeholder="地形 OBJ 路径（可选）" /><button id="reviewOpen" type="submit">打开文件</button></form>
      <label for="motionSearch">已注册轨迹 · 搜索名称或 ID</label><input id="motionSearch" type="search" autocomplete="off" placeholder="搜索轨迹" /><div id="motionPickerList"></div>
    </dialog>`,i.replaceChildren(t.content.cloneNode(!0)),bu()}function bu(){Iv({icons:{Play:Lv,Pause:Pv,StepBack:Nv,StepForward:Dv,Focus:Cv,FolderOpen:Rv,ChevronsLeft:wv,ChevronsRight:Av}})}function ct(i){const t=document.getElementById(i);if(!t)throw new Error(`missing UI element #${i}`);return t}function Hv(){const i=[...document.querySelectorAll("[data-tab]")],t=[...document.querySelectorAll("[data-page]")];for(const e of i)e.addEventListener("click",()=>{const n=e.dataset.tab;for(const s of i)s.classList.toggle("active",s===e);for(const s of t)s.classList.toggle("active",s.dataset.page===n)})}class Gv{constructor(t,e){oe(this,"group",new je);oe(this,"annotations",new je);oe(this,"reference",null);oe(this,"current",null);oe(this,"review");oe(this,"selected");oe(this,"spacing",1);oe(this,"link","left_ankle_roll_link");oe(this,"normals",!0);oe(this,"displacement",!0);oe(this,"scale",3);oe(this,"panel");oe(this,"details");this.scene=t,this.onPose=e,this.panel=document.getElementById("reviewPanel"),this.details=document.createElement("pre"),this.details.style.whiteSpace="pre-wrap",this.scene.add(this.group,this.annotations)}clearAnnotations(){Rs(this.annotations)}setup(t,e,n,s){var R,C;if(this.group.clear(),this.clearAnnotations(),this.panel.replaceChildren(),this.review=t,this.current=e,this.reference=null,this.selected=void 0,this.panel.hidden=!t,!t||t.kind==="support_motion"||t.kind==="support_slip")return;const r=document.createElement("p");if(r.textContent=t.notice,this.panel.append(r),t.kind!=="report"||!((R=t.records)!=null&&R.length))return;const a=t.records,o=a.find(x=>x.label===t.reference)||a[0];this.reference=e.clone(),this.spacing=n?Math.max(1,new Fn().setFromObject(n).getSize(new F).y/2+.5):1;const c=o.q;if(this.reference.position.set(c[0],c[1]-this.spacing,c[2]),this.reference.quaternion.set(c[4],c[5],c[6],c[3]),s.forEach((x,N)=>this.reference.setJointValue(x,c[7+N])),this.group.add(this.reference),n){const x=n.clone();x.position.y-=this.spacing,this.group.add(x),n.position.y+=this.spacing}const l=document.createElement("p");l.textContent="左：原始姿态　右：选中结果",this.panel.append(l);const h=document.createElement("select"),u=document.createElement("select");h.setAttribute("aria-label","实验方法"),h.id="reviewMethod",u.setAttribute("aria-label","保存的迭代"),u.id="reviewIteration",[...new Set(a.map(x=>x.method))].forEach(x=>h.add(new Option(x,x)));const d=()=>{this.selected=a.find(x=>x.label===u.value),this.onPose(this.selected.q)},p=()=>{u.replaceChildren();const x=a.filter(P=>P.method===h.value),N=P=>P.iteration==="initial"?-1:P.iteration==="final"?1/0:Number(P.iteration.replace("step",""))||0;x.sort((P,L)=>N(P)-N(L)),x.forEach(P=>u.add(new Option(P.iteration,P.label))),u.value=(x.find(P=>P.iteration==="final")||x[0]).label,d()};h.onchange=p,u.onchange=d;const g=(x,N)=>{const P=document.createElement("label");P.className="stack-field";const L=document.createElement("span");L.textContent=x,P.append(L,N),this.panel.append(P)};g("实验方法",h),g("保存的迭代",u);const _=document.createElement("select");_.id="reviewLink",_.setAttribute("aria-label","位移部位"),Object.keys(e.links).forEach(x=>_.add(new Option(x,x))),_.value=e.links[this.link]?this.link:Object.keys(e.links)[0],this.link=_.value,_.onchange=()=>{this.link=_.value,this.onPose(this.selected.q)},g("位移部位",_);const m=(x,N,P)=>{const L=document.createElement("label");L.className="toggle";const I=document.createElement("input");I.type="checkbox",I.checked=N,I.onchange=()=>{P(I.checked),this.onPose(this.selected.q)},L.append(I,document.createTextNode(x)),this.panel.append(L)};m("穿透法线（报告记录）",this.normals,x=>this.normals=x),m("材料点实际位移 dx",this.displacement,x=>this.displacement=x);const f=document.createElement("input");f.type="range",f.min="1",f.max="20",f.value=String(this.scale),f.setAttribute("aria-label","位移显示倍率"),f.oninput=()=>{this.scale=Number(f.value),this.onPose(this.selected.q)},g("位移显示倍率",f),this.panel.append(this.details),h.value=((C=a.find(x=>x.label!==o.label))==null?void 0:C.method)||o.method,p()}update(){var l;if(!this.reference||!this.current||!this.selected||!((l=this.review)!=null&&l.records))return;this.current.position.y+=this.spacing,this.current.updateMatrixWorld(!0),this.reference.updateMatrixWorld(!0),this.clearAnnotations();const t=this.review.records.find(h=>h.label===this.review.reference)||this.review.records[0];[t,this.selected].forEach((h,u)=>{const d=u===0?-this.spacing:this.spacing;this.normals&&h.normal_starts.forEach((_,m)=>{const f=new F(...h.normals[m]);f.length()>0&&this.annotations.add(new rr(f.normalize(),new F(_[0],_[1]+d,_[2]),.08,16733525,.015,.008))});const p=h.diagnostics.contact_position_w,g=h.diagnostics.actual_contact;p==null||p.forEach((_,m)=>{if(!(g!=null&&g[m])||!_.every(Number.isFinite))return;const f=new Ae(new ji(.012,8,6),new Un({color:11184810}));f.position.set(_[0],_[1]+d,_[2]),this.annotations.add(f)})});const e=this.reference.links[this.link],n=this.current.links[this.link],s=[],r=[];e==null||e.traverse(h=>{h.isMesh&&s.push(h)}),n==null||n.traverse(h=>{h.isMesh&&r.push(h)});const a=new F;let o=0;s.forEach((h,u)=>{const d=h.geometry.getAttribute("position"),p=r[u];if(!(!d||!p))for(let g=0;g<d.count;g+=Math.max(1,Math.ceil(d.count/12))){const _=new F().fromBufferAttribute(d,g).applyMatrix4(h.matrixWorld);_.y+=2*this.spacing;const f=new F().fromBufferAttribute(d,g).applyMatrix4(p.matrixWorld).sub(_);a.add(f),o++;const R=f.length();this.displacement&&R>1e-8&&this.annotations.add(new rr(f.normalize(),_,R*this.scale,4255632,Math.min(.025,R*this.scale/4),.01))}}),o&&a.multiplyScalar(100/o);const c=h=>{const u=h.diagnostics,d=p=>typeof p=="number"?p.toFixed(3):"未知";return`${h.label}
全身负距离: ${d(u.full_depth_cm)} cm；双足: ${d(u.feet_depth_cm)} cm
验收（报告）: ${u.static_recovery_pass??"未知"}；保留（报告）: ${u.all_required_kept??"未知"}`};this.details.textContent=`${c(t)}

${c(this.selected)}

灰点：报告中的历史接触标记
${this.link} 材料点平均 dx (cm): ${o?a.toArray().map(h=>h.toFixed(3)).join(", "):"无网格采样"}
箭头显示倍率: ${this.scale}×`}bounds(t){this.reference&&t.expandByObject(this.group)}terrainVisible(t){this.group.children.forEach(e=>{e!==this.reference&&(e.visible=t)})}}class Wv{constructor(t,e,n){oe(this,"group",new je);oe(this,"phases",[]);oe(this,"selected",0);oe(this,"marker",null);oe(this,"details",null);oe(this,"rawGroup",new je);oe(this,"showRaw",!0);this.scene=t,this.jump=e,this.focus=n,t.add(this.group)}clear(){Rs(this.group),this.rawGroup.clear(),this.marker=null}setup(t){this.clear(),this.phases=(t==null?void 0:t.kind)==="support_motion"?t.phases??[]:[],this.selected=0;const e=document.getElementById("reviewPanel");if(!this.phases.length)return;e.hidden=!1;const n=document.createElement("p");n.textContent=t.notice,e.append(n);const s=document.createElement("label");s.className="stack-field";const r=document.createElement("span");r.textContent="持续接触阶段";const a=document.createElement("select");a.setAttribute("aria-label","持续接触阶段"),this.phases.forEach((l,h)=>a.add(new Option(`${l.start}–${l.end} ${l.part}`,String(h)))),a.onchange=()=>{this.selected=Number(a.value),this.draw(),this.jump(this.phases[this.selected].start),this.focusSelected()},s.append(r,a),e.append(s);const o=document.createElement("label");o.className="toggle";const c=document.createElement("input");c.type="checkbox",c.checked=this.showRaw,c.onchange=()=>{this.showRaw=c.checked,this.rawGroup.visible=this.showRaw},o.append(c,document.createTextNode("显示原始 rollout 轨迹")),e.append(o),this.details=document.createElement("p"),e.append(this.details),this.draw()}dot(t,e,n){const s=new Ae(new ji(n,10,8),new Un({color:e,depthTest:!1}));return s.position.set(...t),s.renderOrder=20,this.group.add(s),s}draw(){this.clear();const t=this.phases[this.selected];if(!t)return;const e=t.part==="left_foot"?15754313:3254232,n=l=>l.map(h=>new F(...h));this.group.add(this.rawGroup),t.rollouts.forEach(l=>{const h=new dr(new Ie().setFromPoints(n(l)),new zn({color:e,transparent:!0,opacity:.12,depthTest:!1,depthWrite:!1}));h.renderOrder=15,this.rawGroup.add(h)}),this.rawGroup.visible=this.showRaw;const s=n(t.aggregate),r=new S0;for(let l=1;l<s.length;l++)s[l].distanceToSquared(s[l-1])>1e-12&&r.add(new ou(s[l-1],s[l]));if(r.curves.length){const l=new Ae(new Pl(r,Math.max(8,s.length*2),.0015,6,!1),new Un({color:e,depthTest:!1,depthWrite:!1}));l.renderOrder=16,this.group.add(l)}else this.dot(t.aggregate[0],e,.003);const a=s[0],c=s.at(-1).clone().sub(a);if(c.z=0,c.length()>1e-6){const l=new rr(c.clone().normalize(),a,c.length(),e,Math.min(.004,c.length()/6),.0015);l.traverse(h=>{h.material&&(h.material.depthTest=!1)}),l.renderOrder=18,this.group.add(l)}this.marker=this.dot(t.aggregate[0],16777215,.0025),this.details&&(this.details.textContent=`聚合净位移 ${t.aggregate_net_cm.toFixed(2)} cm；原始 rollout 中位数 ${t.rollout_median_net_cm.toFixed(2)} cm。亮色宽线：聚合；淡色细线：原始执行；同一固定材料点。`)}update(t){const e=this.phases[this.selected];!e||!this.marker||(this.marker.visible=t>=e.start&&t<=e.end,this.marker.visible&&this.marker.position.set(...e.aggregate[t-e.start]))}focusSelected(){const t=this.phases[this.selected];t&&this.focus(t.aggregate)}}class Xv{constructor(t,e){oe(this,"group",new je);oe(this,"review");oe(this,"gain",20);oe(this,"frame",0);oe(this,"details",null);this.focus=e,t.add(this.group)}clear(){Rs(this.group)}setup(t){if(this.clear(),this.review=t,this.details=null,(t==null?void 0:t.kind)!=="support_slip")return;const e=document.getElementById("reviewPanel");e.hidden=!1;const n=document.createElement("p");n.textContent=t.notice,e.append(n);const s=document.createElement("label");s.className="stack-field",s.append(document.createTextNode("位移箭头倍率（不改变统计值）"));const r=document.createElement("input");r.type="number",r.min="1",r.max="100",r.value=String(this.gain),r.onchange=()=>{this.gain=Math.max(1,Math.min(100,Number(r.value)||1)),this.update(this.frame)},s.append(r),e.append(s);const a=document.createElement("button");a.textContent="聚焦当前接触点",a.onclick=()=>{var c,l;const o=((l=(c=this.review)==null?void 0:c.slip_frames)==null?void 0:l[this.frame])??[];o.length&&this.focus(o.map(h=>new F(...h.position)))},e.append(a),this.details=document.createElement("p"),e.append(this.details)}update(t){var h,u,d;if(this.frame=t,this.clear(),((h=this.review)==null?void 0:h.kind)!=="support_slip")return;const e=((u=this.review.slip_frames)==null?void 0:u[t])??[],n=["左脚","右脚","左手","右手","左膝","右膝"],s=p=>`${p.part}:${p.surface}`,r=new Map;e.forEach(p=>r.set(s(p),Math.min(r.get(s(p))??1/0,p.motion_cm))),e.forEach(p=>{const g=p.motion_cm===r.get(s(p))?3594650:15967552,_=new F(...p.position),m=new F(...p.delta),f=new Ae(new ji(.0015,8,6),new Un({color:g,depthTest:!1}));f.position.copy(_),f.renderOrder=25,this.group.add(f);const R=m.length()*this.gain;if(R>1e-7){const C=new rr(m.normalize(),_,R,g,Math.min(.004,R*.25),Math.min(.002,R*.12));C.traverse(x=>{x.material&&(x.material.depthTest=!1),x.renderOrder=25}),this.group.add(C)}});const a=[...r].map(([p,g])=>{const[_,m]=p.split(":").map(Number),f=e.filter(x=>s(x)===p).map(x=>x.motion_cm).sort((x,N)=>x-N),R=Math.floor(f.length/2),C=f.length%2?f[R]:(f[R-1]+f[R])/2;return`${n[_]}·面${m}：最低几何运动 ${g.toFixed(3)} / 中位 ${C.toFixed(3)} / 最大 ${f.at(-1).toFixed(3)} cm`}),o=(d=this.review.support_frames)==null?void 0:d[t],c={no_contact:"无有效接触",unloaded:"接触但卸载",load_bearing:"承重",unknown:"未知"},l=o?o.map(p=>{var g,_;return`${n[p.part]} ${c[p.load_state]??"未知"}：${((g=p.normal_force_n)==null?void 0:g.toFixed(1))??"未知"} N，承重点相对切向速度 ${((_=p.loaded_speed_cm_s)==null?void 0:_.toFixed(2))??"未知"} cm/s`}).join("；"):"承重滑移：未知";this.details&&(this.details.textContent=(e.length?`${t}→${t+1}；${a.join("；")}。`:"此帧无可测几何区间，不能记作零滑移。")+l+"。")}}class qv{constructor(t){oe(this,"reference",null);oe(this,"terrain",null);oe(this,"currentTerrain",null);oe(this,"data");oe(this,"materials",[]);oe(this,"spacing",1);oe(this,"mode","beside");oe(this,"terrainShown",!0);oe(this,"opacity",1);oe(this,"available",!0);oe(this,"referenceFrame",null);this.scene=t}clear(){this.reference&&this.scene.remove(this.reference),this.materials.forEach(t=>t.dispose()),this.materials=[],this.terrain&&(this.scene.remove(this.terrain),xi(this.terrain)),this.currentTerrain&&(this.currentTerrain.position.y=0),this.reference=null,this.terrain=null,this.currentTerrain=null,this.data=void 0,this.referenceFrame=null}setup(t,e,n,s){if(this.clear(),this.data=t,this.currentTerrain=n,!t)return;this.reference=e.clone(),this.terrain=s,this.reference.traverse(a=>{if(!a.isMesh)return;const o=c=>{var h;const l=c.clone();return(h=l.color)==null||h.set(5815528),this.materials.push(l),l};a.material=Array.isArray(a.material)?a.material.map(o):o(a.material)});const r=a=>a?new Fn().setFromObject(a).getSize(new F).y:0;this.spacing=Math.max(1,(r(n)+r(s))/4+.5),this.scene.add(this.reference),this.terrain&&this.scene.add(this.terrain),this.setMode(this.mode)}setMode(t){this.mode=t,this.materials.forEach(e=>{e.transparent=t==="overlay",e.opacity=t==="overlay"?.4:1,e.depthWrite=t!=="overlay"}),this.currentTerrain&&(this.currentTerrain.position.y=this.offset)}get layout(){return this.mode}get offset(){return this.data&&this.mode==="beside"?this.spacing:0}update(t,e){if(!this.data||!this.reference)return;e.position.y+=this.offset;const n=(this.data.qpos.length-1)/this.data.fps;if(this.available=t<=n+1e-8,this.reference.visible=this.available,this.referenceFrame=this.available?Math.min(this.data.qpos.length-1,Math.round(t*this.data.fps)):null,this.referenceFrame!==null){const s=this.data.qpos[this.referenceFrame];this.reference.position.set(s[0],s[1]-this.offset,s[2]),this.reference.quaternion.set(s[4],s[5],s[6],s[3]),this.data.joint_names.forEach((r,a)=>this.reference.setJointValue(r,s[7+a])),this.reference.updateMatrixWorld(!0)}this.terrain&&(this.terrain.position.y=-this.offset),this.display(this.terrainShown,this.opacity)}display(t,e){this.terrainShown=t,this.opacity=e,this.terrain&&(this.terrain.visible=t&&this.available&&(this.mode==="beside"||!this.currentTerrain),this.terrain.traverse(n=>{if(!n.isMesh)return;(Array.isArray(n.material)?n.material:[n.material]).forEach(r=>{r.transparent=e<1,r.opacity=e,r.depthWrite=e===1})}))}bounds(t,e=!0){var n,s;(n=this.reference)!=null&&n.visible&&t.expandByObject(this.reference),e&&((s=this.terrain)!=null&&s.visible)&&t.expandByObject(this.terrain)}}const Eu=document.querySelector("#app");if(!Eu)throw new Error("missing #app root");Vv(Eu);Hv();const Vn=document.querySelector("#scene"),_n=document.querySelector("#timeline"),wi=new s0({canvas:Vn,antialias:!0,preserveDrawingBuffer:!0});wi.setPixelRatio(Math.min(window.devicePixelRatio,2));wi.outputColorSpace=Pe;wi.toneMapping=Ch;wi.toneMappingExposure=1.08;wi.shadowMap.enabled=!1;const ke=new iu;ke.background=new $t(725014);ke.fog=new wl(725014,9,20);ke.up.set(0,0,1);const qe=new Je(42,1,.01,100);qe.up.set(0,0,1);qe.position.set(3.2,-4.5,2.7);const bn=new q0(qe,Vn);bn.target.set(0,0,.9);bn.enableDamping=!0;ke.add(new F0(16055295,1515562,2.4));const Tu=new pu(16777215,3.1);Tu.position.set(3,-4,7);ke.add(Tu);const wu=new G0(14,28,3358796,1910576);wu.rotation.x=Math.PI/2;ke.add(wu);let Tt=null,ze=null,Fe=null;const Au=new Xv(ke,i=>{const t=new Fn().setFromPoints(i).getBoundingSphere(new oi),e=qe.position.clone().sub(bn.target).normalize();bn.target.copy(t.center),qe.position.copy(t.center).addScaledVector(e,Math.max(.45,t.radius*5)),bn.update()});let ar=new je;ke.add(ar);let Ai=new je;ke.add(Ai);let pn=null,xe=0,nr=!1,wh=performance.now(),Js=0,or=!1,Ps=!1,Ls=!1,Ns=!1,Ei=!1,ha=null,xa=[];function mr(){return Ls||Ns||Ei}async function jv(i){if(mr())return null;Ns=!0,Xi();try{return await i()}catch(t){return ai(String(t),!0),null}finally{Ns=!1,Xi()}}const lr=new Gv(ke,i=>{Tt&&(Tt.qpos=[i],vn(0))}),Hn=new qv(ke),fl=new Wv(ke,i=>vn(i),i=>{const t=i.map(r=>new F(...r)),e=new Fn().setFromPoints(t).getCenter(new F),n=qe.position.clone().sub(bn.target).normalize(),s=t.at(-1).clone().sub(t[0]);s.z=0,bn.target.copy(e),qe.position.copy(e).addScaledVector(n,s.length()<.05?.48:.95),bn.update()}),ya=[1619620,6869905,15754313,16758087,5213439,11958502,3254232,14703261];let ua=NaN;try{ua=Number(localStorage.getItem("motion-edit.timeline-height"))}catch{}let xs=Number.isFinite(ua)&&ua>0?ua:window.innerWidth<=760?200:230;function Dl(i,t=!1){if(xs=Math.max(150,Math.min(window.innerHeight*.48,Math.round(i))),document.documentElement.style.setProperty("--timeline-height",`${xs}px`),t)try{localStorage.setItem("motion-edit.timeline-height",String(xs))}catch{}Vl()}Dl(xs);async function pl(){const i=await Cs("/api/recent-motions"),t=ct("recentMotionList");ct("recentMotionCount").textContent=String(i.items.length);const e=i.items.map(n=>{const s=document.createElement("button");s.type="button",s.className=`recent-motion-item${n.motion_id===i.active_motion_id?" active":""}`,s.dataset.motionKey=n.motion_id,s.setAttribute("aria-pressed",String(n.motion_id===i.active_motion_id));const r=document.createElement("span");r.className="recent-motion-marker";const a=document.createElement("span");a.className="recent-motion-copy";const o=document.createElement("strong");o.textContent=n.label;const c=document.createElement("small");return c.textContent=n.provenance,a.append(o,c),s.append(r,a),s.onclick=()=>{Il(n.motion_id)},s});return t.replaceChildren(...e),i.items}function Yv(i){document.querySelectorAll(".recent-motion-item").forEach(t=>{const e=t.dataset.motionKey===i;t.classList.toggle("active",e),t.setAttribute("aria-pressed",String(e))})}async function Il(i){if(!(mr()||(Tt==null?void 0:Tt.motion_id)===i)){Ls=!0,On(!1),Xi(),document.querySelectorAll(".recent-motion-item").forEach(t=>t.disabled=!0);try{ai("Opening motion",!1,!0);const t=await Cs("/api/session/load",{method:"POST",body:JSON.stringify({motion_id:i})});xe=0,await Ul(t,!0),await pl(),cr()}catch(t){ai(String(t),!0)}finally{Ls=!1,Xi()}}}async function Cu(){var t,e;ai("Loading motions",!1,!0),ct("retryLoad").hidden=!0,xa=(await Cs("/api/motions")).motions;let i=null;try{i=await Cs("/api/session")}catch(n){if(!(n instanceof xu)||n.status!==404)throw n}if(i)await Ul(i,!0),await pl(),cr();else{const s=((t=(await pl())[0])==null?void 0:t.motion_id)||((e=xa.find(r=>r.ready))==null?void 0:e.motion_id);s?await Il(s):ai("No motion available. Open a registered motion or a playback file.",!1,!0)}}function Ru(){const i=ct("motionSearch").value.trim().toLowerCase(),t=xa.filter(n=>`${n.motion_id} ${n.label} ${n.provenance}`.toLowerCase().includes(i)),e=ct("motionPickerList");e.replaceChildren();for(const n of t){const s=document.createElement("button");s.className="motion-picker-item",s.disabled=!n.ready;const r=document.createElement("strong");r.textContent=n.label;const a=document.createElement("small");a.textContent=`${n.motion_id} · ${n.provenance} · ${n.ready?"可打开":"文件缺失"}`,s.append(r,a),s.onclick=()=>{ct("motionPicker").close(),Il(n.motion_id)},e.append(s)}t.length||(e.textContent="没有匹配的轨迹")}ct("openMotionBtn").onclick=async()=>{if(!mr())try{xa=(await Cs("/api/motions")).motions,ct("motionSearch").value="",Ru(),ct("motionPicker").showModal(),ct("motionSearch").focus()}catch(i){ai(String(i),!0)}};ct("closeMotionPicker").onclick=()=>ct("motionPicker").close();ct("motionSearch").oninput=Ru;ct("retryLoad").onclick=()=>{Cu().catch(Pu)};function Pu(i){ai(String(i),!0),ct("retryLoad").hidden=!1}async function Ul(i,t=!1){var s,r,a,o;const e=Tt;Tt=i,xe=Math.min(xe,i.qpos.length-1),t&&((s=i.review)==null?void 0:s.kind)==="support_motion"&&((r=i.review.phases)!=null&&r.length)&&(xe=i.review.phases[0].start),document.body.classList.toggle("review-mode",!!i.review),document.body.classList.toggle("playback-only",i.read_only),ct("contactHeading").textContent=i.read_only?"Playback":"Contacts",ct("playbackNotice").hidden=!i.read_only||!!((a=i.review)!=null&&a.notice),i.contact_force.mask_status&&i.contact_force.mask_status!=="verified_newton_primary_faces"?(ct("playbackNotice").hidden=!!((o=i.review)!=null&&o.notice),ct("playbackNotice").textContent="接触状态未知：力与旧掩码未经同帧 Newton／主表面验证。"):ct("playbackNotice").textContent="只读轨迹：接触状态以已验证的 Newton 数据为准。",Yv(i.motion_id);const n=i.motion_id;if(ct("motionLabel").textContent=n,ct("viewportMotion").textContent=n,(t||!e)&&(ct("fpsInput").value=String(i.fps)),ix(),kl(),t)try{await $v(i)}catch(c){throw Tt=null,On(!1),ct("retryLoad").hidden=!1,Xi(),c}vn(xe)}async function $v(i){var t,e,n,s;ai("Loading scene",!1,!0),Ei=!0,Xi();try{const r=new uu,a=new Promise((g,_)=>{r.onLoad=()=>g(),r.onError=m=>_(new Error(`failed to load robot asset: ${m}`))}),o=new gv(r);o.loadMeshCb=(g,_,m)=>{if(g.toLowerCase().endsWith(".obj")){new bo(_).load(g,f=>m(f),void 0,f=>m(new je,f));return}m(new je,new Error(`unsupported URDF mesh: ${g}`))};const[c,l]=await Promise.allSettled([o.loadAsync(i.robot_urdf_url),a]);if(c.status==="rejected"||l.status==="rejected")throw c.status==="fulfilled"&&xi(c.value),c.status==="rejected"?c.reason:l.reason;const h=c.value;let u=null,d=null;try{i.terrain_obj_url&&(u=await new bo().loadAsync(i.terrain_obj_url)),(e=(t=i.review)==null?void 0:t.comparison)!=null&&e.terrain_obj_url&&(d=await new bo().loadAsync(i.review.comparison.terrain_obj_url))}catch(g){throw xi(h),u&&xi(u),g}lr.setup(void 0,null,null,[]),Hn.clear(),ze&&(ke.remove(ze),xi(ze)),Fe&&(ke.remove(Fe),xi(Fe)),Rs(ar),Rs(Ai),pn&&(ke.remove(pn),xi(pn),pn=null),ze=h,Fe=u;let p=0;if(ze.traverse(g=>{g.isMesh&&p++}),Vn.dataset.robotMeshes=String(p),ke.add(ze),Fe){const g=new b0({color:5858672,roughness:.86,metalness:.03}),_=new Set;Fe.traverse(m=>{m.isMesh&&((Array.isArray(m.material)?m.material:[m.material]).forEach(f=>_.add(f)),m.material=g)}),_.forEach(m=>m.dispose()),ke.add(Fe)}Ei=!1,lr.setup(i.review,ze,Fe,i.joint_names),fl.setup(i.review),Au.setup(i.review),Hn.setup((n=i.review)==null?void 0:n.comparison,ze,Fe,d),Qv(),Lu(),Jv(),Kv(),Zv(),Pa(),((s=i.review)==null?void 0:s.kind)==="support_motion"&&fl.focusSelected(),La(),cr()}finally{Ei=!1,Xi()}}function Kv(){if(!Tt)return;const i=Tt.qpos.map(t=>new F(t[0],t[1],t[2]));pn=new dr(new Ie().setFromPoints(i),new zn({color:11450306,transparent:!0,opacity:.45})),pn.visible=ct("rootPathToggle").checked,ke.add(pn)}function Jv(){if(Tt)for(const i of Tt.surfaces){if(i.surface_id==="terrain_ground_z0")continue;const t=new F(...i.normal).normalize().multiplyScalar(.002),e=i.metadata.polygon_world;let n;if(e!=null&&e.length&&e.length>=3)n=e.map(o=>new F(o[0],o[1],o[2]).add(t));else{const o=new F(...i.origin),c=new F(...i.tangent_u),l=new F(...i.tangent_v),[h,u]=i.bounds.u,[d,p]=i.bounds.v;n=[o.clone().addScaledVector(c,h).addScaledVector(l,d),o.clone().addScaledVector(c,u).addScaledVector(l,d),o.clone().addScaledVector(c,u).addScaledVector(l,p),o.clone().addScaledVector(c,h).addScaledVector(l,p)].map(g=>g.add(t))}const s=[];for(let o=1;o<n.length-1;o++)s.push(0,o,o+1);const r=new Ie().setFromPoints(n);r.setIndex(s),r.computeVertexNormals();const a=new Ae(r,new Un({color:5087165,transparent:!0,opacity:.08,side:Pn,depthWrite:!1}));a.userData.surface=i,ar.add(a)}}function Zv(){if(!Tt)return;Rs(Ai),Vn.dataset.contactSample="point-force";const i=Tt.contact_force.part_order.length;Vn.dataset.contactPartSlots=String(i);for(let t=0;t<i;t++){const e=new je,n=new Un({color:ya[t%ya.length],depthTest:!0,transparent:!0}),s=new Ae(new ji(.014,14,10),n),r=new Ae(new Fs(.006,.006,1,8),n),a=new Ae(new Rl(.022,.055,10),n);e.add(s,r,a),e.userData.point=s,e.userData.shaft=r,e.userData.head=a,e.userData.material=n,Ai.add(e)}Fl()}function Fl(){if(!Tt)return;const i=Tt;Ai.visible=ct("contactSampleToggle").checked;const t=i.contact_force.positions[xe]||[],e=i.contact_force.forces[xe]||[],n=i.contact_force.masks[xe]||[],s=i.contact_force.position_valid[xe]||[];let r=0;Ai.children.forEach((a,o)=>{const c=a,l=o<t.length&&o<e.length&&!!n[o]&&!!s[o];if(c.visible=l,!l)return;r++;const h=new F(...e[o]),u=h.length(),d=Number(ct("forceScale").value)/100,p=Math.min(.9,u/1200*d),g=Math.min(.055,p*.35),_=Math.max(.001,p-g);c.position.set(...t[o]),u>1e-5&&c.quaternion.setFromUnitVectors(new F(0,1,0),h.normalize());const m=c.userData.shaft,f=c.userData.head;m.scale.set(1,_,1),m.position.set(0,_/2,0),f.scale.set(1,g/.055,1),f.position.set(0,_+g/2,0),m.visible=u>1e-5,f.visible=u>1e-5}),Vn.dataset.visibleContactParts=String(r)}function Lu(){if(!Tt||!ze||Ei)return;const i=Tt.qpos[xe];if(!i)return;const t=ze.position.clone();if(ze.position.set(i[0],i[1],i[2]),ze.quaternion.set(i[4],i[5],i[6],i[3]),Tt.joint_names.forEach((e,n)=>{var s;return(s=ze.setJointValue)==null?void 0:s.call(ze,e,i[7+n])}),lr.update(),Hn.update(xe/Tt.fps,ze),pn&&(pn.position.y=Hn.offset),ct("followRobot").checked){const e=ze.position.clone().sub(t);qe.position.add(e),bn.target.add(e)}fl.update(xe),Au.update(xe),tx()}function vn(i){Tt!=null&&Tt.qpos.length&&(Number.isFinite(i)||(i=xe),xe=Math.max(0,Math.min(Tt.qpos.length-1,Math.round(i))),ct("frameInput").max=String(Tt.qpos.length-1),ct("frameInput").value=String(xe),ct("frameTotal").textContent=`/ ${Tt.qpos.length-1}`,ct("viewportFrame").textContent=`Frame ${xe}`,ct("viewportTime").textContent=`${(xe/Tt.fps).toFixed(2)} s`,Lu(),Fl(),kl())}function Qv(){var a,o,c;const i=Tt==null?void 0:Tt.review,t=i==null?void 0:i.collection;if(ct("collectionPanel").hidden=!t,!t)return;const e=ct("reviewCase"),n=ct("reviewReference");e.replaceChildren(),n.replaceChildren(new Option("不对比","-1")),t.cases.forEach(l=>{e.add(new Option(l.label,String(l.index))),l.index!==t.case_index&&n.add(new Option(l.label,String(l.index)))}),e.value=String(t.case_index),n.value=String(t.reference_index);const s=ct("reviewLayout");s.value=Hn.layout,s.parentElement.hidden=!i.comparison;const r=ct("reviewSegment");r.replaceChildren(),(a=i.segments)==null||a.forEach(l=>r.add(new Option(`${l.label} · ${l.start}–${l.end}`,String(l.start)))),r.parentElement.hidden=!((o=i.segments)!=null&&o.length),ct("reviewWorst").hidden=!((c=i.depth_mm)!=null&&c.length)}function tx(){var n,s;const i=Tt==null?void 0:Tt.review;if(!(i!=null&&i.collection))return;ct("comparisonClock").textContent=i.comparison?`按时间对齐（起点 t=0，最近采样帧）：当前 ${xe}；参考 ${Hn.referenceFrame??"已结束，不显示"}。不自动匹配接触事件。`:"单轨迹回放",ct("scene").dataset.referenceFrame=String(Hn.referenceFrame??"none");const t=(n=i.depth_mm)==null?void 0:n[xe];ct("reviewMetrics").textContent=t===void 0?"当前帧穿透诊断：未提供":`报告穿透深度：${t.toFixed(2)} mm（非接触或承重判定）`;const e=(s=i.segments)==null?void 0:s.find(r=>xe>=r.start&&xe<=r.end);ct("reviewSegment").value=e?String(e.start):""}async function Ol(i,t,e=0,n,s=!1){return jv(async()=>{On(!1),ai("Loading playback",!1,!0);const r=await Cs("/api/review/load",{method:"POST",body:JSON.stringify({path:i,terrain:t,case_index:e,reference_index:n})});s||(xe=0),await Ul(r,!0),ct("reviewPath").value=i,ct("reviewTerrain").value=t||""})}ct("reviewCase").onchange=()=>{var e;const i=(e=Tt==null?void 0:Tt.review)==null?void 0:e.collection;if(!i)return;const t=Number(ct("reviewCase").value);ct("reviewCase").value=String(i.case_index),Ol(i.path,i.terrain_override,t)};ct("reviewReference").onchange=()=>{var e;const i=(e=Tt==null?void 0:Tt.review)==null?void 0:e.collection;if(!i)return;const t=Number(ct("reviewReference").value);ct("reviewReference").value=String(i.reference_index),Ol(i.path,i.terrain_override,i.case_index,t,!0)};ct("reviewLayout").onchange=()=>{Hn.setMode(ct("reviewLayout").value),vn(xe),Pa()};ct("reviewSegment").onchange=()=>{On(!1),vn(Number(ct("reviewSegment").value))};ct("reviewWorst").onclick=()=>{var t;const i=(t=Tt==null?void 0:Tt.review)==null?void 0:t.depth_mm;i!=null&&i.length&&(On(!1),vn(i.reduce((e,n,s)=>n>i[e]?s:e,0)))};function ex(){return yu(_n,(Tt==null?void 0:Tt.contact_force.part_order.length)??8)}function nx(i,t){return vv(i,t,(Tt==null?void 0:Tt.qpos.length)??1)}function kl(){yv({canvas:_n,session:Tt,frame:xe,stableCuts:Bl(),partColors:ya})}function Bl(){var t,e;if(!Tt)return[];const i=new Set;return(e=(t=Tt.review)==null?void 0:t.segments)==null||e.forEach(n=>{i.add(n.start),i.add(n.end)}),Tt.graph.transitions.forEach(n=>{var r;i.add(n.start_frame),i.add(Math.max(n.start_frame,n.end_frame-1));const s=(r=n.metadata)==null?void 0:r.stable_anchor_frames;Array.isArray(s)&&s.forEach(a=>i.add(Number(a)))}),[...i].filter(n=>Number.isInteger(n)&&n>=0&&n<Tt.qpos.length).sort((n,s)=>n-s)}function ix(){Tt&&(ct("partLegend").innerHTML=Tt.contact_force.part_order.map((i,t)=>`<span><b style="background:#${ya[t].toString(16).padStart(6,"0")}"></b>${i}</span>`).join(""))}function Pa(i=!1){const t=new Fn;if(ze&&t.expandByObject(ze),!i&&(Fe!=null&&Fe.visible)&&t.expandByObject(Fe),i||lr.bounds(t),Hn.bounds(t,!i),t.isEmpty())return;const e=t.getBoundingSphere(new oi),n=Math.min(Mi.degToRad(qe.fov/2),Math.atan(Math.tan(Mi.degToRad(qe.fov/2))*qe.aspect)),s=Math.max(.4,e.radius*1.15/Math.sin(n));bn.target.copy(e.center),qe.position.copy(e.center).addScaledVector(new F(1.8,-2.4,1.35).normalize(),s),qe.far=Math.max(100,s+e.radius*4),qe.updateProjectionMatrix(),ke.fog=null,bn.update()}function cr(){ha!==null&&window.clearTimeout(ha),ha=null,ct("sceneStatus").classList.add("hidden")}function ai(i,t=!1,e=!1){cr();const n=ct("sceneStatus");n.textContent=i,n.classList.remove("hidden"),n.classList.toggle("error",t),n.setAttribute("role",t?"alert":"status"),!t&&!e&&(ha=window.setTimeout(cr,3e3))}function La(){const i=ct("terrainToggle").checked;Fe&&(Fe.visible=i),lr.terrainVisible(i);const t=Number(ct("terrainOpacity").value);Fe==null||Fe.traverse(e=>{e.isMesh&&(Array.isArray(e.material)?e.material:[e.material]).forEach(n=>{n.transparent=t<1,n.opacity=t,n.depthWrite=t===1})}),Hn.display(i,t),ar.visible=ct("surfaceToggle").checked,Ai.visible=ct("contactSampleToggle").checked,pn&&(pn.visible=ct("rootPathToggle").checked),Vn.dataset.terrainVisible=String((Fe==null?void 0:Fe.visible)??!1),Vn.dataset.surfacesVisible=String(ar.visible)}function Xi(){const i=!!mr();document.querySelectorAll(".recent-motion-item,#openMotionBtn,#reviewOpen,#collectionPanel select,#collectionPanel button").forEach(t=>t.disabled=i),document.querySelectorAll("#playBtn,#prevBtn,#nextBtn,#prevCut,#nextCut,#frameInput,#fpsInput").forEach(t=>t.disabled=Ls||Ns||Ei||!Tt),Tt&&Tt.qpos.length<2&&(ct("playBtn").disabled=!0),Bl().length||(ct("prevCut").disabled=!0,ct("nextCut").disabled=!0),ct("scene").setAttribute("aria-busy",String(i))}function On(i){nr=i,nr||(Js=0),ct("playBtn").innerHTML=`<i data-lucide="${nr?"pause":"play"}"></i>`,bu()}ct("playBtn").onclick=()=>On(!nr);ct("prevBtn").onclick=()=>{On(!1),vn(xe-1)};ct("nextBtn").onclick=()=>{On(!1),vn(xe+1)};ct("frameInput").onchange=i=>vn(Number(i.target.value));ct("focusBtn").onclick=()=>Pa();ct("focusRobot").onclick=()=>Pa(!0);ct("terrainOpacity").oninput=La;ct("terrainToggle").onchange=La;ct("surfaceToggle").onchange=La;ct("contactSampleToggle").onchange=i=>Ai.visible=i.target.checked;ct("forceScale").oninput=()=>Fl();ct("rootPathToggle").onchange=i=>{pn&&(pn.visible=i.target.checked)};function zl(i){if(Ls||Ns||Ei||!Tt)return;const t=Bl();if(!t.length)return;On(!1);const e=i>0?t:t.slice().reverse();vn(e.find(n=>i>0?n>xe:n<xe)??e[0])}ct("prevCut").onclick=()=>zl(-1);ct("nextCut").onclick=()=>zl(1);function Nu(i,t=!1){if(!Tt)return;const e=_n.getBoundingClientRect(),n=ex(),s=i.clientX-e.left,r=i.clientY-e.top,a=nx(s,n);t&&s>=n.labelW&&vn(a);const o=Math.floor((r-n.rulerH)/n.rowH),c=ct("timelineHover"),l=o>=0&&o<Tt.contact_force.part_order.length?Tt.contact_force.part_order[o]:"Timeline";c.textContent=`${l} · ${Mu(a/Tt.fps)} · frame ${a}`,c.style.left=`${Math.max(n.labelW+44,Math.min(e.width-72,s))}px`,c.classList.remove("hidden")}_n.addEventListener("pointerdown",i=>{Tt&&(On(!1),Ps=!0,_n.setPointerCapture(i.pointerId),_n.classList.add("scrubbing"),Nu(i,!0))});_n.addEventListener("pointermove",i=>Nu(i,Ps));_n.addEventListener("pointerup",i=>{Ps&&(Ps=!1,_n.releasePointerCapture(i.pointerId),_n.classList.remove("scrubbing"))});_n.addEventListener("pointercancel",()=>{Ps=!1,_n.classList.remove("scrubbing")});_n.addEventListener("pointerleave",()=>{Ps||ct("timelineHover").classList.add("hidden")});window.addEventListener("keydown",i=>{Ls||Ns||Ei||ct("motionPicker").open||i.target.closest("input,select,textarea,button,[contenteditable=true]")||["Space","ArrowLeft","ArrowRight","BracketLeft","BracketRight"].includes(i.code)&&(i.preventDefault(),i.code==="Space"?ct("playBtn").click():i.code==="ArrowLeft"||i.code==="ArrowRight"?(On(!1),vn(xe+(i.code==="ArrowLeft"?-1:1)*(i.shiftKey?10:1))):zl(i.code==="BracketLeft"?-1:1))});let Du=0,Iu=0;ct("timelineResizer").addEventListener("pointerdown",i=>{or=!0,Du=i.clientY,Iu=xs,ct("timelineResizer").setPointerCapture(i.pointerId),document.body.classList.add("resizing-timeline")});ct("timelineResizer").addEventListener("pointermove",i=>{or&&Dl(Iu+Du-i.clientY)});ct("timelineResizer").addEventListener("pointerup",i=>{or&&(or=!1,ct("timelineResizer").releasePointerCapture(i.pointerId),document.body.classList.remove("resizing-timeline"),Dl(xs,!0))});function Vl(){const i=Vn.getBoundingClientRect();i.width<=0||i.height<=0||(wi.setSize(i.width,i.height,!1),qe.aspect=i.width/i.height,qe.updateProjectionMatrix(),kl())}window.addEventListener("resize",Vl);Vl();function Uu(i){requestAnimationFrame(Uu);const t=Math.min(.1,Math.max(0,(i-wh)/1e3));if(wh=i,nr&&Tt&&(Js+=t*Fu(!1),Js>=1)){const e=Math.floor(Js);Js-=e,vn((xe+e)%Tt.qpos.length)}bn.update(),wi.render(ke,qe),Vn.dataset.geometryCount=String(wi.info.memory.geometries)}requestAnimationFrame(Uu);function Fu(i=!0){const t=ct("fpsInput"),e=Number(t.value),n=Number.isFinite(e)&&e>=1?Math.min(240,e):Math.min(240,Math.max(1,(Tt==null?void 0:Tt.fps)||50));return i&&(t.value=String(n)),n}ct("fpsInput").onchange=()=>{Fu()};ct("timelineResizer").addEventListener("pointercancel",()=>{or=!1,document.body.classList.remove("resizing-timeline")});ct("reviewOpenForm").onsubmit=i=>{i.preventDefault();const t=ct("reviewPath").value.trim();!t||mr()||(ct("motionPicker").close(),Ol(t,ct("reviewTerrain").value.trim()||null))};Cu().catch(Pu);
