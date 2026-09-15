/* Interactive demo - extracted verbatim from the published artifact,
   with one added guard for the file:// tainted-canvas case (see README). */
(function(){
"use strict";
const N=256, NN=N*N;
const cin=document.getElementById('cin'), cout=document.getElementById('cout');
const gin=cin.getContext('2d',{willReadFrequently:true}), gout=cout.getContext('2d');
const $=id=>document.getElementById(id);
let src=null, ref=null;

/* ---------- helpers ---------- */
function toGray(imgdata){
  const d=imgdata.data, out=new Float32Array(NN);
  for(let i=0,p=0;i<NN;i++,p+=4) out[i]=(0.299*d[p]+0.587*d[p+1]+0.114*d[p+2])/255;
  return out;
}
function norm(a){
  let lo=Infinity,hi=-Infinity;
  for(let i=0;i<a.length;i++){const v=a[i];if(v<lo)lo=v;if(v>hi)hi=v;}
  const r=hi-lo, out=new Float32Array(a.length);
  if(r<1e-9) return out;
  for(let i=0;i<a.length;i++) out[i]=(a[i]-lo)/r;
  return out;
}
function draw(ctx,a){
  const im=ctx.createImageData(N,N), d=im.data;
  for(let i=0,p=0;i<NN;i++,p+=4){
    const v=Math.max(0,Math.min(255,Math.round(a[i]*255)));
    d[p]=d[p+1]=d[p+2]=v; d[p+3]=255;
  }
  ctx.putImageData(im,0,0);
}
/* separable 1-D min/max over a window of radius r, applied rows then cols */
function sep(a,r,fn){
  const t=new Float32Array(NN), o=new Float32Array(NN);
  for(let y=0;y<N;y++){const b=y*N;
    for(let x=0;x<N;x++){
      let m=a[b+x];
      const s=Math.max(0,x-r), e=Math.min(N-1,x+r);
      for(let k=s;k<=e;k++){const v=a[b+k]; if(fn(v,m)) m=v;}
      t[b+x]=m;}}
  for(let x=0;x<N;x++){
    for(let y=0;y<N;y++){
      let m=t[y*N+x];
      const s=Math.max(0,y-r), e=Math.min(N-1,y+r);
      for(let k=s;k<=e;k++){const v=t[k*N+x]; if(fn(v,m)) m=v;}
      o[y*N+x]=m;}}
  return o;
}
const lt=(v,m)=>v<m, gt=(v,m)=>v>m;
const erode=(a,r)=>sep(a,r,lt), dilate=(a,r)=>sep(a,r,gt);
const opening=(a,r)=>dilate(erode(a,r),r);

/* 3rd-order 2D polynomial surface, least squares via normal equations */
function polySurface(a){
  const terms=[];
  for(let i=0;i<=3;i++) for(let j=0;j<=3-i;j++) terms.push([i,j]);
  const M=terms.length, ATA=[], ATb=new Float64Array(M);
  for(let i=0;i<M;i++) ATA.push(new Float64Array(M));
  const bas=new Float64Array(M);
  for(let y=0;y<N;y++) for(let x=0;x<N;x++){
    const xn=x/(N-1), yn=y/(N-1);
    for(let t=0;t<M;t++) bas[t]=Math.pow(xn,terms[t][0])*Math.pow(yn,terms[t][1]);
    const v=a[y*N+x];
    for(let i=0;i<M;i++){ATb[i]+=bas[i]*v; for(let j=i;j<M;j++) ATA[i][j]+=bas[i]*bas[j];}
  }
  for(let i=0;i<M;i++) for(let j=0;j<i;j++) ATA[i][j]=ATA[j][i];
  /* gaussian elimination with partial pivoting */
  const A=ATA.map((row,i)=>{const r=new Float64Array(M+1); r.set(row); r[M]=ATb[i]; return r;});
  for(let c=0;c<M;c++){
    let piv=c; for(let r2=c+1;r2<M;r2++) if(Math.abs(A[r2][c])>Math.abs(A[piv][c])) piv=r2;
    const tmp=A[c]; A[c]=A[piv]; A[piv]=tmp;
    const pv=A[c][c]; if(Math.abs(pv)<1e-12) continue;
    for(let r2=0;r2<M;r2++){ if(r2===c) continue;
      const f=A[r2][c]/pv; if(!f) continue;
      for(let k=c;k<=M;k++) A[r2][k]-=f*A[c][k];}
  }
  const co=new Float64Array(M);
  for(let i=0;i<M;i++) co[i]= Math.abs(A[i][i])<1e-12 ? 0 : A[i][M]/A[i][i];
  const out=new Float32Array(NN);
  for(let y=0;y<N;y++) for(let x=0;x<N;x++){
    const xn=x/(N-1), yn=y/(N-1); let s=0;
    for(let t=0;t<M;t++) s+=co[t]*Math.pow(xn,terms[t][0])*Math.pow(yn,terms[t][1]);
    out[y*N+x]=s;}
  return out;
}
/* match the histogram of a to that of target */
function histMatch(a,target){
  const idx=Array.from({length:NN},(_,i)=>i);
  idx.sort((p,q)=>a[p]-a[q]);
  const sorted=Float32Array.from(target); sorted.sort();
  const out=new Float32Array(NN);
  for(let k=0;k<NN;k++) out[idx[k]]=sorted[k];
  return out;
}

/* ---------- FFT (radix-2, in place) ---------- */
function fft1(re,im,inv){
  const n=re.length;
  for(let i=1,j=0;i<n;i++){
    let bit=n>>1;
    for(;j&bit;bit>>=1) j^=bit;
    j^=bit;
    if(i<j){let t=re[i];re[i]=re[j];re[j]=t; t=im[i];im[i]=im[j];im[j]=t;}
  }
  for(let len=2;len<=n;len<<=1){
    const ang=(inv?2:-2)*Math.PI/len, wr=Math.cos(ang), wi=Math.sin(ang);
    for(let i=0;i<n;i+=len){
      let cr=1, ci=0;
      for(let k=0;k<len/2;k++){
        const ur=re[i+k], ui=im[i+k];
        const vr=re[i+k+len/2]*cr-im[i+k+len/2]*ci;
        const vi=re[i+k+len/2]*ci+im[i+k+len/2]*cr;
        re[i+k]=ur+vr; im[i+k]=ui+vi;
        re[i+k+len/2]=ur-vr; im[i+k+len/2]=ui-vi;
        const nr=cr*wr-ci*wi; ci=cr*wi+ci*wr; cr=nr;
      }
    }
  }
  if(inv) for(let i=0;i<n;i++){re[i]/=n; im[i]/=n;}
}
function fft2(re,im,inv){
  const rr=new Float64Array(N), ri=new Float64Array(N);
  for(let y=0;y<N;y++){
    for(let x=0;x<N;x++){rr[x]=re[y*N+x]; ri[x]=im[y*N+x];}
    fft1(rr,ri,inv);
    for(let x=0;x<N;x++){re[y*N+x]=rr[x]; im[y*N+x]=ri[x];}
  }
  for(let x=0;x<N;x++){
    for(let y=0;y<N;y++){rr[y]=re[y*N+x]; ri[y]=im[y*N+x];}
    fft1(rr,ri,inv);
    for(let y=0;y<N;y++){re[y*N+x]=rr[y]; im[y*N+x]=ri[y];}
  }
}
function butterworth(a,fc,order){
  const re=new Float64Array(NN), im=new Float64Array(NN);
  for(let i=0;i<NN;i++) re[i]=a[i];
  fft2(re,im,false);
  const half=N/2;
  for(let y=0;y<N;y++){
    const dy=(y<half?y:y-N)/N;
    for(let x=0;x<N;x++){
      const dx=(x<half?x:x-N)/N;
      const d=Math.sqrt(dx*dx+dy*dy);
      const H=1/(1+Math.pow(fc/(d+1e-9),2*order));  /* high-pass */
      re[y*N+x]*=H; im[y*N+x]*=H;
    }
  }
  fft2(re,im,true);
  const out=new Float32Array(NN);
  for(let i=0;i<NN;i++) out[i]=re[i];
  return out;
}

/* ---------- metrics ---------- */
function psnr(a,b){
  let se=0; for(let i=0;i<NN;i++){const d=a[i]-b[i]; se+=d*d;}
  const mse=se/NN;
  return mse<1e-12 ? 99 : 10*Math.log10(1/mse);
}
function gauss1(sigma){
  const r=Math.ceil(3*sigma), k=new Float64Array(2*r+1);
  let s=0;
  for(let i=-r;i<=r;i++){const v=Math.exp(-(i*i)/(2*sigma*sigma)); k[i+r]=v; s+=v;}
  for(let i=0;i<k.length;i++) k[i]/=s;
  return {k,r};
}
function blur(a,g){
  const {k,r}=g, t=new Float64Array(NN), o=new Float64Array(NN);
  for(let y=0;y<N;y++) for(let x=0;x<N;x++){
    let s=0; for(let i=-r;i<=r;i++){const xx=Math.min(N-1,Math.max(0,x+i)); s+=a[y*N+xx]*k[i+r];}
    t[y*N+x]=s;}
  for(let x=0;x<N;x++) for(let y=0;y<N;y++){
    let s=0; for(let i=-r;i<=r;i++){const yy=Math.min(N-1,Math.max(0,y+i)); s+=t[yy*N+x]*k[i+r];}
    o[y*N+x]=s;}
  return o;
}
function ssim(a,b){
  const g=gauss1(1.5), C1=0.01*0.01, C2=0.03*0.03;
  const ab=new Float64Array(NN), aa=new Float64Array(NN), bb=new Float64Array(NN);
  for(let i=0;i<NN;i++){ab[i]=a[i]*b[i]; aa[i]=a[i]*a[i]; bb[i]=b[i]*b[i];}
  const mA=blur(a,g), mB=blur(b,g), mAA=blur(aa,g), mBB=blur(bb,g), mAB=blur(ab,g);
  let s=0;
  for(let i=0;i<NN;i++){
    const ma=mA[i], mb=mB[i];
    const va=mAA[i]-ma*ma, vb=mBB[i]-mb*mb, cab=mAB[i]-ma*mb;
    s+=((2*ma*mb+C1)*(2*cab+C2))/((ma*ma+mb*mb+C1)*(va+vb+C2));
  }
  return s/NN;
}

/* ---------- pipeline ---------- */
function buildReference(a){
  const L=polySurface(opening(a,40));
  const flat=new Float32Array(NN);
  for(let i=0;i<NN;i++) flat[i]=Math.min(1,Math.max(0,a[i]-L[i]));
  return histMatch(flat,a);
}
function apply(method,p,a){
  if(method==='ref') return buildReference(a);
  if(method==='rb'){
    /* The illumination field is low-frequency by construction, so the opening used as the
       background estimate is smoothed before subtraction. This also hides the corners of the
       separable square structuring element standing in for the thesis's disk. */
    const bg=blur(opening(a,p),gauss1(Math.max(2,p/3)));
    const c=new Float32Array(NN);
    for(let i=0;i<NN;i++) c[i]=Math.min(1,Math.max(0,a[i]-bg[i]));
    return histMatch(norm(c),a);
  }
  if(method==='th'){
    const o=opening(a,p);
    const c=new Float32Array(NN);
    for(let i=0;i<NN;i++) c[i]=a[i]-o[i];
    return histMatch(norm(c),a);
  }
  if(method==='bw') return histMatch(norm(butterworth(a,p/1000,2)),a);
  return a;
}

const CFG={
  rb:{label:'Radius', min:5, max:90, val:50, unit:' px', out:'Rolling ball — background subtracted'},
  th:{label:'Structuring element', min:3, max:40, val:15, unit:' px', out:'White top-hat'},
  bw:{label:'Cutoff fₔ × 1000', min:10, max:150, val:50, unit:'', out:'Butterworth high-pass, order 2'},
  ref:{label:'(fixed by protocol)', min:40, max:40, val:40, unit:' px', out:'Reference image — §2.5.1'}
};

function run(){
  if(!src) return;
  const m=$('method').value, p=parseInt($('param').value,10);
  const t0=performance.now();
  const out=apply(m,p,src);
  const dt=Math.round(performance.now()-t0);
  draw(gout,out);
  $('outlabel').textContent=CFG[m].out;
  if(m==='ref'){
    $('mpsnr').textContent='—'; $('mssim').textContent='—';
    $('mverdict').innerHTML='<span class="pill">this is the reference</span>';
    $('mtime').textContent=dt+' ms';
    return;
  }
  const P=psnr(out,ref), S=ssim(out,ref);
  $('mpsnr').textContent=P.toFixed(1)+' dB';
  $('mssim').textContent=S.toFixed(3);
  const ok=P>=26&&S>=0.80;
  $('mverdict').innerHTML='<span class="pill '+(ok?'ok':'no')+'">'+(ok?'meets both':'below threshold')+'</span>';
  $('mtime').textContent='computed in '+dt+' ms';
}
function setMethod(){
  const m=$('method').value, c=CFG[m], r=$('param');
  $('plabel').textContent=c.label;
  r.min=c.min; r.max=c.max; r.value=c.val; r.disabled=(m==='ref');
  $('pval').textContent=c.val+c.unit;
  run();
}
function loadImage(url){
  const img=new Image();
  img.onload=function(){
    gin.fillStyle='#fff'; gin.fillRect(0,0,N,N);
    gin.drawImage(img,0,0,N,N);
    let raw;
    try{ raw=gin.getImageData(0,0,N,N); }
    catch(err){
      /* file:// taints the canvas; getImageData throws. Serve over http instead. */
      $('mverdict').innerHTML='<span class="pill no">serve over http</span>';
      $('mtime').textContent='see README \u2014 run a local server';
      $('mpsnr').textContent='\u2014'; $('mssim').textContent='\u2014';
      return;
    }
    src=norm(toGray(raw));
    draw(gin,src);
    ref=buildReference(src);
    run();
  };
  img.src=url;
}
$('method').addEventListener('change',setMethod);
$('param').addEventListener('input',function(){
  $('pval').textContent=this.value+CFG[$('method').value].unit; run();
});
$('upload').addEventListener('change',function(e){
  const f=e.target.files&&e.target.files[0]; if(!f) return;
  const fr=new FileReader(); fr.onload=ev=>loadImage(ev.target.result); fr.readAsDataURL(f);
});
loadImage("assets/sample.jpg");
})();
