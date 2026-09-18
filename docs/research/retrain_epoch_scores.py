import torch, glob, json, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
def fair_crps(m, y):
    M=m.shape[0]; t1=np.abs(m-y[None]).mean(0); t2=np.zeros_like(y)
    for i in range(M): t2+=np.abs(m[i][None]-m).sum(0)
    return float((t1-t2/(2*M*(M-1))).mean())
def rank_tv(m,y,rng):
    M=m.shape[0]; below=(m<y[None]).sum(0); ties=(m==y[None]).sum(0); r=below+np.floor(rng.random(y.shape)*(ties+1)).astype(int)
    h=np.bincount(r,minlength=M+1).astype(float); h/=h.sum(); return float(0.5*np.abs(h-1/(M+1)).sum())
for arm in sys.argv[1:]:
    ds=sorted(glob.glob(f"/home/autoresearch_worktrees/clean_retrain_v1/checkpoints/concat_conditioning/claude_clean_retrain_{arm}/run_*"))
    if not ds: print(arm,"no run"); continue
    d=ds[-1]; meta=json.load(open(d+"/metadata.json")); mu=float(meta["normalization_means"][0]); sd=float(meta["normalization_stds"][0])
    print("==",arm,d.split("/")[-1])
    rng=np.random.default_rng(0)
    for f in sorted(glob.glob(d+"/samples/epoch_*_metric_ensemble.pt")):
        z=torch.load(f,map_location="cpu",weights_only=False)
        S=z["samples"][:,:,0].float().numpy()*sd+mu; Y=z["truth"][:,0].float().numpy()*sd+mu; W=z["valid_mask"][:,0].float().numpy()>0
        S=np.clip(S,0,1); Y=np.clip(Y,0,1)
        cr=[];rm=[];sp=[];z0=[];t0=[];tv=[]
        for i in range(S.shape[0]):
            w=W[i]; m=S[i][:,w]; y=Y[i][w]
            cr.append(fair_crps(m,y)); mean=m.mean(0); rm.append(np.sqrt(np.mean((mean-y)**2))); sp.append(np.sqrt(m.var(0,ddof=1).mean()))
            z0.append(np.mean(m==0)); t0.append(np.mean(y==0)); tv.append(rank_tv(m,y,rng))
        ep=int(z["epoch"]); print(f"  epoch {ep:2d} fairCRPS {np.mean(cr):.4f} meanRMSE {np.mean(rm):.4f} spread {np.mean(sp):.4f} s/r {np.mean(sp)/np.mean(rm):.3f} rankTV {np.mean(tv):.3f} zero member/truth {np.mean(z0):.3f}/{np.mean(t0):.3f}")
