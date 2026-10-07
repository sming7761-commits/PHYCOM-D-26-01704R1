"""Feature reconstruction for PhyGuard."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

KPI_NAMES=["RSRP_dB","RSSI_dB","SINR_dB","EVM","BER","PER_proxy","CQI","MCS","goodput_norm","rho_H"]
PHYSICAL_LABELS=["interference","blockage","mobility","adaptation_mismatch"]

MCS_THRESHOLDS=np.array([-2.0,4.0,10.0,17.0])


def supported_mcs_from_sinr(x):
    x=np.asarray(x)
    return np.sum(x[...,None]>=MCS_THRESHOLDS[None,:],axis=-1)-1


def slope(v):
    v=np.asarray(v,float)
    if len(v)<2:return 0.0
    x=np.arange(len(v),dtype=float); x-=x.mean(); y=v-v.mean()
    d=np.dot(x,x)
    return float(np.dot(x,y)/d) if d>0 else 0.0


def build_one(x,start,end,eps=1e-6):
    ref=x[:24]
    cand=x[start:end]
    med=np.median(ref,axis=0)
    mad=np.median(np.abs(ref-med),axis=0)
    scale=1.4826*mad+eps
    z=np.clip((x-med)/scale,-10,10)
    zc=z[start:end]
    raw=[]
    for j in range(x.shape[1]):
        v=zc[:,j]
        raw.extend([v.mean(),v.std(),np.median(v),np.quantile(v,.1),np.quantile(v,.9),slope(v)])
    raw=np.asarray(raw,float)

    # Oriented robust changes. Positive values mean evidence in the named direction.
    dz=zc.mean(axis=0)
    rsrp_drop=-dz[0]; rssi_rise=dz[1]; sinr_drop=-dz[2]
    evm_rise=dz[3]; ber_rise=dz[4]; per_rise=dz[5]
    cqi_drop=-dz[6]; mcs_drop=-dz[7]; goodput_drop=-dz[8]; rho_drop=-dz[9]
    gap_rise=dz[1]-dz[0]
    used_mcs=float(np.mean(cand[:,7]))
    supported=float(np.mean(supported_mcs_from_sinr(cand[:,2])))
    adapt_gap=max(0.0,used_mcs-supported)
    # Scale MCS gap into roughly robust-z units.
    adapt_gap_z=adapt_gap/0.75

    pos=lambda q:max(float(q),0.0)
    phi=lambda q,s=1.5:float(np.exp(-abs(float(q))/s))
    p_int=(pos(gap_rise)+pos(sinr_drop)+phi(rsrp_drop)+pos(evm_rise)+pos(ber_rise))/5.0
    p_blk=(pos(rsrp_drop)+pos(sinr_drop)+pos(evm_rise)+pos(cqi_drop)+phi(gap_rise))/5.0
    p_mob=(pos(rho_drop)+pos(evm_rise)+pos(ber_rise)+phi(rsrp_drop)+phi(sinr_drop))/5.0
    p_mis=(pos(adapt_gap_z)+pos(sinr_drop)+pos(evm_rise)+pos(ber_rise)+pos(cqi_drop)-0.35*pos(mcs_drop))/5.0
    p_non=(pos(goodput_drop)+phi(rsrp_drop)+phi(sinr_drop)+phi(evm_rise)+phi(ber_rise))/5.0
    scores=np.array([p_int,p_blk,p_mob,p_mis,p_non],float)
    sorted_scores=np.sort(scores[:4])
    underlying=np.array([rsrp_drop,rssi_rise,sinr_drop,evm_rise,ber_rise,per_rise,cqi_drop,mcs_drop,goodput_drop,rho_drop,gap_rise,adapt_gap_z],float)
    phys=np.concatenate([underlying,scores,[scores[:4].max(),sorted_scores[-1]-sorted_scores[-2]]])
    assert raw.shape==(60,) and phys.shape==(19,)
    return raw,phys


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True)
    a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    d=np.load(a.data_dir/'telemetry.npz',allow_pickle=True);X=d['X'];m=pd.read_csv(a.data_dir/'metadata.csv')
    R=[];P=[]
    for i,r in m.iterrows():
        rr,pp=build_one(X[i],int(r.event_start),int(r.event_end));R.append(rr);P.append(pp)
    R=np.stack(R).astype(np.float32);P=np.stack(P).astype(np.float32)
    np.savez_compressed(a.out_dir/'features.npz',raw=R,physical=P,combined=np.concatenate([R,P],axis=1))
    summary={'n':len(R),'raw_shape':list(R.shape),'physical_shape':list(P.shape),'finite':bool(np.isfinite(R).all() and np.isfinite(P).all())}
    (a.out_dir/'feature_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
