"""Reconstructed PhyGuard repeated stratified evaluation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

PHYS=["interference","blockage","mobility","adaptation_mismatch"]


def metrics(y,pred):
    y=np.asarray(y);pred=np.asarray(pred,dtype=object)
    pm=np.isin(y,PHYS);cm=~pm;emit=pred!='ABSTAIN'
    sel=float(np.mean(pred[pm&emit]==y[pm&emit])) if np.any(pm&emit) else 0.0
    cov=float(np.mean(emit[pm]));fsr=float(np.mean(emit[cm]))
    # Abstention is outside the label set and therefore contributes false negatives.
    mf1=float(f1_score(y[pm],pred[pm],labels=PHYS,average='macro',zero_division=0))
    bal=float(recall_score(y[pm],pred[pm],labels=PHYS,average='macro',zero_division=0))
    return {'selected_accuracy':sel,'coverage':cov,'false_specific_rate':fsr,'macro_f1':mf1,'balanced_accuracy':bal}


def choose_thresholds(yv,gp,mp,classes):
    grid=np.round(np.linspace(.05,.95,19),2)
    rows=[]
    for tg in grid:
      for tc in grid:
        mx=mp.max(1);lab=classes[mp.argmax(1)]
        pred=np.where((gp>=tg)&(mx>=tc),lab,'ABSTAIN')
        mm=metrics(yv,pred);mm.update(tau_g=float(tg),tau_c=float(tc))
        feasible=mm['false_specific_rate']<=.05 and mm['selected_accuracy']>=.90
        mm['feasible']=feasible
        # deterministic fallback: total normalized violation, then max coverage.
        mm['penalty']=max(0,mm['false_specific_rate']-.05)/.05+max(0,.90-mm['selected_accuracy'])/.90
        rows.append(mm)
    f=[r for r in rows if r['feasible']]
    if f: best=max(f,key=lambda r:(r['coverage'],r['selected_accuracy'],-r['false_specific_rate'],-r['tau_g']-r['tau_c']))
    else: best=min(rows,key=lambda r:(r['penalty'],-r['coverage'],-r['selected_accuracy'],r['false_specific_rate'],r['tau_g']+r['tau_c']))
    return best,rows


def strat_keys(m):
    return (m.regime.astype(str)+'|'+m.label.astype(str)+'|'+m.severity.astype(str)).values


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--feature-dir',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);ap.add_argument('--repeats',type=int,default=5);ap.add_argument('--seed',type=int,default=20260705)
    a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    m=pd.read_csv(a.data_dir/'metadata.csv');d=np.load(a.feature_dir/'features.npz');P=d['physical'];C=d['combined'];y=m.label.values
    allrows=[];threshold_rows=[]
    for rep in range(a.repeats):
        idx=np.arange(len(y));keys=strat_keys(m)
        trainval,test=train_test_split(idx,test_size=.2,random_state=a.seed+rep,stratify=keys)
        train,val=train_test_split(trainval,test_size=.25,random_state=a.seed+100+rep,stratify=keys[trainval])
        gate=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',solver='lbfgs',random_state=a.seed+rep))
        gate.fit(P[train],np.isin(y[train],PHYS).astype(int))
        phys_train=train[np.isin(y[train],PHYS)]
        res=ExtraTreesClassifier(n_estimators=200,max_depth=None,min_samples_leaf=2,class_weight='balanced',random_state=a.seed+rep,n_jobs=1)
        res.fit(C[phys_train],y[phys_train])
        gv=gate.predict_proba(P[val])[:,1];mv=res.predict_proba(C[val]);best,grid=choose_thresholds(y[val],gv,mv,res.classes_)
        for r in grid:r.update(repeat=rep)
        threshold_rows.extend(grid)
        gt=gate.predict_proba(P[test])[:,1];mt=res.predict_proba(C[test]);mx=mt.max(1);lab=res.classes_[mt.argmax(1)]
        pred=np.where((gt>=best['tau_g'])&(mx>=best['tau_c']),lab,'ABSTAIN')
        mm=metrics(y[test],pred);mm.update(repeat=rep,tau_g=best['tau_g'],tau_c=best['tau_c'],validation_feasible=best['feasible'])
        allrows.append(mm)
    df=pd.DataFrame(allrows);df.to_csv(a.out_dir/'per_repeat_metrics.csv',index=False);pd.DataFrame(threshold_rows).to_csv(a.out_dir/'threshold_grid.csv',index=False)
    summ={c:{'mean':float(df[c].mean()),'std':float(df[c].std(ddof=0))} for c in ['selected_accuracy','coverage','false_specific_rate','macro_f1','balanced_accuracy']}
    summ['all_validation_operating_points_feasible']=bool(df.validation_feasible.all());summ['n_sequences']=len(y)
    (a.out_dir/'summary.json').write_text(json.dumps(summ,indent=2));print(json.dumps(summ,indent=2))
if __name__=='__main__':main()
