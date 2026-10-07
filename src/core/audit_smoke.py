"""Audit directional changes in the reconstructed smoke benchmark."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KPI = ["RSRP_dB","RSSI_dB","SINR_dB","EVM","BER","PER_proxy","CQI","MCS","goodput_norm","rho_H"]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    X=np.load(args.data_dir/'telemetry.npz')['X']
    m=pd.read_csv(args.data_dir/'metadata.csv')
    rows=[]
    for i,r in m.iterrows():
        ref=X[i,:24].mean(axis=0)
        evt=X[i,int(r.event_start):int(r.event_end)].mean(axis=0)
        d=evt-ref
        row={"global_index":int(r.global_index),"regime":r.regime,"label":r.label,"severity":r.severity}
        row.update({f"delta_{k}":float(v) for k,v in zip(KPI,d)})
        row["delta_gap_dB"]=float((evt[1]-evt[0])-(ref[1]-ref[0]))
        rows.append(row)
    df=pd.DataFrame(rows)
    df.to_csv(args.out_dir/'per_sequence_changes.csv',index=False)
    agg=df.groupby('label').mean(numeric_only=True)
    agg.to_csv(args.out_dir/'mean_changes_by_label.csv')

    expectations={
      "interference":{"delta_gap_dB":"positive","delta_SINR_dB":"negative","delta_EVM":"positive","delta_RSRP_dB":"near_zero"},
      "blockage":{"delta_RSRP_dB":"negative","delta_SINR_dB":"negative","delta_EVM":"positive"},
      "mobility":{"delta_rho_H":"negative","delta_EVM":"positive","delta_RSRP_dB":"near_zero"},
      "adaptation_mismatch":{"delta_SINR_dB":"negative","delta_EVM":"positive","delta_MCS":"initially_held"},
      "nonphysical_goodput":{"delta_goodput_norm":"negative","delta_SINR_dB":"near_zero","delta_EVM":"near_zero"},
      "normal":{"all":"near_zero_in_expectation"}
    }
    report={"n_sequences":len(df),"expectations":expectations,"mean_changes":agg.to_dict(orient='index')}
    (args.out_dir/'directional_audit.json').write_text(json.dumps(report,indent=2))
    print(agg[[c for c in ['delta_RSRP_dB','delta_gap_dB','delta_SINR_dB','delta_EVM','delta_BER','delta_PER_proxy','delta_MCS','delta_goodput_norm','delta_rho_H'] if c in agg]].round(4))

if __name__=='__main__': main()
