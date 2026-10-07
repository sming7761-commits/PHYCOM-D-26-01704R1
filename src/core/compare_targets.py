from pathlib import Path
import json,pandas as pd
root=Path('/mnt/data/phyguard_rebuild')
target=pd.read_csv(root/'config/target_metrics.csv')
t=target[(target.protocol=='main')&(target.method=='PhyGuard')].iloc[0]
r=json.loads((root/'results/full/evaluation/summary.json').read_text())
mapc={'selected_accuracy':'selected_accuracy_mean','coverage':'coverage_mean','false_specific_rate':'false_specific_mean','macro_f1':'macro_f1_mean','balanced_accuracy':'balanced_accuracy_mean'}
rows=[]
for new,old in mapc.items():
    recon=100*r[new]['mean']; tar=float(t[old])
    rows.append({'metric':new,'manuscript_target_percent':tar,'reconstructed_percent':recon,'difference_pp':recon-tar})
df=pd.DataFrame(rows);df.to_csv(root/'results/full/evaluation/target_comparison.csv',index=False)
print(df.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
