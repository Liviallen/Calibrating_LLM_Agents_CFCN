from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
base=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
rows=[]
for q in ['Q05','Q50']:
 d=pd.read_csv(base/'validation'/f'{q}_Q00_clean.csv');d=d[(d.subject_type==0)&(d.wait_time==1)].copy()
 d['treat']=(d.treatment_name==q).astype(int)
 f=sm.OLS(d.subject_choice,sm.add_constant(d[['treat']])).fit(cov_type='cluster',cov_kwds={'groups':d.cohort_id,'use_correction':True},use_t=True)
 lo,hi=f.conf_int().loc['treat']
 rows.append(dict(comparison=q+' vs Q00',estimate=f.params['treat'],se=f.bse['treat'],ci_low=lo,ci_high=hi,p_two_sided=f.pvalues['treat'],clusters=d.cohort_id.nunique(),n=len(d),refusal_cohorts=d[d.subject_choice==0].cohort_id.nunique(),refusals=int((d.subject_choice==0).sum())))
r=pd.DataFrame(rows);r['p_holm']=multipletests(r.p_two_sided,method='holm')[1];r.to_csv(out/'h1_lpm_results.csv',index=False);print(r.to_string(index=False))
