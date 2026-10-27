from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
base=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
results=[]
for q in ['Q05','Q50']:
 d=pd.read_csv(base/'validation'/f'{q}_Q00_clean.csv')
 d=d[(d.subject_type==0)&(d.wait_time.isin([1,2]))].copy()
 d['treat']=(d.treatment_name==q).astype(int)
 d['wait2']=(d.wait_time==2).astype(int)
 d['interaction']=d.treat*d.wait2
 X=sm.add_constant(d[['wait2','treat','interaction']])
 fit=sm.OLS(d.subject_choice,X).fit(cov_type='cluster',cov_kwds={'groups':d.cohort_id,'use_correction':True},use_t=True)
 ci=fit.conf_int().loc['interaction']
 rates=d.groupby(['treat','wait2']).subject_choice.mean()
 direct=(rates[1,1]-rates[1,0])-(rates[0,1]-rates[0,0])
 assert np.isclose(direct,fit.params['interaction'])
 results.append(dict(comparison=f'{q} vs Q00',method='LPM waits 1 and 2; cohort-clustered t inference',estimate=fit.params['interaction'],se=fit.bse['interaction'],ci_low=ci[0],ci_high=ci[1],p_two_sided=fit.pvalues['interaction'],n=len(d),clusters=d.cohort_id.nunique(),df_inference=fit.df_resid_inference))
 (out/f'{q}_lpm_summary.txt').write_text(str(fit.summary()))
r=pd.DataFrame(results)
r['p_holm']=multipletests(r.p_two_sided,method='holm')[1]
r.to_csv(out/'lpm_results.csv',index=False)
print(r.to_string(index=False))
