from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
base=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
rows=[]
for q in ['Q05','Q50']:
 raw=pd.read_csv(base/'validation'/f'{q}_Q00_clean.csv');raw=raw[raw.subject_type==0].copy()
 for rule in ['exclude_Q00_wait4','exclude_all_wait4']:
  mask=~((raw.wait_time==4)&((raw.treatment_name=='Q00') if rule=='exclude_Q00_wait4' else True))
  d=raw[mask].copy();d['treat']=(d.treatment_name==q).astype(int);d['interaction']=d.wait_time*d.treat
  X=sm.add_constant(d[['wait_time','treat','interaction']]);y=d.subject_choice
  for method in ['bfgs','lbfgs','newton']:
   r=dict(comparison=q+' vs Q00',restriction=rule,n=len(d),removed=len(raw)-len(d),method=method)
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    try:
     fit=Probit(y,X).fit(method=method,maxiter=5000,disp=False,cov_type='cluster',cov_kwds={'groups':d.cohort_id})
     h=-fit.model.hessian(fit.params)
     r.update(estimate=float(fit.params['interaction']),se=float(fit.bse['interaction']),raw_p=float(fit.pvalues['interaction']),llf=float(fit.llf),converged=bool(fit.mle_retvals.get('converged')),information_condition=float(np.linalg.cond(h)))
    except Exception as e:r['error']=str(e)
    r['warnings']=' | '.join(sorted(set(str(w.message) for w in caught)))
   rows.append(r)
res=pd.DataFrame(rows);res.to_csv(out/'wait4_ordinary_probit_diagnostics.csv',index=False)
print(res.drop(columns=['warnings']).to_string(index=False))
