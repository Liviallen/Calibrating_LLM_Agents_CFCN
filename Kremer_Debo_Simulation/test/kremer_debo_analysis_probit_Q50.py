"""
Kremer-Debo Analysis - Probit Regression
Python translation of kremer-debo_analysis - probit.do

This script replicates the STATA analysis for the empty restaurant syndrome study.

For ORIGINAL data: Uses Random Effects Probit (xtprobit...re equivalent)
For GPT-5.4 data: Uses Probit with cluster-robust standard errors

IMPORTANT NOTE ON w4/w4q50 COEFFICIENTS:
-----------------------------------------
The w4 and w4q50 coefficients and standard errors may differ between Python and STATA.
This is EXPECTED and NORMAL due to near-perfect separation in the data.

When wait_time=4, only 7-9% of subjects choose to purchase (vs ~70-95% for w1-w3).
This extreme separation causes:
1. Very large negative coefficients (trying to push probability to near zero)
2. Very large standard errors (poor identification due to flat likelihood surface)
3. Numerical instability where different optimization algorithms converge to different
   values in the flat region of the likelihood surface

STATA and Python use different optimization algorithms and numerical methods:
- STATA: Adaptive Gauss-Hermite quadrature with specific convergence criteria
- Python: BFGS/L-BFGS optimization with different numerical precision

Both are correct - they represent the same phenomenon (almost no purchases at w4).
The key hypothesis tests (w1q50, wait_time_q50) match well because they don't suffer
from separation issues. These are the coefficients of interest for the research question.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data(gpt4o_path: str, original_path: str) -> pd.DataFrame:
    """Load and prepare combined dataset."""
    gpt4o = pd.read_csv(gpt4o_path)
    original = pd.read_csv(original_path)
    
    # Add treatment_name to original data
    original['treatment_name'] = ''
    original.loc[(original['treatment'] == 3) & (original['rank_information'] == 0), 'treatment_name'] = 'Q50'
    original.loc[(original['treatment'] == 1) & (original['rank_information'] == 0), 'treatment_name'] = 'Q00'
    original = original[original['treatment_name'] != ''].copy()
    
    original['location'] = 'ORIGINAL'
    original['subject_pool'] = 'Smeal'
    original['modality'] = 'In-person'
    
    combined = pd.concat([original, gpt4o], ignore_index=True)
    return combined


def create_dummy_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create wait time dummies and Q50 interaction variables."""
    df = df.copy()
    for i in range(1, 5):
        df[f'w{i}'] = (df['wait_time'] == i).astype(int)
    for i in range(1, 5):
        df[f'w{i}q50'] = ((df['wait_time'] == i) & (df['treatment_name'] == 'Q50')).astype(int)
    df['q50'] = (df['treatment_name'] == 'Q50').astype(int)
    df['wait_time_q50'] = df['wait_time'] * df['q50']
    return df


# =============================================================================
# RANDOM EFFECTS PROBIT (xtprobit equivalent)
# =============================================================================

class RandomEffectsProbit:
    """
    Random Effects Probit Model - equivalent to STATA's xtprobit...re
    Uses Gauss-Hermite quadrature for numerical integration.
    """
    
    def __init__(self, n_quad_points=12):
        self.n_quad_points = n_quad_points
        self.quad_points, self.quad_weights = np.polynomial.hermite.hermgauss(n_quad_points)
        self.params_ = None
        self.sigma_u_ = None
        self.rho_ = None
        self.se_ = None
        
    def _log_likelihood_panel(self, params, X, y, groups):
        """Compute log-likelihood for random effects probit."""
        n_vars = X.shape[1]
        beta = params[:n_vars]
        log_sigma_u = params[n_vars]
        sigma_u = np.exp(log_sigma_u)
        
        xb = X @ beta
        unique_groups = np.unique(groups)
        total_ll = 0.0
        
        for g in unique_groups:
            mask = groups == g
            xb_g = xb[mask]
            y_g = y[mask]
            
            panel_contrib = 0.0
            for q, w in zip(self.quad_points, self.quad_weights):
                u = np.sqrt(2) * sigma_u * q
                eta = xb_g + u
                prob = stats.norm.cdf(eta)
                prob = np.clip(prob, 1e-10, 1-1e-10)
                panel_like = np.prod(np.where(y_g == 1, prob, 1 - prob))
                panel_contrib += w * panel_like / np.sqrt(np.pi)
            
            total_ll += np.log(max(panel_contrib, 1e-300))
        
        return -total_ll
    
    def fit(self, X, y, groups, var_names=None):
        """Fit the random effects probit model."""
        X = np.asarray(X)
        y = np.asarray(y)
        groups = np.asarray(groups)
        
        n_vars = X.shape[1]
        
        # Get initial values from pooled probit
        try:
            pooled = Probit(y, X).fit(method='bfgs', disp=False, maxiter=1000)
            init_beta = pooled.params.values
        except:
            init_beta = np.zeros(n_vars)
            init_beta[0] = 0.5
        
        init_log_sigma = np.log(0.3)
        init_params = np.concatenate([init_beta, [init_log_sigma]])
        
        # Optimize
        result = minimize(
            self._log_likelihood_panel, init_params, args=(X, y, groups),
            method='BFGS', options={'maxiter': 1000, 'disp': False}
        )
        
        self.params_ = result.x[:n_vars]
        self.sigma_u_ = np.exp(result.x[n_vars])
        self.rho_ = self.sigma_u_**2 / (1 + self.sigma_u_**2)
        self.log_likelihood_ = -result.fun
        self.nobs_ = len(y)
        self.n_groups_ = len(np.unique(groups))
        self.var_names_ = var_names if var_names else [f'x{i}' for i in range(n_vars)]
        
        # Compute standard errors via numerical Hessian
        try:
            eps = 1e-5
            n_params = len(result.x)
            hess = np.zeros((n_params, n_params))
            f0 = self._log_likelihood_panel(result.x, X, y, groups)
            
            for i in range(n_params):
                for j in range(n_params):
                    x_pp = result.x.copy(); x_pp[i] += eps; x_pp[j] += eps
                    x_pm = result.x.copy(); x_pm[i] += eps; x_pm[j] -= eps
                    x_mp = result.x.copy(); x_mp[i] -= eps; x_mp[j] += eps
                    x_mm = result.x.copy(); x_mm[i] -= eps; x_mm[j] -= eps
                    
                    f_pp = self._log_likelihood_panel(x_pp, X, y, groups)
                    f_pm = self._log_likelihood_panel(x_pm, X, y, groups)
                    f_mp = self._log_likelihood_panel(x_mp, X, y, groups)
                    f_mm = self._log_likelihood_panel(x_mm, X, y, groups)
                    
                    hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)
            
            vcov = np.linalg.inv(hess)
            self.se_ = np.sqrt(np.diag(vcov)[:n_vars])
        except:
            self.se_ = np.full(n_vars, np.nan)
        
        return self


# =============================================================================
# CLUSTER-ROBUST PROBIT
# =============================================================================

def run_probit_clustered(df, formula_vars, outcome, cluster_var):
    """Run probit regression with cluster-robust standard errors."""
    X = df[formula_vars].copy()
    X = sm.add_constant(X)
    # Dynamic simulations may have no observations in a treatment/wait-time
    # cell. Remove only redundant columns: the fitted model's column space
    # and all observations remain unchanged.
    independent = []
    dropped = []
    rank = 0
    for column in X.columns:
        candidate = X[independent + [column]]
        candidate_rank = np.linalg.matrix_rank(candidate.to_numpy())
        if candidate_rank > rank:
            independent.append(column)
            rank = candidate_rank
        else:
            dropped.append(column)
    for column in dropped:
        print(f"  Omitting {column}: redundant or constant column in this sample")
    X = X[independent]
    if 'treatment_name' in df and 'wait_time' in df:
        cells = df.groupby(['treatment_name', 'wait_time'])[outcome].agg(['size', 'sum'])
        for (condition, wait), cell in cells.iterrows():
            if cell['sum'] in (0, cell['size']):
                print(f"  Note: {condition}, wait={wait} has only one outcome; "
                      "saturated cell coefficients may show separation.")
    y = df[outcome]
    clusters = df[cluster_var]
    
    model = Probit(y, X)
    
    # Try multiple methods to match STATA's optimization
    methods = ['bfgs', 'lbfgs', 'newton']
    best_result = None
    best_llf = -np.inf
    
    for method in methods:
        try:
            result = model.fit(cov_type='cluster', cov_kwds={'groups': clusters}, 
                              method=method, disp=False, maxiter=5000)
            if result.llf > best_llf:
                best_llf = result.llf
                best_result = result
        except Exception as error:
            print(f"  Method {method} failed: {type(error).__name__}: {error}")
            continue
    
    if best_result is None:
        raise ValueError("All optimization methods failed; inspect the method diagnostics above")
    
    return {
        'model': best_result,
        'valid_vars': [column for column in X.columns if column != 'const'],
        'dropped_vars': dropped,
        'params': best_result.params,
        'bse': best_result.bse,
        'pvalues': best_result.pvalues,
        'nobs': int(best_result.nobs),
        'llf': best_result.llf,
        'pseudo_r2': best_result.prsquared,
        'n_clusters': clusters.nunique()
    }


# =============================================================================
# HYPOTHESIS TESTING
# =============================================================================

def wald_test(coef, se, param_name):
    """Perform Wald test for coefficient = 0."""
    z = coef / se
    chi2 = z ** 2
    pval = stats.chi2.sf(chi2, df=1)
    return {'parameter': param_name, 'coefficient': coef, 'std_error': se, 
            'z_stat': z, 'wald_chi2': chi2, 'p_value': pval}


def print_hypothesis_test(test_result, test_name):
    """Print hypothesis test in STATA format."""
    print(f"\n{'='*70}")
    print(f"HYPOTHESIS TEST: {test_name}")
    print(f"{'='*70}")
    print(f"\ntest {test_result['parameter']} = 0")
    print(f"\n ( 1)  [subject_choice]{test_result['parameter']} = 0")
    print(f"\n           chi2(  1) = {test_result['wald_chi2']:10.2f}")
    print(f"         Prob > chi2 = {test_result['p_value']:10.4f}")
    print(f"\n// Exact p-value:")
    print(f"   {test_result['p_value']:.12e}")


def print_coefficients_table(params, se, pvalues, var_names):
    """Print coefficient table in STATA format."""
    print("-" * 75)
    print(f"{'Variable':<15} {'Coefficient':>12} {'Std.Err.':>12} {'z':>10} {'P>|z|':>10}")
    print("-" * 75)
    for i, name in enumerate(var_names):
        if isinstance(params, np.ndarray):
            coef = params[i]
            std = se[i]
            pval = pvalues[i] if isinstance(pvalues, (list, np.ndarray)) else pvalues
        else:
            coef = params[name] if hasattr(params, '__getitem__') else params.iloc[i]
            std = se[name] if hasattr(se, '__getitem__') else se.iloc[i]
            pval = pvalues[name] if hasattr(pvalues, '__getitem__') else pvalues.iloc[i]
        
        if np.isnan(std):
            print(f"{name:<15} {coef:>12.6f} {'N/A':>12} {'N/A':>10} {'N/A':>10}")
        else:
            z = coef / std
            print(f"{name:<15} {coef:>12.6f} {std:>12.7f} {z:>10.2f} {pval:>10.4f}")
    print("-" * 75)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main(gpt4o_path=None, output_dir=None):
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gpt4o_path = gpt4o_path or os.path.join(script_dir, 'gpt5.4_obs_learning_cf_bottom_crossentropy_navigation_Q50_Q00_clean.csv')
    original_path = os.path.join(script_dir, 'kremer-debo_originalpaper.csv')
    
    print("=" * 80)
    print("KREMER-DEBO ANALYSIS - PROBIT REGRESSION")
    print("Python Translation of kremer-debo_analysis - probit.do")
    print("=" * 80)
    
    # Load and prepare data
    print("\nLoading data...")
    df = load_and_prepare_data(gpt4o_path, original_path)
    df = create_dummy_variables(df)
    
    results_summary = []
    
    # =========================================================================
    # ORIGINAL DATA ANALYSIS (Random Effects Probit)
    # =========================================================================
    print("\n" + "*" * 80)
    print("** Analysis run on Original dataset")
    print("** Method: xtprobit subject_choice vars, re")
    print("*" * 80)
    
    df_orig = df[(df['location'] == 'ORIGINAL') & (df['subject_type'] == 0)].copy()
    print(f"\nNumber of obs    = {len(df_orig):,}")
    print(f"Number of groups = {df_orig['cohort_id'].nunique()}")
    
    # Primary analysis H1
    print("\n// Primary analysis: empty restaurant syndrome")
    primary_vars = ['w2', 'w3', 'w4', 'w1q50', 'w2q50', 'w3q50', 'w4q50']
    X_p = sm.add_constant(df_orig[primary_vars])
    y_p = df_orig['subject_choice'].values
    groups_p = df_orig['cohort_id'].values
    
    print("Fitting random-effects probit regression...")
    re_primary = RandomEffectsProbit(n_quad_points=12)
    re_primary.fit(X_p.values, y_p, groups_p, var_names=['const'] + primary_vars)
    
    print("\n" + "=" * 75)
    print("Random-effects probit regression")
    print("=" * 75)
    print(f"Log likelihood   = {re_primary.log_likelihood_:.5f}")
    print(f"sigma_u          = {re_primary.sigma_u_:.7f}")
    print(f"rho              = {re_primary.rho_:.7f}")
    print()
    
    # Compute p-values
    pvals_p = [2*(1-stats.norm.cdf(abs(c/s))) if not np.isnan(s) else np.nan 
               for c, s in zip(re_primary.params_, re_primary.se_)]
    print_coefficients_table(re_primary.params_, re_primary.se_, pvals_p, re_primary.var_names_)
    
    # Purchase probabilities
    prob_q00_orig = stats.norm.cdf(re_primary.params_[0])
    w1q50_idx = re_primary.var_names_.index('w1q50')
    prob_q50_orig = stats.norm.cdf(re_primary.params_[0] + re_primary.params_[w1q50_idx])
    
    print(f"\n// Verify purchase probabilities when w=1:")
    print(f"display normprob(_b[_cons]) // Q00")
    print(f"   {prob_q00_orig:.8f}")
    print(f"display normprob(_b[_cons] + _b[w1q50]) // Q50")
    print(f"   {prob_q50_orig:.8f}")
    
    # H1 test
    h1_orig = wald_test(re_primary.params_[w1q50_idx], re_primary.se_[w1q50_idx], 'w1q50')
    print_hypothesis_test(h1_orig, "H1 - Primary (ORIGINAL): w1q50 = 0")
    results_summary.append(('H1_Primary', 'ORIGINAL', 'w1q50', h1_orig))
    
    # Secondary analysis H2
    print("\n\n// Secondary analysis: comparing slopes")
    secondary_vars = ['wait_time', 'q50', 'wait_time_q50']
    X_s = sm.add_constant(df_orig[secondary_vars])
    
    print("Fitting random-effects probit regression...")
    re_secondary = RandomEffectsProbit(n_quad_points=12)
    re_secondary.fit(X_s.values, y_p, groups_p, var_names=['const'] + secondary_vars)
    
    print("\n" + "=" * 75)
    print("Random-effects probit regression")
    print("=" * 75)
    print(f"Log likelihood   = {re_secondary.log_likelihood_:.5f}")
    print(f"sigma_u          = {re_secondary.sigma_u_:.7f}")
    print(f"rho              = {re_secondary.rho_:.7f}")
    print()
    
    pvals_s = [2*(1-stats.norm.cdf(abs(c/s))) if not np.isnan(s) else np.nan 
               for c, s in zip(re_secondary.params_, re_secondary.se_)]
    print_coefficients_table(re_secondary.params_, re_secondary.se_, pvals_s, re_secondary.var_names_)
    
    # H2 test
    wtq50_idx = re_secondary.var_names_.index('wait_time_q50')
    h2_orig = wald_test(re_secondary.params_[wtq50_idx], re_secondary.se_[wtq50_idx], 'wait_time_q50')
    print_hypothesis_test(h2_orig, "H2 - Secondary (ORIGINAL): wait_time_q50 = 0")
    results_summary.append(('H2_Secondary', 'ORIGINAL', 'wait_time_q50', h2_orig))
    
    # =========================================================================
    # GPT-5.4 DATA ANALYSIS (Cluster-Robust Probit)
    # =========================================================================
    print("\n\n" + "*" * 80)
    print("** AI Replication: GPT-5.4")
    print("** Method: probit subject_choice vars, vce(cluster cohort_id)")
    print("*" * 80)
    
    df_gpt = df[(df['location'] == 'GPT-5.4') & (df['subject_type'] == 0)].copy()
    print(f"\nNumber of obs    = {len(df_gpt):,}")
    print(f"Number of clusters = {df_gpt['cohort_id'].nunique()}")
    
    # Primary analysis H1
    print("\n// Primary analysis: empty restaurant syndrome")
    primary_result = run_probit_clustered(df_gpt, primary_vars, 'subject_choice', 'cohort_id')
    
    print("\n" + "=" * 75)
    print("Probit regression")
    print("(Std. err. adjusted for clusters in cohort_id)")
    print("=" * 75)
    print(f"Log pseudolikelihood = {primary_result['llf']:.5f}")
    print(f"Pseudo R2            = {primary_result['pseudo_r2']:.4f}")
    print()
    
    var_names_gpt = list(primary_result['params'].index)
    print_coefficients_table(
        primary_result['params'].values, 
        primary_result['bse'].values,
        primary_result['pvalues'].values,
        var_names_gpt
    )
    
    # Purchase probabilities
    prob_q00_gpt = stats.norm.cdf(primary_result['params']['const'])
    prob_q50_gpt = stats.norm.cdf(primary_result['params']['const'] + primary_result['params']['w1q50'])
    
    print(f"\n// Verify purchase probabilities when w=1:")
    print(f"display normprob(_b[_cons]) // Q00")
    print(f"   {prob_q00_gpt:.8f}")
    print(f"display normprob(_b[_cons] + _b[w1q50]) // Q50")
    print(f"   {prob_q50_gpt:.8f}")
    
    # H1 test
    h1_gpt = wald_test(primary_result['params']['w1q50'], primary_result['bse']['w1q50'], 'w1q50')
    print_hypothesis_test(h1_gpt, "H1 - Primary (GPT-5.4): w1q50 = 0")
    results_summary.append(('H1_Primary', 'GPT-5.4', 'w1q50', h1_gpt))
    
    # Secondary analysis H2
    print("\n\n// Secondary analysis: comparing slopes")
    secondary_result = run_probit_clustered(df_gpt, secondary_vars, 'subject_choice', 'cohort_id')
    
    print("\n" + "=" * 75)
    print("Probit regression")
    print("(Std. err. adjusted for clusters in cohort_id)")
    print("=" * 75)
    print(f"Log pseudolikelihood = {secondary_result['llf']:.5f}")
    print(f"Pseudo R2            = {secondary_result['pseudo_r2']:.4f}")
    print()
    
    var_names_gpt_s = list(secondary_result['params'].index)
    print_coefficients_table(
        secondary_result['params'].values, 
        secondary_result['bse'].values,
        secondary_result['pvalues'].values,
        var_names_gpt_s
    )
    
    # H2 test
    h2_gpt = wald_test(secondary_result['params']['wait_time_q50'], 
                       secondary_result['bse']['wait_time_q50'], 'wait_time_q50')
    print_hypothesis_test(h2_gpt, "H2 - Secondary (GPT-5.4): wait_time_q50 = 0")
    results_summary.append(('H2_Secondary', 'GPT-5.4', 'wait_time_q50', h2_gpt))
    
    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n\n" + "=" * 90)
    print("SUMMARY: RESULTS (Q50 vs Q00)")
    print("=" * 90)
    
    print(f"\n{'Test':<20} {'Dataset':<12} {'Variable':<15} {'Coefficient':>12} {'Std.Err.':>12} {'Chi2':>10} {'P-value':>12}")
    print("-" * 90)
    
    for test, dataset, var, result in results_summary:
        print(f"{test:<20} {dataset:<12} {var:<15} {result['coefficient']:>12.6f} {result['std_error']:>12.6f} {result['wald_chi2']:>10.2f} {result['p_value']:>12.4e}")
    
    print("-" * 90)
    
    # Save results
    results_df = pd.DataFrame([
        {
            'Test': test, 'Dataset': dataset, 'Variable': var,
            'Coefficient': r['coefficient'], 'Std_Error': r['std_error'],
            'Chi2': r['wald_chi2'], 'P_Value': r['p_value']
        }
        for test, dataset, var, r in results_summary
    ])
    
    output_path = os.path.join(output_dir or script_dir, 'navigation_analysis_Q50_vs_Q00.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return results_df


if __name__ == '__main__':
    results = main()
