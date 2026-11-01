"""Navigation-mixture fitting used by pipeline.py; original numerical routines."""
from file_utils import atomic_to_csv
import os

import numpy as np

import pandas as pd

from scipy.optimize import minimize

from scipy.special import softmax

def get_condition_results_folder(replication_root, condition):
    """Return the folder containing nav simulation results for a condition."""
    return os.path.join(replication_root, condition)

def load_llm_predictions(base_folder, model_name, condition, n_prompts=11):
    """
    Load LLM predictions from all navigation prompt variants.
    
    Args:
        base_folder: Base output folder (e.g., 'replication_results')
        model_name: Model name (e.g., 'gpt4o')
        condition: Condition (e.g., 'Q50')
        n_prompts: Number of prompt variants (default 11)
    
    Returns:
        dict: {prompt_idx: DataFrame} where each DataFrame has 'instance_idx' and 'llm_choice'
    """
    predictions = {}
    
    # Construct folder name pattern
    folder_prefix = f"{model_name}_context_formation_and_navigation_replication_{condition}"
    
    for nav_idx in range(1, n_prompts + 1):
        nav_folder = os.path.join(base_folder, f"{folder_prefix}_nav{nav_idx}")
        # CSV filename pattern: {model_name}_nav{nav_idx}_{condition}_results.csv
        csv_path = os.path.join(nav_folder, f"{model_name}_nav{nav_idx}_{condition}_results.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Ensure llm_choice is numeric (0 or 1)
            df['llm_choice'] = pd.to_numeric(df['llm_choice'], errors='coerce').fillna(0).astype(int)
            predictions[nav_idx] = df[['instance_idx', 'llm_choice']].copy()
    
    return predictions

def load_llm_predictions_for_condition(replication_root, model_name, condition, n_prompts=8):
    """Load nav results from replication_results/{condition}/."""
    condition_folder = get_condition_results_folder(replication_root, condition)
    predictions = load_llm_predictions(condition_folder, model_name, condition, n_prompts)
    if len(predictions) == 0:
        raise ValueError(
            f"No LLM predictions found for {condition} under {condition_folder}. "
            f"Expected folders like {model_name}_context_formation_and_navigation_replication_{condition}_nav1/"
        )
    if len(predictions) != n_prompts:
        raise ValueError(
            f"Expected {n_prompts} navigation variants for {condition}, found {len(predictions)}"
        )
    return predictions

def _read_human_csv(human_csv_path):
    df_human = pd.read_csv(human_csv_path)
    if len(df_human.columns) > 0 and str(df_human.columns[0]).startswith('Unnamed'):
        df_human = df_human.drop(columns=df_human.columns[0])
    return df_human

def prepare_data_for_optimization(human_csv_path, llm_predictions, require_wait_time=False):
    """
    Prepare data matrices for optimization.
    
    Args:
        human_csv_path: Path to human experiment CSV
        llm_predictions: Dict from load_llm_predictions
        require_wait_time: If True, requires 'wait_time' column and filters to {1,2,3,4}
    
    Returns:
        tuple: (Y, Y_hat, valid_instances) or (Y, Y_hat, wait_time_groups, valid_instances)
            Y: numpy array of human choices (N,)
            Y_hat: numpy array of LLM predictions (N, K) where K is number of prompts
            wait_time_groups: numpy array of wait_time values (N,) - only if require_wait_time=True
            valid_instances: list of instance_idx that have all prompts
    
    Raises:
        ValueError: If instances don't match exactly across all prompts, or if wait_time is required but missing
    """
    df_human = _read_human_csv(human_csv_path)
    df_human['human_choice'] = pd.to_numeric(df_human['subject_choice'], errors='coerce').fillna(0).astype(int)
    
    # Handle wait_time if required
    wait_time_groups = None
    if require_wait_time:
        if 'wait_time' not in df_human.columns:
            raise ValueError("Human CSV must contain 'wait_time' column for group-level optimization")
        df_human['wait_time'] = pd.to_numeric(df_human['wait_time'], errors='coerce')
        # Filter to valid wait_time groups (1, 2, 3, 4)
        valid_wait_times = [1, 2, 3, 4]
        df_human = df_human[df_human['wait_time'].isin(valid_wait_times)].copy()
        if len(df_human) == 0:
            raise ValueError(f"No instances found with wait_time in {valid_wait_times}")
    
    # Get all instance indices from human CSV (0-based index)
    all_instances = sorted(df_human.index.tolist())
    n_instances = len(all_instances)
    n_prompts = len(llm_predictions)
    
    # VALIDATION: Check that all prompts have exactly the same instances
    prompt_keys = sorted(llm_predictions.keys())
    reference_instances = None
    
    for prompt_key in prompt_keys:
        df_prompt = llm_predictions[prompt_key]
        prompt_instances = sorted(df_prompt['instance_idx'].unique().tolist())
        
        if reference_instances is None:
            reference_instances = prompt_instances
        else:
            if prompt_instances != reference_instances:
                raise ValueError(f"Instances do not match exactly across prompts. Prompt {prompt_key} differs from reference.")
    
    # Check that reference instances match human CSV instances
    if sorted(all_instances) != sorted(reference_instances):
        raise ValueError("Human CSV instances do not match LLM prediction instances exactly.")
    
    # Initialize matrices
    Y = df_human['human_choice'].values  # (N,)
    Y_hat = np.zeros((n_instances, n_prompts))  # (N, K)
    
    # Fill Y_hat matrix - now we know all instances exist in all prompts
    for k_idx, prompt_key in enumerate(prompt_keys):
        df_prompt = llm_predictions[prompt_key]
        # Match by instance_idx
        for i, instance_idx in enumerate(all_instances):
            prompt_row = df_prompt[df_prompt['instance_idx'] == instance_idx]
            if len(prompt_row) == 0:
                raise ValueError(f"Instance {instance_idx} not found in prompt {prompt_key} (should not happen after validation)")
            elif len(prompt_row) > 1:
                raise ValueError(f"Instance {instance_idx} appears {len(prompt_row)} times in prompt {prompt_key}")
            else:
                Y_hat[i, k_idx] = prompt_row.iloc[0]['llm_choice']
    
    if require_wait_time:
        wait_time_groups = df_human['wait_time'].values  # (N,)
        return Y, Y_hat, wait_time_groups, np.array(all_instances)
    else:
        return Y, Y_hat, np.array(all_instances)

def brier_loss(w, Y, Y_hat):
    """
    Compute Brier loss (squared error) between aggregated prediction and human choice.
    
    Args:
        w: Weight vector (K,) - will be normalized to sum to 1
        Y: Human choices (N,)
        Y_hat: LLM predictions (N, K)
    
    Returns:
        float: Total Brier loss
    """
    # Normalize weights to sum to 1 (simplex constraint)
    w_normalized = w / (w.sum() + 1e-10)
    
    # Compute aggregated prediction: p̂_i(w) = Σ_k w_k * ŷ_ik
    p_hat = Y_hat @ w_normalized  # (N,)
    
    # Brier loss: L(w) = Σ_i (p̂_i(w) - y_i)^2
    loss = np.sum((p_hat - Y) ** 2)
    
    return loss

def cross_entropy_loss(w, Y, Y_hat):
    """
    Compute cross-entropy loss between aggregated prediction and human choice.
    
    Args:
        w: Weight vector (K,) - will be normalized to sum to 1
        Y: Human choices (N,) - binary labels (0 or 1)
        Y_hat: LLM predictions (N, K) - binary predictions (0 or 1)
    
    Returns:
        float: Total cross-entropy loss
    """
    # Normalize weights to sum to 1 (simplex constraint)
    w_normalized = w / (w.sum() + 1e-10)
    
    # Compute aggregated prediction: p̂_i(w) = Σ_k w_k * ŷ_ik
    p_hat = Y_hat @ w_normalized  # (N,)
    
    # Clip to avoid log(0) or log(1) for numerical stability
    p_hat = np.clip(p_hat, 1e-10, 1 - 1e-10)
    
    # Cross-entropy loss: L(w) = -Σ_i [y_i * log(p̂_i) + (1-y_i) * log(1-p̱_i)]
    loss = -np.sum(Y * np.log(p_hat) + (1 - Y) * np.log(1 - p_hat))
    
    return loss

def group_aggregated_loss(w, Y, Y_hat, wait_time_groups):
    """
    Compute group-level aggregated loss (sample-size-weighted squared error).
    
    Groups instances by wait_time (g ∈ {1, 2, 3, 4}) and computes:
    - n_g: number of instances in group g
    - ȳ_g: empirical human purchase rate in group g
    - p̂_gk: prompt-level predicted purchase rate for prompt k in group g
    - p̂_g(w): aggregated prediction for group g using weights w
    - Loss: L_group(w) = Σ_g n_g (ȳ_g - p̂_g(w))^2
    
    Args:
        w: Weight vector (K,) - will be normalized to sum to 1
        Y: Human choices (N,)
        Y_hat: LLM predictions (N, K)
        wait_time_groups: Wait time values (N,) - should be in {1, 2, 3, 4}
    
    Returns:
        float: Total group-level aggregated loss
    """
    # Normalize weights to sum to 1 (simplex constraint)
    w_normalized = w / (w.sum() + 1e-10)
    
    # Get unique wait_time groups (should be {1, 2, 3, 4})
    unique_groups = np.unique(wait_time_groups)
    unique_groups = sorted([g for g in unique_groups if g in [1, 2, 3, 4]])
    
    total_loss = 0.0
    
    for g in unique_groups:
        # Get indices for instances in group g
        group_mask = (wait_time_groups == g)
        n_g = np.sum(group_mask)
        
        if n_g == 0:
            continue
        
        # Compute empirical human purchase rate: ȳ_g = (1/n_g) Σ_{i∈g} y_i
        y_bar_g = np.mean(Y[group_mask])
        
        # Compute prompt-level predicted purchase rates: p̂_gk = (1/n_g) Σ_{i∈g} ŷ_ik
        p_hat_gk = np.mean(Y_hat[group_mask, :], axis=0)  # (K,)
        
        # Compute aggregated prediction: p̂_g(w) = Σ_k w_k p̂_gk
        p_hat_g = np.dot(p_hat_gk, w_normalized)
        
        # Add to loss: n_g * (ȳ_g - p̂_g(w))^2
        group_loss = n_g * (y_bar_g - p_hat_g) ** 2
        total_loss += group_loss
    
    return total_loss

def optimize_prompt_weights(Y, Y_hat, method='softmax', loss_function='brier', wait_time_groups=None):
    """
    Optimize prompt weights to minimize the specified loss function.
    
    Args:
        Y: Human choices (N,)
        Y_hat: LLM predictions (N, K)
        method: 'softmax' (uses softmax parameterization) or 'projected' (uses constraints)
        loss_function: 'brier', 'cross_entropy', or 'brier_aggregated'
        wait_time_groups: Wait time values (N,) - required only for 'brier_aggregated'
    
    Returns:
        dict: {
            'weights': optimized weights (K,),
            'loss': final loss value,
            'method': method used,
            'loss_function': loss function used,
            'optimization_result': scipy optimization result object
        }
    """
    n_prompts = Y_hat.shape[1]
    
    # Validate loss function
    valid_loss_functions = ['brier', 'cross_entropy', 'brier_aggregated']
    if loss_function not in valid_loss_functions:
        raise ValueError(f"loss_function must be one of {valid_loss_functions}, got '{loss_function}'")
    
    # Validate wait_time_groups for brier_aggregated
    if loss_function == 'brier_aggregated':
        if wait_time_groups is None:
            raise ValueError("wait_time_groups is required for 'brier_aggregated' loss function")
    
    # Select loss function
    if loss_function == 'brier':
        def compute_loss(w):
            return brier_loss(w, Y, Y_hat)
    elif loss_function == 'cross_entropy':
        def compute_loss(w):
            return cross_entropy_loss(w, Y, Y_hat)
    else:  # brier_aggregated
        def compute_loss(w):
            return group_aggregated_loss(w, Y, Y_hat, wait_time_groups)
    
    if method == 'softmax':
        # Parameterize via softmax: w_k = exp(θ_k) / Σ_j exp(θ_j)
        # This automatically satisfies w_k >= 0 and Σ w_k = 1
        def loss_fn(theta):
            w = softmax(theta)
            return compute_loss(w)
        
        # Initialize with uniform weights (theta = zeros)
        theta_init = np.zeros(n_prompts)
        
        result = minimize(loss_fn, theta_init, method='BFGS')
        weights_opt = softmax(result.x)
        
    else:  # method == 'projected'
        # Use constraints: w_k >= 0, Σ w_k = 1
        def loss_fn(w):
            return compute_loss(w)
        
        # Initialize with uniform weights
        w_init = np.ones(n_prompts) / n_prompts
        
        # Constraints: sum of weights = 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: w_k >= 0
        bounds = [(0, None) for _ in range(n_prompts)]
        
        result = minimize(loss_fn, w_init, method='SLSQP', 
                         constraints=constraints, bounds=bounds)
        weights_opt = result.x
    
    final_loss = compute_loss(weights_opt)
    
    return {
        'weights': weights_opt,
        'loss': final_loss,
        'method': method,
        'loss_function': loss_function,
        'optimization_result': result
    }

def save_weights_to_csv(results, output_path, prompt_names=None, navigation_prompts=None):
    """
    Save optimized weights to a CSV file.
    
    Args:
        results: Dictionary from optimize_prompt_weights
        output_path: Path where CSV file should be saved
        prompt_names: Optional list of prompt names (default: nav1, nav2, ...)
        navigation_prompts: Optional list of navigation prompt texts (one per prompt variant).
                           If provided, adds a 'prompt_text' column to the CSV for direct use
                           in the 20260114ver simulation code.
    
    Returns:
        str: Path to saved file
    """
    weights = results['weights']
    n_prompts = len(weights)
    
    if prompt_names is None:
        prompt_names = [f'nav{k}' for k in range(1, n_prompts + 1)]
    
    # Create base DataFrame
    weights_df = pd.DataFrame({
        'prompt_variant': prompt_names,
        'optimal_weight': weights,
        'weight_percentage': weights * 100
    })
    
    # Add prompt_text column if navigation_prompts provided
    if navigation_prompts is not None:
        if len(navigation_prompts) == n_prompts:
            # Create a mapping from prompt_name to prompt_text
            prompt_text_map = {f'nav{k}': navigation_prompts[k-1] for k in range(1, n_prompts + 1)}
            weights_df['prompt_text'] = weights_df['prompt_variant'].map(prompt_text_map)
    
    # Sort by weight (descending)
    weights_df = weights_df.sort_values('optimal_weight', ascending=False)
    
    # Save to CSV
    atomic_to_csv(weights_df, output_path, index=False)
    
    return output_path
