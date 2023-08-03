import numpy as np
import pandas as pd
import scipy.stats as t
from sklearn.utils import resample


def find_threshold(df, precision = 0.1, gran = 101, risk = "risk_6", golds = "golds"):
    
    results = np.zeros(gran)
    linsp = np.linspace(0, 1, gran)
    
    idx = 0
    
    for i in linsp:
        
        sub_df = df[df[risk] >= i]
        results[idx] = len(sub_df) and len(sub_df[sub_df[golds] == 1]) / len(sub_df) or -1
        idx += 1
    
    thresh = next((i for i, x in enumerate(results) if x >= precision), 0)
    
    return (linsp[thresh] or 0), results


def compute_precision(scores, y, tau, weights = None):
    
    ## P(Y = 1 l s >= tau)   
    
    scores = np.array(scores)
    y = np.array(y)
    
    if weights is None:
        weights = np.ones(len(scores))
    else:
        weights = np.array(weights)
    try:
        
        return np.average(y[scores >= tau], weights = weights[scores >= tau])
    
    except:
        
        return np.nan
    
def compute_npv(scores, y, tau, weights = None):
    
    ## P(Y = 1 l s <= tau)   
    
    scores = np.array(scores)
    y = np.array(y)
    
    if weights is None:
        weights = np.ones(len(scores))
    else:
        weights = np.array(weights)
    try:
        
        return 1 - np.average(y[scores < tau], weights = weights[scores < tau])
    
    except:
        
        return np.nan    

def find_tau(scores, y, alpha, weights = None):
    
    scores = np.array(scores)
    y = np.array(y)
    
    if weights is None:    
        weights = np.ones(len(scores))
    else:
        weights = np.array(weights)
        scores = scores[weights > 0]
        y = y[weights > 0]
        weights = weights[weights > 0]
        
    weighted_score = weights * y
    
    weighted_score = weighted_score
    
    permutations = np.argsort(scores)[::-1]
    
    weights = weights[permutations]
    scores = scores[permutations]
    y = y[permutations]
    
    c_weights = np.cumsum(weights)
    c_wy = np.cumsum(weights * y)
    
    precisions = c_wy / c_weights
    
    try:
        return scores[np.max(np.where(precisions >= alpha)[0])]
    except: 
        return np.nan
    
def compute_coverage(scores, tau):
    
    scores = np.array(scores)
    
    return len(scores[scores >= tau]) / len(scores)    
    
def weighted_average(values, weights):
    """
    Compute a weighted average ignoring NaN values.
    
    Args:
    values (array-like): The values to be averaged.
    weights (array-like): The corresponding weights for each value.
    
    Returns:
    The weighted average of the values.
    """
    # Convert values and weights to numpy arrays
    values = np.array(values)
    weights = np.array(weights)

    # Mask NaN values in values and corresponding weights
    mask = ~np.isnan(values)
    values = values[mask]
    weights = weights[mask]

    # Compute weighted average
    if np.sum(weights) == 0:
        return np.nan
    else:
        return np.sum(values * weights) / np.sum(weights)
    
def weighted_median(values, weights):
    """
    Compute a weighted average ignoring NaN values.
    
    Args:
    values (array-like): The values to be averaged.
    weights (array-like): The corresponding weights for each value.
    
    Returns:
    The weighted average of the values.
    """
    # Convert values and weights to numpy arrays
    values = np.array(values)
    weights = np.array(weights)

    # Mask NaN values in values and corresponding weights
    mask = ~np.isnan(values)
    values = values[mask]
    weights = weights[mask]
    
    # Sort values and weights based on values
    sorted_indices = np.argsort(values)
    values = values[sorted_indices]
    weights = weights[sorted_indices]

    # Compute cumulative sum of the sorted weights
    cum_sum_weights = np.cumsum(weights)

    # Find the index where cumulative sum becomes greater than or equal to half the total weight sum
    total_weight_sum = np.sum(weights)
    median_index = np.searchsorted(cum_sum_weights, total_weight_sum / 2.0, side='right')

    # Compute the weighted median
    if median_index == 0:
        return np.nan
    elif cum_sum_weights[median_index - 1] == total_weight_sum / 2.0:
        return values[median_index - 1]
    else:
        return values[median_index]
    
def stratified_sample(df, strat_col, frac, state = 0):
    def sampling_strategy(group):
        return group.sample(frac=frac, random_state = state)
    return df.groupby(strat_col, group_keys=False).apply(sampling_strategy)
    
def compute_mean_ci(data, confidence_level=0.95):
    '''
    Computes the confidence interval of the mean for a given vector of data.
    
    Args:
    - data: A numpy array or list of data.
    - confidence_level: The desired level of confidence for the interval. Default is 0.95.
    
    Returns:
    - A tuple containing the lower and upper bounds of the confidence interval.
    '''
    
    # Calculate the sample mean and standard deviation
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    # Calculate the t-value for the specified confidence level with (n-1) degrees of freedom
    n = len(data)
    t_value = t.ppf((1 + confidence_level) / 2, df=n-1)
    
    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = mean - t_value * std / np.sqrt(n)
    upper_bound = mean + t_value * std / np.sqrt(n)
    
    # Return the confidence interval as a tuple
    return (lower_bound, upper_bound)
    
def compute_metrics(df, alpha = 0.1, min_eval = 300, upsample = True, normalize = False, weighted = True, confidence = "CI"):
    
    debug = pipeline(df = df, alpha = alpha, upsample = upsample, normalize = normalize, weighted = weighted)
    
    mean_alpha = np.mean([elem['alpha_emp'] for elem in debug.values() if elem['n_samples'] > min_eval])
    mean_error = mean_absolute_error([elem['alpha_emp'] for elem in debug.values() if elem['n_samples'] > min_eval], 
                    [alpha for elem in debug.values() if elem['n_samples'] > min_eval])
    mean_coverage = np.mean([elem['coverage'] for elem in debug.values() if elem['n_samples'] > min_eval])
    if confidence == "std":
        std_alpha = np.std([np.abs(elem['alpha_emp'] - alpha) 
                            for elem in debug.values() if elem['n_samples'] > min_eval])
    elif confidence == "CI":
        std_alpha = compute_mean_ci([np.abs(elem['alpha_emp'] - alpha) 
                            for elem in debug.values() if elem['n_samples'] > min_eval], 0.95)
    
    return {'alpha': mean_alpha, 'error': mean_error, 'coverage': mean_coverage, 'std_alpha': std_alpha}