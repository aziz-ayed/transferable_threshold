import numpy as np
import pandas as pd
import scipy.stats as t
from sklearn.utils import resample


def compute_ppv(scores, y, tau, weights=None):
    """
    Computes the weighted PPV.
    
    Parameters:
    - scores: Array-like, predictive risk scores for each observation.
    - y: Array-like, actual binary outcomes (1 for positive, 0 for negative).
    - tau: Threshold score for considering a prediction as positive.
    - weights: Optional, array-like, weights for each observation. If None, all observations are equally weighted.
    
    Returns:
    - Weighted precision as a float, or np.nan if computation is not possible.
    """
    
    scores = np.array(scores)
    y = np.array(y)
    weights = np.ones_like(scores) if weights is None else np.array(weights)
    
    # Filter based on the score threshold
    valid_indices = scores >= tau
    scores_filtered = scores[valid_indices]
    y_filtered = y[valid_indices]
    weights_filtered = weights[valid_indices]
    
    # Check if there are any scores above the threshold to avoid division by zero
    if scores_filtered.size > 0:
        return np.average(y_filtered, weights=weights_filtered)
    else:
        return np.nan
    
def compute_npv(scores, y, tau, weights=None):
    """
    Computes the weighted NPV for a given threshold tau.
    
    Parameters:
    - scores: Array-like, predictive scores for each observation.
    - y: Array-like, actual binary outcomes (0 for negative, 1 for positive).
    - tau: Threshold score for considering a prediction as negative.
    - weights: Optional, array-like, weights for each observation. If None, all observations are equally weighted.
    
    Returns:
    - Weighted NPV as a float, or np.nan if computation is not possible.
    """
    
    scores = np.array(scores)
    y = np.array(y)
    weights = np.ones_like(scores) if weights is None else np.array(weights)
    
    # Filter based on the score threshold for negative predictions
    negative_indices = scores < tau
    scores_negative = scores[negative_indices]
    y_negative = y[negative_indices]
    weights_negative = weights[negative_indices]
    
    if scores_negative.size > 0:
        # Compute NPV as 1 - weighted average of y in negatives, since y=1 indicates a false negative
        true_negative_rate = np.average((y_negative == 0).astype(int), weights=weights_negative)
        return true_negative_rate
    else:
        return np.nan

def find_tau(scores, y, alpha, weights=None):
    
    """
    Find the smallest threshold (tau) for scores such that the weighted PPV is at least alpha.
    
    Parameters:
    - scores: Array-like, the risk scores assigned to each observation.
    - y: Array-like, the binary outcome associated with each score (1 for positive, 0 for negative).
    - alpha: The minimum desired PPV.
    - weights: Optional, array-like, the weights for each observation. If not provided, all observations are equally weighted.
    
    Returns:
    - The threshold score (tau) if such a threshold exists; otherwise, np.nan.
    """
    
    scores = np.array(scores)
    y = np.array(y)
    weights = np.ones_like(scores) if weights is None else np.array(weights)
    
    # Filter arrays based on positive weights
    filter_pos_weights = weights > 0
    scores, y, weights = scores[filter_pos_weights], y[filter_pos_weights], weights[filter_pos_weights]
    
    # Sort by scores in descending order
    perm = np.argsort(scores)[::-1]
    scores, y, weights = scores[perm], y[perm], weights[perm]
    
    # Compute cumulative weighted sums
    cum_weights = np.cumsum(weights)
    cum_weighted_y = np.cumsum(weights * y)
    
    # Calculate PPV at each threshold
    ppv = cum_weighted_y / cum_weights
    
    # Find the smallest threshold where PPV meets or exceeds alpha
    valid_indices = np.where(ppv >= alpha)[0]
    if valid_indices.size > 0:
        return scores[np.max(valid_indices)]
    else:
        return np.nan    
        
def compute_coverage(scores, tau):
    """
    Computes the coverage (proportion of scores greater than or equal to tau) of a given set of scores.
    
    Parameters:
    - scores: Array-like, the scores to be evaluated.
    - tau: Float, the threshold score for coverage calculation.
    
    Returns:
    - The coverage as a float, representing the proportion of scores >= tau.
    """
    
    scores = np.array(scores)
    coverage = np.mean(scores >= tau)
    
    return coverage
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
    
def stratified_sample(df, strat_col, frac, state=0):
    """
    Performs stratified sampling from a DataFrame based on a specified column.
    
    Parameters:
    - df: DataFrame from which to sample.
    - strat_col: String, the column name to stratify by.
    - frac: Float, the fraction of the data to sample from each stratum.
    - state: Int, random state for reproducibility.
    
    Returns:
    - DataFrame, stratified sample of the original DataFrame.
    """
    
    return df.groupby(strat_col, group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=state))

    
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