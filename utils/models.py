import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import calibration as cal
from utils.utils import *

class TransferableThreshold:
    
    def perform_sampling(self, source, target, direction, random_state):
        """
        Perform upsampling or downsampling to balance the source and target datasets based on the direction.

        Args:
        - source (array-like): Source dataset.
        - target (array-like): Target dataset.
        - direction (str): Direction of sampling; "up" for upsampling, "down" for downsampling.
        - random_state (int): Seed for the random number generator for reproducibility.

        Returns:
        Tuple[array-like, array-like]: Balanced source and target datasets.
        """
        if direction == "down":
            # Downsampling
            if len(source) <= len(target):
                target = resample(target, n_samples=len(source), random_state=random_state)
            else:
                source = resample(source, n_samples=len(target), random_state=random_state)
        elif direction == "up":
            # Upsampling
            if len(source) < len(target):
                source = resample(source, replace=True, n_samples=len(target), random_state=random_state)
            elif len(target) < len(source):
                target = resample(target, replace=True, n_samples=len(source), random_state=random_state)
        elif direction == 'none':
            #No sampling
            source, target = source, target

        else:
            raise ValueError("Invalid direction. Choose either 'up', 'down', or 'none'.")

        return source, target

    def produce_weights(self, source_upsample, target_upsample, source, domain_col, golds_col, target_domain, random_state):
        """
        This function concatenates the upsampled source and target datasets, trains a logistic regression model
        to distinguish between the two based on the available features, and then calculates importance weights
        for the original source data using the trained model.

        Args:
        - source_upsample (DataFrame): The upsampled source data.
        - target_upsample (DataFrame): The upsampled target data.
        - source (DataFrame): The original source data before upsampling.
        - domain_col (str): The column name indicating the domain.
        - golds_col (str): The column name indicating the gold standard or labels.
        - target_domain (str/int): The value in domain_col that indicates the target domain.
        - random_state (int): The seed for the random number generator.

        Returns:
        - weights (ndarray): An array of importance weights for the source data.
        """
        # Concatenate the upsampled source and target datasets
        mixed_dist = pd.concat([source_upsample, target_upsample])

        # Create the target variable where 1 indicates the target domain and 0 otherwise
        y = (mixed_dist[domain_col] == target_domain).astype(int)

        # Drop the domain and golds columns and impute missing values with the mean
        mixed_dist_prepared = mixed_dist.drop(columns=[domain_col, golds_col]).fillna(mixed_dist.mean())

        # Initialize and fit the logistic regression model
        model = LogisticRegression(random_state=random_state, max_iter=300)
        model.fit(mixed_dist_prepared, y)

        # Prepare the original source data similarly by dropping columns and imputing missing values
        source_prepared = source.drop(columns=[domain_col, golds_col]) #.fillna(mixed_dist.mean())

        # Predict probabilities for the original source data
        probas = model.predict_proba(source_prepared)

        # Calculate importance weights as the ratio of probabilities favoring the target domain
        weights = probas[:, 1] / (probas[:, 0] + 1e-5)

        return weights

    def pipeline(self, df_source, df_target, domain_col, risks_col, golds_col, alpha, sampling_direction = "down", 
             normalize = False, frac_sample = 1, combin_factor = True, include_target = True, random_state = 0):
    
    
        """
        A function that implements the full threshold transfer pipeline. Includes options for upsampling or downsampling,
        normalization, and inclusion of target data and weighting scheme for multiple source domains in the analysis.

        Parameters:
        - df_source (DataFrame): Source dataset containing the domains, risks, and other relevant information.
        - df_target (DataFrame): Target dataset for the analysis.
        - domain_col (str): Column name in the datasets representing the domain.
        - risks_col (str): Column name representing the risk factors or scores.
        - golds_col (str): Column name representing the gold standard or outcome.
        - alpha (float): Target PPV that we are trying to achieve. 
        - sampling_direction (str, optional): Sampling direction to balance the datasets. "down" for downsampling, "up" for upsampling. Defaults to "down".
        - normalize (bool, optional): Flag to normalize importance weights. Defaults to False.
        - frac_sample (float, optional): Fraction (in percent) of the target dataset to sample for stratification. Defaults to 1 to use all the target data available.
        - combin_factor (bool, optional): Flag to use weighting scheme to combine the different source domains for the tau calculation. Defaults to True.
        - include_target (bool, optional): Flag to include target data in the final analysis. Defaults to True.
        - random_state (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 0.

        Returns:
        - dict_results (dict): A dictionary containing the results of the analysis, including metrics like alpha_empirical, tau, coverage, and NPV.
        """

        # Initialize an empty dictionary to store results
        dict_results = {}

        # Extract unique domains from the source data
        domains = np.unique(df_source[domain_col])

        # Ensure the target dataset contains only one unique domain    
        assert df_target[domain_col].nunique() == 1, "More than one target domain"
        target_domain = df_target[domain_col].unique()[0]

        # Initialize lists to store various metrics and weights
        taus, length_j, source_risks, source_y, source_weights = [], [], [], [], []

        # Sample from the target dataset based on the specified fraction. Useful for experiment purposes, 
        # in production frac_sample should always be equal to 1 to use all the target data available.

        target = df_target.sample(frac = frac_sample, random_state = random_state)

        # Iterate through each domain in the source dataset    
        for j in domains:


            # Filter the source data for the current domain
            source = df_source[df_source[domain_col] == j]

            # Perform sampling based on the specified direction
            source_upsample, target_upsample = self.perform_sampling(source, target, 
                                                                direction = sampling_direction, random_state = random_state)

            # Calculate importance weights
            importance_weights = self.produce_weights(source_upsample, target_upsample, source,
                                                 domain_col, golds_col, target_domain, random_state)

            # Normalize weights if specified        
            if normalize == True:
                    importance_weights = importance_weights / np.sum(importance_weights)

            # Extend lists with the current source data and calculated weights
            source_risks.extend(source[risks_col])
            source_y.extend(source[golds_col])
            source_weights.extend(importance_weights)

        # Include target data in the analysis if specified        
        if include_target:

            source_risks.extend(target[risks_col])
            source_y.extend(target[golds_col])
            source_weights.extend(np.ones(len(target)))

        # Calculate tau using the aggregated data and weights
        tau = find_tau(scores = source_risks,
                         y = source_y,
                         alpha = alpha,
                         weights = source_weights)

        # Apply the weighting scheme for the different source domains described in the paper if combin_factor is True   
        if combin_factor:

            start_idx = 0
            new_weights = []

            for j in domains:

                size = len(df_source[df_source[domain_col] == j])
                end_idx = start_idx + size

                fx = (df_source[df_source[domain_col] == j][risks_col] >= tau) * 1
                fx = fx * source_weights[start_idx:end_idx]
                var_fx = np.var(fx)

                chunk = source_weights[start_idx:end_idx]

                multiplied_chunk = [x * 1/(var_fx + 1e-3) for x in chunk]
                new_weights.extend(multiplied_chunk)

                start_idx = end_idx

            if include_target:

                fx = np.sum((target[risks_col] >= tau) * 1) / len(target)
                var_fx = fx * (1 - fx)

                chunk = source_weights[start_idx::]
                multiplied_chunk = [x * 1/(var_fx + 1e-3) for x in chunk]
                new_weights.extend(multiplied_chunk)    

            # Recalculate tau with the adjusted weights after applying the weighting scheme
            tau = find_tau(scores = source_risks,
                             y = source_y,
                             alpha = alpha,
                             weights = new_weights)

        # Calculate metrics for the available target dataset using the final tau value
        alpha_emp = compute_ppv(scores = df_target[risks_col],
                                      y = df_target[golds_col],
                                      tau = tau) 

        npv = compute_npv(scores = df_target[risks_col],
                                      y = df_target[golds_col],
                                      tau = tau) 

        coverage = compute_coverage(scores = df_target[risks_col], tau = tau)

        # Compile results into a dictionary
        dict_results = {'alpha_emp': alpha_emp, 'tau': tau, 
                                'n_samples': len(df_target), 'total_positives': sum(df_target[golds_col])/len(df_target),
                                'coverage': coverage, 'npv': npv, 'taus': taus}

        return dict_results

class PlattEstimator:
    
    def __init__(self):
        pass
    
    def find_threshold_scaling(self, scores, alpha):
        # Sort the scores in ascending order to evaluate thresholds
        sorted_scores = np.sort(scores)
        num_elements = len(sorted_scores)

        # Calculate the initial sum of all scores and the initial count of elements
        # This is for calculating the average above each threshold
        sum_above_threshold = np.sum(sorted_scores)
        count_above_threshold = num_elements

        # Iterate through each score in the sorted list to use it as a potential threshold
        for threshold in sorted_scores:
            avg_above_threshold = sum_above_threshold / count_above_threshold

            if avg_above_threshold > alpha:
                return threshold

            # Update the sum and count for scores above the next threshold
            sum_above_threshold -= threshold  # Remove the current threshold from the sum
            count_above_threshold -= 1  # Decrement the count of elements above the threshold

        # Return np.nan if no threshold satisfies the condition where
        # the average of scores above it is greater than alpha
        return np.nan

    def pipeline(self, df_target, risks_col, golds_col, random_state=0, alpha=0.2, frac_sample=1):
        """
        Apply Platt scaling to adjust risk scores. Find the optimal threshold based on these scores, and compute performance metrics.

        Parameters:
        - df_target: DataFrame containing the target data.
        - risks_col: The column name for risk scores.
        - golds_col: The column name for gold standard (true outcomes).
        - random_state: Seed for the random number generator.
        - alpha: Threshold for selecting the scaling threshold.
        - frac_sample: Fraction of the sample to use for Platt scaling.

        Returns:
        A dictionary with calibration and performance metrics.
        """
        # Sample a fraction of the data for calibration
        target = df_target.sample(frac=frac_sample, random_state=random_state)

        try:

            # Compute Platt scaling coefficients from the available subsample of the data

            platt_scaler = cal.get_platt_scaler(target[risks_col], target[golds_col])
            target_calibrated = platt_scaler(target[risks_col])

            # Apply Platt scaling to the full dataset
            full_target_calibrated = platt_scaler(df_target[risks_col])

            # Find the optimal threshold based on calibrated scores on the available subset of data
            tau = self.find_threshold_scaling(scores=target_calibrated, alpha=alpha)

            # Compute performance metrics using the full dataset and the found threshold
            alpha_emp = compute_ppv(scores=full_target_calibrated, 
                                          y=df_target[golds_col], tau=tau)

            npv = compute_npv(scores=full_target_calibrated, 
                              y=df_target[golds_col], tau=tau)

            coverage = compute_coverage(scores=full_target_calibrated, tau=tau)

        except Exception as e:
            # Handle any errors that occurred during processing, logging the exception if needed
            print(f"An error occurred during processing: {e}")
            alpha_emp, tau, coverage, npv = np.nan, np.nan, np.nan, np.nan

        # Compile results into a dictionary

        dict_results = {

            'alpha_emp': alpha_emp, 
            'tau': tau,
            'n_samples': len(df_target),
            'total_positives': np.mean(df_target[golds_col]),
            'coverage': coverage, 
            'npv': npv
        }

        return dict_results

class RegressionEstimator:
    
    def __init__(self):
        pass
    
    def calculate_tau_values(self, df, domain_col, risks_col, golds_col, alpha_values):
        result_rows = []
        for alpha in alpha_values:
            tau_values = df.groupby(domain_col).apply(lambda x: find_tau(x[risks_col], x[golds_col], alpha))
            result_rows.append(tau_values)

        # Creating the output DataFrame
        reg_df = pd.DataFrame(result_rows).T
        reg_df.columns = [f'tau_{alpha}' for alpha in alpha_values]
        reg_df.reset_index(inplace=True)
        reg_df.rename(columns={'index': domain_col}, inplace=True)
        reg_df = pd.merge(df, reg_df, on=domain_col, how="left")

        return reg_df
    
    def train_linear_models(self, df, domain_col, risks_col, golds_col, alpha_values):
        lin_regs = {}
        scalers = {}
        features = df.drop(domain_col, axis=1).columns.tolist()
        df = self.calculate_tau_values(df, domain_col, risks_col, golds_col, alpha_values)

        for alpha in alpha_values:
            features_a = features + [f"tau_{alpha}"]
            reg_df_full = df[features_a].dropna()

            scaler = StandardScaler()
            X = scaler.fit_transform(reg_df_full[features])
            y = reg_df_full[f'tau_{alpha}']

            model = LinearRegression().fit(X, y)

            lin_regs[alpha] = model
            scalers[alpha] = scaler

        return lin_regs, scalers
    
    def pipeline(self, df_source, df_target, domain_col, risks_col, golds_col, 
                      compute_regs=True, alpha=0.2, random_state=0, frac_sample=1):
        if compute_regs:
            lin_regs, scalers = self.train_linear_models(df_source, domain_col, risks_col, golds_col, [alpha])
        else:
            assert lin_regs is not None and scalers is not None, "If compute_regs = False, you need to provide pretrained linear regressions and scalers."

        # Assuming the rest of the function implementation follows here.
        
        target = df_target.drop(domain_col, axis=1).sample(frac=frac_sample, random_state=random_state)
        
        X = target.dropna()
        X = scalers[alpha].transform(X)
        tau = lin_regs[alpha].predict(X)
        tau = tau.mean()
        
        alpha_emp = compute_ppv(scores=df_target[risks_col], y=df_target[golds_col], tau=tau) 
        npv = compute_npv(scores=df_target[risks_col], y=df_target[golds_col], tau=tau) 
        coverage = compute_coverage(scores=df_target[risks_col], tau=tau)

        dict_results = {'alpha_emp': alpha_emp, 'tau': tau, 'n_samples': len(df_target), 
                        'total_positives': sum(df_target[golds_col])/len(df_target),
                        'coverage': coverage, 'npv': npv}
        
        return dict_results

def simple_estimator(df_target, risks_col, golds_col, alpha=0.2, frac_sample=1, random_state=0):
    """
    Estimate optimal threshold by directly computing it on the available subsample of the target dataset.

    Parameters:
    - df_target: DataFrame containing the target data.
    - risks_col: String name of the column in df_target that contains risk scores.
    - golds_col: String name of the column in df_target that contains gold standard outcomes.
    - alpha: Threshold for determining significant risk scores.
    - frac_sample: Fraction of df_target to use for sampling.
    - random_state: Seed for the random number generator to ensure reproducibility.

    Returns:
    A dictionary with estimated metrics: empirical alpha (precision), threshold (tau),
    negative predictive value (npv), sample size, and coverage.
    """
    # Sample a fraction of the data for initial estimation
    df_sample = df_target.sample(frac=frac_sample, random_state=random_state)
    
    # Find the threshold (tau) using the sampled data
    tau = find_tau(df_sample[risks_col], df_sample[golds_col], weights=None, alpha=alpha)
    
    # Compute the precision, coverage, and NPV on the full dataset using the found threshold
    alpha_emp = compute_ppv(scores=df_target[risks_col], y=df_target[golds_col], tau=tau)
    coverage = compute_coverage(scores=df_target[risks_col], tau=tau)
    npv = compute_npv(scores=df_target[risks_col], y=df_target[golds_col], tau=tau)
    
    # If tau is NaN, indicating that no valid threshold was found, set alpha_emp to NaN as well
    if np.isnan(tau):
        alpha_emp = np.nan
    
    # Compile the computed metrics into a results dictionary
    dict_results = {
        'alpha_emp': alpha_emp, 
        'tau': tau, 
        'npv': npv,
        'n_samples': len(df_target), 
        'coverage': coverage
    }    
    
    return dict_results







