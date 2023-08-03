import numpy as np
import pandas as pd
import scipy.stats as t
from sklearn.linear_model import LogisticRegression
from utils.utils import *

def pipeline_multi(df, df_mgh, domain_col = "institution", upsample = True, weighted = False,
                    random_state = 0, alpha = 0.2, frac_sample = 0.03, option = None, include_target = False):
    
    features = ["risk_" + str(i+1) for i in range(6)] + ["age", "race", "cigsmok", "smokeday", "golds", domain_col]
    
    df = df[features]
    
    domains = np.unique(df[domain_col])
    
    dict_results = {}
    
    test_domain = "mgh"
    taus = []
    length_j = []
    weights_vector = []
        
    for j in domains:

        train_domains = j

        source = df[df[domain_col] == train_domains]
        #source = resample(source, n_samples = 1000, random_state = random_state)

        target = stratified_sample(df_mgh, "golds", frac_sample, state = random_state)

        if upsample:

            if len(source) < len(target):

                source_upsample = resample(source, n_samples = len(target), random_state = random_state)
                target_upsample = target

            elif len(target) < len(source):

                target_upsample = resample(target, n_samples = len(source), random_state = random_state)
                source_upsample = source

        else:

            source_upsample = source
            target_upsample = target

        mixed_dist = pd.concat([source_upsample, target_upsample])

        y = (mixed_dist[domain_col] == test_domain) * 1

        mixed_dist = mixed_dist.drop(domain_col, axis = 1)
        mixed_dist = mixed_dist.fillna(mixed_dist.mean())

        log_reg = LogisticRegression(random_state = 0, max_iter = 300)

        log_reg.fit(mixed_dist.drop(["golds"], axis = 1), y)

        probas = log_reg.predict_proba(source.drop([domain_col] + ["golds"], axis = 1))
        importance_weights = probas[:, 1] #/ (probas[:, 0] + 0.00001)
        
        ess = (np.sum(importance_weights) ** 2) / np.sum(importance_weights ** 2)
        
        tau = find_tau(scores = source["risk_6"],
                 y = source["golds"],
                 alpha = alpha,
                 weights = importance_weights)

        taus.append(tau)
        weights_vector.append(ess)
        length_j.append(len(source))
        #weights_vector.append(np.sum(importance_weights))
        
    if include_target:
        
        tau_target = find_tau(scores = target["risk_6"],
                     y = target["golds"],
                     alpha = alpha,
                     weights = np.ones(len(target)))

        taus.append(tau_target)
        length_j.append(len(target))
        weights_vector.append(len(target))
    
    if option == "std":
        
        taus = np.array(taus)
        length_j = np.array(length_j)
        weights_vector = np.array(weights_vector)
        taus_mean = np.nanmean(taus)
        taus_std = np.nanstd(taus)
        length_j = length_j[(taus >= taus_mean - taus_std) & (taus <= taus_mean + taus_std)]
        weights_vector = weights_vector[(taus >= taus_mean - taus_std) & (taus <= taus_mean + taus_std)]
        taus = taus[(taus >= taus_mean - taus_std) & (taus <= taus_mean + taus_std)]

    
        
    if weighted:
        #tau = weighted_average(taus, np.sqrt(length_j))
        #tau = weighted_average(taus, weights_vector)
        tau = weighted_median(taus, weights_vector)
    else:    
        #tau = np.nanmean(taus)      
        tau = np.nanmedian(taus)
    
    alpha_emp = compute_precision(scores = df_mgh["risk_6"],
                                  y = df_mgh["golds"],
                                  tau = tau) 
    
    npv = compute_npv(scores = df_mgh["risk_6"],
                                  y = df_mgh["golds"],
                                  tau = tau) 

    coverage = compute_coverage(scores = df_mgh["risk_6"], tau = tau)

    dict_results = {'alpha_emp': alpha_emp, 'tau': tau, 'weights_vector': weights_vector,
                            'n_samples': len(df_mgh), 'total_positives': sum(df_mgh['golds'])/len(df_mgh),
                            'coverage': coverage, 'npv': npv, 'taus': taus}
            
    return dict_results

def pipeline_merged(df, df_mgh, domain_col = "institution", upsample = True, 
                    weighted = False, random_state = 0, alpha = 0.2, frac_sample = 0.03):
    
    features = ["risk_" + str(i+1) for i in range(6)] + ["age", "race", "cigsmok", "smokeday", "golds", domain_col]
    
    df = df[features]
    
    domains = np.unique(df[domain_col])
    
    dict_results = {}
    
    test_domain = "mgh"
    taus = []
    length_j = []
    
    source_risks = []
    source_y = []
    source_weights = []
    
    target = stratified_sample(df_mgh, "golds", frac_sample, state = random_state)   
        
    for j in domains:

        train_domains = j

        source = df[df[domain_col] == train_domains]



        if upsample:

            if len(source) < len(target):

                source_upsample = resample(source, n_samples = len(target), random_state = random_state)
                target_upsample = target

            elif len(target) < len(source):

                target_upsample = resample(target, n_samples = len(source), random_state = random_state)
                source_upsample = source

        else:

            source_upsample = source
            target_upsample = target

        mixed_dist = pd.concat([source_upsample, target_upsample])

        y = (mixed_dist[domain_col] == test_domain) * 1

        mixed_dist = mixed_dist.drop(domain_col, axis = 1)
        mixed_dist = mixed_dist.fillna(mixed_dist.mean())

        log_reg = LogisticRegression(random_state = 0, max_iter = 300)

        log_reg.fit(mixed_dist.drop(["golds"], axis = 1), y)

        probas = log_reg.predict_proba(source.drop([domain_col] + ["golds"], axis = 1))
        importance_weights = probas[:, 1] #/ (probas[:, 0] + 0.00001)
            
        source_risks.extend(source["risk_6"])
        source_y.extend(source["golds"])
        source_weights.extend(importance_weights)
    
    #source_risks.extend(target["risk_6"])
    #source_y.extend(target["golds"])
    #source_weights.extend(np.ones(len(target)))
    
    tau = find_tau(scores = source_risks,
                     y = source_y,
                     alpha = alpha,
                     weights = source_weights)  
    
    alpha_emp = compute_precision(scores = df_mgh["risk_6"],
                                  y = df_mgh["golds"],
                                  tau = tau) 

    npv = compute_npv(scores = df_mgh["risk_6"],
                                  y = df_mgh["golds"],
                                  tau = tau) 
    
    coverage = compute_coverage(scores = df_mgh["risk_6"], tau = tau)
    

    dict_results = {'alpha_emp': alpha_emp, 'tau': tau, 
                            'n_samples': len(df_mgh), 'total_positives': sum(df_mgh['golds'])/len(df_mgh),
                            'coverage': coverage, 'npv': npv, 'taus': taus}
            
    return dict_results

def simple_estimator(df, alpha = 0.2, frac_sample = 0.02, random_state=0):
    
    df_copy = stratified_sample(df, "golds", frac = frac_sample, state = random_state)
    
    tau = find_tau(df_copy["risk_6"],
         df_copy["golds"],
         weights=None,
         alpha=alpha)
    
    alpha_emp = compute_precision(scores = df["risk_6"],
                                  y = df["golds"],
                                  tau = tau) 

    coverage = compute_coverage(scores = df["risk_6"], tau = tau)
    
    npv = compute_npv(scores = df["risk_6"],
                                  y = df["golds"],
                                  tau = tau)
    
    if np.isnan(tau) == True:
        alpha_emp = np.nan
        
    dict_results = {'alpha_emp': alpha_emp, 'tau': tau, 'npv': npv,
                            'n_samples': len(df), 'coverage': coverage}    
    
    return dict_results