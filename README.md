# Transferable thresholding for predictive models

Transferring and adjusting decision rules for predictive models in new settings, with seamless adaptation between data-scarce and data-rich environments.

---
> Main contributor: <a href="https://github.com/aziz-ayed" target="_blank" style="text-decoration:none; color: #F08080">Aziz Ayed</a>  
> Email: <a href="mailto:aziz.ayed@mit.edu" style="text-decoration:none; color: #F08080">aziz.ayed@mit.edu</a>  
> Documentation: <a href="https://github.com/aziz-ayed/transferable_threshold/blob/main/Experiments.ipynb" target="_blank" style="text-decoration:none; color: #F08080">Notebook</a>  
> Article: TBD  
> Current release: 03/26/2024
---

When applied to the real-world, predictive risk models often come with a threshold on predicted scores that warrants intervention. These thresholds are often chosen to satisfy a condition on the original validation set, such as achieving a minimum specificity/sensitivity or PPV/NPV.
However, these conditions usually do not hold when applying the same threshold to new datasets, due to distribution shift and calibration drift. 
This calls for recomputing the optimal threshold on every new dataset, which requires large amounts of data.  
In this project, we introduce a novel approach to adjust decision thresholds in every new setting, using very limited amounts of data. The methods also adapts to data-rich settings, with proven convergence to the optimal threshold and minimal variance.  

In the corresponding paper, we specifically explore different clinical scenarios. Indeed, this problem is of paramount importance in the clinical world as a miscalibrated threshold can be particularly harmful to patients (e.g.: increasing false negative rates).  
However, the methodology can be applied on top of any predictive model for any specific task. 

1. [Quick run](#quick-run)
1. [Method](#method)
1. [Results](#results)
   * [Dataset](#dataset)
   * [Experiments](#experiments)


## Quick run

In the <a href="https://github.com/aziz-ayed/transferable_threshold/blob/main/Experiments.ipynb" target="_blank" style="text-decoration:none; color: #F08080">Experiments notebook</a>, we provide a full rundown of the method and the other baselines presented in the paper on a synthetic dataset. We also show how to reproduce the experiments presented in the paper. In this codebase, "tau" is used to characterize a target PPV. This can easily be adjusted for different objectives (NPV, specificity, sensitivity).

For a quick run on your own dataset:

```python
from utils.utils import *
from utils.models import *

# Initialize the method
transfer = TransferableThreshold()

# Compute the optimal threshold to achieve a PPV of alpha
transfer.pipeline(df_source, df_target, domain, risk, 
                                            gold, alpha=alpha)["tau"]
```

The variables "risk" and "gold" should be replaced by the corresponding names in your dataset for respectively: the column containing the risk scores and the one containing the true outcomes. The "domain" variable should indicate the domain of origin. For example, if we are working with different hospitals, this column should contain each hospital's ID.
Finally, df_source is the dataset from which we are transferring the threshold and df_target is the new domain.

## Method

Our approach is based on importance sampling. The key idea for importance sampling is to align the distribution of our original large dataset to the distribution of our new limited dataset. This allows to leverage the large amount of data in the source domain while still optimizing for the specifics of the target. Formally, this can be written as follows, with pi_0 being the pdf of our target, pi_1 the pdf of our source, and f the function we want to optimize (e.g.: PPV):

<p align="center">
<img width="441" alt="Importance sampling" src="https://github.com/aziz-ayed/transferable_threshold/assets/57011275/57e4567b-89a9-4070-b3ba-60e81c354719">
</p>

Here, w corresponds to the density ratio of the two distributions. We provide more explanations on how to estimate w, reduce variance of the overall estimator, and ensure overall convergence in our paper. 

More intuitively, suppose a model was trained on a large hospital which primarily treats younger patients (Hospital A), and is to be deployed in a smaller rural hospital with an older patient population (Hospital B) and very limited data available. Using demographic and clinical data, we can identify subgroups within Hospital A that are similar to Hospital B’s population (older patients with similar genetic features). Then, we can compute the decision thresholds on Hospital A again, this time assigning larger weights to the identified subgroups. 

## Results

In this section, we report the results of our experiments in the specific clinical scenario of lung cancer screening. More experiments and details can be found in the paper. 

### Dataset

Our target set contains 8,821 patients from MGH, screened for lung cancer. These patients are imaged with low-dose lung CT scans, and their outcomes had been monitored for up to 6 years. Our source hospitals are composed of 17 hospitals from the NLST dataset, and we have access to the same features as for the target domain (clinical features and low-dose CT scans).  
The 1-year risk scores are produced using <a href="https://ascopubs.org/doi/10.1200/JCO.22.01345" style="text-decoration:none; color: #F08080">Sybil</a>, a Deep Learning tool that assesses the probability of developing lung cancer based on low-dose CT scans. 

The <a href="https://cdas.cancer.gov/datasets/nlst/" style="text-decoration:none; color: #F08080">NLST dataset</a> is publicly available, upon reasonable request from the National Cancer Institute.  

For illustration purposes, we provide a function to generate synthetic dataset and play with the method in the <a href="https://github.com/aziz-ayed/transferable_threshold/blob/main/Experiments.ipynb" target="_blank" style="text-decoration:none; color: #F08080">Experiments notebook</a>.

### Experiments

In this example, we are trying to find the threshold that will achieve a given target PPV on our new MGH dataset. A threshold too high will have higher PPV but more false negatives, putting patients at risk. A threshold too low will have lower PPV thus more false positives, increasing costs for the hospital. Thus, it is important to be as close to the target PPV as possible.  
We extract a small subsample of size n from the target data and we use it to estimate the optimal threshold, to mimic a data-poor scenario. We then use the full dataset to assess the performance of the selected threshold, in terms of error to the desired PPV (lower is better). We compare our method to different baselines, and we vary n from 100 to 5000 to show the peformance in different data regimes. Each experiment is reproduced 20 times to produce 90% confidence intervals. We observe that our methods achieves similar performance with 200 samples in the new domain as other baselines with 2000 to 4000 samples, reducing the data requirements for threshold adjustment by 10 to 20 times, with significantly reduced variance overall.
<p align=center>
<img width="1066" alt="Results" src="https://github.com/aziz-ayed/transferable_threshold/assets/57011275/3f4f442c-70fa-4e57-90b0-77db1692baf0">
</p>

