# Transferable thresholding of predictive models

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
1. [Datasets](#datasets)
1. [Models](#models)
1. [Results](#results)


## Quick run

In the <a href="https://github.com/aziz-ayed/transferable_threshold/blob/main/Experiments.ipynb" target="_blank" style="text-decoration:none; color: #F08080">Experiments notebook</a>, we provide a full rundown of how to run the method and the other baselines presented in the paper on a synthetic dataset. We also show how to reproduce the experiments presented in the paper. In this codebase, "tau" is used to characterize a target PPV. This can easily be adjusted for different targets (NPV, specificity, sensitivity).

For a quick run on your own dataset:

```python
from utils.utils import *
from utils.models import *

# Initialize the method
transfer = TransferableThreshold()

# Compute the optimal threshold
transfer.pipeline(df_source, df_target, domain, risk, 
                                            gold, alpha=alpha)["tau"]
```

The variables "risk" and "gold" should be replaced by the corresponding names in your dataset for respectively: the column containing the risk scores and the one containing the true outcomes. The "domain" variable should indicate the domain of origin. For example, if we are working with different hospitals, this column should contain each hospital's ID.
Finally, df_source is the dataset from which we are transferring the threshold and df_target is the new domain.

## Method

Our approach is based on importance sampling. The key idea for importance sampling is to align the distribution of our original large dataset to the distribution of our new limited dataset. This allows to leverage the large amount of data in the source domain while still optimizing for the specifics of the target. Formally, this can be written as follows:

## Results

### Datasets


### Experiments



