"""Example of computing precision recall curve for random and ideal 
classificator.
"""
#%%

import matplotlib.pyplot as plt
import numpy as np

import sklearn.metrics as skm


#%% random classifier
# for reproducibility
np.random.seed(5)
N=1000
y_true = np.random.randint(0,2,N)
y_score = np.random.rand(N)

precision, recall, tresholds = skm.precision_recall_curve(y_true, y_score)

#% plot curve

plt.plot(recall, precision, 'bo')
plt.plot(recall, precision, 'b', label='class 1')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'Precision-Recall curve for random classifier, {N} samples')
plt.grid(True)
plt.show()          

 # %%


# %%
