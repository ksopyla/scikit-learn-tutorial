#%%
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm



y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# model1 curve will dominate the model0
# output from model0
y_score0 = np.array([0.75, 0.5, 0.3, 0.35, 0.45, 0.7, 0.3, 0.33, 0.5, 0.8]) 
# output from model1
y_score1 = np.array([0.6, 0.3, 0.3, 0.55, 0.65, 0.4, 0.55, 0.33, 0.75, 0.3])

# model 1 is better
# # output from model0
# y_score0 = np.array([0.75, 0.5, 0.3, 0.35, 0.45, 0.7, 0.3, 0.33, 0.5, 0.8])
# # output from model1
# y_score1 = np.array([0.7, 0.3, 0.3, 0.55, 0.75, 0.4, 0.5, 0.33, 0.72, 0.3])

# looking only at curves it is not so obvious, which one is better
# output from model0
# y_score0 = np.array([0.7, 0.45, 0.3, 0.35, 0.45, 0.7, 0.3, 0.33, 0.55, 0.8])
# # output from model1
# y_score1 = np.array([0.6, 0.3, 0.3, 0.55, 0.65, 0.4, 0.5, 0.33, 0.75, 0.3])


# %


# first model
fpr0, tpr0, tresholds0 = skm.roc_curve(y_true, y_score0)

# second model

fpr1, tpr1, tresholds1 = skm.roc_curve(y_true, y_score1)


auc_roc0 = skm.roc_auc_score(y_true, y_score0)
auc0 = skm.auc(fpr0,tpr0)
print(f"Model 0 auc roc={auc_roc0} area under curve={auc0}")

auc_roc1 = skm.roc_auc_score(y_true, y_score1)
auc1 = skm.auc(fpr1,tpr1)
print(f"Model 0 auc roc={auc_roc1} area under curve={auc1}")


# % plot curve
plt.plot(fpr0,tpr0, 'ro')
plt.plot(fpr0,tpr0, 'r', label='model 0')

plt.plot(fpr1,tpr1, 'bo')
plt.plot(fpr1,tpr1, 'b', label='model 1')

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('ROC curve for 2 ml models')
plt.legend()
plt.show()

# %%
