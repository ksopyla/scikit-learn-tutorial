"""Example of computing precision recall curve
"""
#%%

import matplotlib.pyplot as plt
import numpy as np

import sklearn.metrics as skm

y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1])
y_score = np.array([0.1, 0.4, 0.35, 0.7, 0.2, 0.3, 0.6, 0.8])


# tresholding, above 0.4
y_pred = (y_score>=0.4).astype(int)
print(y_pred)

# tresholding, above 0.6
y_pred = (y_score>=0.6).astype(int)
print(y_pred)

#%%

# if we want to compute prec-recall for class 0, and treat it as positive class we need to negate
# values in y_true that 0 becomes 1 or shift y_scores
precision0, recall0, tresholds0 = skm.precision_recall_curve(y_true, 1-y_score, pos_label=0)

precision1, recall1, tresholds1 = skm.precision_recall_curve(y_true, y_score, pos_label=1)

# print how precision and recall look like at each treshold for class1
for tr, prec, rec in zip(tresholds1, precision1, recall1):
    y_pred = (y_score>=tr).astype(int)
    cm = skm.confusion_matrix(y_true, y_pred)
    # we have to transpose matrix, sklearn shows true values in rows, and predicted in columns
    # I'm used to different format, true values in columns and predicted in rows
    print(cm.T)
    print(skm.classification_report(y_true, y_pred))
    print(f'Treshold={tr},precision={prec}, recall={rec} predictions={y_pred}')

#%% plot curve
plt.plot(recall0, precision0, 'ro')
plt.plot(recall0, precision0, 'r', label='class 0')

plt.plot(recall1, precision1, 'bo')
plt.plot(recall1, precision1, 'b', label='class 1')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve')
plt.legend()
plt.show()          

# %%
