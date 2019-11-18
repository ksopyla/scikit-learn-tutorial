"""Example of computing precision recall curve for random and ideal
classificator.
"""
# %%

import matplotlib.pyplot as plt
import numpy as np

import sklearn.metrics as skm


# %% random classifier balanced data
# set random seed for reproducibility
np.random.seed(5)
N = 1000  # change this number try: 10, 100, 1000

pos_class_prob=1.0/3 # try 1.0/2  1.0/3 2.0/3
# generate N random samples [0,1], positive examples are sampled with probability  'pos_class_prob'
y_true = np.random.choice(np.array([0, 1]),N, p=[1-pos_class_prob,pos_class_prob])
y_score=np.random.rand(N)

precision, recall, tresholds=skm.precision_recall_curve(y_true, y_score)

# % plot curve

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


# %% ideal classifier
# set random seed for reproducibility
np.random.seed(5)
N=50

# in order to generate ideal classifier, we first generate scores and then labels
y_score=np.random.rand(N)
y_true = (y_score>=0.5).astype(int)


precision, recall, tresholds=skm.precision_recall_curve(y_true, y_score)
# print how precision and recall look like at each treshold for class1
for tr, prec, rec in zip(tresholds, precision, recall):
    y_pred = (y_score>=tr).astype(int)
    cm = skm.confusion_matrix(y_true, y_pred)
    # we have to transpose matrix, sklearn shows true values in rows, and predicted in columns
    # I'm used to different format, true values in columns and predicted in rows
    print(cm.T)
    print(skm.classification_report(y_true, y_pred))
    print(f'Treshold={tr},precision={prec}, recall={rec} predictions={y_pred}')

#%% plot curve

plt.plot(recall, precision, 'bo')
plt.plot(recall, precision, 'b', label='class 1')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'Precision-Recall curve for ideal classifier, {N} samples')
plt.grid(True)
plt.show()



# %%
