"""Example of computing precision recall curve
"""
#%%

import matplotlib.pyplot as plt
import numpy as np

import sklearn.metrics as skm

y_true = np.array([0, 0, 0, 0, 0, 1, 1,1])
y_score = np.array([0.1, 0.4, 0.35, 0.7, 0.2, 0.3, 0.6, 0.8])


average_precision = skm.average_precision_score(y_true, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))


precision, recall, tresholds = skm.precision_recall_curve(y_true, y_score)

for tr, prec, rec in zip(tresholds, precision, recall):
    y_pred = (y_score>=tr).astype(int)
    cm = skm.confusion_matrix(y_true, y_pred)
    print(cm)
    print(skm.classification_report(y_true, y_pred))
    print(f'Treshold={tr},precision={prec}, recall={rec} predicions={y_pred}')

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'})
#%%
plt.plot(recall, precision, 'bo')
plt.plot(recall, precision, 'b')
#plt.step(recall, precision, color='b', alpha=0.2, where='post')
#plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
plt.show()          

# %%
