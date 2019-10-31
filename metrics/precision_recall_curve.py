"""Example of computing precision recall curve
"""
#%%
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from inspect import signature
from sklearn.metrics import average_precision_score



y_test = np.array([0, 0, 0, 0, 0, 1, 1,1])
y_score = np.array([0.1, 0.4, 0.35, 0.7, 0.2, 0.3, 0.6, 0.8])


average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))


precision, recall, _ = precision_recall_curve(y_test, y_score)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
plt.show()          