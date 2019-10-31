import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


y_test = np.array([0, 0, 1, 1,1])
y_score = np.array([0.1, 0.4, 0.35, 0.6, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, pos_label=1)

auc_val = metrics.auc(fpr, tpr)


print(f'\nfpr={fpr}\ntpr={tpr}\ntre={thresholds}')
print(f'auc={auc_val}')

plt.plot(fpr, tpr, lw=1)
plt.scatter(fpr,tpr)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
plt.show()