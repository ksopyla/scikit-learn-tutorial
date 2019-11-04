import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

y_true = np.array([0, 0, 1, 1,1])
scores = np.array([0.1, 0.4, 0.35, 0.6, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y_true, scores, pos_label=1)

print(f'\nfpr={fpr}\ntpr={tpr}\ntre={thresholds}')

for tr, tp, fp in zip(thresholds, tpr, fpr):
    y_pred = (scores>=tr).astype(int)
    cm = metrics.confusion_matrix(y_true, y_pred)
    print(cm)
    print(metrics.classification_report(y_true, y_pred))
    print(f'Treshold={tr},tpr={tp}, fpr={fp} predicions={y_pred}')

plt.plot(fpr, tpr, lw=1)
plt.scatter(fpr,tpr)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
plt.show()