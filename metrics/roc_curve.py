"""Example of computing ROC curve
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm

#y_true = np.array([0, 0, 0, 1, 0, 1, 1, 1])
y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1])
y_score = np.array([0.1, 0.4, 0.35, 0.7, 0.2, 0.3, 0.6, 0.8])

#%%
fpr, tpr, tresholds = skm.roc_curve(y_true, y_score, pos_label=1)

print(f'\nfpr={fpr}\ntpr={tpr}\ntre={tresholds}')


for tr, tp, fp in zip(tresholds, tpr, fpr):
    y_pred = (y_score>=tr).astype(int)
    cm = skm.confusion_matrix(y_true, y_pred)
    print(cm)
    print(skm.classification_report(y_true, y_pred))
    print(f'Treshold={tr},tpr={tp}, fpr={fp} predicions={y_pred}')


#%%
plt.plot(fpr, tpr, lw=1)
plt.scatter(fpr,tpr)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
plt.show()

# %%
