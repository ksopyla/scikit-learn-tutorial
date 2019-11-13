#
import sklearn.metrics as skm
import numpy as np


# binary problem

y_true = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 1], dtype=float)
# y_pred = [0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
y_pred = np.array([0.01, 0.12, 0.89, .99, .05, .76, .14, .87, .44, .32])
y_pred = y_pred > 0.5

y_pred = np.array([0, 0, 1, 1])
y_true = np.array([0, 0, 1, 0])

# y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
# y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0])

cm = skm.confusion_matrix(y_true, y_pred)
print(cm)
print(skm.classification_report(y_true, y_pred))

report = skm.classification_report(y_true, y_pred, output_dict=True)
print(report)

#%%  multiclass problem
y_true = [2, 0, 2, 2, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 2, 2, 0, 2, 1, 0, 2, 2]

cm = skm.confusion_matrix(y_true, y_pred)
print(cm)

labels = ["A", "B", "C"]
print(skm.classification_report(y_true, y_pred, target_names=labels))
report = skm.classification_report(y_true, y_pred, output_dict=True)
print(report)



# multilabel problems


# 2 label binary problem
y_true = np.array([
    [0, 0], [1, 1], [0, 0], [1, 1]
])
y_pred = np.array([
    [0, 0], [1, 1], [0, 1], [0, 1]
])

cm = skm.multilabel_confusion_matrix(y_true, y_pred)
print(cm)
print(skm.classification_report(y_true, y_pred,
                                target_names=['label0==1', 'label1==1']))

# 3 label binary problem
y_true = np.array([
    [0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]
])
y_pred = np.array([
    [0, 0, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1]
])

cm = skm.multilabel_confusion_matrix(y_true, y_pred)
print(cm)
print(skm.classification_report(y_true, y_pred,
                                target_names=['label1==1', 'label2==1', 'label3==1']))



# 2 label multiclass problem
# !!!!! multi label multi output not supported
# y_true = np.array([
#                     [0,0], [1,1], [2,2], [1,1]
#                   ])
# y_pred = np.array([
#                     [0,0], [1,1], [2,2], [0,1]
#                   ])
# cm = skm.multilabel_confusion_matrix(y_true, y_pred)
# print(cm)
# print( skm.classification_report(y_true,y_pred, target_names=['label1==1', 'label2==1'] ))


# %%
