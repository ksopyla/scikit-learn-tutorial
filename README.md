#  Scikit-learn tutorial

Set of examples for scikit-learn self-learning.

## Work in progress...

This tutorial is being created. It is not finished.

## How to measure model performance

### Standard metrics precsion, recall, f1 measure - 

The example shows how to compute basic classifier measures like precision, recall, f1

File: [metrics.py](/metrics/metrics.py)


### precision-recall curve

Examples explain how to interpret the precision-recall curve in an ideal, random case. 
What to do if the curve of two models looks similar.

File:
* [precision-recall-curve.py](/metrics/precision-recall-curve.py)
* [precision-recall-curve_edge_case.py](/metrics/precision-recall-curve_edge_case.py)
* [precision-recall-curve_model_comparision.py](/metrics/precision-recall-curve_model_comparision.py)

![Precision-recall curve 2 models- comparision easy](/img/precision_recall_curve_model_comparision_easy.png)

![Precision-recall curve 2 models- comparision not so obvious](/img/precision_recall_curve_model_comparision.png)



## Dev environment

* python > 3.6
* pipenv
* sklearn >0.21.3


