import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_residual(y_true, y_predict):
    tmp = np.ravel(np.hstack([y_true, y_predict]))
    min_val, max_val = tmp.min(), tmp.max()
    
    plt.scatter(y_predict,y_predict-y_true,alpha=0.8,color='#4499dd')
    plt.title("Linear regression")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.hlines(y = 0, xmin = min_val, xmax = max_val, color = "red",alpha=0.3)
    plt.show()

    plt.scatter(y_predict,y_true,alpha=0.8,color='#4499dd')
    plt.title("Linear regression")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.plot([min_val, max_val], [min_val, max_val], c = "red",alpha=0.3)


def plot_roc(y_true,y_predict):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true,y_predict)
    plt.figure(figsize=(6,6))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc_score(y_true,y_predict))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
