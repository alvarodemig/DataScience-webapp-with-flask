# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 13:29:55 2018

@author: Alvaro
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Plot ROC curves
def plot_ROC(X, y, classifier, cv):
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    from scipy import interp
    cv = StratifiedKFold(n_splits=cv)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    #figure = plt.figure()
    plt.gcf().clear()
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png
    
def plot_predVSreal(X, y, classifier, cv):
    from sklearn.model_selection import cross_val_predict
    # cross_val_predict returns an array of the same size as `y` where each entry
    # is a prediction obtained by cross validation:
    predicted = cross_val_predict(classifier, X, y, cv=cv)
    plt.gcf().clear()
    plt.scatter(y, predicted, edgecolors=(0, 0, 0))
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png

def plot_histsmooth(ds, columns):
    sns.set()
    plt.gcf().clear()
    for col in columns:
        sns.distplot(ds[col], label = col)
    from io import BytesIO
    plt.xlabel('')
    plt.legend()
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png

def plot_correlations(ds, corr, corrcat):
    sns.set()
    plt.gcf().clear()
    if corrcat != '': sns.pairplot(ds[corr], hue = corrcat)
    else: sns.pairplot(ds[corr])
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png

def plot_boxplot(ds, cat, num):
    sns.set()
    plt.gcf().clear()
    with sns.axes_style(style='ticks'):
        sns.factorplot(cat, num, data=ds, kind="box")
    from io import BytesIO
    plt.xlabel(cat)
    plt.ylabel(num)
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png


    
    


        