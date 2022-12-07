def evaluation_block_classifier(model, X, y, X_test, y_test):

    """
    :param model: name of classifier
    :param X: name of X data (non-target)
    :param y: name of y data (target)
    :param X_test: name of X test data (non-target)
    :param y_test: name of y test data (target)
    :param roc_graph: boolean to determine if ROC graph is plotted and displayed
    """

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
    from sklearn.model_selection import cross_val_score
    from numpy import mean as npmean
    import matplotlib.pyplot as plt

    # Get y_preds
    y_preds = model.predict(X_test)

    # Evaluation block for singular metrics (non-cross validated)
    print("Singular classifier metrics on test set:")
    print(f'Accuracy: {accuracy_score(y_test, y_preds)*100:.2f}%')
    print(f'Precision: {precision_score(y_test, y_preds)}')
    print(f'Recall: {recall_score(y_test, y_preds)}')
    print(f'F1: {f1_score(y_test, y_preds)}')
    print('')

    # Evaluation block for cross val
    print("Cross validation classifier metrics on test set:")
    print(f'Accuracy: {npmean(cross_val_score(model, X, y, cv=5, scoring="accuracy"))*100:.2f}%')
    print(f'Precision: {npmean(cross_val_score(model, X, y, cv=5, scoring="precision"))}')
    print(f'Recall: {npmean(cross_val_score(model, X, y, cv=5, scoring="recall"))}')
    print(f'F1: {npmean(cross_val_score(model, X, y, cv=5, scoring="f1"))}')
    print('')

    y_probs = model.predict_proba(X_test)
    y_probs_positive = y_probs[:, 1]


    # Define and gather data to plot ROC curve

    fpr, tpr, thresholds = roc_curve(y_test, y_probs_positive)

    def plot_roc_curve(fpr, tpr):

        # Plots our ROC curve given the FPR and TPR
        plt.plot(fpr, tpr, color='orange', label="ROC")

        # Plot line with no predictive power (baseline)
        plt.plot([0,1], [0,1], color='darkblue', linestyle='--', label='Guessing')

        # Customize
        plt.xlabel('False positive rate (FPR)')
        plt.ylabel('True positive rate (TPR)')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    plot_roc_curve(fpr, tpr)
    print(f'ROC AUC score: {roc_auc_score(y_test, y_probs_positive)}')