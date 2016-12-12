import pandas as pd
import numpy as np
from scipy.stats import randint
from sklearn.grid_search import RandomizedSearchCV

decision_tree_params = {
    "max_depth": randint(2, 10),
    "max_features": randint(1, 11),
    "min_samples_split": randint(1, 11),
    "min_samples_leaf": randint(1, 11),
    "criterion": ["gini", "entropy"]
}

svc_params = {
    "C": np.linspace(0.1, 2, 20),
    "loss": ['hinge', 'squared_hinge']
}

kneighbors_params = {
    "n_neighbors": randint(1, 1000),
    "weights": ['uniform', 'distance'],
    "algorithm": ['ball_tree', 'kd_tree'],
    "leaf_size": randint(10, 100),
    "p": [1, 2]
}

params = {
    'DecisionTreeClassifier': decision_tree_params,
    'RandomForestClassifier': decision_tree_params,
    'LinearSVC': svc_params,
    'KNeighborsClassifier': kneighbors_params
}


def grid_search(df, classifiers):

    data = df.as_matrix()
    X = data[:, :-1]
    y = data[:, -1]

    parameters = dict()
    cv_scores = dict()

    for index, classifier in enumerate(classifiers):

        n_iter_search = 20
        name_of_classifier = classifier.__class__.__name__
        print 'Searching for best parameters for {}'.format(name_of_classifier)

        random_search = RandomizedSearchCV(classifier,
                                           param_distributions=params[name_of_classifier],
                                           n_iter=n_iter_search)

        random_search.fit(X, y)
        scores = pd.DataFrame(random_search.grid_scores_)
        best_score = scores.sort_values('mean_validation_score', ascending=False).reset_index()
        best_params = best_score.parameters[0]

        parameters[name_of_classifier] = best_params
        cv_scores[name_of_classifier] = best_score.mean_validation_score[0]

    return parameters

