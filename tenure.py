import itertools
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import cross_val_score

import classifier_tuning

classifiers_to_tune = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LinearSVC(),
    KNeighborsClassifier()
]

datafiles = [
    'hca_all.csv',
    'hca_all_recode.csv',
    'hca_household_vars.csv',
    'hca_household_vars_recode.csv'
]

print "Performing grid search for best parameters"
datafile = pd.read_csv('hca_household_vars_recode.csv')
params = classifier_tuning.grid_search(datafile, classifiers_to_tune)
pd.DataFrame(params).to_csv('tuned_parameters.csv')

print "Grid search completed"

classifiers = [
    DecisionTreeClassifier(**params['DecisionTreeClassifier']),
    RandomForestClassifier(**params['RandomForestClassifier']),
    LinearSVC(**params['LinearSVC']),
    KNeighborsClassifier(**params['KNeighborsClassifier']),
    LogisticRegressionCV()
]

results = pd.DataFrame()

for index, (datafile, classifier) in enumerate(itertools.product(datafiles, classifiers)):

    df = pd.read_csv(datafile)
    data = df.as_matrix()
    X = data[:, :-1]
    y = data[:, -1]

    score = cross_val_score(classifier, X, y, cv=5)

    name_of_dataset = datafile.split('.')[0]

    results.loc[index, 'dataset'] = name_of_dataset
    results.loc[index, 'classifier'] = classifier.__class__.__name__

    for i, scr in enumerate(score):
        results.loc[index, 'score{}'.format(i)] = scr

    results.loc[index, 'mean_score'] = score.mean()

    results.to_csv('results.csv')


