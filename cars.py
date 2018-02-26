import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from itertools import chain


if not os.path.exists('./analysis'):
    os.mkdir('analysis')

if not os.path.exists('./reports'):
    os.mkdir('reports')


# Load dataset
data = pd.read_csv("cars.csv", header=0)

attr_values = {
    'buying price': ['low', 'med', 'high', 'vhigh'],
    'maintenance price': ['low', 'med', 'high', 'vhigh'],
    'number of doors': ['2', '3', '4', '5more'],
    'person capacity': ['2', '4', 'more'],
    'luggage boot': ['small', 'med', 'big'],
    'safety': ['low', 'med', 'high'],
    'acceptability': ['unacc', 'acc', 'good', 'vgood']
}

# Encode Data as integer
data.apply(lambda x: x.replace(attr_values[x.name], range(len(attr_values[x.name])), inplace=True), axis=0)

x, y = data[data.columns[:-1]], np.asarray(data['acceptability'])


###################################
#
# Data analysis and visualization
#
###################################

# Target classes distributions.
# With this plot it is possible to see that the target classes are very unbalanced.
data['acceptability'].value_counts().plot.bar(title='Acceptability', rot=0)
plt.xticks([0, 1, 2, 3], attr_values['acceptability'])
plt.savefig('analysis/acceptability.png')
plt.clf()

# Check for possible important features.
# Based on this plot, it suggest that 'number of doors' and 'luggage boot' are the least important features.
# Removing them may be beneficial for the classifier.
feats = SelectKBest(chi2, k='all')
feats.fit(x, y)

pos = np.arange(len(feats.scores_))
plt.figure(figsize=(10, 5))
plt.bar(pos, feats.scores_, align='center')
plt.xticks(pos, x.columns.values.tolist())
plt.ylabel('Importance score')
plt.title('Feature importance based on the Chi-square test')
plt.tight_layout()
plt.savefig('analysis/feature_importance.png')
plt.clf()

# Attributes distributions
# With this plot it is possible to see that all input features seem evenly distributed.
x.hist(grid=False)
plt.subplots_adjust(hspace=0.55)
plt.savefig('analysis/attributes_distrib.png', bbox_inches='tight')
plt.clf()

# Attributes x acceptability
cols = data['acceptability'].replace((0, 1, 2, 3), ('red', 'black', 'blue', 'green'))
fig, axes = plt.subplots(2, 3)
data.plot.scatter(x='buying price', y='acceptability', c=cols, ax=axes[0][0])
data.plot.scatter(x='maintenance price', y='acceptability', c=cols, ax=axes[0][1])
data.plot.scatter(x='number of doors', y='acceptability', c=cols, ax=axes[0][2])
data.plot.scatter(x='person capacity', y='acceptability', c=cols, ax=axes[1][0])
data.plot.scatter(x='luggage boot', y='acceptability', c=cols, ax=axes[1][1])
data.plot.scatter(x='safety', y='acceptability', c=cols, ax=axes[1][2])
fig.subplots_adjust(hspace=0.55, wspace=0.45)
fig.savefig('analysis/attribute_acceptability_scatter.png', bbox_inches='tight')

# Calculate the frequency of each different attribute related to the different levels of acceptability
# Based on these plots, it is possible to see that that the different levels 'number of doors' have
# similar distribution among the different levels of 'acceptability'. This suggests that this attribute may not
# be so relevant for the classifier.
fig, axes = plt.subplots(2, 3)
fig.set_size_inches(15, 7.5)

for c, ax in zip(attr_values.keys(), list(chain.from_iterable(axes))):
    table = pd.crosstab(index=data[c], columns=data['acceptability']).apply(lambda x: x/x.sum(), axis=0)
    table.columns = attr_values['acceptability']
    table.index = attr_values[c]
    table.plot.bar(title=c, rot=0, ax=ax)

fig.subplots_adjust(hspace=0.55, wspace=0.45)
fig.set_tight_layout(True)
fig.savefig('analysis/attribute_acceptability_bar.png')
plt.clf()


######################
#
# Data preparation
#
######################

# Split Data to Train and Test (80%/20%) keeping the original class distribution
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Use a 5-fold CV procedure to find the best parameter set for the model.
# Each fold has the same classes distribution
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)


#############################
#
# Model setup and evaluation
#
#############################

# Model pipeline with possible dimensionality reduction (PCA or K-best features) step and SVM as the classifier.
# The dimensionality reduction step aims to remove the irrelevant features pointed out in the previous analise.
pipe = Pipeline([
    ('reduce_dim', PCA()),
    ('classifier', SVC())
])

# Set the parameters grid
reduce_dim_opts = [None, PCA(5), PCA(6), SelectKBest(chi2, k=4), SelectKBest(chi2, k=5)]
c_opts = [1, 10, 100]

param_grid = [
    {
        'reduce_dim': reduce_dim_opts,
        'classifier__C': c_opts,
        'classifier__kernel': ['linear'],
    },
    {
        'reduce_dim': reduce_dim_opts,
        'classifier__C': c_opts,
        'classifier__gamma': [1e-1, 1e-2, 1e-3, 1e-4],
        'classifier__kernel': ['rbf'],
    },
]


# Perform a grid search optimization on the training set
clf = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
clf.fit(x_train, y_train)

# Save the best model found
with open('reports/best_model', 'w') as outfile:
    json.dump(clf.best_params_, outfile, indent=4)


def report(y_true, y_pred, file):
    rep = 'Confusion matrix:\n\n'
    rep += pd.DataFrame(confusion_matrix(y_true, y_pred),
                        index=['unacc', 'acc', 'good', 'vgood'],
                        columns=['unacc', 'acc', 'good', 'vgood']).to_string()
    rep += '\n\nClassification metrics:\n\n'
    rep += classification_report(y_true, y_pred, target_names=['unacc', 'acc', 'good', 'vgood'])
    rep += '\nAccuracy: %.4f' % accuracy_score(y_true, y_pred)

    with open(file=file, mode='w') as outfile:
        outfile.write(rep)


# Classification report on the training set
report(y_train, clf.predict(x_train), 'reports/train_performance_report')

# Classification report on the testing set
report(y_test, clf.predict(x_test), 'reports/test_performance_report')
