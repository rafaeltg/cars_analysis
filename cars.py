import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline


# Load dataset
data = pd.read_csv("cars.csv", header=0)

# Encode Data
data['buying price'].replace(('low', 'med', 'high', 'vhigh'), (0, 1, 2, 3,), inplace=True)
data['maintenance price'].replace(('low', 'med', 'high', 'vhigh'), (0, 1, 2, 3), inplace=True)
data['number of doors'].replace(('2', '3', '4', '5more'), (0, 1, 2, 3), inplace=True)
data['person capacity'].replace(('2', '4', 'more'), (0, 1, 2), inplace=True)
data['luggage boot'].replace(('small', 'med', 'big'), (0, 1, 2), inplace=True)
data['safety'].replace(('low', 'med', 'high'), (0, 1, 2), inplace=True)
data['acceptability'].replace(('unacc', 'acc', 'good', 'vgood'), (0, 1, 2, 3), inplace=True)


###################################
#
# Data analysis and visualization
#
###################################

# Calculate the frequency of each different attribute related to the different levels of acceptability
# Based on this analise, it is possible to see that that the different levels 'number of doors' and 'luggage boot' hava
# similar distribution among the different levels of 'acceptability'. This suggests that these two attributes may not
# be so relevant for the classifier.
def freq_table(col_name, idxs):
    table = pd.crosstab(index=data[col_name], columns=data['acceptability'])
    table = table.apply(lambda x: x/x.sum(), axis=1)
    table.columns = ['unacc', 'acc', 'good', 'vgood']
    table.index = idxs
    print(col_name)
    print(table)
    print()


freq_table('buying price', ['low', 'med', 'high', 'vhigh'])
freq_table('maintenance price', ['low', 'med', 'high', 'vhigh'])
freq_table('number of doors', ['2', '3', '4', '5more'])
freq_table('person capacity', ['2', '4', 'more'])
freq_table('luggage boot', ['small', 'med', 'big'])
freq_table('safety', ['low', 'med', 'high'])


# Classes distributions.
# With this plot it is possible to see that the target classes are very unbalanced.
data['acceptability'].hist(grid=False)
plt.savefig('accetaptability_distrib.png')

# Attributes distributions
# With this plot it is possible to see that all input feature seems evenly distributed.
data[data.columns[:-1]].hist(grid=False)
plt.subplots_adjust(hspace=0.55)
plt.savefig('attributes_distrib.png', bbox_inches='tight')

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
fig.savefig('attribute_class.png', bbox_inches='tight')


######################
#
# Data preparation
#
######################

x, y = data[data.columns[:-1]], np.asarray(data['acceptability'])

# Check for possible important features.
# Based on this analise, it suggest that 'number of doors' and 'luggage boot' are the least important features.
# Removing them may be beneficial for the classifier.
feats = SelectKBest(chi2, k='all')
feats.fit(x, y)
print('Feature importance based on the Chi-square test:')
for i, c in enumerate(x.columns.values.tolist()):
    print('%s: %.4f' % (c, feats.scores_[i]))


# Split Data to Train and Test (80%/20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Use a 5-fold CV procedure to find the best parameter set for the model.
# Each fold has the same classes distribution
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)


######################
#
# Model setup
#
######################

# Model pipeline with possible dimensionality reduction (PCA or K-best features) step and SVC as the classifier.
# The dimensionality reduction step aims to remove the irrelevant features pointed out in the previous analise.
pipe = Pipeline([
    ('reduce_dim', PCA()),
    ('classifier', SVC())
])

# Set the parameters grid
reduce_dim_opts = [None, PCA(5), PCA(6), SelectKBest(chi2, k=4), SelectKBest(chi2, k=5)]
c_opts = [1, 10, 100, 1000]

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

print('\n======================================================================\n')
print("Best parameters set found on training set:")
print(clf.best_params_)


def report(y_true, y_pred):
    print(classification_report(y_true, y_pred, target_names=['unacc', 'acc', 'good', 'vgood']))
    print('Accuracy: %.4f' % accuracy_score(y_true, y_pred))
    print('\nConfusion matrix:')
    print(pd.DataFrame(confusion_matrix(y_true, y_pred),
                       index=['unacc', 'acc', 'good', 'vgood'],
                       columns=['unacc', 'acc', 'good', 'vgood']))


print('\n======================================================================\n')
print("Train set classification report:")
report(y_train, clf.predict(x_train))

print('\n======================================================================\n')
print("Test set classification report:")
report(y_test, clf.predict(x_test))
