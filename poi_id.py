#!/usr/bin/python
#import numpy as np
import sys
import pickle
from time import time
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
#from cust_score import custom_scorer
#from sklearn.metrics import make_scorer

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 
#                 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments', 'exercised_stock_options',
#                 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value', 'to_messages', 
#                 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
features_financial = ['poi','salary', 'bonus', 'long_term_incentive', 'deferral_payments', 'other', 'expenses', 
                 'director_fees', 'total_payments', 'total_stock_value']
features_email = ['poi', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 
                  'shared_receipt_with_poi']
features_frac = ['poi', 'fraction_to_poi', 'fraction_from_poi', 'fraction_shared_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset_enhanced.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('BELFER ROBERT', 0)
data_dict.pop('BHATNAGAR SANJAY', 0)

### Task 3: Create new feature(s)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#from sklearn.feature_selection import f_regression
#from sklearn.feature_selection import f_classif
from sklearn.linear_model import Lasso
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import MinMaxScaler

my_dataset = data_dict
data_fin = featureFormat(my_dataset, features_financial, sort_keys = True)
data_em = featureFormat(my_dataset, features_email, sort_keys = True)
data_frac = featureFormat(my_dataset, features_frac, sort_keys = True)
labels_fin, features_fin = targetFeatureSplit(data_fin)
labels_em, features_em = targetFeatureSplit(data_em)
labels_fr, features_fr = targetFeatureSplit(data_frac)

#features_fin_new = SelectKBest(chi2, k = 4).fit_transform(features_fin, labels_fin)
features_fin_new = SelectKBest(chi2, k = 4).fit(features_fin, labels_fin)
#features_em_scaled = Normalizer().fit_transform(features_em)
features_em_new = Lasso(alpha = 0.00001).fit(features_em, labels_em)
features_fr_new = Lasso(alpha = 0.00001).fit(features_fr, labels_fr)
print [round(elem, 10) for elem in features_em_new.coef_], features_em_new.score(features_em, labels_em)
print [round(elem, 10) for elem in features_fr_new.coef_], features_fr_new.score(features_fr, labels_fr)
#pca = PCA(n_components = 4).fit(features)
#print pca.explained_variance_ratio_
print features_fin_new.scores_
features_fin_scaled = MinMaxScaler().fit_transform(features_fin)
## Store to my_dataset for easy export below.
my_dataset = data_dict
features_list = ['poi', 'bonus', 'total_payments', 'total_stock_value', 'fraction_to_poi', 
                 'fraction_from_poi', 'fraction_shared_with_poi']
                 
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_scaled = MinMaxScaler().fit_transform(features)
#### Task 4: Try a varity of classifiers
#### Please name your classifier clf for easy export below.
#### Note that if you want to do PCA or other multi-stage operations,
#### you'll need to use Pipelines. For more info:
#### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features_scaled, labels, test_size=0.25, random_state=42)
##features_train_pca = pca.transform(features_train)
##features_test_pca = pca.transform(features_test)

#from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.grid_search import GridSearchCV

estimator_svm = [('reduce_dim', RandomizedPCA()), ('clf_svm', SVC(kernel= 'rbf', class_weight = 'balanced', gamma=0.1, C=1000))]
estimator_tree = [('reduce_dim', RandomizedPCA()), ('clf_tree', DecisionTreeClassifier(criterion = 'entropy', 
                  max_features = 'sqrt', splitter = 'best'))]
#estimator_knn = [('reduce_dim', PCA()), ('clf_knn', KNeighborsClassifier())]
#estimator_rf = [('reduce_dim', PCA()), ('clf_rf', RandomForestClassifier())]
#estimator_ab = [('reduce_dim', PCA()), ('clf_ab', AdaBoostClassifier())]
                
#param_svm = {
#             'kernel': ['linear', 'poly', 'rbf'], 
#             'C': [10, 100, 1000, 10000], 
#             'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
#            }
param_svm = dict(
             reduce_dim__n_components=[2,3,4,5,6],
             clf_svm__C= [10, 100, 1000, 10000], 
             clf_svm__gamma=[0.0001, 0.001, 0.01, 0.1, 1.0]
            )


#param_tree = {
#              'criterion': ['gini', 'entropy'], 
#                'splitter': ['best', 'random'],
#                'max_features': [None, 'sqrt', 'log2'],
#                'min_samples_split': [2, 5, 10, 15, 25, 40]
#                }
param_tree = dict(reduce_dim__n_components = [2, 3, 4, 5, 6], 
              clf_tree__min_samples_split = [2, 5, 10]
                )
                
param_knn = {
             'n_neighbors': [5, 8, 10, 12, 15],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [5, 10, 15, 25, 30, 40]
                }
#param_knn = dict(
#             reduce_dim__n_components=[2,3,4,5,6],
#             clf_knn__n_neighbors=[5, 8, 10, 12, 15],
#             clf_knn__weights=['uniform', 'distance'],
#             clf_knn__algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'],
#             clf_knn__leaf_size=[5, 10, 15, 25, 30, 40]
#                )
            
param_rf = {
            'n_estimators': [5, 10, 15, 25, 50, 100],
            'criterion': ['gini', 'entropy'],
            'max_features': [None, 'sqrt', 'log2'],
            'min_samples_split': [2, 5, 10, 15, 25, 40]
            }
#param_rf = dict(
#                reduce_dim__n_components=[2,3,4,5,6],
#                clf_rf__n_estimators=[5, 10, 15, 25, 50, 100],
#                clf_rf__criterion=['gini', 'entropy'],
#                clf_rf__max_features=[None, 'sqrt', 'log2'],
#                clf_rf__min_samples_split=[2, 5, 10, 15, 25, 40]
#            )

param_ab = {
                'n_estimators': [5, 10, 15, 25, 50, 100],
                'learning_rate': [0.001, 0.01, 0.1, 1.0, 10.0],
                'algorithm': ['SAMME', 'SAMME.R']
            }            
#param_ab = dict(
#                reduce_dim__n_components=[2,3,4,5,6],
#                clf_ab__n_estimators=[5, 10, 15, 25, 50, 100],
#                clf_ab__learning_rate=[0.001, 0.01, 0.1, 1.0, 10.0],
#                clf_ab__algorithm=['SAMME', 'SAMME.R']
#            )

nb_t0 = time()            
clf_nb = GaussianNB().fit(features_train, labels_train)
pred_nb = clf_nb.predict(features_test)
nb_t1 = time()

#from sklearn.cross_validation import StratifiedShuffleSplit
#sss = StratifiedShuffleSplit(labels_train, test_size = 0.1, n_iter=100)
#score = make_scorer(custom_scorer(labels, predictions), greater_is_better=True)

svm_t0 = time()
#pipe_svm = Pipeline(estimator_svm)
#clf_svm = GridSearchCV(SVC(), param_grid = param_svm)
#clf_svm = GridSearchCV(pipe_svm, param_grid = param_svm) #, scoring = score)
#clf_svm= pipe_svm.fit(features_train, labels_train)
#clf_svm.fit(features_train, labels_train)
#print 'SVM Best Estimators: ', clf_svm.best_estimator_
svm_pca = RandomizedPCA(n_components = 6).fit(features_train)
features_train_pca_svm = svm_pca.transform(features_train)
features_test_pca_svm = svm_pca.transform(features_test)
clf_svm = SVC(kernel= 'rbf', class_weight = 'balanced', gamma=0.1, C=100).fit(features_train_pca_svm, labels_train)
pred_svm = clf_svm.predict(features_test_pca_svm)
svm_t1 = time()

tree_t0 = time()
#pipe_tree = Pipeline(estimator_tree)
#clf_tree = GridSearchCV(DecisionTreeClassifier(criterion = 'entropy', max_features = 'sqrt', 
#                                               splitter = 'best'), param_grid = param_tree)
tree_pca = RandomizedPCA(n_components = 4).fit(features_train)
features_train_pca_tree = tree_pca.transform(features_train)
features_test_pca_tree = tree_pca.transform(features_test)
clf_tree = DecisionTreeClassifier(criterion = 'entropy', max_features = 'sqrt', 
                                  random_state = None, min_samples_split = 4, splitter = 'best')
#clf_tree = GridSearchCV(pipe_tree, param_grid = param_tree)
clf_tree.fit(features_train_pca_tree, labels_train)
#print 'Decision Tree Best Estimators: ', clf_tree.best_estimator_
pred_tree = clf_tree.predict(features_test_pca_tree)
tree_t1 = time()

knn_t0 = time()
#pipe_knn = estimator_knn
#clf_knn = GridSearchCV(KNeighborsClassifier(), param_grid = param_knn)
#clf_knn = GridSearchCV(pipe_knn, param_grid = param_knn)
clf_knn = KNeighborsClassifier(n_neighbors = 8, weights = 'uniform')
clf_knn.fit(features_train, labels_train)
#print 'KNN Best Estimators: ', clf_knn.best_estimator_
pred_knn = clf_knn.predict(features_test)
knn_t1 = time()

rf_t0 = time()
#pipe_rf = Pipeline(estimator_rf)
#clf_rf = GridSearchCV(RandomForestClassifier(), param_grid = param_rf)
#clf_rf = GridSearchCV(pipe_rf, param_grid = param_rf)
clf_rf = RandomForestClassifier(n_estimators = 40, criterion = 'entropy', 
                                random_state = 5, max_features = 'sqrt', min_samples_split = 5)
clf_rf.fit(features_train, labels_train)
#print 'Random Forest Best Estimators: ', clf_rf.best_estimator_
pred_rf = clf_rf.predict(features_test)
rf_t1 = time()

ab_t0 = time()
#pipe_ab = Pipeline(estimator_ab)
#clf_ab = GridSearchCV(AdaBoostClassifier(), param_grid = param_ab)
#clf_ab = GridSearchCV(pipe_ab, param_grid = param_ab)
clf_ab = AdaBoostClassifier(algorithm = 'SAMME', n_estimators = 50, learning_rate = 0.1)
clf_ab.fit(features_train, labels_train)
#print 'AbaBoost Best Estimators: ', clf_ab.best_estimator_
pred_ab = clf_ab.predict(features_test)
ab_t1 = time()

# Provided to give you a starting point. Try a variety of classifiers.


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# Compute Metrics for each algorithm and display them:
acc_nb = accuracy_score(labels_test, pred_nb)
acc_svm = accuracy_score(labels_test, pred_svm)
acc_tree = accuracy_score(labels_test, pred_tree)
acc_knn = accuracy_score(labels_test, pred_knn)
acc_rf = accuracy_score(labels_test, pred_rf)
acc_ab = accuracy_score(labels_test, pred_ab)

rec_nb = recall_score(labels_test, pred_nb)
rec_svm = recall_score(labels_test, pred_svm)
rec_tree = recall_score(labels_test, pred_tree)
rec_knn = recall_score(labels_test, pred_knn)
rec_rf = recall_score(labels_test, pred_rf)
rec_ab = recall_score(labels_test, pred_ab)

prec_nb = precision_score(labels_test, pred_nb)
prec_svm = precision_score(labels_test, pred_svm)
prec_tree = precision_score(labels_test, pred_tree)
prec_knn = precision_score(labels_test, pred_knn)
prec_rf = precision_score(labels_test, pred_rf)
prec_ab = precision_score(labels_test, pred_ab)

score_nb = acc_nb*rec_nb*prec_nb
score_svm = acc_svm*rec_svm*prec_svm
score_tree = acc_tree*rec_tree*prec_tree
score_knn = acc_knn*rec_knn*prec_knn
score_rf = acc_rf*rec_rf*prec_rf
score_ab = acc_ab*rec_ab*prec_ab

print '\t'.join(['Algorithm Type', 'Accuracy', 'Recall', 'Precision', 'Score Metric', 'Time Taken (s)'])
print '\t'.join(['Naive Bayes', str('%.2f' % acc_nb), str('%.2f' % rec_nb), str('%.2f' % prec_nb), str('%.3f' % score_nb), str('%.2f' % (nb_t1-nb_t0))])
print '\t'.join(['Support VM', str('%.2f' % acc_svm), str('%.2f' % rec_svm), str('%.2f' % prec_svm), str('%.3f' % score_svm), str('%.2f' % (svm_t1-svm_t0))])
print '\t'.join(['D. Trees', str('%.2f' % acc_tree), str('%.2f' % rec_tree), str('%.2f' % prec_tree), str('%.3f' % score_tree), str('%.2f' % (tree_t1 - tree_t0))])
print '\t'.join(['K. Nearest', str('%.2f' % acc_knn), str('%.2f' % rec_knn), str('%.2f' % prec_knn), str('%.3f' % score_knn), str('%.2f' % (knn_t1 - knn_t0))])
print '\t'.join(['R. Forest', str('%.2f' % acc_rf), str('%.2f' % rec_rf), str('%.2f' % prec_rf), str('%.3f' % score_rf), str('%.2f' % (rf_t1 - rf_t0))])
print '\t'.join(['AdaBoost', str('%.2f' % acc_ab), str('%.2f' % rec_ab), str('%.2f' % prec_ab), str('%.3f' % score_ab), str('%.2f' % (ab_t1 - ab_t0))])

#score_array = np.array([score_nb, score_svm, score_tree, score_knn, score_rf, score_ab], dtype = float))
#score_array = np.array([score_nb, score_svm, score_tree, score_knn], dtype=float)
#clf_list = [clf_nb, clf_svm, clf_tree, clf_knn, clf_rf, clf_ab]
#clf_list = [clf_nb, clf_svm, clf_tree, clf_knn]
#max_index = np.argmax(score_array)

#clf = clf_list[max_index]
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
#clf = clf_tree

dump_classifier_and_data(clf_nb, my_dataset, features_list, "my_classifier_nb.pkl")
dump_classifier_and_data(clf_svm, my_dataset, features_list, "my_classifier_svm.pkl")
dump_classifier_and_data(clf_tree, my_dataset, features_list, "my_classifier_tree.pkl")
dump_classifier_and_data(clf_knn, my_dataset, features_list, "my_classifier_knn.pkl")
dump_classifier_and_data(clf_rf, my_dataset, features_list, "my_classifier_rf.pkl")
dump_classifier_and_data(clf_ab, my_dataset, features_list, "my_classifier_ab.pkl")