'''Here I analyze the three year loan book from 2007-12. The models will be
tested on a holdout set, and further validated on 2013 data.

First, I want to see for the continuous variables if, now that there is more data,
if there is visible separability when plotting.'''

#libraries
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import TruncatedSVD
from sklearn.cross_validation import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import roc_curve, auc
from scipy import interp
from mlxtend.evaluate import plot_learning_curves
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import plot_sequential_feature_selection
import pickle
from textblob import TextBlob
%matplotlib inline

loans_3yr_until_2013 = pd.read_csv('../../Data/second_run/loanbook_3yr_until_2013.csv')

loans_3yr_until_2013.drop('row_num.1', axis = 1, inplace = True)

loans_3yr_until_2013.columns.values

loans_3yr_until_2013.set_index('id', inplace = True)

'''Columns to create df for later eval'''
#Creating new gross return column
loans_3yr_until_2013['loan_return'] = loans_3yr_until_2013['total_pymnt'] - loans_3yr_until_2013['funded_amnt']
loans_3yr_until_2013['loan_return_scaled'] = loans_3yr_until_2013['loan_return'].divide(loans_3yr_until_2013['funded_amnt'])



id_cols = ['year', 'grade', 'gross_return_scaled', 'loan_return', 'loan_return_scaled', 'int_rate', 'loan_status', 'loan_status_encoded','sub_grade']
true_outcomes = loans_3yr_until_2013[id_cols]
true_outcomes.groupby('grade').int_rate.mean()


'''Setting up y. This is where I would set the logic for subsetting on year...
Grades A through F
Data through 2012.'''

y_gradeA = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'A') & (loans_3yr_until_2013.year<=2012)]['loan_status_encoded']
y_gradeB = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'B') & (loans_3yr_until_2013.year<=2012)]['loan_status_encoded']
y_gradeC = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'C') & (loans_3yr_until_2013.year<=2012)]['loan_status_encoded']
y_gradeD = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'D') & (loans_3yr_until_2013.year<=2012)]['loan_status_encoded']
y_gradeE = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'E') & (loans_3yr_until_2013.year<=2012)]['loan_status_encoded']
y_gradeF = loans_3yr_until_2013[(loans_3yr_until_2013.grade.isin(['F', 'G'])) & (loans_3yr_until_2013.year<=2012)]['loan_status_encoded']
y_gradeA = y_gradeA.replace(to_replace={0: 0, 1: 0, 2: 1, 3: 1})
y_gradeB = y_gradeB.replace(to_replace={0: 0, 1: 0, 2: 1, 3: 1})
y_gradeC = y_gradeC.replace(to_replace={0: 0, 1: 0, 2: 1, 3: 1})
y_gradeD = y_gradeD.replace(to_replace={0: 0, 1: 0, 2: 1, 3: 1})
y_gradeE = y_gradeE.replace(to_replace={0: 0, 1: 0, 2: 1, 3: 1})
y_gradeF = y_gradeF.replace(to_replace={0: 0, 1: 0, 2: 1, 3: 1})

'''Obtaining baseline non-default rate'''
defaults = []
for y in [y_gradeA, y_gradeB, y_gradeC, y_gradeD, y_gradeE, y_gradeF]:
    defaults.append([y.value_counts(True)[1], len(y)])



baseline= pd.DataFrame(defaults, index = ['A', 'B', 'C', 'D', 'E', 'F'])
baseline = baseline.rename(columns = {0: 'default_rate', 1: 'num_obs'})
baseline.to_csv('../../Results/second_run/three_year/2007-12 training/default_rate_and_obs_descriptive_stats.csv', sep = ',', index = True, index__label = 'grade')
for index1, row in baseline.iterrows():
    print 'Baseline default for %s is %.3f' % (index1, row.default_rate)

#installment is out b/c of derivation from amnt_requested and int_rate. total_acc is out b/c high correlation (70%+ with open acc)
notscale_cols = ['row_num', 'installment', 'total_acc', 'mths_since_last_delinq', \
'mths_since_last_record', 'mths_since_last_major_derog', 'funded_amnt','grade', 'sub_grade', 'date', 'title', 'zip_code', 'term', \
'emp_title', 'earliest_cr_line', 'loan_status_encoded','collections_12_mths_ex_med',\
'annual_inc_joint', 'dti_joint', 'verification_status_joint_Verified', \
'verification_status_joint_Source Verified','acc_now_delinq', 'delinq_amnt', \
'loan_status', 'out_prncp', 'out_prncp_inv', 'recoveries', 'total_pymnt',\
'total_pymnt_inv', 'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp', 'gross_return',\
'gross_return_scaled', 'date', 'month', 'year',\
'State_AL','State_AR','State_AZ','State_CA','State_CO','State_CT',\
 'State_DC','State_DE','State_FL','State_GA','State_HI','State_IA','State_ID',\
 'State_IL','State_IN','State_KS','State_KY','State_LA','State_MA','State_MD',\
 'State_ME','State_MI','State_MN','State_MO','State_MS','State_MT','State_NC',\
 'State_ND','State_NE','State_NH','State_NJ','State_NM','State_NV','State_NY',\
 'State_OH','State_OK','State_OR','State_PA','State_RI','State_SC','State_SD',\
 'State_TN', 'State_TX','State_UT','State_VA','State_VT','State_WA','State_WI',\
 'State_WV','State_WY']

'''Set up my X for 2007 to 2012, by grade'''
notscalemask = loans_3yr_until_2013.columns.isin(notscale_cols)
scalecols = loans_3yr_until_2013.columns[~notscalemask]

X_gradeA = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'A') & (loans_3yr_until_2013.year<=2012)][scalecols]
X_gradeB = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'B') & (loans_3yr_until_2013.year<=2012)][scalecols]
X_gradeC = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'C') & (loans_3yr_until_2013.year<=2012)][scalecols]
X_gradeD = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'D') & (loans_3yr_until_2013.year<=2012)][scalecols]
X_gradeE = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'E') & (loans_3yr_until_2013.year<=2012)][scalecols]
X_gradeF = loans_3yr_until_2013[(loans_3yr_until_2013.grade.isin(['F', 'G'])) & (loans_3yr_until_2013.year<=2012)][scalecols]

for data in [X_gradeA, X_gradeB, X_gradeC, X_gradeD, X_gradeE, X_gradeF]:
    print "observations: ", data.shape[0]


'''Grades E and F might not have enough obs. As for subgrades, I could consider doing analysis by subgrade through
grade D.'''

credit_cols = ['annual_inc','dti',
       'delinq_2yrs', 'inq_last_6mths',  'open_acc', 'pub_rec', 'revol_bal',
       'revol_util', 'total_acc', 'Risk_Score',
       'diff_earliest_app_date', 'work_prop',
       'magnitude_ann_inc_to_median']

'''Train Test Split the 2012 Data'''

X_gradeA_train, X_gradeA_test, y_gradeA_train, y_gradeA_test = train_test_split(X_gradeA, y_gradeA, test_size = .2, random_state = 31, stratify = y_gradeA)
X_gradeB_train, X_gradeB_test, y_gradeB_train, y_gradeB_test = train_test_split(X_gradeB, y_gradeB, test_size = .2, random_state = 31, stratify = y_gradeB)
X_gradeC_train, X_gradeC_test, y_gradeC_train, y_gradeC_test = train_test_split(X_gradeC, y_gradeC, test_size = .2, random_state = 31, stratify = y_gradeC)
X_gradeD_train, X_gradeD_test, y_gradeD_train, y_gradeD_test = train_test_split(X_gradeD, y_gradeD, test_size = .2, random_state = 31, stratify = y_gradeD)
X_gradeE_train, X_gradeE_test, y_gradeE_train, y_gradeE_test = train_test_split(X_gradeE, y_gradeE, test_size = .2, random_state = 31, stratify = y_gradeE)
X_gradeF_train, X_gradeF_test, y_gradeF_train, y_gradeF_test = train_test_split(X_gradeF, y_gradeF, test_size = .2, random_state = 31, stratify = y_gradeF)

'''Helper Functions'''
def evaluate_model(model, train_data, target_train, test_data, target_test):
    mean_recall = cross_val_score(model, train_data, target_train, cv = StratifiedKFold(target_train, 5, shuffle = True, random_state = 31), scoring = 'recall').mean()
    mean_roc_auc = cross_val_score(model, train_data, target_train, cv = StratifiedKFold(target_train, 5, shuffle = True, random_state = 31), scoring = 'roc_auc').mean()
    model.fit(train_data, target_train)
    y_pred = model.predict(test_data)
    cm = confusion_matrix(target_test, model.predict(test_data))
    cr = classification_report(target_test, y_pred)
    test_roc_auc = roc_auc_score(target_test, model.predict_proba(test_data)[:,1])
    pred_default_rate = sum(cm[:,1])/float(sum(sum(cm)))
    act_default_rate = sum(cm[1])/float(sum(sum(cm)))
    observed_default_rate = float(cm[1, 0])/sum(cm[:, 0])
    print 'Confusion Matrix'
    print cm
    print cr
    print "Predicted Default Rate:", pred_default_rate
    print "Actual Default Rate for the Baseline:", act_default_rate
    print "Observed Default Rate for only lending to those you think are good credits:", observed_default_rate
    print "Mean recall, cross-val, is:", mean_recall
    print "Mean roc_auc, cross-val, is:", mean_roc_auc
    print "ROC_AUC from test data is:", test_roc_auc
    return mean_recall, mean_roc_auc, test_roc_auc, pred_default_rate, act_default_rate, observed_default_rate

def eval_grid(grid_model, training_data, training_target, testing_data, testing_target, features_cols):
    grid_model.fit(training_data[features_cols], training_target)
    print "The best roc_auc score is:", grid_model.best_score_
    print "Optimal parameters are:", grid_model.best_params_
    return evaluate_model(grid_model.best_estimator_, training_data[features_cols], training_target, testing_data[features_cols], testing_target)


def eval_2013(model, data, target):
    y_pred = model.predict(data)
    y_pred_proba = model.predict_proba(data)[:,1]
    cm = confusion_matrix(target, y_pred)
    cr = classification_report(target, y_pred)
    roc_auc = roc_auc_score(target,y_pred_proba)
    pred_default_rate = sum(cm[:,1])/float(sum(sum(cm)))
    act_default_rate = sum(cm[1])/float(sum(sum(cm)))
    observed_default_rate = float(cm[1, 0])/sum(cm[:, 0])
    print 'Confusion Matrix for ' + str(model)
    print cm
    print cr
    print "Predicted Default Rate:", pred_default_rate
    print "Actual Default Rate for the Baseline:", act_default_rate
    print "Observed Default Rate for only lending to those you think are good credits:", observed_default_rate
    print "ROC_AUC from 2013 is:", roc_auc
    return roc_auc, pred_default_rate, act_default_rate, observed_default_rate

def plot_roc_curve_credit_cols(X, y, model, model_name, grade):
    cv = StratifiedKFold(y, n_folds=5)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas_ = model.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for ' + model_name + ' ' + grade + ' loans, 2007-12')
    plt.legend(loc="lower right", fontsize = 'x-small')
    plt.savefig('../Results/second_run/three_year/2007-12 training/Feature Selection/' + grade + '/ROC_' + model_name + ' ' + grade + '.png', bbox_inches = 'tight')

def plot_roc_curve_models(X, y, X_test, y_test,model, model_name, grade_name, grade):
    colors = ['black', 'orange', 'blue', 'red', 'green', 'magenta']
    linestyles = [':', '-', '--', '-.', '-', ':']
    for clf, label, clr, ls in zip(model, model_name, colors, linestyles):
        y_pred_proba = clf.fit(X, y).predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(y_true = y_test, y_score = y_pred_proba)
        roc_auc = auc(x = fpr, y= tpr)
        plt.plot(fpr, tpr, color = clr, linestyle = ls, label = '%s (auc = %.2f)' % (label, roc_auc))

    plt.legend(loc = 'lower right', fontsize = 'small')
    plt.plot([0,1], [0,1], linestyle = '--', color = 'gray', linewidth = 2, label = 'Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Models for ' + grade_name + ' loans, 2007-12')
    plt.savefig('../Results/second_run/three_year/2007-12 training/Feature Selection/' + grade + '/ROC_curves' + grade + '.png', bbox_inches = 'tight')


'''Grade F - Now we'll have dummy variables also, and continuous variables.
Given the dummies, if not doing PCA, we can just use MinMaxScaler to have a pipeline.

Given small obs, and performance of AdaBoost with Logistic Regression on PCA, try
similar model, or plain old Logistic Regression.

Feature Selection:
1) Look at RF Feature importances
2) SBS from Sebastian Raschka using Logistic Regression L2.

In all cases, let's use credit cols + dummies as starting point.
'''
'''Method 1 for Grade F'''
#Random Forest Feature Importances...
rf_gradeF = RandomForestClassifier(n_estimators=10000, random_state = 0, n_jobs = -1)
rf_gradeF.fit(X_gradeF_train, y_gradeF_train)
feature_importances_gradeF = pd.DataFrame(rf_gradeF.feature_importances_, index = X_gradeF_train.columns).sort_values(0, ascending = False)
feature_importances_gradeF.rename(columns = {0: 'Importance'}, inplace = True)

#getting the number of features that would be selected for threshold.
gradeF_selected = SelectFromModel(rf_gradeF, threshold = .01, prefit = True).transform(X_gradeF_train)
gradeF_selected.shape

#Creating column list for feature selection from RF.
features_selected_gradeF_cols = feature_importances_gradeF.iloc[range(0,20)]['Unnamed: 0']
features_selected_gradeF_columns = list(features_selected_gradeF_cols)
#Saving overall features importances for grade F
feature_importances_gradeF.to_csv('../Results/second_run/three_year/2007-12 training/Feature Selection/gradeF/rf_feature_importances_sorted.csv', sep = ',', index = True)
feature_importances_gradeF = pd.read_csv('../Results/second_run/three_year/2007-12 training/Feature Selection/gradeF/rf_feature_importances_sorted.csv')

'''Method 2 for Grade F'''
#SBS, using Logistic Regression L2
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.pipeline import Pipeline

lrl2_gradeF = make_pipeline(MinMaxScaler(), LogisticRegression(C =1, class_weight = 'balanced', penalty = 'l2'))
#base estimators for SBS
rf = RandomForestClassifier(max_depth = 5, min_samples_leaf  = 6, n_estimators = 20)

sbs_gradeF = SFS(LogisticRegression(C =1, class_weight = 'balanced', penalty = 'l2'), k_features = 29, forward = False, floating = False,\
 scoring = 'roc_auc', print_progress = True, cv = 5)
sbs_gradeF_rf = SFS(rf, k_features = 29, forward = False, floating = False,\
 scoring = 'roc_auc', print_progress = True, cv = 5)

#pipelines for SBS
gradeF_pipe = Pipeline([('minmax', MinMaxScaler()),('sbs', sbs_gradeF), ('lrl2', LogisticRegression(C =1, class_weight = 'balanced', penalty = 'l2'))])
gradeF_pipe_rf = Pipeline([('minmax', MinMaxScaler()),('sbs', sbs_gradeF), ('rf', rf)])

#grid search set up for two searches for number of features and features.
param_grid = [{'sbs__k_features': [15, 18, 20, 22, 29]}]
gs_gradeF = GridSearchCV(estimator = gradeF_pipe, param_grid = param_grid, scoring = 'roc_auc', cv = 5, n_jobs = -1, verbose = True, refit = True)
gs_gradeF_rf = GridSearchCV(estimator = gradeF_pipe_rf, param_grid = param_grid, scoring = 'roc_auc', cv = 5, n_jobs = -1, verbose = True, refit = True)

gs_gradeF_lr = gs_gradeF.fit(X_gradeF_train, y_gradeF_train)
gs_gradeF_rf_searched = gs_gradeF_rf.fit(X_gradeF_train, y_gradeF_train)

print("Best parameters via GridSearch with Log Reg", gs_gradeF_lr.best_params_)
print('Best features with Log Reg:', gs_gradeF_lr.best_estimator_.steps[1][1].k_feature_idx_)

print("Best parameters via GridSearch with RF", gs_gradeF_rf_searched.best_params_)
print('Best features with RF:', gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_)

rf_sbs_features = pd.DataFrame(X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_].columns.values)
rf_featimp = pd.DataFrame(features_selected_gradeF_cols)
rf_sbs_features.rename(columns = {0: 'sbs'}, inplace = True)
rf_featimp.rename(columns = {'Unnamed: 0': 'rf_featimp'}, inplace = True)

#features from Feat Imp that are in SBS_rf - note that Risk Score and other cont are not in this.
rf_featimp_in_sbs = pd.merge(rf_sbs_features, rf_featimp, how = 'left', left_on = 'sbs', right_on = 'rf_featimp')
rf_featimp_in_sbs.to_csv('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeF/RF feature imps in SBS.csv', sep = ',', index = True, index_label = 'sbs')

rf_sbs_in_featimp = pd.merge(rf_sbs_features, rf_featimp, how = 'right', left_on = 'sbs', right_on = 'rf_featimp')
rf_sbs_in_featimp.to_csv('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeF/SBS feats in RF feature imps.csv', sep = ',', index = True, index_label = 'rf_featimp')

#printing performance of results on base estimators. Not too meaningful because we need to optimize, but still interesting.
from mlxtend.feature_selection import plot_sequential_feature_selection as plot_sfs

fig1 = plot_sfs(gs_gradeF_lr.best_estimator_.steps[1][1].get_metric_dict(), kind='std_dev')

plt.ylim([0.4, 1])
plt.title('Sequential Backward Selection with Logistic Regression (w. StdDev)')
plt.grid()
plt.savefig('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeF/SBS_lr.png')
fig2 = plot_sfs(gs_gradeF_rf_searched.best_estimator_.steps[1][1].get_metric_dict(), kind='std_dev')

plt.ylim([0.4, 1])
plt.title('Sequential Backward Selection with Random Forest (w. StdDev)')
plt.grid()
plt.savefig('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeF/SBS_rf.png')

'''Looking at benchmark ROC AUC for feature selection. Using Log Reg'''
features = [gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_, gs_gradeF_lr.best_estimator_.steps[1][1].k_feature_idx_]
roc_auc_bench = {}
for model, name in zip([pipe_lrl2_F, pipe_et_F, pipe_svc_F],['lrl2', 'et', 'svc']):
    _, roc_auc_bench[name+'rf_feat_imp'], _, _, _, _ = evaluate_model(model, X_gradeF_train[features_selected_gradeF_columns], y_gradeF_train, X_gradeF_test[features_selected_gradeF_columns], y_gradeF_test)

names = ['rf_sbs', 'lr_sbs']
for model, model_name in zip([pipe_lrl2_F, pipe_et_F, pipe_svc_F],['lrl2', 'et', 'svc']):
    for feature_cols, name in zip(features, names):
        _, roc_auc_bench[model_name + name], _, _, _, _ = evaluate_model(model, X_gradeF_train.iloc[:,feature_cols], y_gradeF_train, X_gradeF_test.iloc[:,feature_cols], y_gradeF_test)

pd.DataFrame(roc_auc_bench, index = ['cross_val_roc_auc']).T.sort_values('cross_val_roc_auc', ascending = False)

rf_sbs_features.to_csv('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeF/SBS_features_via_RF.csv', sep = ',', index = True)

rf_sbs_features = pd.read_csv('../Results/second_run/three_year/2007-12 training/Feature Selection/gradeF/SBS_features_via_RF.csv')
'''Looking at the benchmark ROC AUC comparing the features from RF feature importance to SBS from Log Reg and RF,
the Log Reg and RF ones do best, and the RF ones are in a higher cross validated ROC AUC position overall than the Log Reg ones.

Hence, for grade F, I would use the ones from the RF SBS.'''


#Now that I have the best features for grade F, let's use zip to get the scores for them
#for grade F.
scores_defaults_F = {}
pipe_svc_F = make_pipeline(MinMaxScaler(), SVC(C = 1, kernel = 'rbf', max_iter = 30000, probability = True, class_weight = 'balanced', random_state = 31))
pipe_dt_F = make_pipeline(MinMaxScaler(), DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 4, min_samples_split = 8, min_samples_leaf = 4))
pipe_dtbag_F = make_pipeline(MinMaxScaler(), BaggingClassifier(DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 3, min_samples_split = 6, min_samples_leaf = 8), random_state = 31, n_estimators = 50))
pipe_rf_F = make_pipeline(MinMaxScaler(), RandomForestClassifier(random_state = 31, class_weight = 'balanced', n_estimators = 100, max_depth = 4, min_samples_split = 8))
pipe_et_F = make_pipeline(MinMaxScaler(), ExtraTreesClassifier(random_state = 31, class_weight = 'balanced', max_depth = 4, min_samples_split = 8, min_samples_leaf = 8))
pipe_lrl1_F = make_pipeline(MinMaxScaler(), LogisticRegression(C = 100, class_weight = 'balanced', penalty = 'l1'))
pipe_lrl2_F = make_pipeline(MinMaxScaler(), LogisticRegression(C = 50, class_weight = 'balanced', penalty = 'l2'))
pipe_adaboost_F = make_pipeline(MinMaxScaler(), AdaBoostClassifier(DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 5, min_samples_split = 6), random_state = 31))
pipe_adaboost_F_lr = make_pipeline(MinMaxScaler(), AdaBoostClassifier(LogisticRegression(C = 50, class_weight = 'balanced', penalty = 'l2', solver = 'lbfgs', max_iter = 1000), random_state = 31, learning_rate = .2, n_estimators = 200))
pipe_gradient_F_dev = make_pipeline(MinMaxScaler(), GradientBoostingClassifier(loss = 'deviance', learning_rate = .1, n_estimators = 200, max_depth = 4, min_samples_leaf = 6, verbose = True))
pipe_gradient_F_exp = make_pipeline(MinMaxScaler(), GradientBoostingClassifier(loss = 'exponential', learning_rate = .1, n_estimators = 200, max_depth = 4, min_samples_leaf = 6,  verbose = True))
pipe_voting_F = VotingClassifier(estimators = [('svc', pipe_svc_F), ('dt', pipe_dt_F), ('dtbag', pipe_dtbag_F), ('lrl2',pipe_lrl2_F), ('et', pipe_et_F), ('gradient_exp', pipe_gradient_F_exp), ('ada_lr', pipe_adaboost_F_lr)], voting = 'soft')

models = [pipe_svc_F, pipe_dt_F, pipe_dtbag_F, pipe_rf_F, pipe_et_F, pipe_lrl1_F, pipe_lrl2_F, pipe_adaboost_F, pipe_adaboost_F_lr, pipe_gradient_F_dev, pipe_gradient_F_exp, pipe_voting_F]
model_names = ['SVC', 'DT', 'Bagged DT', 'Random Forest', 'Extra Trees', 'Logistic Regression L1', 'Logistic Regression L2', 'AdaBoost', 'AdaBoost with LogReg', 'Gradient Deviance', 'Gradient Exp', 'Voting Untuned']

for model, model_name in zip(models, model_names):
    print ""
    print model_name + " Output for Grade F Loans, 2007 - 12"
    print ""
    scores_defaults_F[model_name] = evaluate_model(model, X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train, X_gradeF_test.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_test)

scores_defaults_F_df = pd.DataFrame(scores_defaults_F, index = ['mean_cv_recall', 'mean_cv_roc_auc', 'test_roc_auc', 'pred_default_rate', 'act_default_rate', 'observed_default_rate']).T.sort_values('test_roc_auc', ascending = False)
scores_defaults_F_df.to_csv('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeF/scores_default_gradeF.csv', sep = ',', index = True)

'''For Grade F, let's try to optimize DT, then Gradient, then Bagged DT, then AdaBoost, Log Reg.
Then plot learning curves for each'''
#DECISION TREE
dt_params = {'decisiontreeclassifier__max_depth': [1, 2, 3, 4,5,6,7,8,910], 'decisiontreeclassifier__min_samples_leaf':[2,4,6], \
'decisiontreeclassifier__min_samples_split':[2,4,6,8,10],\
'decisiontreeclassifier__criterion': ['gini', 'entropy'], 'decisiontreeclassifier__splitter':['random', 'best']}

gs_dt_gradeF = GridSearchCV(pipe_dt_F, dt_params, cv = StratifiedKFold(y_gradeF_train, 5, shuffle = True), verbose = True, scoring = 'roc_auc', n_jobs = -1)
gs_dt_gradeF.fit(X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train)
print "Best roc_auc score:", gs_dt_gradeF.best_score_
print "Best params:", gs_dt_gradeF.best_params_
np.mean(cross_val_score(gs_dt_gradeF.best_estimator_, X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train, cv = 5, scoring = 'roc_auc'))
scores_defaults_F['gs_dt'] = evaluate_model(gs_dt_gradeF.best_estimator_, X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train, X_gradeF_test.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_test)

#Bagged DT
pipe_dtbag_F_optdt = make_pipeline(MinMaxScaler(), BaggingClassifier(DecisionTreeClassifier(class_weight = 'balanced', criterion = 'gini', splitter = 'random', random_state = 31, max_depth = 2, min_samples_split = 2, min_samples_leaf = 2), random_state = 31))
bagging_params = {'baggingclassifier__n_estimators': [50, 100, 150, 200,400], 'baggingclassifier__bootstrap_features': [True, False]}
gs_pipe_dtbag_F = GridSearchCV(pipe_dtbag_F_optdt, bagging_params, cv = StratifiedKFold(y_gradeF_train, 5, shuffle = True), verbose = True, scoring = 'roc_auc', n_jobs = -1)
gs_pipe_dtbag_F.fit(X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train)
print "Best roc_auc score:", gs_pipe_dtbag_F.best_score_
print "Best params:", gs_pipe_dtbag_F.best_params_
np.mean(cross_val_score(gs_pipe_dtbag_F.best_estimator_, X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train, cv = 5, scoring = 'roc_auc'))
scores_defaults_F['gs_bagdt'] = evaluate_model(gs_pipe_dtbag_F.best_estimator_, X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train, X_gradeF_test.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_test)

#Gradient Exp
#pipe_gradient_F_dev = make_pipeline(MinMaxScaler(), GradientBoostingClassifier(loss = 'deviance', learning_rate = .1, n_estimators = 200, max_depth = 4, min_samples_leaf = 6, verbose = True))
pipe_gradient_F_exp_optdt = make_pipeline(MinMaxScaler(), GradientBoostingClassifier(loss = 'exponential', learning_rate = .1, n_estimators = 200, max_depth = 2, min_samples_leaf = 2, min_samples_split = 2))
gradient_params = {'gradientboostingclassifier__loss': ['deviance', 'exponential'], 'gradientboostingclassifier__learning_rate': [.3, .5, 1], 'gradientboostingclassifier__n_estimators': [100, 200, 300, 400, 500]}
gs_gradient_F = GridSearchCV(pipe_gradient_F_exp_optdt, gradient_params, cv = StratifiedKFold(y_gradeF_train, 5, shuffle = True), verbose = True, scoring = 'roc_auc', n_jobs = -1)
gs_gradient_F.fit(X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train)

print "Best roc_auc score:", gs_gradient_F.best_score_
print "Best params:", gs_gradient_F.best_params_
np.mean(cross_val_score(gs_gradient_F.best_estimator_, X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train, cv = 5, scoring = 'roc_auc'))
scores_defaults_F['gs_Gradient'] = evaluate_model(gs_gradient_F.best_estimator_, X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train, X_gradeF_test.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_test)

#Adaboost with opt dt
pipe_adaboost_F_optdt = make_pipeline(MinMaxScaler(), AdaBoostClassifier(DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 2, min_samples_split = 2, min_samples_leaf = 2), random_state = 31))
adaboost_params = {'adaboostclassifier__n_estimators': [100, 200, 300, 400, 500], 'adaboostclassifier__learning_rate': [.1, .3, .5, .7, 1]}
gs_adaboost_F = GridSearchCV(pipe_adaboost_F_optdt, adaboost_params, cv = StratifiedKFold(y_gradeF_train, 5, shuffle = True), verbose = True, scoring = 'roc_auc', n_jobs = -1)
gs_adaboost_F.fit(X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train)
print "Best roc_auc score:", gs_adaboost_F.best_score_
print "Best params:", gs_adaboost_F.best_params_
np.mean(cross_val_score(gs_adaboost_F.best_estimator_, X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train, cv = 5, scoring = 'roc_auc'))
scores_defaults_F['gs_AdaBoost'] = evaluate_model(gs_adaboost_F.best_estimator_, X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train, X_gradeF_test.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_test)

#Logistic Regression L2 C = 1 seems best...
pipe_lrl2_F1 = make_pipeline(MinMaxScaler(), LogisticRegression(C = 1, class_weight = 'balanced', penalty = 'l2'))
scores_defaults_F['Opt Logistic Regression L2'] = evaluate_model(pipe_lrl2_F1, X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train, X_gradeF_test.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_test)

#Logistic Regression L1 C = 1 seems best...
pipe_lrl1_F1 = make_pipeline(MinMaxScaler(), LogisticRegression(C = 1, class_weight = 'balanced', penalty = 'l1'))
scores_defaults_F['Opt Logistic Regression L1'] = evaluate_model(pipe_lrl1_F1, X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train, X_gradeF_test.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_test)

#Random Forest
pipe_rf_F_opt = make_pipeline(MinMaxScaler(), RandomForestClassifier(random_state = 31, class_weight = 'balanced', n_estimators = 100, max_depth = 2, min_samples_split = 2, min_samples_leaf =2))
random_params = {'randomforestclassifier__criterion': ['gini', 'entropy'], 'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2', None], 'randomforestclassifier__n_estimators': [100, 200, 300]}
gs_rf_F = GridSearchCV(pipe_rf_F_opt, random_params, cv = StratifiedKFold(y_gradeF_train, 5, shuffle = True), verbose = True, scoring = 'roc_auc', n_jobs = -1)
gs_rf_F.fit(X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train)
print "Best roc_auc score:", gs_rf_F.best_score_
print "Best params:", gs_rf_F.best_params_
np.mean(cross_val_score(gs_rf_F.best_estimator_, X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train, cv = 5, scoring = 'roc_auc'))
scores_defaults_F['gs_Random Forest'] = evaluate_model(gs_rf_F.best_estimator_, X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train, X_gradeF_test.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_test)

#SVC
svc_params = {'svc__C': [.001, .01,.1, 1, 10, 100]}
gs_svc_F = GridSearchCV(pipe_svc_F, svc_params, cv = StratifiedKFold(y_gradeF_train, 5, shuffle = True), verbose = True, n_jobs = -1, scoring = 'roc_auc')
gs_svc_F.fit(X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train)
print "Best roc_auc score:", gs_svc_F.best_score_
print "Best params:", gs_svc_F.best_params_

#ET
pipe_et_F_opt = make_pipeline(MinMaxScaler(), ExtraTreesClassifier(random_state = 31, class_weight = 'balanced', n_estimators = 100, max_depth = 2, min_samples_split = 2, min_samples_leaf =2))
et_params = {'extratreesclassifier__criterion': ['gini', 'entropy'], 'extratreesclassifier__max_features': ['auto', 'sqrt', 'log2', None], 'extratreesclassifier__n_estimators': [200, 300, 400, 500]}
gs_et_F = GridSearchCV(pipe_et_F_opt, et_params, cv = StratifiedKFold(y_gradeF_train, 5, shuffle = True), verbose = True, scoring = 'roc_auc', n_jobs = -1)
gs_et_F.fit(X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train)
print "Best roc_auc score:", gs_et_F.best_score_
print "Best params:", gs_et_F.best_params_
np.mean(cross_val_score(gs_et_F.best_estimator_, X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train, cv = 5, scoring = 'roc_auc'))
scores_defaults_F['gs_Extra Trees'] = evaluate_model(gs_et_F.best_estimator_, X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train, X_gradeF_test.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_test)

'''Learning Curves for Models Grade F'''
pipe_voting_F_opt_models = make_pipeline(MinMaxScaler(), VotingClassifier(estimators = [('lrl1', pipe_lrl1_F1), ('dtbag', gs_pipe_dtbag_F.best_estimator_), ('et', gs_et_F.best_estimator_), ('rf',gs_rf_F.best_estimator_), ('svc', pipe_svc_F), ('gradient',gs_gradient_F.best_estimator_), ('ada', gs_adaboost_F.best_estimator_), ('dt',pipe_dt_F)], voting = 'soft'))
#While lots of bias and variance, in general, the learning curves for these look better than for PCA for Grade F.
models_opt = [pipe_lrl2_F1, pipe_lrl1_F1, gs_pipe_dtbag_F.best_estimator_, gs_et_F.best_estimator_, gs_rf_F.best_estimator_, pipe_svc_F, gs_gradient_F.best_estimator_, gs_adaboost_F.best_estimator_, pipe_dt_F, pipe_voting_F_opt_models]
models_names = ['Logistic Regression L2', 'Logistic Regression L1', 'Bagged DT', 'Extra Trees', 'Random Forest', 'SVM', 'Gradient Boosting', 'AdaBoost', 'Decision Tree', 'Ensemble Voting']

for model_opt, model_name in zip(models_opt, models_names):
    plot_learning_curves(X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train, X_gradeF_test.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_test, clf = model_opt, print_model = False, scoring = 'roc_auc')
    plt.title('Learning Curve for ' + model_name + ' on Grade F Loans, 2007-12')
    plt.savefig(''.join(['../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeF/gradeF_learning_curve ',model_name,'.png']), bbox_inches = 'tight')
    plt.show()

'''ROC Curves for Grade F'''
models =[pipe_svc_F, pipe_lrl2_F1, gs_pipe_dtbag_F.best_estimator_, gs_et_F.best_estimator_, pipe_voting_F_opt_models, gs_gradient_F.best_estimator_]
model_names = ['SVC', 'Log Reg L2', 'Bagged DT', 'Extra Trees', 'Ensemble Voting, Soft', 'Gradient']

plot_roc_curve_models(X_gradeF_train.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_train, X_gradeF_test.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF_test ,models, model_names, 'Grade F', 'gradeF')

'''All of the models perform better on AUC than the PCA only ones on the continuous data.
Retrain the following models on all of data for 2012, validate on 2013, knowing the whole ground truth is not known.
-SVC, C = 1
-LR L2, C = 1
-Bagged DT
-Extra Trees
-Ensemble Voting

'''
#Getting ROC AUC Scores in chart
plt.figure(figsize = (16, 8))
scores_defaults_F_df[['mean_cv_roc_auc', 'test_roc_auc']].sort_values('mean_cv_roc_auc', ascending = False).plot(kind = 'bar', fontsize = 12)
plt.title('ROC AUC Scores for Models on Grade F loans, 2007-12', fontsize = 14)
plt.legend(loc = 'lower right')
plt.savefig('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeF/roc_auc_scores_train_test.png', bbox_inches = 'tight')

#Default rates in chart
plt.figure(figsize = (16, 8))
scores_defaults_F_df[['observed_default_rate', 'pred_default_rate']].sort_values('observed_default_rate', ascending = True).plot(kind = 'bar', fontsize = 12)
plt.title('Default Rates for Models for Grade F Loans on Test Data', fontsize = 12)
plt.axhline(y = scores_defaults_F_df['act_default_rate'][0], linewidth = 4, color = 'black')
plt.legend(loc = 'lower left', fontsize = 'small')
plt.savefig('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeF/defaults_test.png', bbox_inches = 'tight')

#gross returns for chart
models =[pipe_svc_F, pipe_lrl2_F1, gs_pipe_dtbag_F.best_estimator_, gs_et_F.best_estimator_, pipe_voting_F_opt_models, gs_gradient_F.best_estimator_]
model_col = ['svc_pred', 'logreg_pred', 'dtbag_pred', 'et_pred', 'ensemble_pred', 'gradient_pred']

y_pred = {}
for model, col in zip(models, model_col):
    y_pred[col] = model.predict(X_gradeF_test.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_])

predicted_gradeF = pd.DataFrame(y_pred, index = X_gradeF_test.index)
predicted_gradeF_actual = pd.merge(predicted_gradeF, pd.DataFrame(y_gradeF_test, index = y_gradeF_test.index), how = 'inner', right_index = True, left_index = True)
predicted_act_returns_gradeF = pd.merge(predicted_gradeF_actual, true_outcomes[['sub_grade', 'gross_return_scaled']], how = 'inner', right_index = True, left_index = True)

returns_mean = {}
for pred in ['dtbag_pred', 'ensemble_pred', 'et_pred', 'gradient_pred',
       'logreg_pred', 'svc_pred']:
       returns_mean[pred] = predicted_act_returns_gradeF[predicted_act_returns_gradeF[pred]==0]['gross_return_scaled'].mean()
returns_mean['whole_gradeF'] = predicted_act_returns_gradeF.gross_return_scaled.mean()
returns_gradeF_test = pd.DataFrame(returns_mean, index = ['mean_return']).T
plt.figure()
returns_gradeF_test.sort_values('mean_return', ascending = False).plot(kind = 'bar', color = 'green')
plt.axhline(y = returns_gradeF_test.loc['whole_gradeF'][0], color = 'black', linewidth = 4)
plt.title('Average Returns for Grade F loans, Predicted Non-Default vs. Test')
plt.savefig('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeF/returns_model_non_default_predicted.png', bbox_inches = 'tight')

#refitting models for production
model_fit_names = ['svc_gradeF', 'logreg_gradeF', 'dtbag_gradeF', 'et_gradeF', 'ensemble_gradeF', 'gradient_gradeF']
for model, fit_name in zip(models, model_fit_names):
    model_fitted = model.fit(X_gradeF.iloc[:,gs_gradeF_rf_searched.best_estimator_.steps[1][1].k_feature_idx_], y_gradeF)
    pickle_file = open(''.join(['../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeF/pickled models/', fit_name, '.pk1']), 'wb')
    pickle.dump(model_fitted, pickle_file, -1)
    pickle_file.close()


#Obtaining returns by sub-grade for Q&A
mean_model_sub_grade = {}
counts_sub_grade = {}
for pred in ['dtbag_pred', 'ensemble_pred', 'et_pred', 'gradient_pred',
       'logreg_pred', 'svc_pred']:
       mean_model_sub_grade[pred] =predicted_act_returns_gradeF[predicted_act_returns_gradeF[pred]==0].groupby('sub_grade').gross_return_scaled.mean()
       counts_sub_grade[pred] = predicted_act_returns_gradeF[predicted_act_returns_gradeF[pred]==0].groupby('sub_grade').gross_return_scaled.count()
gradeF_returns_subgrade = pd.merge(pd.DataFrame(mean_model_sub_grade), pd.DataFrame(predicted_act_returns_gradeF.groupby('sub_grade').gross_return_scaled.mean()), how = 'right', right_index = True, left_index = True)
gradeF_returns_counts = pd.merge(gradeF_returns_subgrade, pd.DataFrame(counts_sub_grade), how = 'left', right_index = True, left_index = True)
gradeF_returns_counts.rename(columns = {'dtbag_pred_x': 'dtbag_pred', 'ensemble_pred_x': 'ensemble_pred', 'et_pred_x': 'et_pred', 'gradient_pred_x': 'gradient_pred',
       'logreg_pred_x':'logreg_pred', 'svc_pred_x': 'svc_pred', 'gross_return_scaled':'returns',
       'dtbag_pred_y': 'dtbag_count', 'ensemble_pred_y': 'ensemble_count', 'et_pred_y': 'et count', 'gradient_pred_y': 'gradient_count',
       'logreg_pred_y':'logreg_count', 'svc_pred_y':'svc_count'}, inplace = True)
total_loans_gradeF = pd.DataFrame(predicted_act_returns_gradeF.groupby('sub_grade').gross_return_scaled.count())
total_loans_gradeF.rename(columns = {'gross_return_scaled': 'count_total_loans'}, inplace = True)
gradeF_returns_counts = pd.merge(gradeF_returns_counts, total_loans_gradeF, how = 'right', right_index = True, left_index = True)
gradeF_returns_counts.rename(columns = {'returns':'all_loans_returns'}, inplace = True)

gradeF_returns_counts.to_csv('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeF/returns_by_sub_grade_including_counts.csv', sep = ',', index = True, index_label = 'sub_grade')

################################################################################
################################################################################

'''Grade E

Small # of obs: < 2000.

Feature Selection:
1) Look at RF Feature importances
2) SBS from Sebastian Raschka using Logistic Regression L2.

'''
'''Method 1 for Grade E'''
#Random Forest Feature Importances...
rf_gradeE = RandomForestClassifier(n_estimators=10000, random_state = 0, n_jobs = -1)
rf_gradeE.fit(X_gradeE_train, y_gradeE_train)
feature_importances_gradeE = pd.DataFrame(rf_gradeE.feature_importances_, index = X_gradeE_train.columns).sort_values(0, ascending = False)
feature_importances_gradeE.rename(columns = {0: 'Importance'}, inplace = True)

#Take a look at the feature importances. Definitely some similarity to F, but
#different order of features.
feature_importances_gradeE

#getting the number of features that would be selected for threshold. Choosing 0.01 b/c highest importance is <.07 due to 58 features.
gradeE_selected = SelectFromModel(rf_gradeE, threshold = .01, prefit = True).transform(X_gradeE_train)
# num of obs and the number of features. 18 in this case. Was 22 for F?
gradeE_selected.shape

#Creating column list for feature selection from RF.
features_selected_gradeE_cols = feature_importances_gradeE.iloc[range(0,18)]
features_selected_gradeE_columns = list(features_selected_gradeE_cols.index)
#Saving overall features importances for grade E
feature_importances_gradeE.to_csv('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/rf_feature_importances_sorted.csv', sep = ',', index = True)

'''Method 2 for Grade E'''
#SBS, using Logistic Regression L2 and RF

rf = RandomForestClassifier(max_depth = 5, class_weight = 'balanced', min_samples_leaf  = 6, n_estimators = 20)

sbs_gradeE = SFS(LogisticRegression(C =1, class_weight = 'balanced', penalty = 'l2'), k_features = 29, forward = False, floating = False,\
 scoring = 'roc_auc', print_progress = True, cv = 5)
sbs_gradeE_rf = SFS(rf, k_features = 29, forward = False, floating = False,\
 scoring = 'roc_auc', print_progress = True, cv = 5)

#pipelines for SBS
gradeE_pipe = Pipeline([('minmax', MinMaxScaler()),('sbs', sbs_gradeE), ('lrl2', LogisticRegression(C =1, class_weight = 'balanced', penalty = 'l2'))])
gradeE_pipe_rf = Pipeline([('minmax', MinMaxScaler()),('sbs', sbs_gradeE_rf), ('rf', rf)])

#grid search set up for two searches for number of features and features.
param_grid = [{'sbs__k_features': [15, 18, 20, 25, 29]}]
gs_gradeE = GridSearchCV(estimator = gradeE_pipe, param_grid = param_grid, scoring = 'roc_auc', cv = 5, n_jobs = -1, verbose = True, refit = True)
gs_gradeE_rf = GridSearchCV(estimator = gradeE_pipe_rf, param_grid = param_grid, scoring = 'roc_auc', cv = 5, n_jobs = -1, verbose = True, refit = True)

gs_gradeE_lr = gs_gradeE.fit(X_gradeE_train, y_gradeE_train)
gs_gradeE_rf_searched = gs_gradeE_rf.fit(X_gradeE_train, y_gradeE_train)

print("Best parameters via GridSearch with Log Reg", gs_gradeE_lr.best_params_)
print('Best features with Log Reg:', gs_gradeE_lr.best_estimator_.steps[1][1].k_feature_idx_)


pd.DataFrame(X_gradeE_train.iloc[:, logreg_best_features_index].columns).to_csv('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/logreg_params.csv', sep = ',', index = True)
logreg_best_features_index = pd.read_csv('../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/logreg_params.csv')
logreg_best_features_index = list(logreg_best_features_index['0'])

print("Best parameters via GridSearch with RF", gs_gradeE_rf_searched.best_params_)
print('Best features with RF:', gs_gradeE_rf_searched.best_estimator_.steps[1][1].k_feature_idx_)
pd.DataFrame(X_gradeE_train.iloc[:, gs_gradeE_rf_searched.best_estimator_.steps[1][1].k_feature_idx_].columns).to_csv('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/rf_SBS_feature_selection.csv', sep = ',', index = True)
rf_sbs_features = pd.DataFrame(X_gradeE_train.iloc[:,gs_gradeE_rf_searched.best_estimator_.steps[1][1].k_feature_idx_].columns.values)
rf_featimp = pd.DataFrame(features_selected_gradeE_cols)
rf_sbs_features.rename(columns = {0: 'sbs'}, inplace = True)
rf_featimp.rename(columns = {'Unnamed: 0': 'rf_featimp'}, inplace = True)

#features from Feat Imp that are in SBS_rf - note that Risk Score and other cont are not in this.
rf_featimp_in_sbs = pd.merge(rf_sbs_features, rf_featimp, how = 'left', left_on = 'sbs', right_index = True)
rf_featimp_in_sbs.to_csv('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/RF feature imps in SBS.csv', sep = ',', index = True, index_label = 'sbs')

rf_sbs_in_featimp = pd.merge(rf_sbs_features, rf_featimp, how = 'right', left_on = 'sbs', right_index = True)
rf_sbs_in_featimp.to_csv('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/SBS feats in RF feature imps.csv', sep = ',', index = True, index_label = 'rf_featimp')

#printing performance of results on base estimators. Not too meaningful because we need to optimize, but still interesting.
from mlxtend.feature_selection import plot_sequential_feature_selection as plot_sfs

fig1 = plot_sfs(gs_gradeE_lr.best_estimator_.steps[1][1].get_metric_dict(), kind='std_dev')

plt.ylim([0.4, 1])
plt.title('Sequential Backward Selection with Logistic Regression (w. StdDev)')
plt.grid()
plt.savefig('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/SBS_lr.png')

fig2 = plot_sfs(gs_gradeE_rf_searched.best_estimator_.steps[1][1].get_metric_dict(), kind='std_dev')

plt.ylim([0.4, 1])
plt.title('Sequential Backward Selection with Random Forest (w. StdDev)')
plt.grid()
plt.savefig('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/SBS_rf.png')

'''Looking at benchmark ROC AUC for feature selection. Using Log Reg'''
features = [gs_gradeE_rf_searched.best_estimator_.steps[1][1].k_feature_idx_, gs_gradeE_lr.best_estimator_.steps[1][1].k_feature_idx_]
roc_auc_bench = {}
for model, name in zip([pipe_lrl2_E, pipe_et_E, pipe_svc_E],['lrl2', 'et', 'svc']):
    _, roc_auc_bench[name+'rf_feat_imp'], _, _, _, _ = evaluate_model(model, X_gradeE_train[features_selected_gradeE_columns], y_gradeE_train, X_gradeE_test[features_selected_gradeE_columns], y_gradeE_test)

names = ['rf_sbs', 'lr_sbs']
for model, model_name in zip([pipe_lrl2_E, pipe_et_E, pipe_svc_E],['lrl2', 'et', 'svc']):
    for feature_col, name in zip(features, names):
        _, roc_auc_bench[model_name + name], _, _, _, _ = evaluate_model(model, X_gradeE_train.iloc[:,feature_col], y_gradeE_train, X_gradeE_test.iloc[:,feature_col], y_gradeE_test)

pd.DataFrame(roc_auc_bench, index = ['cross_val_roc_auc']).T.sort_values('cross_val_roc_auc', ascending = False).to_csv('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/logreg_SBS_wins_feature_selection.csv', sep = ',', index = True)
gradeE_best_features_comp = pd.read_csv('../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/logreg_SBS_wins_feature_selection.csv')

'''Looking at the benchmark ROC AUC comparing the features from RF feature importance to SBS from Log Reg and RF,
the Log Reg do best, unequivocally.

Hence, for grade E, I would use the ones from the LR SBS.'''


#Now that I have the best features for grade E, let's use zip to get the scores for them
#for grade E.
scores_defaults_E = {}
pipe_svc_E = make_pipeline(MinMaxScaler(), SVC(C = 1, kernel = 'rbf', max_iter = 30000, probability = True, class_weight = 'balanced', random_state = 31))
pipe_dt_E = make_pipeline(MinMaxScaler(), DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 4, min_samples_split = 8, min_samples_leaf = 4))
pipe_dtbag_E = make_pipeline(MinMaxScaler(), BaggingClassifier(DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 3, min_samples_split = 6, min_samples_leaf = 8), random_state = 31, n_estimators = 50))
pipe_rf_E = make_pipeline(MinMaxScaler(), RandomForestClassifier(random_state = 31, class_weight = 'balanced', n_estimators = 100, max_depth = 4, min_samples_split = 8))
pipe_et_E = make_pipeline(MinMaxScaler(), ExtraTreesClassifier(random_state = 31, class_weight = 'balanced', max_depth = 3, min_samples_split = 3, min_samples_leaf = 3))
pipe_lrl1_E = make_pipeline(MinMaxScaler(), LogisticRegression(C = 1, class_weight = 'balanced', penalty = 'l1'))
pipe_lrl2_E = make_pipeline(MinMaxScaler(), LogisticRegression(C = 1, class_weight = 'balanced', penalty = 'l2'))
pipe_adaboost_E = make_pipeline(MinMaxScaler(), AdaBoostClassifier(DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 5, min_samples_split = 6), random_state = 31))
pipe_adaboost_E_lr = make_pipeline(MinMaxScaler(), AdaBoostClassifier(LogisticRegression(C = 1, class_weight = 'balanced', penalty = 'l2', solver = 'lbfgs', max_iter = 1000), random_state = 31, learning_rate = .2, n_estimators = 200))
pipe_gradient_E_dev = make_pipeline(MinMaxScaler(), GradientBoostingClassifier(loss = 'deviance', learning_rate = .1, n_estimators = 200, max_depth = 4, min_samples_leaf = 6, verbose = True))
pipe_gradient_E_exp = make_pipeline(MinMaxScaler(), GradientBoostingClassifier(loss = 'exponential', learning_rate = .1, n_estimators = 200, max_depth = 4, min_samples_leaf = 6,  verbose = True))
pipe_voting_E = VotingClassifier(estimators = [('svc', pipe_svc_E), ('dtbag', pipe_dtbag_E), ('lrl2',pipe_lrl2_E), ('lrl1', pipe_lrl1_E), ('rf', pipe_rf_E), ('et', pipe_et_E), ('ada_lr', pipe_adaboost_E_lr)], voting = 'soft')

models = [pipe_svc_E, pipe_dt_E, pipe_dtbag_E, pipe_rf_E, pipe_et_E, pipe_lrl1_E, pipe_lrl2_E, pipe_adaboost_E, pipe_adaboost_E_lr, pipe_gradient_E_dev, pipe_gradient_E_exp, pipe_voting_E]
model_names = ['SVC', 'DT', 'Bagged DT', 'Random Forest', 'Extra Trees', 'Logistic Regression L1', 'Logistic Regression L2', 'AdaBoost', 'AdaBoost with LogReg', 'Gradient Deviance', 'Gradient Exp', 'Voting Untuned']
#Bagged DT with fewer branches does better. Prune the DT for GS? Log Reg and SVC do well.
for model, model_name in zip(models, model_names):
    print ""
    print model_name + " Output for Grade E Loans, 2007 - 12"
    print ""
    scores_defaults_E[model_name] = evaluate_model(model, X_gradeE_train[logreg_best_features_index], y_gradeE_train, X_gradeE_test[logreg_best_features_index], y_gradeE_test)

scores_defaults_E_df = pd.DataFrame(scores_defaults_E, index = ['mean_cv_recall', 'mean_cv_roc_auc', 'test_roc_auc', 'pred_default_rate', 'act_default_rate', 'observed_default_rate']).T.sort_values('test_roc_auc', ascending = False)
scores_defaults_E_df.to_csv('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/scores_default_gradeE.csv', sep = ',', index = True)

'''For Grade E, let's try to optimize DT, then Bagged DT, then AdaBoost, Log Reg, SVC, ET, RF.
Then plot learning curves for each'''
#DECISION TREE
dt_params = {'decisiontreeclassifier__max_depth': [4,5, 6, 8, 10, 12], 'decisiontreeclassifier__min_samples_leaf':[2,4,6, 8, 10, 12], \
'decisiontreeclassifier__min_samples_split':[2,5,10],\
'decisiontreeclassifier__criterion': ['gini', 'entropy'], 'decisiontreeclassifier__splitter':['random', 'best']}

gs_dt_gradeE = GridSearchCV(pipe_dt_E, dt_params, cv = StratifiedKFold(y_gradeE_train, 5, shuffle = True), verbose = True, scoring = 'roc_auc', n_jobs = -1)
gs_dt_gradeE.fit(X_gradeE_train[logreg_best_features_index], y_gradeE_train)
print "Best roc_auc score:", gs_dt_gradeE.best_score_
print "Best params:", gs_dt_gradeE.best_params_
np.mean(cross_val_score(gs_dt_gradeE.best_estimator_, X_gradeE_train.iloc[:,logreg_best_features_index], y_gradeE_train, cv = 5, scoring = 'roc_auc'))
scores_defaults_E['gs_dt'] = evaluate_model(gs_dt_gradeE.best_estimator_, X_gradeE_train[logreg_best_features_index], y_gradeE_train, X_gradeE_test[logreg_best_features_index], y_gradeE_test)

#Bagged DTBest dt params: {'decisiontreeclassifier__min_samples_leaf': 2, 'decisiontreeclassifier__min_samples_split': 10, 'decisiontreeclassifier__max_depth': 4, 'decisiontreeclassifier__criterion': 'gini', 'decisiontreeclassifier__splitter': 'random'}
pipe_dtbag_E_optdt = make_pipeline(MinMaxScaler(), BaggingClassifier(DecisionTreeClassifier(class_weight = 'balanced', criterion = 'gini', splitter = 'random', random_state = 31, max_depth = 4, min_samples_split = 10, min_samples_leaf = 2), random_state = 31))
bagging_params = {'baggingclassifier__n_estimators': [ 600, 800, 1000, 1200], 'baggingclassifier__bootstrap_features': [True, False]}
gs_pipe_dtbag_E = GridSearchCV(pipe_dtbag_E_optdt, bagging_params, cv = StratifiedKFold(y_gradeE_train, 5, shuffle = True), verbose = True, scoring = 'roc_auc', n_jobs = -1)
gs_pipe_dtbag_E.fit(X_gradeE_train[logreg_best_features_index], y_gradeE_train)
print "Best roc_auc score:", gs_pipe_dtbag_E.best_score_
print "Best params:", gs_pipe_dtbag_E.best_params_
np.mean(cross_val_score(gs_pipe_dtbag_E.best_estimator_, X_gradeE_train[logreg_best_features_index], y_gradeE_train, cv = 5, scoring = 'roc_auc'))
scores_defaults_E['gs_bagdt'] = evaluate_model(gs_pipe_dtbag_E.best_estimator_, X_gradeE_train[logreg_best_features_index], y_gradeE_train, X_gradeE_test[logreg_best_features_index], y_gradeE_test)

#Gradient Exp
#pipe_gradient_F_dev = make_pipeline(MinMaxScaler(), GradientBoostingClassifier(loss = 'deviance', learning_rate = .1, n_estimators = 200, max_depth = 4, min_samples_leaf = 6, verbose = True))
pipe_gradient_E_exp_optdt = make_pipeline(MinMaxScaler(), GradientBoostingClassifier(loss = 'exponential', learning_rate = .1, n_estimators = 200, max_depth = 4, min_samples_leaf = 10, min_samples_split = 2))
gradient_params = {'gradientboostingclassifier__loss': ['deviance', 'exponential'], 'gradientboostingclassifier__learning_rate': [.3, .5, 1], 'gradientboostingclassifier__n_estimators': [500, 700, 900, 1100]}
gs_gradient_E = GridSearchCV(pipe_gradient_E_exp_optdt, gradient_params, cv = StratifiedKFold(y_gradeE_train, 5, shuffle = True), verbose = True, scoring = 'roc_auc', n_jobs = -1)
gs_gradient_E.fit(X_gradeE_train[logreg_best_features_index], y_gradeE_train)

print "Best roc_auc score:", gs_gradient_E.best_score_
print "Best params:", gs_gradient_E.best_params_
np.mean(cross_val_score(gs_gradient_E.best_estimator_, X_gradeE_train[logreg_best_features_index], y_gradeE_train, cv = 5, scoring = 'roc_auc'))
scores_defaults_E['gs_Gradient'] = evaluate_model(gs_gradient_E.best_estimator_, X_gradeE_train[logreg_best_features_index], y_gradeE_train, X_gradeE_test.iloc[:,logreg_best_features_index], y_gradeE_test)

#Adaboost with opt dt
pipe_adaboost_E_optdt = make_pipeline(MinMaxScaler(), AdaBoostClassifier(DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 4, min_samples_split = 10, min_samples_leaf = 2), random_state = 31))
adaboost_params = {'adaboostclassifier__n_estimators': [300, 400, 500], 'adaboostclassifier__learning_rate': [.7, 1, 1.3, 1.5, 2]}
gs_adaboost_E = GridSearchCV(pipe_adaboost_E_optdt, adaboost_params, cv = StratifiedKFold(y_gradeE_train, 5, shuffle = True), verbose = True, scoring = 'roc_auc', n_jobs = -1)
gs_adaboost_E.fit(X_gradeE_train[logreg_best_features_index], y_gradeE_train)
print "Best roc_auc score:", gs_adaboost_E.best_score_
print "Best params:", gs_adaboost_E.best_params_
np.mean(cross_val_score(gs_adaboost_E.best_estimator_, X_gradeE_train[logreg_best_features_index], y_gradeE_train, cv = 5, scoring = 'roc_auc'))
scores_defaults_E['gs_AdaBoost'] = evaluate_model(gs_adaboost_E.best_estimator_, X_gradeE_train[logreg_best_features_index], y_gradeE_train, X_gradeE_test[logreg_best_features_index], y_gradeE_test)

#Logistic Regression L2 C = 1 seems best...
scores = {}
for c in [0.001, 0.01, 0.1, 100, 1000, 10000]:
    pipe_lrl2_E = make_pipeline(MinMaxScaler(), LogisticRegression(C = c, class_weight = 'balanced', penalty = 'l2'))
    score = np.mean(cross_val_score(pipe_lrl2_E, X_gradeE_train.iloc[:,logreg_best_features_index], y_gradeE_train, cv = 5, scoring = 'roc_auc'))
    scores[c] = score
pipe_lrl2_E = make_pipeline(MinMaxScaler(), LogisticRegression(C = 1, class_weight = 'balanced', penalty = 'l2'))

#Logistic Regression L1 C = 1000 seems best...
scores = {}
for c in [0.001, 0.01, 0.1,1, 100, 1000, 10000]:
    pipe_lrl1_E = make_pipeline(MinMaxScaler(), LogisticRegression(C = c, class_weight = 'balanced', penalty = 'l1'))
    score = np.mean(cross_val_score(pipe_lrl1_E, X_gradeE_train[logreg_best_features_index], y_gradeE_train, cv = 5, scoring = 'roc_auc'))
    scores[c] = score
pipe_lrl1_E = make_pipeline(MinMaxScaler(), LogisticRegression(C = 1000, class_weight = 'balanced', penalty = 'l1'))
scores_defaults_E['Opt Logistic Regression L1'] = evaluate_model(pipe_lrl1_E, X_gradeE_train[logreg_best_features_index], y_gradeE_train, X_gradeE_test[logreg_best_features_index], y_gradeE_test)
scores_defaults_E['AdaBoost with Log Reg L2'] = evaluate_model(pipe_adaboost_E_lr, X_gradeE_train[logreg_best_features_index], y_gradeE_train, X_gradeE_test[logreg_best_features_index], y_gradeE_test)

#Random Forest
pipe_rf_E_opt = make_pipeline(MinMaxScaler(), RandomForestClassifier(random_state = 31, class_weight = 'balanced', n_estimators = 100, max_depth = 4, min_samples_split = 10, min_samples_leaf =2))
random_params = {'randomforestclassifier__criterion': ['gini', 'entropy'], 'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2', None], 'randomforestclassifier__n_estimators': [100, 200, 300]}
gs_rf_E = GridSearchCV(pipe_rf_E_opt, random_params, cv = StratifiedKFold(y_gradeE_train, 5, shuffle = True), verbose = True, scoring = 'roc_auc', n_jobs = -1)
gs_rf_E.fit(X_gradeE_train[logreg_best_features_index], y_gradeE_train)
print "Best roc_auc score:", gs_rf_E.best_score_
print "Best params:", gs_rf_E.best_params_
np.mean(cross_val_score(gs_rf_E.best_estimator_, X_gradeE_train[logreg_best_features_index], y_gradeE_train, cv = 5, scoring = 'roc_auc'))
scores_defaults_E['gs_Random Forest'] = evaluate_model(gs_rf_E.best_estimator_, X_gradeE_train[logreg_best_features_index], y_gradeE_train, X_gradeE_test[logreg_best_features_index], y_gradeE_test)

#SVC
svc_params = {'svc__C': [.001, .01,.1, 1, 10, 100]}
gs_svc_E = GridSearchCV(pipe_svc_E, svc_params, cv = StratifiedKFold(y_gradeE_train, 5, shuffle = True), verbose = True, n_jobs = -1, scoring = 'roc_auc')
gs_svc_E.fit(X_gradeE_train.iloc[:,logreg_best_features_index], y_gradeE_train)
print "Best roc_auc score:", gs_svc_E.best_score_
print "Best params:", gs_svc_E.best_params_

#ET
pipe_et_E_opt = make_pipeline(MinMaxScaler(), ExtraTreesClassifier(random_state = 31, class_weight = 'balanced', n_estimators = 100, max_depth = 4, min_samples_split = 10, min_samples_leaf =2))
et_params = {'extratreesclassifier__criterion': ['gini', 'entropy'], 'extratreesclassifier__max_features': ['auto', 'sqrt', 'log2', None], 'extratreesclassifier__n_estimators': [200, 300, 400, 500]}
gs_et_E = GridSearchCV(pipe_et_E_opt, et_params, cv = StratifiedKFold(y_gradeE_train, 5, shuffle = True), verbose = True, scoring = 'roc_auc', n_jobs = -1)
gs_et_E.fit(X_gradeE_train[logreg_best_features_index], y_gradeE_train)
print "Best roc_auc score:", gs_et_E.best_score_
print "Best params:", gs_et_E.best_params_
np.mean(cross_val_score(gs_et_E.best_estimator_, X_gradeE_train[logreg_best_features_index], y_gradeE_train, cv = 5, scoring = 'roc_auc'))
scores_defaults_E['gs_Extra Trees'] = evaluate_model(gs_et_E.best_estimator_, X_gradeE_train[logreg_best_features_index], y_gradeE_train, X_gradeE_test[logreg_best_features_index], y_gradeE_test)

'''Learning Curves for Models Grade E'''
pipe_voting_E_opt_models = make_pipeline(MinMaxScaler(), VotingClassifier(estimators = [('lrl2', pipe_lrl2_E), ('dtbag', gs_pipe_dtbag_E.best_estimator_), ('et', gs_et_E.best_estimator_), ('rf',gs_rf_E.best_estimator_), ('svc', pipe_svc_E), ('lrl1',pipe_lrl1_E)], voting = 'soft'))
evaluate_model(pipe_voting_E_opt_models, X_gradeE_train[logreg_best_features_index], y_gradeE_train, X_gradeE_test[logreg_best_features_index], y_gradeE_test)
#While lots of bias and variance, in general, the learning curves for these look better than for PCA for Grade E.
models_opt = [pipe_lrl2_E, pipe_lrl1_E, gs_pipe_dtbag_E.best_estimator_, gs_et_E.best_estimator_, gs_rf_E.best_estimator_, pipe_svc_E, gs_gradient_E.best_estimator_, gs_adaboost_E.best_estimator_, pipe_dt_E, pipe_voting_E_opt_models, pipe_adaboost_E_lr]
models_names = ['Logistic Regression L2', 'Logistic Regression L1', 'Bagged DT', 'Extra Trees', 'Random Forest', 'SVM', 'Gradient Boosting', 'AdaBoost', 'Decision Tree', 'Ensemble Voting', 'AdaBoost LR']

for model_opt, model_name in zip(models_opt, models_names):
    plot_learning_curves(X_gradeE_train.iloc[:,logreg_best_features_index], y_gradeE_train, X_gradeE_test.iloc[:,logreg_best_features_index], y_gradeE_test, clf = model_opt, print_model = False, scoring = 'roc_auc')
    plt.title('Learning Curve for ' + model_name + ' on Grade E Loans, 2007-12')
    plt.savefig(''.join(['../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/gradeE_learning_curve ',model_name,'.png']), bbox_inches = 'tight')
    plt.show()

'''ROC Curves for Grade E'''
models =[pipe_svc_E, pipe_lrl2_E, gs_pipe_dtbag_E.best_estimator_, gs_et_E.best_estimator_, pipe_voting_E_opt_models, pipe_adaboost_E_lr]
model_names = ['SVC', 'Log Reg L2', 'Bagged DT', 'Extra Trees', 'Ensemble Voting, Soft', 'AdaBoost - LR L2']

plot_roc_curve_models(X_gradeE_train.iloc[:,logreg_best_features_index], y_gradeE_train, X_gradeE_test.iloc[:,logreg_best_features_index], y_gradeE_test ,models, model_names, 'Grade E', 'gradeE')

'''All of the models perform better on AUC than the PCA only ones on the continuous data.
Retrain the following models on all of data for 2012, validate on 2013, knowing the whole ground truth is not known.
-SVC, C = 1
-LR L2, C = 1
-Bagged DT
-Extra Trees
-Ensemble Voting

'''
#Getting ROC AUC Scores in chart
plt.figure(figsize = (16, 8))
scores_defaults_E_df[['mean_cv_roc_auc', 'test_roc_auc']].sort_values('mean_cv_roc_auc', ascending = False).plot(kind = 'bar', fontsize = 12)
plt.title('ROC AUC Scores for Models on Grade E loans, 2007-12', fontsize = 14)
plt.ylim([.4, .8])
plt.legend(loc = 'upper right')
plt.savefig('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/roc_auc_scores_train_test.png', bbox_inches = 'tight')

#Default rates in chart
plt.figure(figsize = (16, 8))
scores_defaults_E_df[['observed_default_rate', 'pred_default_rate']].sort_values('observed_default_rate', ascending = True).plot(kind = 'bar', fontsize = 12)
plt.title('Default Rates for Models for Grade E Loans on Test Data', fontsize = 12)
plt.axhline(y = scores_defaults_E_df['act_default_rate'][0], linewidth = 4, color = 'black')
plt.legend(loc = 'lower left', fontsize = 'small')
plt.savefig('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/defaults_test.png', bbox_inches = 'tight')

#gross returns for chart
models =[pipe_svc_E, pipe_lrl2_E, pipe_lrl1_E, pipe_adaboost_E_lr, gs_rf_E.best_estimator_, gs_pipe_dtbag_E.best_estimator_, gs_et_E.best_estimator_, pipe_voting_E_opt_models]
model_col = ['svc_pred', 'logreg_l2_pred','logreg_l1_pred', 'adaboost_lr_pred', 'rf_pred','dtbag_pred', 'et_pred', 'ensemble_pred']

y_pred = {}
for model, col in zip(models, model_col):
    y_pred[col] = model.predict(X_gradeE_test[logreg_best_features_index])

predicted_gradeE = pd.DataFrame(y_pred, index = X_gradeE_test.index)
predicted_gradeE_actual = pd.merge(predicted_gradeE, pd.DataFrame(y_gradeE_test, index = y_gradeE_test.index), how = 'inner', right_index = True, left_index = True)
predicted_act_returns_gradeE = pd.merge(predicted_gradeE_actual, true_outcomes[['sub_grade', 'gross_return_scaled']], how = 'inner', right_index = True, left_index = True)

returns_mean = {}
for pred in predicted_gradeE.columns.values:
       returns_mean[pred] = predicted_act_returns_gradeE[predicted_act_returns_gradeE[pred]==0]['gross_return_scaled'].mean()
returns_mean['whole_gradeE'] = predicted_act_returns_gradeE.gross_return_scaled.mean()
returns_gradeE_test = pd.DataFrame(returns_mean, index = ['mean_return']).T
plt.figure()
returns_gradeE_test.sort_values('mean_return', ascending = False).plot(kind = 'bar', color = 'green')
plt.axhline(y = returns_gradeE_test.loc['whole_gradeE'][0], color = 'black', linewidth = 4)
plt.title('Average Returns for Grade E loans, Predicted Non-Default vs. Test')
plt.savefig('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/returns_model_non_default_predicted.png', bbox_inches = 'tight')
returns_gradeE_test.to_csv('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/gross returns by model.csv', sep = ',', index = True, index_label = 'model')
print "Mean return for top three models vs baseline", returns_gradeE_test.ix[0:3].mean()/returns_gradeE_test.ix[-1]-1
print "Mean return for top model vs baseline", returns_gradeE_test.ix[0][0]/returns_gradeE_test.ix[-1][0]-1
#refitting models for production
models =[pipe_svc_E, pipe_lrl2_E, pipe_lrl1_E, pipe_adaboost_E_lr, gs_rf_E.best_estimator_, gs_pipe_dtbag_E.best_estimator_, gs_et_E.best_estimator_, pipe_voting_E_opt_models]
model_fit_names = ['svc_gradeE', 'logreg_l2_gradeE', 'logreg_l1_gradeE', 'adaboost_lr_gradeE','rf_gradeE', 'dtbag_gradeE', 'et_gradeE', 'ensemble_gradeE']
for model, fit_name in zip(models, model_fit_names):
    model_fitted = model.fit(X_gradeE[logreg_best_features_index], y_gradeE)
    pickle_file = open(''.join(['../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/pickled models/', fit_name, '.pk1']), 'wb')
    pickle.dump(model_fitted, pickle_file, -1)
    pickle_file.close()

gradeE_returns_counts.columns.values
#Obtaining returns by sub-grade for Q&A
mean_model_sub_grade = {}
counts_sub_grade = {}
for pred in ['dtbag_pred', 'ensemble_pred', 'et_pred', 'adaboost_lr_pred', 'rf_pred',
       'logreg_l1_pred','logreg_l2_pred', 'svc_pred']:
       mean_model_sub_grade[pred] =predicted_act_returns_gradeE[predicted_act_returns_gradeE[pred]==0].groupby('sub_grade').gross_return_scaled.mean()
       counts_sub_grade[pred] = predicted_act_returns_gradeE[predicted_act_returns_gradeE[pred]==0].groupby('sub_grade').gross_return_scaled.count()
gradeE_returns_subgrade = pd.merge(pd.DataFrame(mean_model_sub_grade), pd.DataFrame(predicted_act_returns_gradeE.groupby('sub_grade').gross_return_scaled.mean()), how = 'right', right_index = True, left_index = True)
gradeE_returns_counts = pd.merge(gradeE_returns_subgrade, pd.DataFrame(counts_sub_grade), how = 'left', right_index = True, left_index = True)
gradeE_returns_counts.rename(columns = {'adaboost_lr_pred_x': 'adaboost_lr_pred', 'dtbag_pred_x': 'dtbag_pred', 'ensemble_pred_x': 'ensemble_pred', 'et_pred_x': 'et_pred',
       'logreg_l1_pred_x':'logreg_l1_pred', 'logreg_l2_pred_x':'logreg_l2_pred','rf_pred_x': 'rf_pred', 'svc_pred_x': 'svc_pred', 'gross_return_scaled':'returns',
       'adaboost_lr_pred_y': 'adaboost_lr_count', 'dtbag_pred_y': 'dtbag_count', 'ensemble_pred_y': 'ensemble_count', 'et_pred_y': 'et count',
       'logreg_l1_pred_y':'logreg_l1_count', 'logreg_l2_pred_y':'logreg_l2_count','rf_pred_y': 'rf_count','svc_pred_y':'svc_count'}, inplace = True)
total_loans_gradeE = pd.DataFrame(predicted_act_returns_gradeE.groupby('sub_grade').gross_return_scaled.count())
total_loans_gradeE.rename(columns = {'gross_return_scaled': 'count_total_loans'}, inplace = True)
gradeF_returns_counts = pd.merge(gradeE_returns_counts, total_loans_gradeE, how = 'right', right_index = True, left_index = True)
gradeF_returns_counts.rename(columns = {'returns':'all_loans_returns'}, inplace = True)

gradeE_returns_counts.to_csv('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/returns_by_sub_grade_including_counts.csv', sep = ',', index = True, index_label = 'sub_grade')

################################################################################
################################################################################

'''Grade D

Decent # of obs: < 9000.

Feature Selection:
1) Look at RF Feature importances
2) SBS from Sebastian Raschka using Logistic Regression L2.

'''
'''Method 1 for Grade D'''
#Random Forest Feature Importances...
rf_gradeD = RandomForestClassifier(n_estimators=10000, random_state = 0, n_jobs = -1)
rf_gradeD.fit(X_gradeD_train, y_gradeD_train)
feature_importances_gradeD = pd.DataFrame(rf_gradeD.feature_importances_, index = X_gradeE_train.columns).sort_values(0, ascending = False)
feature_importances_gradeD.rename(columns = {0: 'Importance'}, inplace = True)

#Take a look at the feature importances. Definitely some similarity to F, but
#different order of features.
feature_importances_gradeD

#getting the number of features that would be selected for threshold. Choosing 0.01 b/c highest importance is <.07 due to 58 features.
gradeD_selected = SelectFromModel(rf_gradeD, threshold = .01, prefit = True).transform(X_gradeD_train)
# num of obs and the number of features. 18 in this case. Was 22 for F?. 17 for D.
gradeD_selected.shape

#Creating column list for feature selection from RF.
features_selected_gradeD_cols = feature_importances_gradeD.iloc[range(0,17)]
features_selected_gradeD_columns = list(features_selected_gradeD_cols.index)
#Saving overall features importances for grade E
feature_importances_gradeD.to_csv('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeD/rf_feature_importances_sorted.csv', sep = ',', index = True)
feature_importances_gradeD = pd.read_csv('../Results/second_run/three_year/2007-12 training/Feature Selection/gradeD/rf_feature_importances_sorted.csv')
feature_importances_gradeD.rename(columns = {'Unnamed: 0': 'feature'}, inplace = True)
'''Method 2 for Grade D'''
#SBS, using Logistic Regression L2 and RF
from sklearn.pipeline import Pipeline
rf = RandomForestClassifier(max_depth = 5, class_weight = 'balanced', min_samples_leaf  = 6, n_estimators = 20)

sbs_gradeD = SFS(LogisticRegression(C =1, class_weight = 'balanced', penalty = 'l2'), k_features = 17, forward = False, floating = False,\
 scoring = 'roc_auc', print_progress = True, cv = 3)
sbs_gradeD_rf = SFS(rf, k_features = 17, forward = False, floating = False,\
 scoring = 'roc_auc', print_progress = True, cv = 3)
sbs_gradeD_25 = SFS(LogisticRegression(C =1, class_weight = 'balanced', penalty = 'l2'), k_features = 25, forward = False, floating = False,\
 scoring = 'roc_auc', print_progress = True, cv = 3)
sbs_gradeD_rf_25 = SFS(rf, k_features = 25, forward = False, floating = False,\
 scoring = 'roc_auc', print_progress = True, cv = 3)
#pipelines for SBS
gradeD_pipe = Pipeline([('minmax', MinMaxScaler()),('sbs', sbs_gradeD), ('lrl2', LogisticRegression(C =1, class_weight = 'balanced', penalty = 'l2'))])
gradeD_pipe_rf = Pipeline([('minmax', MinMaxScaler()),('sbs', sbs_gradeD_rf), ('rf', rf)])
gradeD_pipe_25 = Pipeline([('minmax', MinMaxScaler()),('sbs', sbs_gradeD_25), ('lrl2', LogisticRegression(C =1, class_weight = 'balanced', penalty = 'l2'))])
gradeD_pipe_rf_25 = Pipeline([('minmax', MinMaxScaler()),('sbs', sbs_gradeD_rf_25), ('rf', rf)])

sbs_lr = gradeD_pipe.fit(X_gradeD_train, y_gradeD_train)
sbs_lr_25 = gradeD_pipe_25.fit(X_gradeD_train, y_gradeD_train)
sbs_rf = gradeD_pipe_rf.fit(X_gradeD_train, y_gradeD_train)
sbs_rf_25 = gradeD_pipe_rf_25.fit(X_gradeD_train, y_gradeD_train)

sbs_lr.steps
print('Best features with Log Reg:', sbs_lr.steps[1][1].k_feature_idx_)

logreg_best_features_index = sbs_lr.steps[1][1].k_feature_idx_
pd.DataFrame(X_gradeD_train.iloc[:, logreg_best_features_index].columns).to_csv('../Results/second_run/three_year/2007-12 training/Feature Selection/gradeD/logreg_params.csv', sep = ',', index = True)
logreg_best_features_index = pd.read_csv('../Results/second_run/three_year/2007-12 training/Feature Selection/gradeD/logreg_params.csv')
logreg_best_features_index = list(logreg_best_features_index['0'])

logreg_best_features_index_25 = sbs_lr_25.steps[1][1].k_feature_idx_
pd.DataFrame(X_gradeD_train.iloc[:, logreg_best_features_index_25].columns).to_csv('../Results/second_run/three_year/2007-12 training/Feature Selection/gradeD/logreg_params_k_25.csv', sep = ',', index = True)
logreg_best_features_index_25 = pd.read_csv('../Results/second_run/three_year/2007-12 training/Feature Selection/gradeD/logreg_params_k_25.csv')
logreg_best_features_index_25 = list(logreg_best_features_index_25['0'])

print('Best features with RF:', sbs_rf.steps[1][1].k_feature_idx_)
pd.DataFrame(X_gradeD_train.iloc[:, sbs_rf.steps[1][1].k_feature_idx_].columns).to_csv('../Results/second_run/three_year/2007-12 training/Feature Selection/gradeD/rf_SBS_feature_selection.csv', sep = ',', index = True)
rf_sbs_features = pd.DataFrame(X_gradeD_train.iloc[:,sbs_rf.steps[1][1].k_feature_idx_].columns.values)
rf_featimp = pd.DataFrame(features_selected_gradeD_cols)
rf_sbs_features.rename(columns = {0: 'sbs'}, inplace = True)
rf_featimp.rename(columns = {'Unnamed: 0': 'rf_featimp'}, inplace = True)

print('Best features with RF, k = 25:', sbs_rf_25.steps[1][1].k_feature_idx_)
pd.DataFrame(X_gradeD_train.iloc[:, sbs_rf_25.steps[1][1].k_feature_idx_].columns).to_csv('../Results/second_run/three_year/2007-12 training/Feature Selection/gradeD/rf_SBS_feature_selection_k_25.csv', sep = ',', index = True)
rf_sbs_features = pd.DataFrame(X_gradeD_train.iloc[:,sbs_rf_25.steps[1][1].k_feature_idx_].columns.values)

#printing performance of results on base estimators. Not too meaningful because we need to optimize, but still interesting.
from mlxtend.feature_selection import plot_sequential_feature_selection as plot_sfs

fig1 = plot_sfs(sbs_lr.steps[1][1].get_metric_dict(), kind='std_dev')

plt.ylim([0.4, 1])
plt.title('Sequential Backward Selection with Logistic Regression (w. StdDev)')
plt.grid()
plt.savefig('../Results/second_run/three_year/2007-12 training/Feature Selection/gradeD/SBS_lr.png')

fig2 = plot_sfs(sbs_rf.steps[1][1].get_metric_dict(), kind='std_dev')

plt.ylim([0.4, 1])
plt.title('Sequential Backward Selection with Random Forest (w. StdDev)')
plt.grid()
plt.savefig('../Results/second_run/three_year/2007-12 training/Feature Selection/gradeD/SBS_rf.png')

fig3 = plot_sfs(sbs_lr_25.steps[1][1].get_metric_dict(), kind='std_dev')

plt.ylim([0.4, 1])
plt.title('Sequential Backward Selection with Logistic Regression (w. StdDev)')
plt.grid()
plt.savefig('../Results/second_run/three_year/2007-12 training/Feature Selection/gradeD/SBS_lr_25.png')

fig4 = plot_sfs(sbs_rf_25.steps[1][1].get_metric_dict(), kind='std_dev')

plt.ylim([0.4, 1])
plt.title('Sequential Backward Selection with Random Forest, k = 25 (w. StdDev)')
plt.grid()
plt.savefig('../Results/second_run/three_year/2007-12 training/Feature Selection/gradeD/SBS_rf_25.png')


'''Looking at benchmark ROC AUC for feature selection. Using Log Reg'''
features = [sbs_lr.steps[1][1].k_feature_idx_, sbs_lr_25.steps[1][1].k_feature_idx_, sbs_rf.steps[1][1].k_feature_idx_, sbs_rf_25.steps[1][1].k_feature_idx_]
roc_auc_bench = {}
for model, name in zip([pipe_lrl2_D, pipe_et_D, pipe_svc_D],['lrl2', 'et', 'svc']):
    _, roc_auc_bench[name+'rf_feat_imp'], _, _, _, _ = evaluate_model(model, X_gradeD_train[features_selected_gradeD_columns], y_gradeD_train, X_gradeD_test[features_selected_gradeD_columns], y_gradeD_test)

names = ['lr_sbs', 'lr_sbs_25', 'rf_sbs', 'rf_sbs_25']
for model, model_name in zip([pipe_lrl2_D, pipe_et_D, pipe_svc_D],['lrl2', 'et', 'svc']):
    for feature_col, name in zip(features, names):
        _, roc_auc_bench[model_name + name], _, _, _, _ = evaluate_model(model, X_gradeD_train.iloc[:,feature_col], y_gradeD_train, X_gradeD_test.iloc[:,feature_col], y_gradeD_test)

pd.DataFrame(roc_auc_bench, index = ['cross_val_roc_auc']).T.sort_values('cross_val_roc_auc', ascending = False).to_csv('../Results/second_run/three_year/2007-12 training/Feature Selection/gradeD/feature_selection_eval.csv', sep = ',', index = True)
pd.DataFrame(roc_auc_bench, index = ['cross_val_roc_auc']).T.sort_values('cross_val_roc_auc', ascending = False)

'''Looking at the benchmark ROC AUC comparing the features from RF feature importance to SBS from Log Reg 17 and 25,
the Log Reg 17 beats the Log Reg 25, barely..

Hence, for grade D, I would use the ones from the LR SBS with 17 features.'''


#Now that I have the best features for grade E, let's use zip to get the scores for them
#for grade E.
scores_defaults_D = {}
pipe_svc_D = make_pipeline(MinMaxScaler(), SVC(C = 1, kernel = 'rbf', max_iter = 30000, probability = True, class_weight = 'balanced', random_state = 31))
pipe_dt_D = make_pipeline(MinMaxScaler(), DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 4, min_samples_split = 8, min_samples_leaf = 4))
pipe_dtbag_D = make_pipeline(MinMaxScaler(), BaggingClassifier(DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 3, min_samples_split = 6, min_samples_leaf = 8), random_state = 31, n_estimators = 50))
pipe_rf_D = make_pipeline(MinMaxScaler(), RandomForestClassifier(random_state = 31, class_weight = 'balanced', n_estimators = 100, max_depth = 4, min_samples_split = 8))
pipe_et_D = make_pipeline(MinMaxScaler(), ExtraTreesClassifier(random_state = 31, class_weight = 'balanced', max_depth = 3, min_samples_split = 3, min_samples_leaf = 3))
pipe_lrl1_D = make_pipeline(MinMaxScaler(), LogisticRegression(C = 1, class_weight = 'balanced', penalty = 'l1'))
pipe_lrl2_D = make_pipeline(MinMaxScaler(), LogisticRegression(C = 1, class_weight = 'balanced', penalty = 'l2'))
pipe_adaboost_D = make_pipeline(MinMaxScaler(), AdaBoostClassifier(DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 5, min_samples_split = 6), random_state = 31))
pipe_adaboost_D_lr = make_pipeline(MinMaxScaler(), AdaBoostClassifier(LogisticRegression(C = 1, class_weight = 'balanced', penalty = 'l2', solver = 'lbfgs', max_iter = 1000), random_state = 31, learning_rate = .2, n_estimators = 200))
pipe_gradient_D_dev = make_pipeline(MinMaxScaler(), GradientBoostingClassifier(loss = 'deviance', learning_rate = .1, n_estimators = 200, max_depth = 4, min_samples_leaf = 6, verbose = True))
pipe_gradient_D_exp = make_pipeline(MinMaxScaler(), GradientBoostingClassifier(loss = 'exponential', learning_rate = .1, n_estimators = 200, max_depth = 4, min_samples_leaf = 6,  verbose = True))
pipe_voting_D = VotingClassifier(estimators = [('svc', pipe_svc_D), ('dtbag', pipe_dtbag_D), ('lrl2',pipe_lrl2_D), ('lrl1', pipe_lrl1_D), ('rf', pipe_rf_D), ('et', pipe_et_D), ('ada_lr', pipe_adaboost_D_lr)], voting = 'soft')

models = [pipe_svc_D, pipe_dt_D, pipe_dtbag_D, pipe_rf_D, pipe_et_D, pipe_lrl1_D, pipe_lrl2_D, pipe_adaboost_D, pipe_adaboost_D_lr, pipe_gradient_D_dev, pipe_gradient_D_exp, pipe_voting_D]
model_names = ['SVC', 'DT', 'Bagged DT', 'Random Forest', 'Extra Trees', 'Logistic Regression L1', 'Logistic Regression L2', 'AdaBoost', 'AdaBoost with LogReg', 'Gradient Deviance', 'Gradient Exp', 'Voting Untuned']
#Bagged DT with fewer branches does better. Prune the DT for GS? Log Reg and SVC do well.
for model, model_name in zip(models, model_names):
    print ""
    print model_name + " Output for Grade D Loans, 2007 - 12"
    print ""
    scores_defaults_D[model_name] = evaluate_model(model, X_gradeD_train[logreg_best_features_index], y_gradeD_train, X_gradeD_test[logreg_best_features_index], y_gradeD_test)

scores_defaults_D_df = pd.DataFrame(scores_defaults_D, index = ['mean_cv_recall', 'mean_cv_roc_auc', 'test_roc_auc', 'pred_default_rate', 'act_default_rate', 'observed_default_rate']).T.sort_values('test_roc_auc', ascending = False)
scores_defaults_D_df.to_csv('../Results/second_run/three_year/2007-12 training/Feature Selection/gradeD/scores_default_gradeD.csv', sep = ',', index = True)

'''For Grade D, let's try to optimize DT, then Bagged DT, then AdaBoost, Log Reg, SVC, ET, RF.
Then plot learning curves for each'''
#DECISION TREE
dt_params = {'decisiontreeclassifier__max_depth': [2, 3,4,5, 6, 8, 10, 12], 'decisiontreeclassifier__min_samples_leaf':[2,4,6, 8, 10, 12], \
'decisiontreeclassifier__min_samples_split':[2,5,10],\
'decisiontreeclassifier__criterion': ['gini', 'entropy'], 'decisiontreeclassifier__splitter':['random', 'best']}

gs_dt_gradeD = GridSearchCV(pipe_dt_D, dt_params, cv = StratifiedKFold(y_gradeD_train, 5, shuffle = True), verbose = True, scoring = 'roc_auc', n_jobs = -1)
gs_dt_gradeD.fit(X_gradeD_train[logreg_best_features_index], y_gradeD_train)
print "Best roc_auc score:", gs_dt_gradeD.best_score_
print "Best params:", gs_dt_gradeD.best_params_
np.mean(cross_val_score(gs_dt_gradeD.best_estimator_, X_gradeD_train[logreg_best_features_index], y_gradeD_train, cv = 5, scoring = 'roc_auc'))
scores_defaults_D['gs_dt'] = evaluate_model(gs_dt_gradeD.best_estimator_, X_gradeD_train[logreg_best_features_index], y_gradeD_train, X_gradeD_test[logreg_best_features_index], y_gradeD_test)
eval_grid
#Bagged DTBest dt params: Best params: {'decisiontreeclassifier__min_samples_leaf': 8, 'decisiontreeclassifier__min_samples_split': 2, 'decisiontreeclassifier__max_depth': 5, 'decisiontreeclassifier__criterion': 'gini', 'decisiontreeclassifier__splitter': 'random'}
pipe_dtbag_D_optdt = make_pipeline(MinMaxScaler(), BaggingClassifier(DecisionTreeClassifier(class_weight = 'balanced', criterion = 'gini', splitter = 'random', random_state = 31, max_depth = 5, min_samples_split = 2, min_samples_leaf = 8), random_state = 31))
bagging_params = {'baggingclassifier__n_estimators': [50, 200, 400, 600, 800, 1000, 1200], 'baggingclassifier__bootstrap_features': [True, False]}
gs_pipe_dtbag_D = GridSearchCV(pipe_dtbag_E_optdt, bagging_params, cv = StratifiedKFold(y_gradeE_train, 5, shuffle = True), verbose = True, scoring = 'roc_auc', n_jobs = -1)
scores_defaults_D['gs_bagdt'] = eval_grid(gs_pipe_dtbag_D, X_gradeD_train, y_gradeD_train, X_gradeD_test, y_gradeD_test, logreg_best_features_index)


#Gradient Exp
pipe_gradient_D_exp_optdt = make_pipeline(MinMaxScaler(), GradientBoostingClassifier(loss = 'exponential', learning_rate = .1, n_estimators = 200, max_depth = 5, min_samples_leaf = 8, min_samples_split = 2))
gradient_params = {'gradientboostingclassifier__loss': ['deviance', 'exponential'], 'gradientboostingclassifier__learning_rate': [.3, .5, 1], 'gradientboostingclassifier__n_estimators': [100, 300, 500, 700, 900, 1100]}
gs_gradient_D = GridSearchCV(pipe_gradient_D_exp_optdt, gradient_params, cv = StratifiedKFold(y_gradeD_train, 5, shuffle = True), verbose = True, scoring = 'roc_auc', n_jobs = -1)
scores_defaults_D['gs_Gradient'] = eval_grid(gs_gradient_D, X_gradeD_train, y_gradeD_train, X_gradeD_test, y_gradeD_test, logreg_best_features_index)

#Adaboost with opt dt
pipe_adaboost_D_optdt = make_pipeline(MinMaxScaler(), AdaBoostClassifier(DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 5, min_samples_split = 2, min_samples_leaf = 8), random_state = 31))
adaboost_params = {'adaboostclassifier__n_estimators': [100, 200, 300, 400, 500], 'adaboostclassifier__learning_rate': [.7, 1, 1.3, 1.5, 2]}
gs_adaboost_D = GridSearchCV(pipe_adaboost_D_optdt, adaboost_params, cv = StratifiedKFold(y_gradeD_train, 5, shuffle = True), verbose = True, scoring = 'roc_auc', n_jobs = -1)
scores_defaults_D['gs_AdaBoost'] = eval_grid(gs_adaboost_D, X_gradeD_train, y_gradeD_train, X_gradeD_test, y_gradeD_test, logreg_best_features_index)

#Logistic Regression L2 C = 1 seems best... Note., Logistic Regression via GridSearchCV does not work on my computer.
scores = {}
for c in [0.001, 0.01, 0.1, 100, 1000, 10000]:
    pipe_lrl2_D = make_pipeline(MinMaxScaler(), LogisticRegression(C = c, class_weight = 'balanced', penalty = 'l2'))
    score = np.mean(cross_val_score(pipe_lrl2_D, X_gradeD_train[logreg_best_features_index], y_gradeD_train, cv = StratifiedKFolds(y_gradeD_train, 5, shuffle = True), scoring = 'roc_auc', random_state = 31))
    scores[c] = score
pipe_lrl2_D = make_pipeline(MinMaxScaler(), LogisticRegression(C = 1, class_weight = 'balanced', penalty = 'l2'))

#Logistic Regression L1 C = 1000 seems best...
scores = {}
for c in [0.001, 0.01, 0.1,1, 100, 1000, 10000]:
    pipe_lrl1_E = make_pipeline(MinMaxScaler(), LogisticRegression(C = c, class_weight = 'balanced', penalty = 'l1'))
    score = np.mean(cross_val_score(pipe_lrl1_E, X_gradeE_train[logreg_best_features_index], y_gradeE_train, cv = 5, scoring = 'roc_auc'))
    scores[c] = score
pipe_lrl1_E = make_pipeline(MinMaxScaler(), LogisticRegression(C = 1000, class_weight = 'balanced', penalty = 'l1'))
scores_defaults_E['Opt Logistic Regression L1'] = evaluate_model(pipe_lrl1_E, X_gradeE_train[logreg_best_features_index], y_gradeE_train, X_gradeE_test[logreg_best_features_index], y_gradeE_test)
scores_defaults_E['AdaBoost with Log Reg L2'] = evaluate_model(pipe_adaboost_E_lr, X_gradeE_train[logreg_best_features_index], y_gradeE_train, X_gradeE_test[logreg_best_features_index], y_gradeE_test)

#Random Forest
pipe_rf_E_opt = make_pipeline(MinMaxScaler(), RandomForestClassifier(random_state = 31, class_weight = 'balanced', n_estimators = 100, max_depth = 4, min_samples_split = 10, min_samples_leaf =2))
random_params = {'randomforestclassifier__criterion': ['gini', 'entropy'], 'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2', None], 'randomforestclassifier__n_estimators': [100, 200, 300]}
gs_rf_E = GridSearchCV(pipe_rf_E_opt, random_params, cv = StratifiedKFold(y_gradeE_train, 5, shuffle = True), verbose = True, scoring = 'roc_auc', n_jobs = -1)
gs_rf_E.fit(X_gradeE_train[logreg_best_features_index], y_gradeE_train)
print "Best roc_auc score:", gs_rf_E.best_score_
print "Best params:", gs_rf_E.best_params_
np.mean(cross_val_score(gs_rf_E.best_estimator_, X_gradeE_train[logreg_best_features_index], y_gradeE_train, cv = 5, scoring = 'roc_auc'))
scores_defaults_E['gs_Random Forest'] = evaluate_model(gs_rf_E.best_estimator_, X_gradeE_train[logreg_best_features_index], y_gradeE_train, X_gradeE_test[logreg_best_features_index], y_gradeE_test)

#SVC
svc_params = {'svc__C': [.001, .01,.1, 1, 10, 100]}
gs_svc_E = GridSearchCV(pipe_svc_E, svc_params, cv = StratifiedKFold(y_gradeE_train, 5, shuffle = True), verbose = True, n_jobs = -1, scoring = 'roc_auc')
gs_svc_E.fit(X_gradeE_train.iloc[:,logreg_best_features_index], y_gradeE_train)
print "Best roc_auc score:", gs_svc_E.best_score_
print "Best params:", gs_svc_E.best_params_

#ET
pipe_et_E_opt = make_pipeline(MinMaxScaler(), ExtraTreesClassifier(random_state = 31, class_weight = 'balanced', n_estimators = 100, max_depth = 4, min_samples_split = 10, min_samples_leaf =2))
et_params = {'extratreesclassifier__criterion': ['gini', 'entropy'], 'extratreesclassifier__max_features': ['auto', 'sqrt', 'log2', None], 'extratreesclassifier__n_estimators': [200, 300, 400, 500]}
gs_et_E = GridSearchCV(pipe_et_E_opt, et_params, cv = StratifiedKFold(y_gradeE_train, 5, shuffle = True), verbose = True, scoring = 'roc_auc', n_jobs = -1)
gs_et_E.fit(X_gradeE_train[logreg_best_features_index], y_gradeE_train)
print "Best roc_auc score:", gs_et_E.best_score_
print "Best params:", gs_et_E.best_params_
np.mean(cross_val_score(gs_et_E.best_estimator_, X_gradeE_train[logreg_best_features_index], y_gradeE_train, cv = 5, scoring = 'roc_auc'))
scores_defaults_E['gs_Extra Trees'] = evaluate_model(gs_et_E.best_estimator_, X_gradeE_train[logreg_best_features_index], y_gradeE_train, X_gradeE_test[logreg_best_features_index], y_gradeE_test)

'''Learning Curves for Models Grade E'''
pipe_voting_E_opt_models = make_pipeline(MinMaxScaler(), VotingClassifier(estimators = [('lrl2', pipe_lrl2_E), ('dtbag', gs_pipe_dtbag_E.best_estimator_), ('et', gs_et_E.best_estimator_), ('rf',gs_rf_E.best_estimator_), ('svc', pipe_svc_E), ('lrl1',pipe_lrl1_E)], voting = 'soft'))
evaluate_model(pipe_voting_E_opt_models, X_gradeE_train[logreg_best_features_index], y_gradeE_train, X_gradeE_test[logreg_best_features_index], y_gradeE_test)
#While lots of bias and variance, in general, the learning curves for these look better than for PCA for Grade E.
models_opt = [pipe_lrl2_E, pipe_lrl1_E, gs_pipe_dtbag_E.best_estimator_, gs_et_E.best_estimator_, gs_rf_E.best_estimator_, pipe_svc_E, gs_gradient_E.best_estimator_, gs_adaboost_E.best_estimator_, pipe_dt_E, pipe_voting_E_opt_models, pipe_adaboost_E_lr]
models_names = ['Logistic Regression L2', 'Logistic Regression L1', 'Bagged DT', 'Extra Trees', 'Random Forest', 'SVM', 'Gradient Boosting', 'AdaBoost', 'Decision Tree', 'Ensemble Voting', 'AdaBoost LR']

for model_opt, model_name in zip(models_opt, models_names):
    plot_learning_curves(X_gradeE_train.iloc[:,logreg_best_features_index], y_gradeE_train, X_gradeE_test.iloc[:,logreg_best_features_index], y_gradeE_test, clf = model_opt, print_model = False, scoring = 'roc_auc')
    plt.title('Learning Curve for ' + model_name + ' on Grade E Loans, 2007-12')
    plt.savefig(''.join(['../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/gradeE_learning_curve ',model_name,'.png']), bbox_inches = 'tight')
    plt.show()

'''ROC Curves for Grade E'''
models =[pipe_svc_E, pipe_lrl2_E, gs_pipe_dtbag_E.best_estimator_, gs_et_E.best_estimator_, pipe_voting_E_opt_models, pipe_adaboost_E_lr]
model_names = ['SVC', 'Log Reg L2', 'Bagged DT', 'Extra Trees', 'Ensemble Voting, Soft', 'AdaBoost - LR L2']

plot_roc_curve_models(X_gradeE_train.iloc[:,logreg_best_features_index], y_gradeE_train, X_gradeE_test.iloc[:,logreg_best_features_index], y_gradeE_test ,models, model_names, 'Grade E', 'gradeE')

'''All of the models perform better on AUC than the PCA only ones on the continuous data.
Retrain the following models on all of data for 2012, validate on 2013, knowing the whole ground truth is not known.
-SVC, C = 1
-LR L2, C = 1
-Bagged DT
-Extra Trees
-Ensemble Voting

'''
#Getting ROC AUC Scores in chart
plt.figure(figsize = (16, 8))
scores_defaults_E_df[['mean_cv_roc_auc', 'test_roc_auc']].sort_values('mean_cv_roc_auc', ascending = False).plot(kind = 'bar', fontsize = 12)
plt.title('ROC AUC Scores for Models on Grade E loans, 2007-12', fontsize = 14)
plt.ylim([.4, .8])
plt.legend(loc = 'upper right')
plt.savefig('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/roc_auc_scores_train_test.png', bbox_inches = 'tight')

#Default rates in chart
plt.figure(figsize = (16, 8))
scores_defaults_E_df[['observed_default_rate', 'pred_default_rate']].sort_values('observed_default_rate', ascending = True).plot(kind = 'bar', fontsize = 12)
plt.title('Default Rates for Models for Grade E Loans on Test Data', fontsize = 12)
plt.axhline(y = scores_defaults_E_df['act_default_rate'][0], linewidth = 4, color = 'black')
plt.legend(loc = 'lower left', fontsize = 'small')
plt.savefig('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/defaults_test.png', bbox_inches = 'tight')

#gross returns for chart
models =[pipe_svc_E, pipe_lrl2_E, pipe_lrl1_E, pipe_adaboost_E_lr, gs_rf_E.best_estimator_, gs_pipe_dtbag_E.best_estimator_, gs_et_E.best_estimator_, pipe_voting_E_opt_models]
model_col = ['svc_pred', 'logreg_l2_pred','logreg_l1_pred', 'adaboost_lr_pred', 'rf_pred','dtbag_pred', 'et_pred', 'ensemble_pred']

y_pred = {}
for model, col in zip(models, model_col):
    y_pred[col] = model.predict(X_gradeE_test[logreg_best_features_index])

predicted_gradeE = pd.DataFrame(y_pred, index = X_gradeE_test.index)
predicted_gradeE_actual = pd.merge(predicted_gradeE, pd.DataFrame(y_gradeE_test, index = y_gradeE_test.index), how = 'inner', right_index = True, left_index = True)
predicted_act_returns_gradeE = pd.merge(predicted_gradeE_actual, true_outcomes[['sub_grade', 'gross_return_scaled']], how = 'inner', right_index = True, left_index = True)

returns_mean = {}
for pred in predicted_gradeE.columns.values:
       returns_mean[pred] = predicted_act_returns_gradeE[predicted_act_returns_gradeE[pred]==0]['gross_return_scaled'].mean()
returns_mean['whole_gradeE'] = predicted_act_returns_gradeE.gross_return_scaled.mean()
returns_gradeE_test = pd.DataFrame(returns_mean, index = ['mean_return']).T
plt.figure()
returns_gradeE_test.sort_values('mean_return', ascending = False).plot(kind = 'bar', color = 'green')
plt.axhline(y = returns_gradeE_test.loc['whole_gradeE'][0], color = 'black', linewidth = 4)
plt.title('Average Returns for Grade E loans, Predicted Non-Default vs. Test')
plt.savefig('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/returns_model_non_default_predicted.png', bbox_inches = 'tight')
returns_gradeE_test.to_csv('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/gross returns by model.csv', sep = ',', index = True, index_label = 'model')
print "Mean return for top three models vs baseline", returns_gradeE_test.ix[0:3].mean()/returns_gradeE_test.ix[-1]-1
print "Mean return for top model vs baseline", returns_gradeE_test.ix[0][0]/returns_gradeE_test.ix[-1][0]-1
#refitting models for production
models =[pipe_svc_E, pipe_lrl2_E, pipe_lrl1_E, pipe_adaboost_E_lr, gs_rf_E.best_estimator_, gs_pipe_dtbag_E.best_estimator_, gs_et_E.best_estimator_, pipe_voting_E_opt_models]
model_fit_names = ['svc_gradeE', 'logreg_l2_gradeE', 'logreg_l1_gradeE', 'adaboost_lr_gradeE','rf_gradeE', 'dtbag_gradeE', 'et_gradeE', 'ensemble_gradeE']
for model, fit_name in zip(models, model_fit_names):
    model_fitted = model.fit(X_gradeE[logreg_best_features_index], y_gradeE)
    pickle_file = open(''.join(['../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/pickled models/', fit_name, '.pk1']), 'wb')
    pickle.dump(model_fitted, pickle_file, -1)
    pickle_file.close()

gradeE_returns_counts.columns.values
#Obtaining returns by sub-grade for Q&A
mean_model_sub_grade = {}
counts_sub_grade = {}
for pred in ['dtbag_pred', 'ensemble_pred', 'et_pred', 'adaboost_lr_pred', 'rf_pred',
       'logreg_l1_pred','logreg_l2_pred', 'svc_pred']:
       mean_model_sub_grade[pred] =predicted_act_returns_gradeE[predicted_act_returns_gradeE[pred]==0].groupby('sub_grade').gross_return_scaled.mean()
       counts_sub_grade[pred] = predicted_act_returns_gradeE[predicted_act_returns_gradeE[pred]==0].groupby('sub_grade').gross_return_scaled.count()
gradeE_returns_subgrade = pd.merge(pd.DataFrame(mean_model_sub_grade), pd.DataFrame(predicted_act_returns_gradeE.groupby('sub_grade').gross_return_scaled.mean()), how = 'right', right_index = True, left_index = True)
gradeE_returns_counts = pd.merge(gradeE_returns_subgrade, pd.DataFrame(counts_sub_grade), how = 'left', right_index = True, left_index = True)
gradeE_returns_counts.rename(columns = {'adaboost_lr_pred_x': 'adaboost_lr_pred', 'dtbag_pred_x': 'dtbag_pred', 'ensemble_pred_x': 'ensemble_pred', 'et_pred_x': 'et_pred',
       'logreg_l1_pred_x':'logreg_l1_pred', 'logreg_l2_pred_x':'logreg_l2_pred','rf_pred_x': 'rf_pred', 'svc_pred_x': 'svc_pred', 'gross_return_scaled':'returns',
       'adaboost_lr_pred_y': 'adaboost_lr_count', 'dtbag_pred_y': 'dtbag_count', 'ensemble_pred_y': 'ensemble_count', 'et_pred_y': 'et count',
       'logreg_l1_pred_y':'logreg_l1_count', 'logreg_l2_pred_y':'logreg_l2_count','rf_pred_y': 'rf_count','svc_pred_y':'svc_count'}, inplace = True)
total_loans_gradeE = pd.DataFrame(predicted_act_returns_gradeE.groupby('sub_grade').gross_return_scaled.count())
total_loans_gradeE.rename(columns = {'gross_return_scaled': 'count_total_loans'}, inplace = True)
gradeF_returns_counts = pd.merge(gradeE_returns_counts, total_loans_gradeE, how = 'right', right_index = True, left_index = True)
gradeF_returns_counts.rename(columns = {'returns':'all_loans_returns'}, inplace = True)

gradeE_returns_counts.to_csv('../../Results/second_run/three_year/2007-12 training/Feature Selection/gradeE/returns_by_sub_grade_including_counts.csv', sep = ',', index = True, index_label = 'sub_grade')
