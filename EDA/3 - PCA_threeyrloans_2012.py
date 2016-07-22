'''PCA on the continuous data, mainly credit and income, to see how explanatory
the models could be just with that. Short answer is not good enough, but still
interesting results.'''

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
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import roc_curve, auc
from scipy import interp
from mlxtend.evaluate import plot_learning_curves

import pickle
from textblob import TextBlob
%matplotlib inline

loans_3yr_until_2013 = pd.read_csv('../../Data/second_run/loanbook_3yr_until_2013.csv')

loans_3yr_until_2013.drop('row_num.1', axis = 1, inplace = True)

loans_3yr_until_2013.columns.values

loans_3yr_until_2013.set_index('id', inplace = True)

'''Columns to create df for later eval'''
id_cols = ['year', 'grade', 'gross_return_scaled', 'loan_status', 'loan_status_encoded','sub_grade']
true_outcomes = loans_3yr_until_2013[id_cols]


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

for index1, row in baseline.iterrows():
    print 'Baseline default for %s is %.3f' % (index1, row.default_rate)


notscale_cols = ['row_num', 'funded_amnt','grade', 'sub_grade', 'date', 'title', 'zip_code', 'term', \
'emp_title', 'earliest_cr_line', 'loan_status_encoded',\
'annual_inc_joint', 'dti_joint', 'verification_status_joint_Verified', \
'verification_status_joint_Source Verified', \
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



'''For Grade A, what is the # of obs per sub-grade?
Enough probably for PCA on individual subgrades'''
for sub_grade in [3, 0, 2, 4, 1]:
    print "Grade A, Sub_grade" + str(sub_grade) + " # of obs:", X_gradeA[X_gradeA['sub_grade_encoded']==sub_grade].shape[0]
for sub_grade in [6, 9, 5, 7, 8]:
    print "Grade B, Sub_grade" + str(sub_grade) + " # of obs:", X_gradeB[X_gradeB['sub_grade_encoded']==sub_grade].shape[0]
for sub_grade in [14, 10, 13, 11, 12]:
    print "Grade C, Sub_grade" + str(sub_grade) + " # of obs:", X_gradeC[X_gradeC['sub_grade_encoded']==sub_grade].shape[0]
for sub_grade in [15, 19, 16, 17, 18]:
    print "Grade D, Sub_grade" + str(sub_grade) + " # of obs:", X_gradeD[X_gradeD['sub_grade_encoded']==sub_grade].shape[0]
for sub_grade in [20, 21, 22, 24, 23]:
    print "Grade E, Sub_grade" + str(sub_grade) + " # of obs:", X_gradeE[X_gradeE['sub_grade_encoded']==sub_grade].shape[0]
for sub_grade in [25, 29, 28, 26, 27, 30, 31, 32, 33, 34]:
    print "Grade F, Sub_grade" + str(sub_grade) + " # of obs:", X_gradeF[X_gradeF['sub_grade_encoded']==sub_grade].shape[0]

'''Grades E and F might not have enough obs. As for subgrades, I could consider doing analysis by subgrade through
grade D.'''

credit_cols = ['annual_inc','dti',
       'delinq_2yrs', 'inq_last_6mths',  'open_acc', 'pub_rec', 'revol_bal',
       'revol_util', 'total_acc', 'Risk_Score',
       'diff_earliest_app_date', 'work_prop',
       'magnitude_ann_inc_to_median']
#'annual_inc','mths_since_last_delinq',
#'mths_since_last_record','mths_since_last_major_derog',
#'sub_grade_encoded'



'''PCA on Grade A, credit data
'''
gradeA_pca = PCA(n_components = None)
scaler_gradeA = RobustScaler()
gradeA_pca.fit_transform(scaler_gradeA.fit_transform(X_gradeA[credit_cols]))


cum_var = np.cumsum(gradeA_pca.explained_variance_ratio_)
plt.figure(figsize=(9,7))

component_number = range(1, 14)

plt.plot(component_number, cum_var, lw=7)

plt.axhline(y=0, linewidth=5, color='grey', ls='dashed')
plt.axhline(y=1, linewidth=3, color='grey', ls='dashed')
plt.axhline(y = .90, linewidth = 3, color = 'green', ls = 'dashed')
plt.axvline(x = 8, linewidth = 3, color = 'black', ls = 'solid')
plt.style.use('fivethirtyeight')
ax = plt.gca()
ax.set_xlim([1,13])
ax.set_ylim([-.05,1.05])

ax.set_ylabel('cumulative variance explained', fontsize=12)
ax.set_xlabel('component', fontsize=12)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(10)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(10)

ax.set_title('Principal Component vs. Cumulative Explained Variance\n', fontsize=16)
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeA_PCA_explained_variance.png', bbox_inches = 'tight')

'''Let's also see what composes the first two PCs.'''
pca_2 = PCA(n_components = 8)
PCs_whole = pca_2.fit_transform(scaler_gradeA.fit_transform(X_gradeA[credit_cols]))
prin_comps = pd.DataFrame(PCs_whole, columns = ['PC' + str(i) for i in range(1,9)], index = X_gradeA[credit_cols].index)
loan_status_pcs_year_A = pd.merge(true_outcomes[['loan_status_encoded', 'year']], prin_comps, left_index = True, right_index = True)

'''Plotting the PCs.'''
colors = ['b', 'r']
markers = ['s', 'o']
for l, c, m in zip(np.unique(y_gradeA), colors, markers):
    plt.scatter(loan_status_pcs_year_A[y_gradeA==l]['PC1'], loan_status_pcs_year_A[y_gradeA==l]['PC2'], c=c, label = l, marker = m)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('PCA on Grade A Loans, PC1 and PC2')
    plt.legend(loc = 'lower left')
    #plt.savefig('../../EDA/PCA/nomonthvar/regularPCA_gradeA_PC7and8.png')

prin_comps_features = pd.merge(X_gradeA[credit_cols], prin_comps, left_index = True, right_index = True)

corr_prin_comps = prin_comps_features.corr().drop(['PC1', 'PC2', 'PC8', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'], axis = 0)
corr_prin_comps.drop(credit_cols, axis = 1, inplace = True)
corr_prin_comps.to_csv('../../EDA/PCA/nomonthvar/prin_comps_gradeA_3yrloans.csv', sep = ',', index = True)

vars_with_high_power = [annual_inc, dti, diff_earliest_app_date, total_acc, open_acc, inq_last_6mths, revol_bal, revol_util, Risk_Score, magnitude_ann_inc_to_median, work_prop, median_est_income_hh]

'''While with PCA we do not see linear separability, we see separability in the clusters.
Hence, we should look at Kernel PCA.'''

'''PCA on Grade B, credit data
'''
gradeB_pca = PCA(n_components = 8)
scaler_gradeB = MinMaxScaler()
PCs_whole = gradeB_pca.fit_transform(scaler_gradeB.fit_transform(X_gradeB[credit_cols]))
prin_comps = pd.DataFrame(PCs_whole, columns = ['PC' + str(i) for i in range(1,9)], index = X_gradeB[credit_cols].index)
loan_status_pcs_year_B = pd.merge(true_outcomes[['loan_status_encoded', 'year']], prin_comps, left_index = True, right_index = True)

#Looking at the PCs to see what sort of separability there is.
colors = ['g', 'r']
markers = ['s', 'o']
for l, c, m in zip(np.unique(y_gradeB), colors, markers):
    plt.scatter(loan_status_pcs_year_B[y_gradeB==l]['PC7'], loan_status_pcs_year_B[y_gradeB==l]['PC8'], c=c, label = l, marker = m)
    plt.xlabel('PC 7')
    plt.ylabel('PC 8')
    plt.legend(loc = 'lower left')
    plt.savefig('../../EDA/PCA/nomonthvar/gradeB/regularPCA_gradeB_PC7and8.png')
plt.show()

'''Similar to Grade A.'''


'''PCA on Grade C, credit data
'''
gradeC_pca = PCA(n_components = 8)
scaler_gradeC = MinMaxScaler()
PCs_whole = gradeC_pca.fit_transform(scaler_gradeC.fit_transform(X_gradeC[credit_cols]))
prin_comps = pd.DataFrame(PCs_whole, columns = ['PC' + str(i) for i in range(1,9)], index = X_gradeC[credit_cols].index)
loan_status_pcs_year_C = pd.merge(true_outcomes[['loan_status_encoded', 'year']], prin_comps, left_index = True, right_index = True)

#Looking at the PCs to see what sort of separability there is.
colors = ['g', 'r']
markers = ['s', 'o']
for l, c, m in zip(np.unique(y_gradeC), colors, markers):
    plt.scatter(loan_status_pcs_year_C[y_gradeC==l]['PC7'], loan_status_pcs_year_C[y_gradeC==l]['PC8'], c=c, label = l, marker = m)
    plt.xlabel('PC 7')
    plt.ylabel('PC 8')
    plt.legend(loc = 'lower left')
    plt.savefig('../../EDA/PCA/nomonthvar/gradeC/regularPCA_gradeC_PC7and8.png')
plt.show()

'''Not as similar to Grades A and B, but still shows non-linear separability.'''

'''PCA on Grade D, credit data
'''
gradeD_pca = PCA(n_components = 8)
scaler_gradeD = MinMaxScaler()
PCs_whole = gradeD_pca.fit_transform(scaler_gradeD.fit_transform(X_gradeD[credit_cols]))
prin_comps = pd.DataFrame(PCs_whole, columns = ['PC' + str(i) for i in range(1,9)], index = X_gradeD[credit_cols].index)
loan_status_pcs_year_D = pd.merge(true_outcomes[['loan_status_encoded', 'year']], prin_comps, left_index = True, right_index = True)

#Looking at the PCs to see what sort of separability there is.
colors = ['g', 'r']
markers = ['s', 'o']
for l, c, m in zip(np.unique(y_gradeD), colors, markers):
    plt.scatter(loan_status_pcs_year_D[y_gradeD==l]['PC7'], loan_status_pcs_year_D[y_gradeD==l]['PC8'], c=c, label = l, marker = m)
    plt.xlabel('PC 7')
    plt.ylabel('PC 8')
    plt.legend(loc = 'lower left')
    plt.savefig('../../EDA/PCA/nomonthvar/gradeD/regularPCA_gradeD_PC7and8.png')
plt.show()
'''Getting prin comp vs original features for Grade D'''
prin_comps_features = pd.merge(X_gradeD[credit_cols], prin_comps, left_index = True, right_index = True)

corr_prin_comps = prin_comps_features.corr().drop(['PC1', 'PC2', 'PC8', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'], axis = 0)
corr_prin_comps.drop(credit_cols, axis = 1, inplace = True)
corr_prin_comps.to_csv('../../EDA/PCA/nomonthvar/gradeD/prin_comps_gradeD_3yrloans.csv', sep = ',', index = True)
'''Can see from this how the most important features are different for Grade D vs. Grade A.
Income was not in the first most important PC for grade A, but is for Grade D'''

'''Also not linearly separable.'''

'''PCA on Grade E, credit data
'''
gradeE_pca = PCA(n_components = 8)
scaler_gradeE = MinMaxScaler()
PCs_whole = gradeE_pca.fit_transform(scaler_gradeE.fit_transform(X_gradeE[credit_cols]))
prin_comps = pd.DataFrame(PCs_whole, columns = ['PC' + str(i) for i in range(1,9)], index = X_gradeE[credit_cols].index)
loan_status_pcs_year_E = pd.merge(true_outcomes[['loan_status_encoded', 'year']], prin_comps, left_index = True, right_index = True)

#Looking at the PCs to see what sort of separability there is.
colors = ['g', 'r']
markers = ['s', 'o']
for l, c, m in zip(np.unique(y_gradeE), colors, markers):
    plt.scatter(loan_status_pcs_year_E[y_gradeE==l]['PC7'], loan_status_pcs_year_E[y_gradeE==l]['PC8'], c=c, label = l, marker = m)
    plt.xlabel('PC 7')
    plt.ylabel('PC 8')
    plt.legend(loc = 'lower left')
    plt.savefig('../../EDA/PCA/nomonthvar/gradeE/regularPCA_gradeE_PC7and8.png')
plt.show()

prin_comps_features = pd.merge(X_gradeE[credit_cols], prin_comps, left_index = True, right_index = True)

corr_prin_comps = prin_comps_features.corr().drop(['PC1', 'PC2', 'PC8', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'], axis = 0)
corr_prin_comps.drop(credit_cols, axis = 1, inplace = True)
corr_prin_comps.to_csv('../../EDA/PCA/nomonthvar/gradeE/prin_comps_gradeE_3yrloans.csv', sep = ',', index = True)

''' not linearly seperably.'''


'''PCA on Grade F, credit data
'''
gradeF_pca = PCA(n_components = 8)
scaler_gradeF = MinMaxScaler()
PCs_whole = gradeF_pca.fit_transform(scaler_gradeF.fit_transform(X_gradeF[credit_cols]))
prin_comps = pd.DataFrame(PCs_whole, columns = ['PC' + str(i) for i in range(1,9)], index = X_gradeF[credit_cols].index)
loan_status_pcs_year_F = pd.merge(true_outcomes[['loan_status_encoded', 'year']], prin_comps, left_index = True, right_index = True)

#Looking at the PCs to see what sort of separability there is.
colors = ['g', 'r']
markers = ['s', 'o']
for l, c, m in zip(np.unique(y_gradeF), colors, markers):
    plt.scatter(loan_status_pcs_year_F[y_gradeF==l]['PC5'], loan_status_pcs_year_F[y_gradeF==l]['PC6'], c=c, label = l, marker = m)
    plt.xlabel('PC 5')
    plt.ylabel('PC 6')
    plt.legend(loc = 'lower left')
    plt.savefig('../../EDA/PCA/nomonthvar/gradeF/regularPCA_gradeF_PC5and6.png')
plt.show()

prin_comps_features = pd.merge(X_gradeF[credit_cols], prin_comps, left_index = True, right_index = True)

corr_prin_comps = prin_comps_features.corr().drop(['PC1', 'PC2', 'PC8', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'], axis = 0)
corr_prin_comps.drop(credit_cols, axis = 1, inplace = True)
corr_prin_comps.to_csv('../../EDA/PCA/nomonthvar/gradeE/prin_comps_gradeE_3yrloans.csv', sep = ',', index = True)


''' Does not look similar to any other grade, nor is it linearly seperably. Wonder if any
kernel trick would make it separable. It looks like it may not be easily separable at all.
Kernel PCA however does not seem to be working well on this machine. The data
might not be separable via a kernel.

However, since PCA seems to make the data into something distinct, but not by a linear function,
let's look, for each grade, how the following separate it just on the PCs with regular PCA.
Kernel PCA takes too long.

Classifiers to test: SVC, Decision Trees, Bagged Decision Tree, RandomForest, AdaBoost. Doubt Logistic, but will test it.'''

'''Helper Functions'''
'''Cross Validation Scores for each'''
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

def eval_grid(grid_model, train_data, train_target, test_data, test_target):
    grid_model.fit(train_data, train_target)
    print "The best score is:", grid_model.best_score_
    print "Optimal parameters are:", grid_model.best_params_
    evaluate_model(grid_model.best_estimator_, train_data, train_target, test_data, test_target)
    return grid_model.best_params_

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
        probas_ = model.fit(X[credit_cols].iloc[train], y.iloc[train]).predict_proba(X[credit_cols].iloc[test])
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
    plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/ROC_' + model_name + ' ' + grade + '.png', bbox_inches = 'tight')

def plot_roc_curve_credit_cols_models(X, y, X_test, y_test,model, model_name, grade):
    colors = ['black', 'orange', 'blue', 'red', 'green', 'purple']
    linestyles = [':', '-', '--', '-.', '-', ':']
    for clf, label, clr, ls in zip(model, model_name, colors, linestyles):
        y_pred_proba = clf.fit(X[credit_cols], y).predict_proba(X_test[credit_cols])[:,1]
        fpr, tpr, thresholds = roc_curve(y_true = y_test, y_score = y_pred_proba)
        roc_auc = auc(x = fpr, y= tpr)
        plt.plot(fpr, tpr, color = clr, linestyle = ls, label = '%s (auc = %.2f)' % (label, roc_auc))

    plt.legend(loc = 'lower right', fontsize = 'small')
    plt.plot([0,1], [0,1], linestyle = '--', color = 'gray', linewidth = 2, label = 'Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Models for ' + grade + ' loans, 2007-12')
    plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/ROC_curves' + grade + '.png', bbox_inches = 'tight')


scores_defaults = {}

'''Train Test Split the 2012 Data'''

X_gradeA_train, X_gradeA_test, y_gradeA_train, y_gradeA_test = train_test_split(X_gradeA, y_gradeA, test_size = .2, random_state = 31, stratify = y_gradeA)
X_gradeB_train, X_gradeB_test, y_gradeB_train, y_gradeB_test = train_test_split(X_gradeB, y_gradeB, test_size = .2, random_state = 31, stratify = y_gradeB)
X_gradeC_train, X_gradeC_test, y_gradeC_train, y_gradeC_test = train_test_split(X_gradeC, y_gradeC, test_size = .2, random_state = 31, stratify = y_gradeC)
X_gradeD_train, X_gradeD_test, y_gradeD_train, y_gradeD_test = train_test_split(X_gradeD, y_gradeD, test_size = .2, random_state = 31, stratify = y_gradeD)
X_gradeE_train, X_gradeE_test, y_gradeE_train, y_gradeE_test = train_test_split(X_gradeE, y_gradeE, test_size = .2, random_state = 31, stratify = y_gradeE)
X_gradeF_train, X_gradeF_test, y_gradeF_train, y_gradeF_test = train_test_split(X_gradeF, y_gradeF, test_size = .2, random_state = 31, stratify = y_gradeF)

'''2013 Data'''
X_gradeA_2013 = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'A') & (loans_3yr_until_2013.year==2013)][scalecols]
X_gradeB_2013 = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'B') & (loans_3yr_until_2013.year==2013)][scalecols]
X_gradeC_2013 = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'C') & (loans_3yr_until_2013.year==2013)][scalecols]
X_gradeD_2013 = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'D') & (loans_3yr_until_2013.year==2013)][scalecols]
X_gradeE_2013 = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'E') & (loans_3yr_until_2013.year==2013)][scalecols]
X_gradeF_2013 = loans_3yr_until_2013[(loans_3yr_until_2013.grade.isin(['F', 'G'])) & (loans_3yr_until_2013.year==2013)][scalecols]

y_gradeA_2013 = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'A') & (loans_3yr_until_2013.year==2013)]['loan_status_encoded']
y_gradeB_2013 = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'B') & (loans_3yr_until_2013.year==2013)]['loan_status_encoded']
y_gradeC_2013 = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'C') & (loans_3yr_until_2013.year==2013)]['loan_status_encoded']
y_gradeD_2013 = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'D') & (loans_3yr_until_2013.year==2013)]['loan_status_encoded']
y_gradeE_2013 = loans_3yr_until_2013[(loans_3yr_until_2013.grade == 'E') & (loans_3yr_until_2013.year==2013)]['loan_status_encoded']
y_gradeF_2013 = loans_3yr_until_2013[(loans_3yr_until_2013.grade.isin(['F', 'G'])) & (loans_3yr_until_2013.year==2013)]['loan_status_encoded']
y_gradeA_2013 = y_gradeA_2013.replace(to_replace={0: 0, 1: 0, 2: 1, 3: 1})
y_gradeB_2013 = y_gradeB_2013.replace(to_replace={0: 0, 1: 0, 2: 1, 3: 1})
y_gradeC_2013 = y_gradeC_2013.replace(to_replace={0: 0, 1: 0, 2: 1, 3: 1})
y_gradeD_2013 = y_gradeD_2013.replace(to_replace={0: 0, 1: 0, 2: 1, 3: 1})
y_gradeE_2013 = y_gradeE_2013.replace(to_replace={0: 0, 1: 0, 2: 1, 3: 1})
y_gradeF_2013 = y_gradeF_2013.replace(to_replace={0: 0, 1: 0, 2: 1, 3: 1})

'''Grade A
Procedure. See how PCs alone do with our classifiers.
In the next script, we'll use feature selection (no PCA) and compare to ROC AUC with these.

Try to find optimized parameters for the dt for bagging, and try for RF and AdaBoost too. Potential to beat the SVM.
I.e. if max_depth of 10 improves estimator for Bagged DT with 50 estimators, use for RF too, as RF does this too.
Consider also a BaggingClassifier on top of KNN?

Also, what would an optimal C for SVC be?

Finally, take AdaBoost out of voting classifier. Does not seem to be doing much good.
Besides, works, better for shallower trees. Look at RF performance vs DT Bagged. might be that one or both as they are close are
good for the voting classifier.
'''

pipe_svc = make_pipeline(MinMaxScaler(), PCA(n_components = 8), SVC(C= 10, kernel = 'rbf', max_iter = 30000, probability = True, class_weight = 'balanced', random_state = 31))
pipe_dt = make_pipeline(RobustScaler(), PCA(n_components = 8), DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 5, min_samples_split = 10, min_samples_leaf = 8))
pipe_dtbag = make_pipeline(RobustScaler(), PCA(n_components = 8), BaggingClassifier(DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 5, min_samples_split = 10, min_samples_leaf = 2), random_state = 31, n_estimators = 100))
pipe_rf = make_pipeline(RobustScaler(), PCA(n_components = 8), RandomForestClassifier(random_state = 31, class_weight = 'balanced', max_depth = 5, min_samples_split = 10, min_samples_leaf = 2, n_estimators = 100))
pipe_et = make_pipeline(RobustScaler(), PCA(n_components = 8), ExtraTreesClassifier(random_state = 31, class_weight = 'balanced', max_depth = 5, min_samples_split = 8, min_samples_leaf = 2, n_estimators = 200))
pipe_lrl1 = make_pipeline(RobustScaler(), PCA(n_components = 8), LogisticRegression(C = 50, class_weight = 'balanced', penalty = 'l1'))
pipe_lrl2 = make_pipeline(RobustScaler(), PCA(n_components = 8), LogisticRegression(C = 50, class_weight = 'balanced', penalty = 'l2'))
pipe_adaboost = make_pipeline(RobustScaler(), PCA(n_components = 8), AdaBoostClassifier(DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 5, min_samples_split = 10, min_samples_leaf = 8), random_state = 31))
pipe_voting = VotingClassifier(estimators = [('svc', pipe_svc), ('lrl1', pipe_lrl1), ('lrl2',pipe_lrl2), ('dtbag', pipe_dtbag), ('dt', pipe_dt), ('rf', pipe_rf)], voting = 'soft', weights = [2, .5, 2, 1.5, 1.5, 1])


scores_defaults['dt']= evaluate_model(pipe_dt, X_gradeA_train[credit_cols], y_gradeA_train, X_gradeA_test[credit_cols], y_gradeA_test)
scores_defaults['rf'] = evaluate_model(pipe_rf, X_gradeA_train[credit_cols], y_gradeA_train, X_gradeA_test[credit_cols], y_gradeA_test)
scores_defaults['et'] = evaluate_model(pipe_et, X_gradeA_train[credit_cols], y_gradeA_train, X_gradeA_test[credit_cols], y_gradeA_test)
scores_defaults['dt_bagged'] = evaluate_model(pipe_dtbag, X_gradeA_train[credit_cols], y_gradeA_train, X_gradeA_test[credit_cols], y_gradeA_test)
scores_defaults['svm'] = evaluate_model(pipe_svc, X_gradeA_train[credit_cols], y_gradeA_train, X_gradeA_test[credit_cols], y_gradeA_test)
scores_defaults['lr_l1'] = evaluate_model(pipe_lrl1, X_gradeA_train[credit_cols], y_gradeA_train, X_gradeA_test[credit_cols], y_gradeA_test)
scores_defaults['lr_l2'] = evaluate_model(pipe_lrl2, X_gradeA_train[credit_cols], y_gradeA_train, X_gradeA_test[credit_cols], y_gradeA_test)
scores_defaults['voting_untuned'] = evaluate_model(pipe_voting, X_gradeA_train[credit_cols], y_gradeA_train, X_gradeA_test[credit_cols], y_gradeA_test)
scores_defaults['adaboost'] = evaluate_model(pipe_adaboost, X_gradeA_train[credit_cols], y_gradeA_train, X_gradeA_test[credit_cols], y_gradeA_test)


scores_defaults_A = pd.DataFrame(scores_defaults, index = ['mean_cv_recall', 'mean_cv_roc_auc', 'test_roc_auc', 'pred_default_rate', 'act_default_rate', 'observed_default_rate']).T.sort_values('test_roc_auc', ascending = False)
scores_defaults_A.to_csv('../Results/second_run/three_year/2007-12 training/PCA/scores_default_gradeA.csv', sep = ',', index = True)

#pickle models - save this for training on the whole data
svm_gradeA_file = open('../Results/second_run/three_year/2007-12 training/PCA/pickled models/gradeA/svm_gradeA_PCA.pk1', 'wb')
pickle.dump(pipe_svc, svm_gradeA_file, -1)
svm_gradeA_file.close()

dtbag_gradeA_file = open('../Results/second_run/three_year/2007-12 training/PCA/pickled_models/gradeA/dtbag_gradeA_PCA.pk1', 'wb')
pickle.dump(pipe_dtbag, dtbag_gradeA_file, -1)
dtbag_gradeA_file.close()

rf_gradeA_file = open('../Results/second_run/three_year/2007-12 training/PCA/pickled_models/gradeA/rf_gradeA_PCA.pk1', 'wb')
pickle.dump(pipe_rf, rf_gradeA_file, -1)
rf_gradeA_file.close()

et_gradeA_file = open('../Results/second_run/three_year/2007-12 training/PCA/pickled_models/gradeA/rf_gradeA_PCA.pk1', 'wb')
pickle.dump(pipe_et, et_gradeA_file, -1)
et_gradeA_file.close()

lrl2_gradeA_file = open('../Results/second_run/three_year/2007-12 training/PCA/pickled_models/gradeA/lrl2_gradeA_PCA.pk1', 'wb')
pickle.dump(pipe_lrl2, lrl2_gradeA_file, -1)
lrl2_gradeA_file.close()

voting_gradeA_file = open('../Results/second_run/three_year/2007-12 training/PCA/pickled_models/gradeA/voting_gradeA_PCA.pk1', 'wb')
pickle.dump(pipe_voting, voting_gradeA_file, -1)
voting_gradeA_file.close()
#Grid search
#dt_params = {'decisiontreeclassifier__criterion': ['entropy', 'gini'], 'decisiontreeclassifier__max_depth': [5, 10, 20], 'decisiontreeclassifier__min_samples_split': [2, 10], 'decisiontreeclassifier__splitter': ['best', 'random']}

#gsdt_gradeA = GridSearchCV(pipe_dt, dt_params, cv = StratifiedKFold(y_gradeA_train, 5, shuffle = True), verbose = True, scoring = 'roc_auc', n_jobs = -1)
#eval_grid(gsdt_gradeA, X_gradeA_train[credit_cols], y_gradeA_train, X_gradeA_test[credit_cols], y_gradeA_test)

gradeA_2013_scores = {}
gradeA_2013_scores['dt'] = eval_2013(pipe_dt, X_gradeA_2013[credit_cols], y_gradeA_2013)
gradeA_2013_scores['rf'] = eval_2013(pipe_rf, X_gradeA_2013[credit_cols], y_gradeA_2013)
gradeA_2013_scores['et'] = eval_2013(pipe_et, X_gradeA_2013[credit_cols], y_gradeA_2013)
gradeA_2013_scores['dtbag'] = eval_2013(pipe_dtbag, X_gradeA_2013[credit_cols], y_gradeA_2013)
gradeA_2013_scores['svm'] = eval_2013(pipe_svc, X_gradeA_2013[credit_cols], y_gradeA_2013)
gradeA_2013_scores['lrl1'] = eval_2013(pipe_lrl1, X_gradeA_2013[credit_cols], y_gradeA_2013)
gradeA_2013_scores['lrl2'] = eval_2013(pipe_lrl2, X_gradeA_2013[credit_cols], y_gradeA_2013)
gradeA_2013_scores['voting'] = eval_2013(pipe_voting, X_gradeA_2013[credit_cols], y_gradeA_2013)
gradeA_2013_scores['adaboost'] = eval_2013(pipe_adaboost, X_gradeA_2013[credit_cols], y_gradeA_2013)

gradeA_2013_scores_df = pd.DataFrame(gradeA_2013_scores, index = ['roc_auc', 'predicted_default', 'actual_default', 'observed_default']).T.sort_values('roc_auc', ascending = False)
gradeA_2013_scores_df
gradeA_2013_scores_df.to_csv('../Results/second_run/three_year/2007-12 training/PCA/gradeA_2013_scores.csv', sep = ',', index = True)

'''Grade A Learning Curves
Let's look at these for the best performing models for 2012: svm, voting, bagged dt, rf, et, lr_l2'''

#Logistic Regresion L2 penalty
from mlxtend.evaluate import plot_learning_curves
plot_learning_curves(X_gradeA_train[credit_cols].as_matrix(), y_gradeA_train.as_matrix(), X_gradeA_test[credit_cols].as_matrix(), y_gradeA_test, clf = pipe_lrl2, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Logistic Regression on Grade A Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeA_learning_curve_logregl2.png', bbox_inches = 'tight')

#Bagged DT
plot_learning_curves(X_gradeA_train[credit_cols].as_matrix(), y_gradeA_train.as_matrix(), X_gradeA_test[credit_cols].as_matrix(), y_gradeA_test, clf = pipe_dtbag, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Bagged DT on Grade A Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeA_learning_curve_bagdt.png', bbox_inches = 'tight')

#Random Forest
plot_learning_curves(X_gradeA_train[credit_cols].as_matrix(), y_gradeA_train.as_matrix(), X_gradeA_test[credit_cols].as_matrix(), y_gradeA_test, clf = pipe_rf, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Random Forest Classifier on Grade A Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeA_learning_curve_rf.png', bbox_inches = 'tight')

#Extra Trees
plot_learning_curves(X_gradeA_train[credit_cols].as_matrix(), y_gradeA_train.as_matrix(), X_gradeA_test[credit_cols].as_matrix(), y_gradeA_test, clf = pipe_et, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Extra Trees Classifier on Grade A Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeA_learning_curve_et.png', bbox_inches = 'tight')

#Voting classifiers
plot_learning_curves(X_gradeA_train[credit_cols].as_matrix(), y_gradeA_train.as_matrix(), X_gradeA_test[credit_cols].as_matrix(), y_gradeA_test, clf = pipe_voting, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Logistic Regression on Grade A Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeA_learning_curve_logregl2.png', bbox_inches = 'tight')

#SVM
plot_learning_curves(X_gradeA_train[credit_cols].as_matrix(), y_gradeA_train.as_matrix(), X_gradeA_test[credit_cols].as_matrix(), y_gradeA_test, clf = pipe_svc, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for SVM on Grade A Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeA_learning_curve_svm.png', bbox_inches = 'tight')

'''ROC Curves for certain estimators'''

#function adapted from sklearn example: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#example-model-selection-plot-roc-crossval-py

plot_roc_curve_credit_cols(X_gradeA, y_gradeA, pipe_svc, 'Support Vector Machine', 'Grade A')
plot_roc_curve_credit_cols(X_gradeA, y_gradeA, pipe_lrl2, 'Logistic Regression - L2', 'Grade A')
plot_roc_curve_credit_cols(X_gradeA, y_gradeA, pipe_dt, 'Decision Tree', 'Grade A')
plot_roc_curve_credit_cols(X_gradeA, y_gradeA, pipe_dtbag, 'Bagged Decision Tree', 'Grade A')
plot_roc_curve_credit_cols(X_gradeA, y_gradeA, pipe_et, 'Extra Trees Classifier', 'Grade A')
plot_roc_curve_credit_cols(X_gradeA, y_gradeA, pipe_rf, 'Random Forest', 'Grade A')
plot_roc_curve_credit_cols(X_gradeA, y_gradeA, pipe_voting, 'Ensemble Voting, Soft,', 'Grade A')

#overall ROC Curve - adapted from Sebastian Raschka's Python Machine Learning.
models_A = [pipe_svc, pipe_lrl2, pipe_dtbag, pipe_et, pipe_rf, pipe_voting]
model_A_names = ['SVM', 'LogReg L2', 'Bagged DT', 'Extra Trees', 'Random Forest', 'Ensemble Voting, Soft']
plot_roc_curve_credit_cols_models(X_gradeA_train, y_gradeA_train, X_gradeA_test, y_gradeA_test,models_A, model_A_names, 'Grade A')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/ROC_Curves_gradeA.png')

'''Grade B
Procedure. See how PCs alone do with our classifiers.
In the next script, we'll use feature selection (no PCA) and compare to ROC AUC with these.
'''
scores_defaults_B = {}

pipe_svc_B = make_pipeline(MinMaxScaler(), PCA(n_components = 8), SVC(C = 10, kernel = 'rbf', max_iter = 30000, probability = True, class_weight = 'balanced', random_state = 31))
pipe_dt_B = make_pipeline(RobustScaler(), PCA(n_components = 8), DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 6, min_samples_split = 8, min_samples_leaf = 4))
pipe_dtbag_B = make_pipeline(RobustScaler(), PCA(n_components = 8), BaggingClassifier(DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 6, min_samples_split = 6, min_samples_leaf = 4), random_state = 31, n_estimators = 50))
pipe_rf_B = make_pipeline(RobustScaler(), PCA(n_components = 8), RandomForestClassifier(random_state = 31, class_weight = 'balanced', n_estimators = 100, max_depth = 5, min_samples_split = 8, min_samples_leaf = 4))
pipe_et_B = make_pipeline(RobustScaler(), PCA(n_components = 8), ExtraTreesClassifier(random_state = 31, class_weight = 'balanced', max_depth = 6, min_samples_split = 6, min_samples_leaf = 4))
pipe_lrl1_B = make_pipeline(RobustScaler(), PCA(n_components = 8), LogisticRegression(C = 50, class_weight = 'balanced', penalty = 'l1'))
pipe_lrl2_B = make_pipeline(RobustScaler(), PCA(n_components = 8), LogisticRegression(C = 50, class_weight = 'balanced', penalty = 'l2'))
pipe_adaboost_B = make_pipeline(RobustScaler(), PCA(n_components = 8), AdaBoostClassifier(DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 10, min_samples_split = 10), random_state = 31, n_estimators = 200, learning_rate = 1))
pipe_voting_B = VotingClassifier(estimators = [('svc', pipe_svc), ('dt', pipe_dt), ('dtbag', pipe_dtbag), ('rf', pipe_rf), ('lrl1', pipe_lrl1), ('lrl2',pipe_lrl2)], voting = 'soft')

scores_defaults_B['dt']= evaluate_model(pipe_dt_B, X_gradeB_train[credit_cols], y_gradeB_train, X_gradeB_test[credit_cols], y_gradeB_test)
scores_defaults_B['rf'] = evaluate_model(pipe_rf_B, X_gradeB_train[credit_cols], y_gradeB_train, X_gradeB_test[credit_cols], y_gradeB_test)
scores_defaults_B['et'] = evaluate_model(pipe_et_B, X_gradeB_train[credit_cols], y_gradeB_train, X_gradeB_test[credit_cols], y_gradeB_test)
scores_defaults_B['dt_bagged'] = evaluate_model(pipe_dtbag_B, X_gradeB_train[credit_cols], y_gradeB_train, X_gradeB_test[credit_cols], y_gradeB_test)
scores_defaults_B['svm'] = evaluate_model(pipe_svc_B, X_gradeB_train[credit_cols], y_gradeB_train, X_gradeB_test[credit_cols], y_gradeB_test)
scores_defaults_B['lr_l1'] = evaluate_model(pipe_lrl1_B, X_gradeB_train[credit_cols], y_gradeB_train, X_gradeB_test[credit_cols], y_gradeB_test)
scores_defaults_B['lr_l2'] = evaluate_model(pipe_lrl2_B, X_gradeB_train[credit_cols], y_gradeB_train, X_gradeB_test[credit_cols], y_gradeB_test)
scores_defaults_B['voting_untuned'] = evaluate_model(pipe_voting_B, X_gradeB_train[credit_cols], y_gradeB_train, X_gradeB_test[credit_cols], y_gradeB_test)
scores_defaults_B['adaboost'] = evaluate_model(pipe_adaboost_B, X_gradeB_train[credit_cols], y_gradeB_train, X_gradeB_test[credit_cols], y_gradeB_test)

scores_defaults_B_df = pd.DataFrame(scores_defaults_B, index = ['mean_cv_recall', 'mean_cv_roc_auc', 'test_roc_auc', 'pred_default_rate', 'act_default_rate', 'observed_default_rate']).T.sort_values('test_roc_auc', ascending = False)
scores_defaults_B_df.to_csv('../Results/second_run/three_year/2007-12 training/PCA/scores_default_gradeB.csv', sep = ',', index = True)

gradeB_2013_scores = {}
gradeB_2013_scores['dt'] = eval_2013(pipe_dt_B, X_gradeB_2013[credit_cols], y_gradeB_2013)
gradeB_2013_scores['rf'] = eval_2013(pipe_rf_B, X_gradeB_2013[credit_cols], y_gradeB_2013)
gradeB_2013_scores['et'] = eval_2013(pipe_et_B, X_gradeB_2013[credit_cols], y_gradeB_2013)
gradeB_2013_scores['dtbag'] = eval_2013(pipe_dtbag_B, X_gradeB_2013[credit_cols], y_gradeB_2013)
gradeB_2013_scores['svm'] = eval_2013(pipe_svc_B, X_gradeB_2013[credit_cols], y_gradeB_2013)
gradeB_2013_scores['lrl1'] = eval_2013(pipe_lrl1_B, X_gradeB_2013[credit_cols], y_gradeB_2013)
gradeB_2013_scores['lrl2'] = eval_2013(pipe_lrl2_B, X_gradeB_2013[credit_cols], y_gradeB_2013)
gradeB_2013_scores['voting'] = eval_2013(pipe_voting_B, X_gradeB_2013[credit_cols], y_gradeB_2013)
gradeB_2013_scores['adaboost'] = eval_2013(pipe_adaboost_B, X_gradeB_2013[credit_cols], y_gradeB_2013)

gradeB_2013_scores = pd.DataFrame(gradeB_2013_scores).T
gradeB_2013_scores.rename(columns = {0:'roc_auc', 1:'predicted_default', 2:'actual_default', 3:'observed_default'}, inplace = True)
gradeB_2013_scores.sort_values('roc_auc', ascending = False)
gradeB_2013_scores.sort_values('roc_auc', ascending = False).to_csv('../Results/second_run/three_year/2007-12 training/PCA/gradeB_2013_scores.csv', sep = ',', index = True)

#Logistic Regression
plot_learning_curves(X_gradeB_train[credit_cols].as_matrix(), y_gradeB_train.as_matrix(), X_gradeB_test[credit_cols].as_matrix(), y_gradeB_test, clf = pipe_lrl2_B, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Logistic Regression, L2, on Grade B Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeB/gradeB_learning_curve_logregl2.png', bbox_inches = 'tight')

plot_learning_curves(X_gradeB_train[credit_cols].as_matrix(), y_gradeB_train.as_matrix(), X_gradeB_test[credit_cols].as_matrix(), y_gradeB_test, clf = pipe_lrl1_B, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Logistic Regression, L1, on Grade B Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeB/gradeB_learning_curve_logregl1.png', bbox_inches = 'tight')

#Bagged DT -high bias and growing variance
plot_learning_curves(X_gradeB_train[credit_cols].as_matrix(), y_gradeB_train.as_matrix(), X_gradeB_test[credit_cols].as_matrix(), y_gradeB_test, clf = pipe_dtbag_B, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Bagged DT on Grade B Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeB/gradeB_learning_curve_bagdt.png', bbox_inches = 'tight')

#Random Forest - high bias and growing variance
plot_learning_curves(X_gradeB_train[credit_cols].as_matrix(), y_gradeB_train.as_matrix(), X_gradeB_test[credit_cols].as_matrix(), y_gradeB_test, clf = pipe_rf_B, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Random Forest Classifier on Grade B Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeB/gradeB_learning_curve_rf.png', bbox_inches = 'tight')

#Extra Trees - high bias, low variance until whole data set
plot_learning_curves(X_gradeB_train[credit_cols].as_matrix(), y_gradeB_train.as_matrix(), X_gradeB_test[credit_cols].as_matrix(), y_gradeB_test, clf = pipe_et_B, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Extra Trees Classifier on Grade B Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeB/gradeB_learning_curve_et.png', bbox_inches = 'tight')

#Voting classifiers - high bias, low variance until whole dataset.
plot_learning_curves(X_gradeB_train[credit_cols].as_matrix(), y_gradeB_train.as_matrix(), X_gradeB_test[credit_cols].as_matrix(), y_gradeB_test, clf = pipe_voting_B, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Voting Classifier on Grade B Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeB/gradeB_learning_curve_voting.png', bbox_inches = 'tight')

#SVM
plot_learning_curves(X_gradeB_train[credit_cols].as_matrix(), y_gradeB_train.as_matrix(), X_gradeB_test[credit_cols].as_matrix(), y_gradeB_test, clf = pipe_svc_B, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for SVM on Grade B Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeB/gradeB_learning_curve_svm.png', bbox_inches = 'tight')

'''ROC Curves for certain estimators'''

plot_roc_curve_credit_cols(X_gradeB, y_gradeB, pipe_svc_B, 'Support Vector Machine', 'Grade B')
plot_roc_curve_credit_cols(X_gradeB, y_gradeB, pipe_lrl2_B, 'Logistic Regression - L2', 'Grade B')
plot_roc_curve_credit_cols(X_gradeB, y_gradeB, pipe_lrl1_B, 'Logistic Regression - L2', 'Grade B')
plot_roc_curve_credit_cols(X_gradeB, y_gradeB, pipe_dt_B, 'Decision Tree', 'Grade B')
plot_roc_curve_credit_cols(X_gradeB, y_gradeB, pipe_dtbag_B, 'Bagged Decision Tree', 'Grade B')
plot_roc_curve_credit_cols(X_gradeB, y_gradeB, pipe_et_B, 'Extra Trees Classifier', 'Grade B')
plot_roc_curve_credit_cols(X_gradeB, y_gradeB, pipe_rf_B, 'Random Forest', 'Grade B')
plot_roc_curve_credit_cols(X_gradeB, y_gradeB, pipe_voting_B, 'Ensemble Voting, Soft,', 'Grade B')

models_B = [pipe_svc_B, pipe_lrl2_B, pipe_dtbag_B, pipe_et_B, pipe_rf_B, pipe_voting_B]
model_B_names = ['SVM', 'LogReg L2', 'Bagged DT', 'Extra Trees', 'Random Forest', 'Ensemble Voting, Soft']
plot_roc_curve_credit_cols_models(X_gradeB_train, y_gradeB_train, X_gradeB_test, y_gradeB_test,models_B, model_B_names, 'Grade B')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeB/ROC_Curves_gradeB.png')

'''Grade C
Procedure. See how PCs alone do with our classifiers.
In the next script, we'll use feature selection (no PCA) and compare to ROC AUC with these.
For SVC at least, C <= 10 seem optimal.
For DT, since ET did well with max depth of 6, try on DT.
'''

scores_defaults_C = {}
pipe_svc_C = make_pipeline(MinMaxScaler(), PCA(n_components = 8), SVC(C = 10, kernel = 'rbf', max_iter = 30000, probability = True, class_weight = 'balanced', random_state = 31))
pipe_svc_C_1 = make_pipeline(MinMaxScaler(), PCA(n_components = 8), SVC(C = 1, kernel = 'rbf', max_iter = 30000, probability = True, class_weight = 'balanced', random_state = 31))
pipe_dt_C = make_pipeline(RobustScaler(), PCA(n_components = 8), DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 10, min_samples_split = 10, min_samples_leaf = 4))
pipe_dtbag_C = make_pipeline(RobustScaler(), PCA(n_components = 8), BaggingClassifier(DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 8, min_samples_split = 6, min_samples_leaf = 4), random_state = 31, n_estimators = 50))
pipe_rf_C = make_pipeline(RobustScaler(), PCA(n_components = 8), RandomForestClassifier(random_state = 31, class_weight = 'balanced', n_estimators = 100, max_depth = 6, min_samples_split = 10, min_samples_leaf = 6))
pipe_et_C = make_pipeline(RobustScaler(), PCA(n_components = 8), ExtraTreesClassifier(random_state = 31, class_weight = 'balanced', max_depth = 8, min_samples_split = 10, min_samples_leaf = 4))
pipe_lrl1_C = make_pipeline(RobustScaler(), PCA(n_components = 8), LogisticRegression(C = 50, class_weight = 'balanced', penalty = 'l1'))
pipe_lrl2_C = make_pipeline(RobustScaler(), PCA(n_components = 8), LogisticRegression(C = 50, class_weight = 'balanced', penalty = 'l2'))
pipe_adaboost_C = make_pipeline(RobustScaler(), PCA(n_components = 8), AdaBoostClassifier(LogisticRegression(C=50, class_weight = 'balanced', penalty = 'l2', solver = 'lbfgs', max_iter = 500, verbose = True), random_state = 31, n_estimators = 500, learning_rate = .1))
pipe_adaboost_C_dt = make_pipeline(RobustScaler(), PCA(n_components = 8), AdaBoostClassifier(DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 8, min_samples_split = 10, min_samples_leaf = 4), random_state = 31, n_estimators = 200, learning_rate = .4))
pipe_gradient_C = make_pipeline(RobustScaler(), PCA(n_components = 8), GradientBoostingClassifier(loss = 'exponential', max_depth = 8, min_samples_split = 4, verbose = True, random_state = 31, n_estimators = 200, learning_rate = .4))
pipe_voting_C = VotingClassifier(estimators = [('svc', pipe_svc_C), ('svc_1', pipe_svc_C_1), ('dtbag', pipe_dtbag_C), ('lrl1', pipe_lrl1_C), ('lrl2',pipe_lrl2_C), ('et', pipe_et_C)], voting = 'soft', weights = [2,1,1,2,1,2])

scores_defaults_C['dt']= evaluate_model(pipe_dt_C, X_gradeC_train[credit_cols], y_gradeC_train, X_gradeC_test[credit_cols], y_gradeC_test)
scores_defaults_C['rf'] = evaluate_model(pipe_rf_C, X_gradeC_train[credit_cols], y_gradeC_train, X_gradeC_test[credit_cols], y_gradeC_test)
scores_defaults_C['et'] = evaluate_model(pipe_et_C, X_gradeC_train[credit_cols], y_gradeC_train, X_gradeC_test[credit_cols], y_gradeC_test)
scores_defaults_C['dt_bagged'] = evaluate_model(pipe_dtbag_C, X_gradeC_train[credit_cols], y_gradeC_train, X_gradeC_test[credit_cols], y_gradeC_test)
scores_defaults_C['svm'] = evaluate_model(pipe_svc_C, X_gradeC_train[credit_cols], y_gradeC_train, X_gradeC_test[credit_cols], y_gradeC_test)
scores_defaults_C['svm_1'] = evaluate_model(pipe_svc_C_1, X_gradeC_train[credit_cols], y_gradeC_train, X_gradeC_test[credit_cols], y_gradeC_test)

scores_defaults_C['lr_l1'] = evaluate_model(pipe_lrl1_C, X_gradeC_train[credit_cols], y_gradeC_train, X_gradeC_test[credit_cols], y_gradeC_test)
scores_defaults_C['lr_l2'] = evaluate_model(pipe_lrl2_C, X_gradeC_train[credit_cols], y_gradeC_train, X_gradeC_test[credit_cols], y_gradeC_test)
scores_defaults_C['voting_untuned'] = evaluate_model(pipe_voting_C, X_gradeC_train[credit_cols], y_gradeC_train, X_gradeC_test[credit_cols], y_gradeC_test)
scores_defaults_C['adaboost'] = evaluate_model(pipe_adaboost_C, X_gradeC_train[credit_cols], y_gradeC_train, X_gradeC_test[credit_cols], y_gradeC_test)
scores_defaults_C['adaboost_dt'] = evaluate_model(pipe_adaboost_C_dt, X_gradeC_train[credit_cols], y_gradeC_train, X_gradeC_test[credit_cols], y_gradeC_test)
scores_defaults_C['gradient_dt'] = evaluate_model(pipe_gradient_C, X_gradeC_train[credit_cols], y_gradeC_train, X_gradeC_test[credit_cols], y_gradeC_test)
scores_defaults_C_df = pd.DataFrame(scores_defaults_C, index = ['mean_cv_recall', 'mean_cv_roc_auc', 'test_roc_auc', 'pred_default_rate', 'act_default_rate', 'observed_default_rate']).T.sort_values('test_roc_auc', ascending = False)

scores_defaults_C_df.to_csv('../Results/second_run/three_year/2007-12 training/PCA/gradeC/scores_default_gradeC.csv', sep = ',', index = True)

#Logistic Regression - high bias - underfitting for both.
plot_learning_curves(X_gradeC_train[credit_cols].as_matrix(), y_gradeC_train.as_matrix(), X_gradeC_test[credit_cols].as_matrix(), y_gradeC_test, clf = pipe_lrl2_C, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Logistic Regression, L2, on Grade C Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeC/gradeC_learning_curve_logregl2.png', bbox_inches = 'tight')

plot_learning_curves(X_gradeC_train[credit_cols].as_matrix(), y_gradeC_train.as_matrix(), X_gradeC_test[credit_cols].as_matrix(), y_gradeC_test, clf = pipe_lrl1_C, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Logistic Regression, L1, on Grade C Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeC/gradeC_learning_curve_logregl1.png', bbox_inches = 'tight')

#Bagged DT - high variance. Also bias.
plot_learning_curves(X_gradeC_train[credit_cols].as_matrix(), y_gradeC_train.as_matrix(), X_gradeC_test[credit_cols].as_matrix(), y_gradeC_test, clf = pipe_dtbag_C, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Bagged DT on Grade C Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeC/gradeC_learning_curve_bagdt.png', bbox_inches = 'tight')

#Random Forest - high variance and bias.
plot_learning_curves(X_gradeC_train[credit_cols].as_matrix(), y_gradeC_train.as_matrix(), X_gradeC_test[credit_cols].as_matrix(), y_gradeC_test, clf = pipe_rf_C, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Random Forest Classifier on Grade C Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeC/gradeC_learning_curve_rf.png', bbox_inches = 'tight')

#Extra Trees - high bias. Higher variance as training increases.
plot_learning_curves(X_gradeC_train[credit_cols].as_matrix(), y_gradeC_train.as_matrix(), X_gradeC_test[credit_cols].as_matrix(), y_gradeC_test, clf = pipe_et_C, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Extra Trees Classifier on Grade C Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeC/gradeC_learning_curve_et.png', bbox_inches = 'tight')

#Voting classifiers
plot_learning_curves(X_gradeC_train[credit_cols].as_matrix(), y_gradeC_train.as_matrix(), X_gradeC_test[credit_cols].as_matrix(), y_gradeC_test, clf = pipe_voting_C, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Voting Classifier on Grade C Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeC/gradeC_learning_curve_voting.png', bbox_inches = 'tight')

#SVM - High Bias.
plot_learning_curves(X_gradeC_train[credit_cols].as_matrix(), y_gradeC_train.as_matrix(), X_gradeC_test[credit_cols].as_matrix(), y_gradeC_test, clf = pipe_svc_C_1, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for SVM on Grade C Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeC/gradeC_learning_curve_svm.png', bbox_inches = 'tight')

#AdaBoost with LogReg L2, LBFGS. High Bias. Low Variance.
plot_learning_curves(X_gradeC_train[credit_cols].as_matrix(), y_gradeC_train.as_matrix(), X_gradeC_test[credit_cols].as_matrix(), y_gradeC_test, clf = pipe_adaboost_C, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for AdaBoost with LogReg L2 Penalty on Grade C Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeC/gradeC_learning_curve_adaboost_lrl2.png', bbox_inches = 'tight')

#AdaBoost with DT High Bias. Low Variance.
plot_learning_curves(X_gradeC_train[credit_cols].as_matrix(), y_gradeC_train.as_matrix(), X_gradeC_test[credit_cols].as_matrix(), y_gradeC_test, clf = pipe_adaboost_C_dt, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for AdaBoost with DT on Grade C Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeC/gradeC_learning_curve_adaboost_dt.png', bbox_inches = 'tight')


'''ROC Curves for certain estimators'''

models =[pipe_svc_C_1, pipe_lrl2_C, pipe_dtbag_C, pipe_et_C, pipe_adaboost_C, pipe_rf_C]
model_names = ['SVC', 'Log Reg L2', 'Bagged DT', 'Extra Trees', 'Adaboost with LogReg', 'Random Forest']

plot_roc_curve_credit_cols_models(X_gradeC_train, y_gradeC_train, X_gradeC_test, y_gradeC_test ,models, model_names, 'Grade C')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeC/ROC_Curves_gradeC.png')

plot_roc_curve_credit_cols(X_gradeC, y_gradeC, pipe_svc_C_1, 'Support Vector Machine', 'Grade C')
plot_roc_curve_credit_cols(X_gradeC, y_gradeC, pipe_lrl2_C, 'Logistic Regression - L2', 'Grade C')
plot_roc_curve_credit_cols(X_gradeC, y_gradeC, pipe_lrl1_C, 'Logistic Regression - L2', 'Grade C')
plot_roc_curve_credit_cols(X_gradeC, y_gradeC, pipe_dt_C, 'Decision Tree', 'Grade C')
plot_roc_curve_credit_cols(X_gradeC, y_gradeC, pipe_dtbag_C, 'Bagged Decision Tree', 'Grade C')
plot_roc_curve_credit_cols(X_gradeC, y_gradeC, pipe_et_C, 'Extra Trees Classifier', 'Grade C')
plot_roc_curve_credit_cols(X_gradeC, y_gradeC, pipe_rf_C, 'Random Forest', 'Grade C')
plot_roc_curve_credit_cols(X_gradeC, y_gradeC, pipe_voting_C, 'Ensemble Voting, Soft,', 'Grade C')



gradeC_2013_scores = {}
gradeC_2013_scores['dt'] = eval_2013(pipe_dt_C, X_gradeC_2013[credit_cols], y_gradeC_2013)
gradeC_2013_scores['rf'] = eval_2013(pipe_rf_C, X_gradeC_2013[credit_cols], y_gradeC_2013)
gradeC_2013_scores['et'] = eval_2013(pipe_et_C, X_gradeC_2013[credit_cols], y_gradeC_2013)
gradeC_2013_scores['dtbag'] = eval_2013(pipe_dtbag_C, X_gradeC_2013[credit_cols], y_gradeC_2013)
gradeC_2013_scores['svm'] = eval_2013(pipe_svc_C, X_gradeC_2013[credit_cols], y_gradeC_2013)
gradeC_2013_scores['svm_C1'] = eval_2013(pipe_svc_C_1, X_gradeC_2013[credit_cols], y_gradeC_2013)
gradeC_2013_scores['lrl1'] = eval_2013(pipe_lrl1_C, X_gradeC_2013[credit_cols], y_gradeC_2013)
gradeC_2013_scores['lrl2'] = eval_2013(pipe_lrl2_C, X_gradeC_2013[credit_cols], y_gradeC_2013)
gradeC_2013_scores['voting'] = eval_2013(pipe_voting_C, X_gradeC_2013[credit_cols], y_gradeC_2013)
gradeC_2013_scores['adaboost'] = eval_2013(pipe_adaboost_C, X_gradeC_2013[credit_cols], y_gradeC_2013)

gradeC_2013_scores_df = pd.DataFrame(gradeC_2013_scores, index = ['roc_auc', 'predicted_default', 'actual_default', 'observed_default']).T.sort_values('roc_auc', ascending = False)
gradeC_2013_scores_df
gradeC_2013_scores_df.to_csv('../Results/second_run/three_year/2007-12 training/PCA/gradeC/gradeC_2013_scores.csv', sep = ',', index = True)


'''Grade D
Procedure. See how PCs alone do with our classifiers.
In the next script, we'll use feature selection (no PCA) and compare to ROC AUC with these.
'''
scores_defaults_D = {}
pipe_svc_D = make_pipeline(MinMaxScaler(), PCA(n_components = 8), SVC(C = 10, kernel = 'rbf', max_iter = 30000, probability = True, class_weight = 'balanced', random_state = 31))
pipe_dt_D = make_pipeline(RobustScaler(), PCA(n_components = 8), DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 4, min_samples_split = 6, min_samples_leaf = 4))
pipe_dtbag_D = make_pipeline(RobustScaler(), PCA(n_components = 8), BaggingClassifier(DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 4, min_samples_leaf = 4), random_state = 31, n_estimators = 50))
pipe_rf_D = make_pipeline(RobustScaler(), PCA(n_components = 8), RandomForestClassifier(random_state = 31, class_weight = 'balanced', n_estimators = 200, max_depth = 4, min_samples_leaf = 4))
pipe_et_D = make_pipeline(RobustScaler(), PCA(n_components = 8), ExtraTreesClassifier(random_state = 31, class_weight = 'balanced', max_depth = 4,min_samples_split = 6,  min_samples_leaf = 4))
pipe_lrl1_D = make_pipeline(RobustScaler(), PCA(n_components = 8), LogisticRegression(C= 50,class_weight = 'balanced', penalty = 'l1'))
pipe_lrl2_D = make_pipeline(RobustScaler(), PCA(n_components = 8), LogisticRegression(C=50,class_weight = 'balanced', penalty = 'l2'))
pipe_adaboost_D = make_pipeline(RobustScaler(), PCA(n_components = 8), AdaBoostClassifier(LogisticRegression(C=1,class_weight = 'balanced', penalty = 'l2', solver = 'lbfgs', max_iter = 1000), random_state = 31, n_estimators = 200, learning_rate = .3))
pipe_voting_D = VotingClassifier(estimators = [('svc', pipe_svc_D), ('dt', pipe_dt_D), ('dtbag', pipe_dtbag_D), ('lrl2',pipe_lrl2_D), ('et', pipe_et_D), ('rf', pipe_rf_D)], voting = 'soft', weights = [2,1,2,2,1,2])

scores_defaults_D['dt']= evaluate_model(pipe_dt_D, X_gradeD_train[credit_cols], y_gradeD_train, X_gradeD_test[credit_cols], y_gradeD_test)
scores_defaults_D['rf'] = evaluate_model(pipe_rf_D, X_gradeD_train[credit_cols], y_gradeD_train, X_gradeD_test[credit_cols], y_gradeD_test)
scores_defaults_D['et'] = evaluate_model(pipe_et_D, X_gradeD_train[credit_cols], y_gradeD_train, X_gradeD_test[credit_cols], y_gradeD_test)
scores_defaults_D['dt_bagged'] = evaluate_model(pipe_dtbag_D, X_gradeD_train[credit_cols], y_gradeD_train, X_gradeD_test[credit_cols], y_gradeD_test)
scores_defaults_D['svm'] = evaluate_model(pipe_svc_D, X_gradeD_train[credit_cols], y_gradeD_train, X_gradeD_test[credit_cols], y_gradeD_test)
scores_defaults_D['lr_l1'] = evaluate_model(pipe_lrl1_D, X_gradeD_train[credit_cols], y_gradeD_train, X_gradeD_test[credit_cols], y_gradeD_test)
scores_defaults_D['lr_l2'] = evaluate_model(pipe_lrl2_D, X_gradeD_train[credit_cols], y_gradeD_train, X_gradeD_test[credit_cols], y_gradeD_test)
scores_defaults_D['voting_untuned'] = evaluate_model(pipe_voting_D, X_gradeD_train[credit_cols], y_gradeD_train, X_gradeD_test[credit_cols], y_gradeD_test)
scores_defaults_D['adaboost'] = evaluate_model(pipe_adaboost_D, X_gradeD_train[credit_cols], y_gradeD_train, X_gradeD_test[credit_cols], y_gradeD_test)

scores_defaults_D_df = pd.DataFrame(scores_defaults_D, index = ['mean_cv_recall', 'mean_cv_roc_auc', 'test_roc_auc', 'pred_default_rate', 'act_default_rate', 'observed_default_rate']).T.sort_values('test_roc_auc', ascending = False)
scores_defaults_D_df
scores_defaults_D_df.to_csv('../Results/second_run/three_year/2007-12 training/PCA/gradeD/scores_default_gradeD.csv', sep = ',', index = True)


#Logistic Regression - high bias - underfitting for both.
plot_learning_curves(X_gradeD_train[credit_cols].as_matrix(), y_gradeD_train.as_matrix(), X_gradeD_test[credit_cols].as_matrix(), y_gradeD_test, clf = pipe_lrl2_D, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Logistic Regression, L2, on Grade D Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeD/gradeD_learning_curve_logregl2.png', bbox_inches = 'tight')

plot_learning_curves(X_gradeD_train[credit_cols].as_matrix(), y_gradeD_train.as_matrix(), X_gradeD_test[credit_cols].as_matrix(), y_gradeD_test, clf = pipe_lrl1_D, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Logistic Regression, L1, on Grade D Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeD/gradeD_learning_curve_logregl1.png', bbox_inches = 'tight')

#Bagged DT - high variance. Also bias.
plot_learning_curves(X_gradeD_train[credit_cols].as_matrix(), y_gradeD_train.as_matrix(), X_gradeD_test[credit_cols].as_matrix(), y_gradeD_test, clf = pipe_dtbag_D, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Bagged DT on Grade D Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeD/gradeD_learning_curve_bagdt.png', bbox_inches = 'tight')

#Random Forest - high variance and bias.
plot_learning_curves(X_gradeD_train[credit_cols].as_matrix(), y_gradeD_train.as_matrix(), X_gradeD_test[credit_cols].as_matrix(), y_gradeD_test, clf = pipe_rf_D, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Random Forest Classifier on Grade D Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeD/gradeD_learning_curve_rf.png', bbox_inches = 'tight')

#Extra Trees - high bias. Higher variance as training increases.
plot_learning_curves(X_gradeD_train[credit_cols].as_matrix(), y_gradeD_train.as_matrix(), X_gradeD_test[credit_cols].as_matrix(), y_gradeD_test, clf = pipe_et_D, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Extra Trees Classifier on Grade D Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeD/gradeD_learning_curve_et.png', bbox_inches = 'tight')

#Voting classifiers
plot_learning_curves(X_gradeD_train[credit_cols].as_matrix(), y_gradeD_train.as_matrix(), X_gradeD_test[credit_cols].as_matrix(), y_gradeD_test, clf = pipe_voting_D, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Voting Classifier on Grade D Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeD/gradeD_learning_curve_voting.png', bbox_inches = 'tight')

#SVM - High Bias.
plot_learning_curves(X_gradeD_train[credit_cols].as_matrix(), y_gradeD_train.as_matrix(), X_gradeD_test[credit_cols].as_matrix(), y_gradeD_test, clf = pipe_svc_D, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for SVM on Grade D Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeD/gradeD_learning_curve_svm.png', bbox_inches = 'tight')

#AdaBoost with LogReg L2, LBFGS. High Bias. Low Variance.
plot_learning_curves(X_gradeD_train[credit_cols].as_matrix(), y_gradeD_train.as_matrix(), X_gradeD_test[credit_cols].as_matrix(), y_gradeD_test, clf = pipe_adaboost_D, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for AdaBoost with LogReg L2 Penalty on Grade D Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeD/gradeD_learning_curve_adaboost.png', bbox_inches = 'tight')

'''ROC Curves for Grade D'''
models =[pipe_svc_D, pipe_lrl2_D, pipe_dtbag_D, pipe_et_D, pipe_voting_D, pipe_rf_D]
model_names = ['SVC', 'Log Reg L2', 'Bagged DT', 'Extra Trees', 'Ensemble Voting, Soft', 'Random Forest']

plot_roc_curve_credit_cols_models(X_gradeD_train, y_gradeD_train, X_gradeD_test, y_gradeD_test ,models, model_names, 'Grade D')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeD/ROC_Curves_gradeD.png')


gradeD_2013_scores = {}
gradeD_2013_scores['dt'] = eval_2013(pipe_dt_D, X_gradeD_2013[credit_cols], y_gradeD_2013)
gradeD_2013_scores['rf'] = eval_2013(pipe_rf_D, X_gradeD_2013[credit_cols], y_gradeD_2013)
gradeD_2013_scores['et'] = eval_2013(pipe_et_D, X_gradeD_2013[credit_cols], y_gradeD_2013)
gradeD_2013_scores['dtbag'] = eval_2013(pipe_dtbag_D, X_gradeD_2013[credit_cols], y_gradeD_2013)
gradeD_2013_scores['svm'] = eval_2013(pipe_svc_D, X_gradeD_2013[credit_cols], y_gradeD_2013)
gradeD_2013_scores['lrl1'] = eval_2013(pipe_lrl1_D, X_gradeD_2013[credit_cols], y_gradeD_2013)
gradeD_2013_scores['lrl2'] = eval_2013(pipe_lrl2_D, X_gradeD_2013[credit_cols], y_gradeD_2013)
gradeD_2013_scores['voting'] = eval_2013(pipe_voting_D, X_gradeD_2013[credit_cols], y_gradeD_2013)
gradeD_2013_scores['adaboost'] = eval_2013(pipe_adaboost_D, X_gradeD_2013[credit_cols], y_gradeD_2013)

gradeD_2013_scores_df = pd.DataFrame(gradeD_2013_scores, index = ['roc_auc', 'predicted_default', 'actual_default', 'observed_default']).T.sort_values('roc_auc', ascending = False)
gradeD_2013_scores_df
gradeD_2013_scores_df.to_csv('../Results/second_run/three_year/2007-12 training/PCA/gradeD_2013_scores.csv', sep = ',', index = True)

'''Grade E
Procedure. See how PCs alone do with our classifiers.
In the next script, we'll use feature selection (no PCA) and compare to ROC AUC with these.
'''
scores_defaults_E = {}
pipe_svc_E = make_pipeline(MinMaxScaler(), PCA(n_components = 8), SVC(C = 10, kernel = 'rbf', max_iter = 30000, probability = True, class_weight = 'balanced', random_state = 31))
pipe_dt_E = make_pipeline(RobustScaler(), PCA(n_components = 8), DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 4, min_samples_leaf = 4, min_samples_split = 8))
pipe_dtbag_E = make_pipeline(RobustScaler(), PCA(n_components = 8), BaggingClassifier(DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 4, min_samples_split = 8, min_samples_leaf = 4), random_state = 31, n_estimators= 50))
pipe_rf_E = make_pipeline(RobustScaler(), PCA(n_components = 8), RandomForestClassifier(random_state = 31, class_weight = 'balanced', n_estimators = 100, max_depth = 4, min_samples_split = 8, min_samples_leaf = 4))
pipe_et_E = make_pipeline(RobustScaler(), PCA(n_components = 8), ExtraTreesClassifier(random_state = 31, class_weight = 'balanced', max_depth = 4, min_samples_split = 8, min_samples_leaf = 4))
pipe_lrl1_E = make_pipeline(RobustScaler(), PCA(n_components = 8), LogisticRegression(C = 50, class_weight = 'balanced', penalty = 'l1'))
pipe_lrl2_E = make_pipeline(RobustScaler(), PCA(n_components = 8), LogisticRegression(C = 50, class_weight = 'balanced', penalty = 'l2'))
pipe_adaboost_E = make_pipeline(RobustScaler(), PCA(n_components = 8), AdaBoostClassifier(LogisticRegression(C = 100, class_weight = 'balanced', penalty = 'l2', solver = 'lbfgs', max_iter = 1000), random_state = 31, n_estimators = 200, learning_rate = .2))
pipe_voting_E = VotingClassifier(estimators = [('svc', pipe_svc_E), ('dt', pipe_dt_E), ('dtbag', pipe_dtbag_E), ('lrl2',pipe_lrl2_E), ('rf', pipe_rf_D), ('ada', pipe_adaboost_E)], voting = 'soft')

scores_defaults_E['dt']= evaluate_model(pipe_dt_E, X_gradeE_train[credit_cols], y_gradeE_train, X_gradeE_test[credit_cols], y_gradeE_test)
scores_defaults_E['rf'] = evaluate_model(pipe_rf_E, X_gradeE_train[credit_cols], y_gradeE_train, X_gradeE_test[credit_cols], y_gradeE_test)
scores_defaults_E['et'] = evaluate_model(pipe_et_E, X_gradeE_train[credit_cols], y_gradeE_train, X_gradeE_test[credit_cols], y_gradeE_test)
scores_defaults_E['dt_bagged'] = evaluate_model(pipe_dtbag_E, X_gradeE_train[credit_cols], y_gradeE_train, X_gradeE_test[credit_cols], y_gradeE_test)
scores_defaults_E['svm'] = evaluate_model(pipe_svc_E, X_gradeE_train[credit_cols], y_gradeE_train, X_gradeE_test[credit_cols], y_gradeE_test)
scores_defaults_E['lr_l1'] = evaluate_model(pipe_lrl1_E, X_gradeE_train[credit_cols], y_gradeE_train, X_gradeE_test[credit_cols], y_gradeE_test)
scores_defaults_E['lr_l2'] = evaluate_model(pipe_lrl2_E, X_gradeE_train[credit_cols], y_gradeE_train, X_gradeE_test[credit_cols], y_gradeE_test)
scores_defaults_E['voting_untuned'] = evaluate_model(pipe_voting_E, X_gradeE_train[credit_cols], y_gradeE_train, X_gradeE_test[credit_cols], y_gradeE_test)
scores_defaults_E['adaboost'] = evaluate_model(pipe_adaboost_E, X_gradeE_train[credit_cols], y_gradeE_train, X_gradeE_test[credit_cols], y_gradeE_test)

scores_defaults_E_df = pd.DataFrame(scores_defaults_E, index = ['mean_cv_recall', 'mean_cv_roc_auc', 'test_roc_auc', 'pred_default_rate', 'act_default_rate', 'observed_default_rate']).T.sort_values('test_roc_auc', ascending = False)
scores_defaults_E_df.to_csv('../Results/second_run/three_year/2007-12 training/PCA/gradeE/scores_default_gradeE.csv', sep = ',', index = True)
scores_defaults_E_df


#Logistic Regression - high bias - underfitting for both.
plot_learning_curves(X_gradeE_train[credit_cols].as_matrix(), y_gradeE_train.as_matrix(), X_gradeE_test[credit_cols].as_matrix(), y_gradeE_test, clf = pipe_lrl2_E, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Logistic Regression, L2, on Grade E Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeE/gradeE_learning_curve_logregl2.png', bbox_inches = 'tight')

plot_learning_curves(X_gradeE_train[credit_cols].as_matrix(), y_gradeE_train.as_matrix(), X_gradeE_test[credit_cols].as_matrix(), y_gradeE_test, clf = pipe_lrl1_E, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Logistic Regression, L1, on Grade E Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeE/gradeE_learning_curve_logregl1.png', bbox_inches = 'tight')

#Bagged DT - high variance. Also bias.
plot_learning_curves(X_gradeD_train[credit_cols].as_matrix(), y_gradeD_train.as_matrix(), X_gradeD_test[credit_cols].as_matrix(), y_gradeD_test, clf = pipe_dtbag_D, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Bagged DT on Grade D Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeD/gradeD_learning_curve_bagdt.png', bbox_inches = 'tight')

#Random Forest - high variance and bias.
plot_learning_curves(X_gradeE_train[credit_cols].as_matrix(), y_gradeE_train.as_matrix(), X_gradeE_test[credit_cols].as_matrix(), y_gradeE_test, clf = pipe_rf_E, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Random Forest Classifier on Grade E Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeE/gradeE_learning_curve_rf.png', bbox_inches = 'tight')

#Extra Trees - high bias. Higher variance as training increases.
plot_learning_curves(X_gradeE_train[credit_cols].as_matrix(), y_gradeE_train.as_matrix(), X_gradeE_test[credit_cols].as_matrix(), y_gradeE_test, clf = pipe_et_E, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Extra Trees Classifier on Grade E Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeE/gradeE_learning_curve_et.png', bbox_inches = 'tight')

#Voting classifiers - high bias and variance at end
plot_learning_curves(X_gradeE_train[credit_cols].as_matrix(), y_gradeE_train.as_matrix(), X_gradeE_test[credit_cols].as_matrix(), y_gradeE_test, clf = pipe_voting_E, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Voting Classifier on Grade E Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeE/gradeE_learning_curve_voting.png', bbox_inches = 'tight')

#SVM - High Bias.
plot_learning_curves(X_gradeE_train[credit_cols].as_matrix(), y_gradeE_train.as_matrix(), X_gradeE_test[credit_cols].as_matrix(), y_gradeE_test, clf = pipe_svc_E, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for SVM on Grade E Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeE/gradeE_learning_curve_svm.png', bbox_inches = 'tight')

#AdaBoost with LogReg L2, LBFGS. High Bias. Low Variance except for very end.
plot_learning_curves(X_gradeE_train[credit_cols].as_matrix(), y_gradeE_train.as_matrix(), X_gradeE_test[credit_cols].as_matrix(), y_gradeE_test, clf = pipe_adaboost_E, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for AdaBoost with LogReg L2 Penalty on Grade E Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeE/gradeE_learning_curve_adaboost.png', bbox_inches = 'tight')
scores_defaults_E_df
'''ROC Curves for Grade E'''
models =[pipe_svc_E, pipe_lrl2_E, pipe_dtbag_E, pipe_et_E, pipe_voting_E, pipe_adaboost_E]
model_names = ['SVC', 'Log Reg L2', 'Bagged DT', 'Extra Trees', 'Ensemble Voting, Soft', 'Adaboost']

plot_roc_curve_credit_cols_models(X_gradeE_train, y_gradeE_train, X_gradeE_test, y_gradeE_test ,models, model_names, 'Grade E')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeE/ROC_Curves_gradeE.png')


gradeE_2013_scores = {}
gradeE_2013_scores['dt'] = eval_2013(pipe_dt_E, X_gradeE_2013[credit_cols], y_gradeE_2013)
gradeE_2013_scores['rf'] = eval_2013(pipe_rf_E, X_gradeE_2013[credit_cols], y_gradeE_2013)
gradeE_2013_scores['et'] = eval_2013(pipe_et_E, X_gradeE_2013[credit_cols], y_gradeE_2013)
gradeE_2013_scores['dtbag'] = eval_2013(pipe_dtbag_E, X_gradeE_2013[credit_cols], y_gradeE_2013)
gradeE_2013_scores['svm'] = eval_2013(pipe_svc_E, X_gradeE_2013[credit_cols], y_gradeE_2013)
gradeE_2013_scores['lrl1'] = eval_2013(pipe_lrl1_E, X_gradeE_2013[credit_cols], y_gradeE_2013)
gradeE_2013_scores['lrl2'] = eval_2013(pipe_lrl2_E, X_gradeE_2013[credit_cols], y_gradeE_2013)
gradeE_2013_scores['voting'] = eval_2013(pipe_voting_E, X_gradeE_2013[credit_cols], y_gradeE_2013)
gradeE_2013_scores['adaboost'] = eval_2013(pipe_adaboost_E, X_gradeE_2013[credit_cols], y_gradeE_2013)

gradeE_2013_scores_df = pd.DataFrame(gradeE_2013_scores, index = ['roc_auc', 'predicted_default', 'actual_default', 'observed_default']).T.sort_values('roc_auc', ascending = False)
gradeE_2013_scores_df.to_csv('../Results/second_run/three_year/2007-12 training/PCA/gradeE/gradeE_2013_scores.csv', sep = ',', index = True)
gradeE_2013_scores_df

'''Grade F
Procedure. See how PCs alone do with our classifiers.
In the next script, we'll use feature selection (no PCA) and compare to ROC AUC with these.
'''
scores_defaults_F = {}
pipe_svc_F = make_pipeline(MinMaxScaler(), PCA(n_components = 8), SVC(C = 10, kernel = 'rbf', max_iter = 30000, probability = True, class_weight = 'balanced', random_state = 31))
pipe_svc_F_100 = make_pipeline(MinMaxScaler(), PCA(n_components = 8), SVC(C = 100, kernel = 'rbf', max_iter = 30000, probability = True, class_weight = 'balanced', random_state = 31))
pipe_dt_F = make_pipeline(RobustScaler(), PCA(n_components = 8), DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 4, min_samples_split = 8, min_samples_leaf = 4))
pipe_dtbag_F = make_pipeline(RobustScaler(), PCA(n_components = 8), BaggingClassifier(DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 3, min_samples_split = 6, min_samples_leaf = 8), random_state = 31, n_estimators = 50))
pipe_rf_F = make_pipeline(RobustScaler(), PCA(n_components = 8), RandomForestClassifier(random_state = 31, class_weight = 'balanced', n_estimators = 100, max_depth = 4, min_samples_split = 8))
pipe_et_F = make_pipeline(RobustScaler(), PCA(n_components = 8), ExtraTreesClassifier(random_state = 31, class_weight = 'balanced', max_depth = 4, min_samples_split = 8, min_samples_leaf = 8))
pipe_lrl1_F = make_pipeline(RobustScaler(), PCA(n_components = 8), LogisticRegression(C = 100, class_weight = 'balanced', penalty = 'l1'))
pipe_lrl2_F = make_pipeline(RobustScaler(), PCA(n_components = 8), LogisticRegression(C = 50, class_weight = 'balanced', penalty = 'l2'))
pipe_adaboost_F = make_pipeline(RobustScaler(), PCA(n_components = 8), AdaBoostClassifier(DecisionTreeClassifier(class_weight = 'balanced', random_state = 31, max_depth = 5, min_samples_split = 6), random_state = 31))
pipe_adaboost_F_lr = make_pipeline(RobustScaler(), PCA(n_components = 8), AdaBoostClassifier(LogisticRegression(C = 50, class_weight = 'balanced', penalty = 'l2', solver = 'lbfgs', max_iter = 1000), random_state = 31, learning_rate = .2, n_estimators = 200))
pipe_voting_F = VotingClassifier(estimators = [('svc', pipe_svc_F), ('dt', pipe_dt_F), ('dtbag', pipe_dtbag_F), ('lrl2',pipe_lrl2_F), ('et', pipe_et_F)], voting = 'soft')

scores_defaults_F['dt']= evaluate_model(pipe_dt_F, X_gradeF_train[credit_cols], y_gradeF_train, X_gradeF_test[credit_cols], y_gradeF_test)
scores_defaults_F['rf'] = evaluate_model(pipe_rf_F, X_gradeF_train[credit_cols], y_gradeF_train, X_gradeF_test[credit_cols], y_gradeF_test)
scores_defaults_F['et'] = evaluate_model(pipe_et_F, X_gradeF_train[credit_cols], y_gradeF_train, X_gradeF_test[credit_cols], y_gradeF_test)
scores_defaults_F['dt_bagged'] = evaluate_model(pipe_dtbag_F, X_gradeF_train[credit_cols], y_gradeF_train, X_gradeF_test[credit_cols], y_gradeF_test)
scores_defaults_F['svm'] = evaluate_model(pipe_svc_F, X_gradeF_train[credit_cols], y_gradeF_train, X_gradeF_test[credit_cols], y_gradeF_test)
scores_defaults_F['svm_C_100'] = evaluate_model(pipe_svc_F_100, X_gradeF_train[credit_cols], y_gradeF_train, X_gradeF_test[credit_cols], y_gradeF_test)
scores_defaults_F['lr_l1'] = evaluate_model(pipe_lrl1_F, X_gradeF_train[credit_cols], y_gradeF_train, X_gradeF_test[credit_cols], y_gradeF_test)
scores_defaults_F['lr_l2'] = evaluate_model(pipe_lrl2_F, X_gradeF_train[credit_cols], y_gradeF_train, X_gradeF_test[credit_cols], y_gradeF_test)
scores_defaults_F['voting_untuned'] = evaluate_model(pipe_voting_F, X_gradeF_train[credit_cols], y_gradeF_train, X_gradeF_test[credit_cols], y_gradeF_test)
scores_defaults_F['adaboost'] = evaluate_model(pipe_adaboost_F, X_gradeF_train[credit_cols], y_gradeF_train, X_gradeF_test[credit_cols], y_gradeF_test)
scores_defaults_F['adaboost_lr'] = evaluate_model(pipe_adaboost_F_lr, X_gradeF_train[credit_cols], y_gradeF_train, X_gradeF_test[credit_cols], y_gradeF_test)
scores_defaults_F_df = pd.DataFrame(scores_defaults_F, index = ['mean_cv_recall', 'mean_cv_roc_auc', 'test_roc_auc', 'pred_default_rate', 'act_default_rate', 'observed_default_rate']).T.sort_values('test_roc_auc', ascending = True)
scores_defaults_F_df.to_csv('../Results/second_run/three_year/2007-12 training/PCA/gradeF/scores_default_gradeF.csv', sep = ',', index = True)

#Logistic Regression - high bias - underfitting for both.
plot_learning_curves(X_gradeF_train[credit_cols].as_matrix(), y_gradeF_train.as_matrix(), X_gradeF_test[credit_cols].as_matrix(), y_gradeF_test, clf = pipe_lrl2_F, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Logistic Regression, L2, on Grade F Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeF/gradeF_learning_curve_logregl2.png', bbox_inches = 'tight')

plot_learning_curves(X_gradeF_train[credit_cols].as_matrix(), y_gradeF_train.as_matrix(), X_gradeF_test[credit_cols].as_matrix(), y_gradeF_test, clf = pipe_lrl1_F, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Logistic Regression, L1, on Grade F Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeF/gradeF_learning_curve_logregl1.png', bbox_inches = 'tight')

#Bagged DT - high variance. Also bias.
plot_learning_curves(X_gradeF_train[credit_cols].as_matrix(), y_gradeF_train.as_matrix(), X_gradeF_test[credit_cols].as_matrix(), y_gradeF_test, clf = pipe_dtbag_F, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Bagged DT on Grade F Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeF/gradeF_learning_curve_bagdt.png', bbox_inches = 'tight')

#Random Forest - high variance and bias.
plot_learning_curves(X_gradeF_train[credit_cols].as_matrix(), y_gradeF_train.as_matrix(), X_gradeF_test[credit_cols].as_matrix(), y_gradeF_test, clf = pipe_rf_F, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Random Forest Classifier on Grade F Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeF/gradeF_learning_curve_rf.png', bbox_inches = 'tight')

#Extra Trees - high bias. Higher variance as training increases.
plot_learning_curves(X_gradeF_train[credit_cols].as_matrix(), y_gradeF_train.as_matrix(), X_gradeF_test[credit_cols].as_matrix(), y_gradeF_test, clf = pipe_et_F, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Extra Trees Classifier on Grade F Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeF/gradeF_learning_curve_et.png', bbox_inches = 'tight')

#Voting classifiers - high bias and variance at end
plot_learning_curves(X_gradeF_train[credit_cols].as_matrix(), y_gradeF_train.as_matrix(), X_gradeF_test[credit_cols].as_matrix(), y_gradeF_test, clf = pipe_voting_F, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for Voting Classifier on Grade F Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeF/gradeF_learning_curve_voting.png', bbox_inches = 'tight')

#SVM - High Bias.
plot_learning_curves(X_gradeF_train[credit_cols].as_matrix(), y_gradeF_train.as_matrix(), X_gradeF_test[credit_cols].as_matrix(), y_gradeF_test, clf = pipe_svc_F, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for SVM on Grade F Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeF/gradeF_learning_curve_svm.png', bbox_inches = 'tight')

#AdaBoost with LogReg L2, LBFGS. High Bias. Low Variance except for very end.
plot_learning_curves(X_gradeF_train[credit_cols].as_matrix(), y_gradeF_train.as_matrix(), X_gradeF_test[credit_cols].as_matrix(), y_gradeF_test, clf = pipe_adaboost_F_lr, print_model = False, scoring = 'roc_auc')
plt.title('Learning Curve for AdaBoost with LogReg L2 Penalty on Grade F Loans, 2007-12')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeF/gradeF_learning_curve_adaboost.png', bbox_inches = 'tight')

'''ROC Curves for Grade F'''
models =[pipe_svc_F, pipe_lrl2_F, pipe_dtbag_F, pipe_et_F, pipe_voting_F, pipe_adaboost_F]
model_names = ['SVC', 'Log Reg L2', 'Bagged DT', 'Extra Trees', 'Ensemble Voting, Soft', 'Adaboost']

plot_roc_curve_credit_cols_models(X_gradeF_train, y_gradeF_train, X_gradeF_test, y_gradeF_test ,models, model_names, 'Grade F')
plt.savefig('../Results/second_run/three_year/2007-12 training/PCA/gradeF/ROC_Curves_gradeF.png')




gradeF_2013_scores = {}
gradeF_2013_scores['dt'] = eval_2013(pipe_dt_F, X_gradeF_2013[credit_cols], y_gradeF_2013)
gradeF_2013_scores['rf'] = eval_2013(pipe_rf_F, X_gradeF_2013[credit_cols], y_gradeF_2013)
gradeF_2013_scores['et'] = eval_2013(pipe_et_F, X_gradeF_2013[credit_cols], y_gradeF_2013)
gradeF_2013_scores['dtbag'] = eval_2013(pipe_dtbag_F, X_gradeF_2013[credit_cols], y_gradeF_2013)
gradeF_2013_scores['svm'] = eval_2013(pipe_svc_F, X_gradeF_2013[credit_cols], y_gradeF_2013)
gradeF_2013_scores['lrl1'] = eval_2013(pipe_lrl1_F, X_gradeF_2013[credit_cols], y_gradeF_2013)
gradeF_2013_scores['lrl2'] = eval_2013(pipe_lrl2_F, X_gradeF_2013[credit_cols], y_gradeF_2013)
gradeF_2013_scores['voting'] = eval_2013(pipe_voting_F, X_gradeF_2013[credit_cols], y_gradeF_2013)
gradeF_2013_scores['adaboost'] = eval_2013(pipe_adaboost_F, X_gradeF_2013[credit_cols], y_gradeF_2013)
gradeF_2013_scores['adaboost_lr'] = eval_2013(pipe_adaboost_F_lr, X_gradeF_2013[credit_cols], y_gradeF_2013)

gradeF_2013_scores_df = pd.DataFrame(gradeF_2013_scores, index = ['roc_auc', 'predicted_default', 'actual_default', 'observed_default']).T.sort_values('roc_auc', ascending = False)
gradeF_2013_scores_df.to_csv('../Results/second_run/three_year/2007-12 training/PCA/gradeF/gradeF_2013_scores.csv', sep = ',', index = True)
gradeF_2013_scores_df
