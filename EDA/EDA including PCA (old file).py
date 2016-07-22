'''Questions to ask for EDA:

Three Year Loans

1) How is the credit data distributed? Normally?
For the most part yes if taking out the outliers.

2) If PCA is performed, what shows?
    -Answer: It looks like for the continuous variables that are roughly normal,
    when looking, you can see credit segments emerging by grade (i.e. you can
    see the subgrades), but not necessarily a clear split between fully paid and default.

3) When does default occur in the loan life? 80% of defaults in 2 years.

4) If I clustered loans from 2007-9 and loans from 2012 on some credit metrics,
how much would they overlap (followup)?

5) How does the outlier data cluster?

6) What does clustering reveal? Does it reveal several groups for analysis (i.e.
market segments?)? Do they match grade, loan status, or something else?
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import TruncatedSVD, PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import ttest_rel
import pickle
%matplotlib inline

loan_only = pd.read_csv('../../Data/LoanStatsstage1.csv', dtype = {'State': str,\
'emp_length': str, 'desc': str, 'next_pymnt_d': str, 'verification_status_joint': str})

loan_only.columns.values

loan_only['date'] = pd.to_datetime(loan_only['date'], infer_datetime_format = True)
loan_only['earliest_cr_line'] = pd.to_datetime(loan_only['earliest_cr_line'], infer_datetime_format = True)
loan_only['last_pymnt_d'] = pd.to_datetime(loan_only['last_pymnt_d'], infer_datetime_format = True)
loan_only['earliest_cr_line'].head()


#IRS zips
loan_only = loan_only[(loan_only.zip_code != '055xx') & (loan_only.zip_code != '192xx') & (loan_only.zip_code != '375xx') & \
(loan_only.zip_code != '399xx') & (loan_only.zip_code != '459xx') & (loan_only.zip_code != '569xx') & (loan_only.zip_code != '649xx') & \
(loan_only.zip_code != '733xx') & (loan_only.zip_code != '842xx') & (loan_only.zip_code != '938xx')]

loan_only = loan_only[loan_only.zip_code != '969xx']
loan_only = loan_only[loan_only.zip_code != '008xx']
#these are military base zips. incidentally, the default rate is quite high for these, so a general rule
#would be to not lend if you see those zips? Would like to impute average income for these.

loan_only = loan_only[(loan_only.zip_code != '962xx') & (loan_only.zip_code != '963xx') &\
(loan_only.zip_code!= '964xx')& (loan_only.zip_code != '965xx') & (loan_only.zip_code != '966xx')\
 & (loan_only.zip_code != '340xx') & (loan_only.zip_code != '090xx') & (loan_only.zip_code != '091xx') & \
 (loan_only.zip_code != '092xx') & (loan_only.zip_code != '093xx') & (loan_only.zip_code != '094xx') & \
 (loan_only.zip_code != '095xx') & (loan_only.zip_code != '096xx') & (loan_only.zip_code != '097xx') & \
 (loan_only.zip_code != '098xx')]

#Scrubbing for zips not in use. Yes, that takes out actual loans from the data,
#but should still leave plenty...
loan_only = loan_only[(loan_only.zip_code != '000xx') & (loan_only.zip_code != '001xx') & (loan_only.zip_code != '002xx') & \
(loan_only.zip_code != '004xx') & (loan_only.zip_code != '099xx') & (loan_only.zip_code != '213xx') & (loan_only.zip_code != '269xx') & \
(loan_only.zip_code != '343xx') & (loan_only.zip_code != '345xx') & (loan_only.zip_code != '348xx') & (loan_only.zip_code != '353xx') & \
(loan_only.zip_code != '419xx') & (loan_only.zip_code != '428xx') & (loan_only.zip_code != '429xx') & (loan_only.zip_code != '517xx') & \
(loan_only.zip_code != '518xx') & (loan_only.zip_code != '519xx') & (loan_only.zip_code != '529xx') & (loan_only.zip_code != '533xx') & \
(loan_only.zip_code != '536xx') & (loan_only.zip_code != '552xx') &(loan_only.zip_code != '568xx') & (loan_only.zip_code != '578xx') & \
(loan_only.zip_code != '579xx') & (loan_only.zip_code != '589xx') & (loan_only.zip_code != '621xx') & (loan_only.zip_code != '632xx') & \
(loan_only.zip_code != '642xx') & (loan_only.zip_code != '643xx') & (loan_only.zip_code != '659xx') & (loan_only.zip_code != '663xx') & \
(loan_only.zip_code != '682xx') & (loan_only.zip_code != '694xx') & (loan_only.zip_code != '695xx') & (loan_only.zip_code != '696xx') & \
(loan_only.zip_code != '697xx') & (loan_only.zip_code != '698xx') & (loan_only.zip_code != '699xx') & (loan_only.zip_code != '702xx') & \
(loan_only.zip_code != '709xx') & (loan_only.zip_code != '715xx') & (loan_only.zip_code != '732xx') & (loan_only.zip_code != '742xx') & \
(loan_only.zip_code != '771xx') & (loan_only.zip_code != '817xx') & (loan_only.zip_code != '818xx') & (loan_only.zip_code != '819xx') & \
(loan_only.zip_code != '839xx') & (loan_only.zip_code != '848xx') & (loan_only.zip_code != '849xx') & (loan_only.zip_code != '854xx') & \
(loan_only.zip_code != '858xx') & (loan_only.zip_code != '861xx') & (loan_only.zip_code != '862xx') & (loan_only.zip_code != '866xx') & \
(loan_only.zip_code != '867xx') & (loan_only.zip_code != '868xx') & (loan_only.zip_code != '869xx') & (loan_only.zip_code != '876xx') & \
(loan_only.zip_code != '886xx') & (loan_only.zip_code != '887xx') & (loan_only.zip_code != '888xx') & (loan_only.zip_code != '892xx') & \
(loan_only.zip_code != '896xx') & (loan_only.zip_code != '899xx') & (loan_only.zip_code != '909xx') & (loan_only.zip_code != '929xx') & \
(loan_only.zip_code != '987xx')]


'''Three Year Loans first.'''

credit_cols = ['funded_amnt','annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc',
       'revol_bal', 'revol_util', 'total_acc',
        'Risk_Score', 'sub_grade_encoded',
       'diff_earliest_app_date']

credit_int_cols = ['funded_amnt','int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc',
       'revol_bal', 'revol_util', 'total_acc',
        'Risk_Score', 'sub_grade_encoded',
       'diff_earliest_app_date']

#loan_only[loan_only.year.isin([2007, 2008, 2009])][credit_cols].corr()

#'magnitude_ann_inc_to_median'

def do_plot_hist(feat):
    plt.figure(figsize = (10,5))
    plt.subplot(121)
    loan_only[loan_only.term == '36 months'][feat].plot(kind='hist', alpha = 0.3, normed = True)
    plt.xlabel(feat.capitalize())
    plt.title(feat.capitalize() + ' Distribution, whole set')

    plt.subplot(122)
    loan_only[loan_only.loan_status_encoded == 0][feat].plot(kind='hist', alpha = 0.3, normed = True)
    loan_only[loan_only.loan_status_encoded == 3][feat].plot(kind='hist', alpha = 0.3, normed = True)
    plt.legend(['Fully Paid', 'Default'])
    plt.xlabel(feat.capitalize())
    plt.title(feat.capitalize() + ' Distribution, per target')
    plt.savefig(' '.join(['3 year Loan', feat.capitalize(), 'Distributions.png']))

for c in credit_cols:
    do_plot_hist(c)

def do_plot_hist_5yr(feat):
    plt.figure(figsize = (10,5))
    plt.subplot(121)
    loan_only[loan_only.term=='60 months'][feat].plot(kind='hist', alpha = 0.3, normed = True)
    plt.xlabel(feat.capitalize())
    plt.title(feat.capitalize() + ' Distribution, whole set')

    plt.subplot(122)
    loan_only[loan_only.term=='60 months'][loan_only.loan_status == 0][feat].plot(kind='hist', alpha = 0.3, normed = True)
    loan_only[loan_only.term=='60 months'][loan_only.loan_status == 3][feat].plot(kind='hist', alpha = 0.3, normed = True)
    plt.legend(['Fully Paid', 'Default'])
    plt.xlabel(feat.capitalize())
    plt.title(feat.capitalize() + ' Distribution, per target')
    plt.savefig(' '.join(['5 year Loan',feat.capitalize(),'Distributions.png']))

for c in credit_cols:
    do_plot_hist_5yr(c)

'''What about default vs fully paid when looking at some of my dummy like variables?'''

loan_only.loan_status_encoded = loan_only.loan_status_encoded.replace(to_replace={0:0, 1:0, 2:1, 3:1})
pd.DataFrame(loan_only[(loan_only.term=='36 months')& (loan_only.year.isin([2007, 2008, 2009, 2010, 2011, 2012, 2013]))].groupby('purpose').loan_status_encoded.value_counts(True)).to_csv('../EDA/default_rates_by_loan_purpose_2007_13_36mo.csv', sep = ',')
pd.DataFrame(loan_only[(loan_only.term=='60 months') & (loan_only.year.isin([2007, 2008, 2009, 2010, 2011, 2012, 2012]))].groupby('purpose').loan_status_encoded.value_counts(True)).to_csv('../EDA/default_rates_by_loan_purpose_2010_12_36mo.csv', sep = ',')

def scatter_class(data, feat):
    plt.figure()
    data.plot(kind = 'scatter', x = feat, y = 'loan_status_encoded')
    plt.xlabel(feat.capitalize())
    plt.title(feat.capitalize() + 'Relationship to Default (Class = 1)')
    plt.ylabel('Class (0 = Paid, 1 = Default)')

other_cols = ['acc_now_delinq', 'pub_rec', 'collections_12_mths_ex_med', 'public_record_ind',\
'mths_since_last_delinq', 'ever_delinq_dummy','mths_since_last_major_derog','mths_since_last_record',\
'public_record_ind', 'derog_record_present', 'credit_card',\
'debt_consolidation', 'educational', 'home_improvement', 'house',\
'major_purchase', 'medical', 'moving', 'other', 'renewable_energy',\
'small_business', 'vacation', 'wedding', 'MORTGAGE', 'NONE',\
'OTHER', 'OWN', 'RENT']

for feat in other_cols:
    scatter_class(loan_only, feat)
'''You can not tell for the dummy variables by scatter plot how they separate classes.
However, some have higher proportions of default, and hence probably help with separability.'''


'''Default rate by State.'''
def_rate_states_through_2012 = pd.DataFrame(loan_only[loan_only.year.isin([2007, 2008, 2009, 2010, 2011, 2012])].groupby('State').loan_status.value_counts(True))
def_rate_states_through_2012 = def_rate_states_through_2012.rename(columns = {0:'frequency'})
def_rate_states_through_2012.to_csv('../EDA/def_rates_3yr_to2012_by_state.csv', sep = ',', index = True)

def_rate_states_through_2012_by_year = pd.DataFrame(loan_only[loan_only.year.isin([2007, 2008, 2009, 2010, 2011, 2012])].groupby(['State', 'year']).loan_status.value_counts(True))
def_rate_states_through_2012_by_year = def_rate_states_through_2012_by_year.rename(columns = {0:'frequency'})
def_rate_states_through_2012_by_year.to_csv('../EDA/def_rates_3yr_to2012_by_state_by_year.csv', sep = ',', index = True)



def_rate_through_2012_by_year = pd.DataFrame(loan_only[loan_only.year.isin([2007, 2008, 2009, 2010, 2011, 2012])][loan_only.term=='36 months'].groupby(['year', 'State']).loan_status.value_counts(True))
def_rate_through_2012_by_year = def_rate_through_2012_by_year.rename(columns = {0:'frequency'})
def_rate_through_2012_by_year.to_csv('../EDA/def_rates_3yr_to2012_by_year.csv', sep = ',', index = True)
test = pd.DataFrame(def_rate_through_2012_by_year.ix[2008].frequency[:, 0])
test = test.assign(def_rate_2009 = def_rate_through_2012_by_year.ix[2009].frequency[:, 0])
test = test.assign(def_rate_2010 = def_rate_through_2012_by_year.ix[2010].frequency[:, 0])
test = test.assign(def_rate_2011 = def_rate_through_2012_by_year.ix[2011].frequency[:, 0])
test = test.assign(def_rate_2012 = def_rate_through_2012_by_year.ix[2012].frequency[:, 0])
test = test.rename(columns = {'frequency':'def_rate_2008'})
plt.figure()
test.ix[:10].plot.bar(figsize = (20, 10), fontsize = 15)
plt.title('Default Rates by Year, First 10 States of US')

test.to_csv('../EDA/def_rates_3yr_to2012.csv', sep = ',', index = True)

def_rate_5yr_loans = pd.DataFrame(loan_only[loan_only.year.isin([2007, 2008, 2009, 2010, 2011, 2012])][loan_only.term=='60 months'].groupby(['year', 'State']).loan_status.value_counts(True))
def_rate_5yr_loans = def_rate_5yr_loans.rename(columns = {0:'frequency'})
test = pd.DataFrame(def_rate_5yr_loans.ix[2010].frequency[:, 0])
test = test.assign(def_rate_2011 = def_rate_5yr_loans.ix[2011].frequency[:, 0])
test = test.assign(def_rate_2012 = def_rate_5yr_loans.ix[2012].frequency[:, 0])
test = test.rename(columns = {'frequency':'def_rate_2010'})
test.to_csv('../EDA/def_rates_5yr_loans.csv', sep = ',', index = True)
'''ARe the default rates diff y-o-y - Basically 2010 is an odd year...'''
for i in range(len(test.columns.values)):
    if i > 0:
        print ttest_rel(test[test.columns.values[0]],test[test.columns.values[i]])



'''Default Rates over lifecycles'''

default_timing = loan_only[loan_only.year.isin([2007, 2008, 2009, 2010, 2011, 2012])][loan_only.loan_status.isin([2,3])][['date','last_pymnt_d','term']]

#three year.



default_timing['diff'] = default_timing['last_pymnt_d'] - default_timing['date']
default_timing['diff'].fillna(0, inplace = True)
default_timing['diff'] = default_timing['diff'].apply(convert_to_dayint)

default_timing[default_timing.term=='36 months'].diff_scaled.plot(kind = 'hist', alpha = 0.3, normed = True, color = 'r')
for index1, row in default_timing.iterrows():
    if row.term == '36 months':
        default_timing.set_value(index1, 'diff_scaled', row['diff']/(float(1095)))
    if row.term == '60 months':
        default_timing.set_value(index1, 'diff_scaled', row['diff']/(float(1825)))

default_timing[default_timing.term=='36 months'].diff_scaled.plot(kind = 'hist', alpha = 0.3, normed = True, color = 'r', cumulative = True)
plt.title('Timing of Default vizualized with CDF, 36 month loans')
plt.savefig('timingofdefault_36monthloansthrough2012.png')

default_timing[default_timing.term=='60 months'].diff_scaled.plot(kind = 'hist', alpha = 0.3, normed = True, color = 'r', cumulative = True)
plt.title('Timing of Default vizualized with CDF, 60 month loans')
plt.savefig('timingofdefault_60monthloansthrough2012.png')


'''At two year mark, a lot of defaults have occured. Nearly 80%. One year mark, 40%.
Implies that could train on data through 2010 to test on 2012 dataset, and roll into prod.

5 year loans: at 4 year mark, more than 90% defaulted. at three year mark, more than 80%.
Implies that if training on data, could use 2011 data for testing OOS for 2010 5 year loans since
majority of defaults will have occurred and hence ground truth likely revealed.

Similarly, could incorporate 2010 and 11 data for training for production model.
'''
import seaborn as sns
sns.distplot(default_timing[default_timing.term=='36 months'].diff_scaled, \
hist = True, norm_hist=True, color = 'r')

'''Filtering out outliers'''

test = loan_only[(loan_only.delinq_amnt > .2)]
test2 = loan_only[(loan_only.revol_util > 2)]
test3 = loan_only[loan_only.acc_now_delinq >=1]
test4 =  loan_only[loan_only.revol_bal > 100000]
test5 = loan_only[loan_only.total_acc > 60]
test6 = loan_only[loan_only.open_acc > 25]
test7 = loan_only[loan_only.annual_inc > 200000]
test8 = loan_only[loan_only.delinq_2yrs> 3]
test9 = loan_only[loan_only.pub_rec>2]
test10 = loan_only[loan_only.collections_12_mths_ex_med>= 2]

loan_only_outliers = pd.concat([test, test2, test3, test4, test5, test6, test7, test8, test9, test10])
loan_only_outliers.info()

loan_only_outliers.drop_duplicates(inplace = True)
loan_only_no_outliers = loan_only[~loan_only.row_num.isin(loan_only_outliers.row_num)]


'''PCA on credit data - 2007-10 data, Three Year Loans:

1) The credit data has plenty of normal-like distributed features (though plenty
that appear binary too).
2) Instead of feature selection, let's consider feature engineering on the credit data. '''

scaler_eda = RobustScaler()
loans_3yr_07_10_scaled_A = pd.DataFrame(scaler_eda.fit_transform(loan_only_no_outliers[(loan_only_no_outliers.year.isin([2007, 2008, 2009, 2010])) & (loan_only_no_outliers.term == '36 months') & (loan_only_no_outliers.grade == 'A')][credit_int_cols]), columns = credit_int_cols, index = loan_only_no_outliers[(loan_only_no_outliers.year.isin([2007, 2008, 2009, 2010])) & (loan_only_no_outliers.term== '36 months') & (loan_only_no_outliers.grade == 'A')].index)
pca = PCA(n_components = None)
loans_3yr_credit_pca = pca.fit_transform(loans_3yr_07_10_scaled_A)

cum_var = np.cumsum(pca.explained_variance_ratio_) #two components
pca.n_components_
plt.figure(figsize=(9,7))

component_number = range(1, 15)

plt.plot(component_number, cum_var, lw=7)

plt.axhline(y=0, linewidth=5, color='grey', ls='dashed')
plt.axhline(y=1, linewidth=3, color='grey', ls='dashed')
plt.axhline(y = .95, linewidth = 3, color = 'green', ls = 'dashed')

ax = plt.gca()
ax.set_xlim([1,14])
ax.set_ylim([-.05,1.05])

ax.set_ylabel('cumulative variance explained', fontsize=16)
ax.set_xlabel('component', fontsize=16)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)

ax.set_title('component vs cumulative variance explained\n', fontsize=20)

'''Let's also see what composes the first two PCs.'''
pca_2 = PCA(n_components = 8)
PCs_whole = pca_2.fit_transform(loans_3yr_07_10_scaled_A)
prin_comps = pd.DataFrame(PCs_whole, columns = ['PC' + str(i) for i in range(1,9)], index = loans_3yr_07_10_scaled_A.index)
loan_status_pcs_year_A = pd.merge(loan_only_no_outliers[['loan_status_encoded', 'year']], prin_comps, left_index = True, right_index = True)

graph_pc1pc2_2009_A = loan_status_pcs_year_A[loan_status_pcs_year_A.year.isin([2007,2008,2009])].plot(kind = 'scatter', x = 'PC1', y = 'PC2', figsize = (16, 8))

for i, loan_status in enumerate(loan_status_pcs_year_A[loan_status_pcs_year_A.year.isin([2007,2008,2009])]['loan_status_encoded']):
    graph_pc1pc2_2009_A.annotate(loan_status, (loan_status_pcs_year_A[loan_status_pcs_year_A.year.isin([2007,2008,2009])].iloc[i].PC1, loan_status_pcs_year_A[loan_status_pcs_year_A.year.isin([2007,2008,2009])].iloc[i].PC2))

loan_status_pcs_year_A.plot(kind = 'scatter', x = 'PC8', y = 'loan_status_encoded')
#Not separable with PCA linearly, either with Robust or Standard Scaler

'''Outliers are throwing our PCA off.'''
prin_comps_features = pd.merge(loans_3yr_07_10_scaled, prin_comps, left_index = True, right_index = True)

corr_prin_comps = prin_comps_features.corr().drop(['PC1', 'PC2'], axis = 0)
corr_prin_comps.drop(credit_cols, axis = 1, inplace = True)
corr_prin_comps


'''PCA does not show a way for this to be separable, not linearly. Try kernel PCA'''

'''LDA on the data. 2007-10 data, Three Year Loans:

1) The credit data has plenty of normal-like distributed features.
2) Instead of feature selection, let's consider feature engineering on the credit data. '''

scaler_eda = RobustScaler()
loans_3yr_07_10_scaled_A = pd.DataFrame(scaler_eda.fit_transform(loan_only_no_outliers[(loan_only_no_outliers.year.isin([2007, 2008, 2009, 2010])) & (loan_only_no_outliers.term == '36 months') & (loan_only_no_outliers.grade == 'A')][credit_int_cols]), columns = credit_int_cols, index = loan_only_no_outliers[(loan_only_no_outliers.year.isin([2007, 2008, 2009, 2010])) & (loan_only_no_outliers.term== '36 months') & (loan_only_no_outliers.grade == 'A')].index)
pca_2 = PCA(n_components = None)
loans_3yr_credit_pca_2 = pca_2.fit_transform(loans_3yr_07_10_scaled_A)

cum_var = np.cumsum(pca_2.explained_variance_ratio_)

'''Let's also see what composes the first eight PCs.'''
pca_2 = PCA(n_components = 8)
PCs_whole = pca_2.fit_transform(loans_3yr_07_10_scaled_A)
prin_comps = pd.DataFrame(PCs_whole, columns = ['PC' + str(i) for i in range(1,9)], index = loans_3yr_07_10_scaled_A.index)
loan_status_pcs_year_A = pd.merge(loan_only_no_outliers[['loan_status_encoded', 'year']], prin_comps, left_index = True, right_index = True)

graph_pc1pc2_2009_A = loan_status_pcs_year_A[loan_status_pcs_year_A.year.isin([2007,2008,2009])].plot(kind = 'scatter', x = 'PC1', y = 'PC2', figsize = (16, 8))

for i, loan_status in enumerate(loan_status_pcs_year_A[loan_status_pcs_year_A.year.isin([2007,2008,2009])]['loan_status_encoded']):
    graph_pc1pc2_2009_A.annotate(loan_status, (loan_status_pcs_year_A[loan_status_pcs_year_A.year.isin([2007,2008,2009])].iloc[i].PC1, loan_status_pcs_year_A[loan_status_pcs_year_A.year.isin([2007,2008,2009])].iloc[i].PC2))

loan_status_pcs_year_A.plot(kind = 'scatter', x = 'PC2', y = 'loan_status_encoded')
#Not separable with PCA linearly, either with Robust or Standard Scaler


'''Kernel PCA on credit data - 2007-10 data, Three Year Loans:

1) The credit data has plenty of normal-like distributed features (though plenty
that appear binary too).
2) Instead of feature selection, let's consider feature engineering on the credit data. '''

from matplotlib.colors import ListedColorMap

def plot_decision_regions(X, y, classifier, resolution = 0.02):
    markers = ('s', 'x')
    colors = ('red', 'blue')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:,1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx2.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0], y = X[y ==c1, 1], alpha = 0.8, c = cmap(idx), marker = markers[idx], label = c1)

'''Testing out various kernel PCAs.
RBF does not seem to work well for separation.
Poly does not work. I suspect none of them do.
Hence, I have to move on to filtering my dataset for feature importances by grade, and then testing models.
To do it properly, let's use the code I have for cross val from the 5 year loan set.

'''
scaler_eda = RobustScaler()
loans_3yr_07_10_scaled_A = pd.DataFrame(scaler_eda.fit_transform(loan_only_no_outliers[(loan_only_no_outliers.year.isin([2007, 2008, 2009, 2010])) & (loan_only_no_outliers.term == '36 months') & (loan_only_no_outliers.grade == 'A')][credit_int_cols]), columns = credit_int_cols, index = loan_only_no_outliers[(loan_only_no_outliers.year.isin([2007, 2008, 2009, 2010])) & (loan_only_no_outliers.term== '36 months') & (loan_only_no_outliers.grade == 'A')].index)
y = loan_only_no_outliers[['loan_status_encoded']].iloc[loans_3yr_07_10_scaled_A.index]
#poly might be best.

for i in ['sigmoid', 'cosine']:
    scikit_kpca = KernelPCA(n_components = None, kernel = i,)
    X_kernelpca = scikit_kpca.fit_transform(loans_3yr_07_10_scaled_A)
    plt.scatter(X_kernelpca[y==0, 0], X_kernelpca[y==0, 1], color = 'red', marker = '^', alpha = 0.5)
    plt.scatter(X_kernelpca[y==1, 0], X_kernelpca[y==1, 1], color = 'blue', marker = 'o', alpha = 0.5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

for i in ['rbf', 'poly']:
    for g in range(1, 6):
        if i == 'rbf':
            scikit_kpca = KernelPCA(n_components = None, kernel = i, gamma = g)
            X_kernelpca = scikit_kpca.fit_transform(loans_3yr_07_10_scaled_A)
            plt.scatter(X_kernelpca[y==0, 0], X_kernelpca[y==0, 1], color = 'red', marker = '^', alpha = 0.5)
            plt.scatter(X_kernelpca[y==1, 0], X_kernelpca[y==1, 1], color = 'blue', marker = 'o', alpha = 0.5)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title(''.join([(i.capitalize()),' PCs with gamma of ', str(g)]))
            plt.show()
        else:
            for d in range(1, 6):
                scikit_kpca = KernelPCA(n_components = None, kernel = i, gamma = g, degree = d)
                X_kernelpca = scikit_kpca.fit_transform(loans_3yr_07_10_scaled_A)
                plt.scatter(X_kernelpca[y==0, 0], X_kernelpca[y==0, 1], color = 'red', marker = '^', alpha = 0.5)
                plt.scatter(X_kernelpca[y==1, 0], X_kernelpca[y==1, 1], color = 'blue', marker = 'o', alpha = 0.5)
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.title(''.join([(i.capitalize()) + ' PCs with gamma of ' + str(g) + ' and degree of ' + str(d)]))
                plt.show()


'''Also, whatever model works, a script for a website will have to replicate the cleaning steps.'''
