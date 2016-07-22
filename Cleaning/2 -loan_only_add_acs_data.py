'''Part 2 of Cleaning:
Adding in the ACS Income Data, and Dummifying certain categorical vars.

So I'm going to be creating a file where the max date is Dec 31, 2012. This will
allow me to see the full loan aging for three year loans.

I'll also be creating a file where the max date is Dec 31, 2010, for the five year loans.

Whatever model from those two, I'll test on post 2012 data.

I'll also explore when default occurs typically, and if it usually is early on,
then I'll also use two post 2013-09-23 files, one for three year loans, and one for five year loans,
for testing the old models, and then creating new models from the 2013 data to test on 2014 loans.

Note: Need to preserve row
'''


#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string




usecols = ['id', 'row_num', 'sub_grade', 'amnt_request', 'funded_amnt', 'annual_inc', 'credit_card', 'debt_consolidation',
'educational', 'home_improvement', 'house', 'major_purchase',
'medical', 'moving', 'other', 'renewable_energy', 'small_business',
'vacation', 'wedding', 'sub_grade_encoded', 'term', 'installment', 'int_rate',\
 'grade', 'emp_title', 'dti', 'zip_code', 'title', 'Risk_Score', 'delinq_amnt', 'delinq_2yrs', \
 'earliest_cr_line', 'inq_last_6mths', 'mths_since_last_delinq', \
 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', \
 'total_acc', 'ever_delinq_dummy','derog_record_present',\
 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'acc_now_delinq',\
  'public_record_ind', 'diff_earliest_app_date', 'loan_status_encoded',\
 'MORTGAGE', 'NONE', 'OTHER', 'OWN', 'RENT', 'date', 'year', 'month', \
 'loan_status', 'out_prncp', 'out_prncp_inv', 'recoveries', 'total_pymnt',\
 'total_pymnt_inv', 'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp', \
 'gross_return','gross_return_scaled', 'annual_inc_joint', 'dti_joint', \
  'emp_length_1', 'emp_length_10', 'emp_length_2', 'emp_length_3',
  'emp_length_4', 'emp_length_5', 'emp_length_6', 'emp_length_7',
  'emp_length_8', 'emp_length_9', 'emp_length_n/a', 'State_AL',
  'State_AR', 'State_AZ', 'State_CA', 'State_CO', 'State_CT',
  'State_DC', 'State_DE', 'State_FL', 'State_GA', 'State_HI',
  'State_IA', 'State_ID', 'State_IL', 'State_IN', 'State_KS',
  'State_KY', 'State_LA', 'State_MA', 'State_MD', 'State_ME',
  'State_MI', 'State_MN', 'State_MO', 'State_MS', 'State_MT',
  'State_NC', 'State_ND', 'State_NE', 'State_NH', 'State_NJ',
  'State_NM', 'State_NV', 'State_NY', 'State_OH', 'State_OK',
  'State_OR', 'State_PA', 'State_RI', 'State_SC', 'State_SD',
  'State_TN', 'State_TX', 'State_UT', 'State_VA', 'State_VT',
  'State_WA', 'State_WI', 'State_WV', 'State_WY',
  'verification_status_joint_Source Verified',
  'verification_status_joint_Verified',
  'verification_status_Source Verified',
  'verification_status_Verified']


loan_only = pd.read_csv('../../Data/LoanStatsstage1.csv', usecols = usecols, dtype = \
{'State': str})

loan_only['date'] = pd.to_datetime(loan_only['date'], infer_datetime_format = True)
loan_only['earliest_cr_line'] = pd.to_datetime(loan_only['earliest_cr_line'], infer_datetime_format = True)
loan_only['earliest_cr_line'].head()




#since I want to use public facing data, let's subset the public facing data
#and outcome variables..

features_joint = ['id', 'row_num', 'sub_grade', 'amnt_request', 'funded_amnt', 'annual_inc', 'credit_card', 'debt_consolidation',
'educational', 'home_improvement', 'house', 'major_purchase',
'medical', 'moving', 'other', 'renewable_energy', 'small_business',
'vacation', 'wedding', 'sub_grade_encoded', 'term', 'installment', 'int_rate',\
 'grade', 'emp_title', 'dti', 'zip_code', 'title', 'Risk_Score', 'delinq_2yrs', \
 'earliest_cr_line', 'inq_last_6mths', 'mths_since_last_delinq', \
 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', \
 'total_acc', 'ever_delinq_dummy','derog_record_present',\
 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'acc_now_delinq',\
  'public_record_ind', 'diff_earliest_app_date', 'loan_status_encoded',\
 'MORTGAGE', 'NONE', 'OTHER', 'OWN', 'RENT', 'date', 'year', 'month', \
 'annual_inc_joint', 'dti_joint', \
  'emp_length_1', 'emp_length_10', 'emp_length_2', 'emp_length_3',
  'emp_length_4', 'emp_length_5', 'emp_length_6', 'emp_length_7',
  'emp_length_8', 'emp_length_9', 'emp_length_n/a', 'State_AL',
  'State_AR', 'State_AZ', 'State_CA', 'State_CO', 'State_CT',
  'State_DC', 'State_DE', 'State_FL', 'State_GA', 'State_HI',
  'State_IA', 'State_ID', 'State_IL', 'State_IN', 'State_KS',
  'State_KY', 'State_LA', 'State_MA', 'State_MD', 'State_ME',
  'State_MI', 'State_MN', 'State_MO', 'State_MS', 'State_MT',
  'State_NC', 'State_ND', 'State_NE', 'State_NH', 'State_NJ',
  'State_NM', 'State_NV', 'State_NY', 'State_OH', 'State_OK',
  'State_OR', 'State_PA', 'State_RI', 'State_SC', 'State_SD',
  'State_TN', 'State_TX', 'State_UT', 'State_VA', 'State_VT',
  'State_WA', 'State_WI', 'State_WV', 'State_WY',
  'verification_status_joint_Source Verified',
  'verification_status_joint_Verified',
  'verification_status_Source Verified',
  'verification_status_Verified']

outcome_cols = ['id', 'funded_amnt', 'sub_grade', 'loan_status', 'loan_status_encoded', 'out_prncp', 'out_prncp_inv', 'recoveries', 'total_pymnt',\
'total_pymnt_inv', 'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp', 'gross_return',\
'gross_return_scaled']



#ensuring no null values for title
loan_only.title.fillna('none', axis = 0, inplace = True)

'''Data for median income:
Group by:
-Zip Code and State
-Emp Length - not in ACS
-Wish home-ownership was in there.

HC02_EST_VC02 - median income, household, estimate.
GEO.id
GEO.id2

HC01_EST_VC08 - did not work %. 1 - this is the % who did work, full or part time.
        I can also get the % full/part time split, but I do not know if this is worth
        putting in the data.

'''

'''Obtaining Median Income from ACS.'''
acs_0711 = pd.read_csv('../../Data/ACS_11_5YR_S1903/ACS_11_5YR_S1903_with_ann.csv', \
skiprows = 2, usecols = [1,5], names = ['zipcode', 'median_est_income_household'], \
dtype = {'zipcode': str, 'median_est_income_household': str}, na_values = ['-', 'N', '(X)'])
acs_0711.median_est_income_household.isnull().value_counts()
acs_0711.dropna(axis = 0, inplace = True)
replace_last_2 = lambda x: ''.join([x[:3], 'xx'])
acs_0711['zipcode'] = acs_0711.zipcode.apply(replace_last_2)
acs_0711['median_est_income_household'] = acs_0711['median_est_income_household'].str.rstrip('.0')
acs_0711['median_est_income_household'] = acs_0711['median_est_income_household'].str.rstrip('+')
acs_0711['median_est_income_household'] = acs_0711['median_est_income_household'].str.rstrip('-')
acs_0711['median_est_income_household'] = acs_0711['median_est_income_household'].str.replace(',', '')
acs_0711['median_est_income_household'] = pd.to_numeric(acs_0711['median_est_income_household'])
acs_0711_df = pd.pivot_table(acs_0711, index = 'zipcode', values = 'median_est_income_household')
acs_0711_df = pd.DataFrame(acs_0711.groupby('zipcode').median_est_income_household.mean())

acs_0812 = pd.read_csv('../../Data/ACS_12_5YR_S1901/ACS_12_5YR_S1901_with_ann.csv', \
 skiprows = 2, usecols = [1, 91], names = ['zipcode', 'median_est_income_household_0812'], \
dtype = {'zipcode': str, 'median_est_income_household_0812': str}, na_values = ['-', 'N', '(X)'])
acs_0812.median_est_income_household_0812.isnull().value_counts()
acs_0812.dropna(axis = 0, inplace = True)
acs_0812['zipcode'] = acs_0812.zipcode.apply(replace_last_2)
acs_0812['median_est_income_household_0812'] = acs_0812['median_est_income_household_0812'].str.rstrip('.0')
acs_0812['median_est_income_household_0812'] = acs_0812['median_est_income_household_0812'].str.rstrip('+')
acs_0812['median_est_income_household_0812'] = acs_0812['median_est_income_household_0812'].str.rstrip('-')
acs_0812['median_est_income_household_0812'] = acs_0812['median_est_income_household_0812'].str.replace(',', '')
acs_0812['median_est_income_household_0812'] = pd.to_numeric(acs_0812['median_est_income_household_0812'])
acs_0812_df = pd.pivot_table(acs_0812, index = 'zipcode', values = 'median_est_income_household_0812')
acs_0812_df = pd.DataFrame(acs_0812.groupby('zipcode').median_est_income_household_0812.mean())


acs_0913 = pd.read_csv('../../Data/ACS_13_5YR_S1901/ACS_13_5YR_S1901_with_ann.csv', \
 skiprows = 2, usecols = [1, 91], names = ['zipcode', 'median_est_income_household_0913'], \
dtype = {'zipcode': str, 'median_est_income_household_0913': str}, na_values = ['-', 'N', '(X)'])
acs_0913.median_est_income_household_0913.isnull().value_counts()
acs_0913.dropna(axis = 0, inplace = True)
acs_0913['zipcode'] = acs_0913.zipcode.apply(replace_last_2)
acs_0913['median_est_income_household_0913'] = acs_0913['median_est_income_household_0913'].str.rstrip('.0')
acs_0913['median_est_income_household_0913'] = acs_0913['median_est_income_household_0913'].str.rstrip('+')
acs_0913['median_est_income_household_0913'] = acs_0913['median_est_income_household_0913'].str.rstrip('-')
acs_0913['median_est_income_household_0913'] = acs_0913['median_est_income_household_0913'].str.replace(',', '')
acs_0913['median_est_income_household_0913'] = pd.to_numeric(acs_0913['median_est_income_household_0913'])
acs_0913_df = pd.DataFrame(pd.pivot_table(acs_0913, index = 'zipcode', values = 'median_est_income_household_0913'))

acs_1014 = pd.read_csv('../../Data/ACS_14_5YR_S1901/ACS_14_5YR_S1901_with_ann.csv', \
 skiprows = 2, usecols = [1, 91], names = ['zipcode', 'median_est_income_household_1014'], \
dtype = {'zipcode': str, 'median_est_income_household_1014': str}, na_values = ['-', 'N', '(X)'])
acs_1014.median_est_income_household_1014.isnull().value_counts()

acs_1014.dropna(axis = 0, inplace = True)
acs_1014['zipcode'] = acs_1014.zipcode.apply(replace_last_2)
acs_1014['median_est_income_household_1014'] = acs_1014['median_est_income_household_1014'].str.rstrip('.0')
acs_1014['median_est_income_household_1014'] = acs_1014['median_est_income_household_1014'].str.rstrip('+')
acs_1014['median_est_income_household_1014'] = acs_1014['median_est_income_household_1014'].str.rstrip('-')
acs_1014['median_est_income_household_1014'] = acs_1014['median_est_income_household_1014'].str.replace(',', '')
acs_1014['median_est_income_household_1014'] = pd.to_numeric(acs_1014['median_est_income_household_1014'])
acs_1014_df = pd.DataFrame(pd.pivot_table(acs_1014, index = 'zipcode', values = 'median_est_income_household_1014'))

acs = pd.merge(acs_0711_df, acs_0812_df, how = 'inner', right_index = True, left_index = True)
acs = pd.merge(acs, acs_0913_df, how = 'inner', right_index = True, left_index = True)
acs = pd.merge(acs, acs_1014_df, how = 'inner', right_index = True, left_index = True)
import scipy.stats
scipy.stats.ttest_rel(acs.median_est_income_household_0812, acs.median_est_income_household_0913)

acs.head()


'''Obtaining % who worked by zip code. Unsurprising how bad it is for PR given the news.'''
acs_0711_work = pd.read_csv('../../Data/Work Status ACS/ACS_11_5YR_S2303/ACS_11_5YR_S2303_with_ann.csv', \
skiprows = 2, usecols = [1,39], names = ['zipcode', 'did_not_work'], \
dtype = {'zipcode': str, 'did_not_work': str}, na_values = ['-', 'N', '(X)'])
acs_0711_work.did_not_work.isnull().value_counts()
acs_0711_work.dropna(axis = 0, inplace = True)
replace_last_2 = lambda x: ''.join([x[:3], 'xx'])
work_val = lambda x: 100-x
divide_100 = lambda x: float(x)/100
acs_0711_work['zipcode'] = acs_0711_work.zipcode.apply(replace_last_2)
acs_0711_work['did_not_work'] = acs_0711_work['did_not_work'].str.rstrip('+')
acs_0711_work['did_not_work'] = acs_0711_work['did_not_work'].str.rstrip('-')
acs_0711_work['did_not_work'] = pd.to_numeric(acs_0711_work['did_not_work'])
acs_0711_work['did_not_work'] = acs_0711_work['did_not_work'].apply(work_val)
acs_0711_work['did_not_work'] = acs_0711_work['did_not_work'].apply(divide_100)
acs_0711_work = acs_0711_work.rename(columns = {'did_not_work': 'work_prop'})
acs_0711_work_df = pd.DataFrame(acs_0711_work.groupby('zipcode').work_prop.mean())

acs_0812_work = pd.read_csv('../../Data/Work Status ACS/ACS_12_5YR_S2303/ACS_12_5YR_S2303_with_ann.csv', \
skiprows = 2, usecols = [1,39], names = ['zipcode', 'did_not_work'], \
dtype = {'zipcode': str, 'did_not_work': str}, na_values = ['-', 'N', '(X)'])
acs_0812_work.did_not_work.isnull().value_counts()
acs_0812_work.dropna(axis = 0, inplace = True)
acs_0812_work['zipcode'] = acs_0812_work.zipcode.apply(replace_last_2)
acs_0812_work['did_not_work'] = acs_0812_work['did_not_work'].str.rstrip('+')
acs_0812_work['did_not_work'] = acs_0812_work['did_not_work'].str.rstrip('-')
acs_0812_work['did_not_work'] = pd.to_numeric(acs_0812_work['did_not_work'])
acs_0812_work['did_not_work'] = acs_0812_work['did_not_work'].apply(work_val)
acs_0812_work['did_not_work'] = acs_0812_work['did_not_work'].apply(divide_100)
acs_0812_work = acs_0812_work.rename(columns = {'did_not_work': 'work_prop_0812'})
acs_0812_work_df = pd.DataFrame(acs_0812_work.groupby('zipcode').work_prop_0812.mean())

acs_0913_work = pd.read_csv('../../Data/Work Status ACS/ACS_13_5YR_S2303/ACS_13_5YR_S2303_with_ann.csv', \
skiprows = 2, usecols = [1,39], names = ['zipcode', 'did_not_work'], \
dtype = {'zipcode': str, 'did_not_work': str}, na_values = ['-', 'N', '(X)'])
acs_0913_work.did_not_work.isnull().value_counts()
acs_0913_work.dropna(axis = 0, inplace = True)
acs_0913_work['zipcode'] = acs_0913_work.zipcode.apply(replace_last_2)
acs_0913_work['did_not_work'] = acs_0913_work['did_not_work'].str.rstrip('+')
acs_0913_work['did_not_work'] = acs_0913_work['did_not_work'].str.rstrip('-')
acs_0913_work['did_not_work'] = pd.to_numeric(acs_0913_work['did_not_work'])
acs_0913_work['did_not_work'] = acs_0913_work['did_not_work'].apply(work_val)
acs_0913_work['did_not_work'] = acs_0913_work['did_not_work'].apply(divide_100)
acs_0913_work = acs_0913_work.rename(columns = {'did_not_work': 'work_prop_0913'})
acs_0913_work_df = pd.DataFrame(acs_0913_work.groupby('zipcode').work_prop_0913.mean())

acs_1014_work = pd.read_csv('../../Data/Work Status ACS/ACS_14_5YR_S2303/ACS_14_5YR_S2303_with_ann.csv', \
skiprows = 2, usecols = [1,39], names = ['zipcode', 'did_not_work'], \
dtype = {'zipcode': str, 'did_not_work': str}, na_values = ['-', 'N', '(X)'])
acs_1014_work.did_not_work.isnull().value_counts()
acs_1014_work.dropna(axis = 0, inplace = True)
acs_1014_work['zipcode'] = acs_1014_work.zipcode.apply(replace_last_2)
acs_1014_work['did_not_work'] = acs_1014_work['did_not_work'].str.rstrip('+')
acs_1014_work['did_not_work'] = acs_1014_work['did_not_work'].str.rstrip('-')
acs_1014_work['did_not_work'] = pd.to_numeric(acs_1014_work['did_not_work'])
acs_1014_work['did_not_work'] = acs_1014_work['did_not_work'].apply(work_val)
acs_1014_work['did_not_work'] = acs_1014_work['did_not_work'].apply(divide_100)
acs_1014_work = acs_1014_work.rename(columns = {'did_not_work': 'work_prop_1014'})
acs_1014_work_df = pd.DataFrame(acs_1014_work.groupby('zipcode').work_prop_1014.mean())

acs_work = pd.merge(acs_0711_work_df, acs_0812_work_df, how = 'inner', right_index = True, left_index = True)
acs_work = pd.merge(acs_work, acs_0913_work_df, how = 'inner', right_index = True, left_index = True)
acs_work = pd.merge(acs_work, acs_1014_work_df, how = 'inner', right_index = True, left_index = True)
acs_work.head()



acs_join = pd.merge(acs, acs_work, how = 'inner', right_index = True, left_index = True)
#Fill 332 with 331xx since both are Miami.

acs_join[acs_join.index=='772xx'].iloc[0]
acs_join[(acs_join.index=='500xx') | (acs_join.index=='501xx') | (acs_join.index=='502xx') | (acs_join.index=='503xx')].iloc[:,[0,4]]
for index1, row in acs_join.iterrows():
    if index1 == '332xx':
        acs_join.set_value(index1, ['median_est_income_household', 'work_prop'], acs_join[acs_join.index=='331xx'].iloc[0,[0,4]])
        acs_join.set_value(index1, ['median_est_income_household_0812', 'work_prop_0812'], acs_join[acs_join.index=='331xx'].iloc[0,[1,5]])
        acs_join.set_value(index1, ['median_est_income_household_0913', 'work_prop_0913'], acs_join[acs_join.index=='331xx'].iloc[0,[2,6]])
        acs_join.set_value(index1, ['median_est_income_household_1014', 'work_prop_1014'], acs_join[acs_join.index=='331xx'].iloc[0,[3,7]])

#Assuming Miami zips are similar
acs_join.set_value('332xx', ['median_est_income_household', 'work_prop'], acs_join[acs_join.index=='331xx'].iloc[0,[0,4]])
acs_join.set_value('332xx', ['median_est_income_household_0812', 'work_prop_0812'], acs_join[acs_join.index=='331xx'].iloc[0,[1,5]])
acs_join.set_value('332xx', ['median_est_income_household_0913', 'work_prop_0913'], acs_join[acs_join.index=='331xx'].iloc[0,[2,6]])
acs_join.set_value('332xx', ['median_est_income_household_1014', 'work_prop_1014'], acs_join[acs_join.index=='331xx'].iloc[0,[3,7]])


#Assuming Houston Main zips are similar
acs_join.set_value('772xx', ['median_est_income_household', 'work_prop'], acs_join[acs_join.index=='770xx'].iloc[0,[0,4]])
acs_join.set_value('772xx', ['median_est_income_household_0812', 'work_prop_0812'], acs_join[acs_join.index=='770xx'].iloc[0,[1,5]])
acs_join.set_value('772xx', ['median_est_income_household_0913', 'work_prop_0913'], acs_join[acs_join.index=='770xx'].iloc[0,[2,6]])
acs_join.set_value('772xx', ['median_est_income_household_1014', 'work_prop_1014'], acs_join[acs_join.index=='770xx'].iloc[0,[3,7]])

#Assuming Dallas Main zips are similar

acs_join.set_value('753xx', ['median_est_income_household', 'work_prop'], acs_join[acs_join.index=='752xx'].iloc[0,[0,4]])
acs_join.set_value('753xx', ['median_est_income_household_0812', 'work_prop_0812'], acs_join[acs_join.index=='752xx'].iloc[0,[1,5]])
acs_join.set_value('753xx', ['median_est_income_household_0913', 'work_prop_0913'], acs_join[acs_join.index=='752xx'].iloc[0,[2,6]])
acs_join.set_value('753xx', ['median_est_income_household_1014', 'work_prop_1014'], acs_join[acs_join.index=='752xx'].iloc[0,[3,7]])


#Assuming Govt DC zips are similar
acs_join.set_value('204xx', ['median_est_income_household', 'work_prop'], acs_join[acs_join.index=='203xx'].iloc[0,[0,4]])
acs_join.set_value('204xx', ['median_est_income_household_0812', 'work_prop_0812'], acs_join[acs_join.index=='203xx'].iloc[0,[1,5]])
acs_join.set_value('204xx', ['median_est_income_household_0913', 'work_prop_0913'], acs_join[acs_join.index=='203xx'].iloc[0,[2,6]])
acs_join.set_value('204xx', ['median_est_income_household_1014', 'work_prop_1014'], acs_join[acs_join.index=='203xx'].iloc[0,[3,7]])

acs_join.set_value('202xx', ['median_est_income_household', 'work_prop'], acs_join[acs_join.index=='203xx'].iloc[0,[0,4]])
acs_join.set_value('202xx', ['median_est_income_household_0812', 'work_prop_0812'], acs_join[acs_join.index=='203xx'].iloc[0,[1,5]])
acs_join.set_value('202xx', ['median_est_income_household_0913', 'work_prop_0913'], acs_join[acs_join.index=='203xx'].iloc[0,[2,6]])
acs_join.set_value('202xx', ['median_est_income_household_1014', 'work_prop_1014'], acs_join[acs_join.index=='203xx'].iloc[0,[3,7]])

acs_join.set_value('205xx', ['median_est_income_household', 'work_prop'], acs_join[acs_join.index=='203xx'].iloc[0,[0,4]])
acs_join.set_value('205xx', ['median_est_income_household_0812', 'work_prop_0812'], acs_join[acs_join.index=='203xx'].iloc[0,[1,5]])
acs_join.set_value('205xx', ['median_est_income_household_0913', 'work_prop_0913'], acs_join[acs_join.index=='203xx'].iloc[0,[2,6]])
acs_join.set_value('205xx', ['median_est_income_household_1014', 'work_prop_1014'], acs_join[acs_join.index=='203xx'].iloc[0,[3,7]])


#Assume Sacramento 942xx is similar to Sacramento 958xx
acs_join.set_value('942xx', ['median_est_income_household', 'work_prop'], acs_join[acs_join.index=='958xx'].iloc[0,[0,4]])
acs_join.set_value('942xx', ['median_est_income_household_0812', 'work_prop_0812'], acs_join[acs_join.index=='958xx'].iloc[0,[1,5]])
acs_join.set_value('942xx', ['median_est_income_household_0913', 'work_prop_0913'], acs_join[acs_join.index=='958xx'].iloc[0,[2,6]])
acs_join.set_value('942xx', ['median_est_income_household_1014', 'work_prop_1014'], acs_join[acs_join.index=='958xx'].iloc[0,[3,7]])

#Assume Atlanta 311xx is similar to Atlanta 303xx
acs_join.set_value('311xx', ['median_est_income_household', 'work_prop'], acs_join[acs_join.index=='303xx'].iloc[0,[0,4]])
acs_join.set_value('311xx', ['median_est_income_household_0812', 'work_prop_0812'], acs_join[acs_join.index=='303xx'].iloc[0,[1,5]])
acs_join.set_value('311xx', ['median_est_income_household_0913', 'work_prop_0913'], acs_join[acs_join.index=='303xx'].iloc[0,[2,6]])
acs_join.set_value('311xx', ['median_est_income_household_1014', 'work_prop_1014'], acs_join[acs_join.index=='303xx'].iloc[0,[3,7]])


#Use 891xx for 889xx Las Vegas
acs_join.set_value('889xx', ['median_est_income_household', 'work_prop'], acs_join[acs_join.index=='891xx'].iloc[0,[0,4]])
acs_join.set_value('889xx', ['median_est_income_household_0812', 'work_prop_0812'], acs_join[acs_join.index=='891xx'].iloc[0,[1,5]])
acs_join.set_value('889xx', ['median_est_income_household_0913', 'work_prop_0913'], acs_join[acs_join.index=='891xx'].iloc[0,[2,6]])
acs_join.set_value('889xx', ['median_est_income_household_1014', 'work_prop_1014'], acs_join[acs_join.index=='891xx'].iloc[0,[3,7]])


#Using 900xx for 901xx LA
acs_join.set_value('901xx', ['median_est_income_household', 'work_prop'], acs_join[acs_join.index=='900xx'].iloc[0,[0,4]])
acs_join.set_value('901xx', ['median_est_income_household_0812', 'work_prop_0812'], acs_join[acs_join.index=='900xx'].iloc[0,[1,5]])
acs_join.set_value('901xx', ['median_est_income_household_0913', 'work_prop_0913'], acs_join[acs_join.index=='900xx'].iloc[0,[2,6]])
acs_join.set_value('901xx', ['median_est_income_household_1014', 'work_prop_1014'], acs_join[acs_join.index=='900xx'].iloc[0,[3,7]])


#For 509xx Des Moines, use average of 500-503xx
acs_join.set_value('509xx', ['median_est_income_household', 'work_prop'], acs_join[(acs_join.index=='500xx') | (acs_join.index=='501xx') | (acs_join.index=='502xx') | (acs_join.index=='503xx')].iloc[:,[0,4]].mean())
acs_join.set_value('509xx', ['median_est_income_household_0812', 'work_prop_0812'], acs_join[(acs_join.index=='500xx') | (acs_join.index=='501xx') | (acs_join.index=='502xx') | (acs_join.index=='503xx')].iloc[:,[1,5]].mean())
acs_join.set_value('509xx', ['median_est_income_household_0913', 'work_prop_0913'], acs_join[(acs_join.index=='500xx') | (acs_join.index=='501xx') | (acs_join.index=='502xx') | (acs_join.index=='503xx')].iloc[:,[2,6]].mean())
acs_join.set_value('509xx', ['median_est_income_household_1014', 'work_prop_1014'], acs_join[(acs_join.index=='500xx') | (acs_join.index=='501xx') | (acs_join.index=='502xx') | (acs_join.index=='503xx')].iloc[:,[3,7]].mean())


#Assume 555xx is 553xx for minneapolis west.

acs_join.set_value('555xx', ['median_est_income_household', 'work_prop'], acs_join[acs_join.index=='553xx'].iloc[0,[0,4]])
acs_join.set_value('555xx', ['median_est_income_household_0812', 'work_prop_0812'], acs_join[acs_join.index=='553xx'].iloc[0,[1,5]])
acs_join.set_value('555xx', ['median_est_income_household_0913', 'work_prop_0913'], acs_join[acs_join.index=='553xx'].iloc[0,[2,6]])
acs_join.set_value('555xx', ['median_est_income_household_1014', 'work_prop_1014'], acs_join[acs_join.index=='553xx'].iloc[0,[3,7]])


#Only the 1014 df seems to be significantly different when doing a paired sample t-test.
#Based on this, using this for 2014 forward is what I'll do. Wonder if this will help at all.

'''With the acs five year estimates - realize that it is not the best call but five year estimates beyond the 2011 end point are gigs of data.
I will use these to create a new column getting the scaled diff in annual income by zip for borrowers.
The acs_0711 data will be used through 2011, and then the 2012 and 2013 ones for those respective years.
2014 for the rest.
'''
#drop zip 969xx and 008xx because this is for the Marshall Islands and Virgin Islands. Do not have ACS data for these.
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

#IRS zips
loan_only = loan_only[(loan_only.zip_code != '055xx') & (loan_only.zip_code != '192xx') & (loan_only.zip_code != '375xx') & \
(loan_only.zip_code != '399xx') & (loan_only.zip_code != '459xx') & (loan_only.zip_code != '569xx') & (loan_only.zip_code != '649xx') & \
(loan_only.zip_code != '733xx') & (loan_only.zip_code != '842xx') & (loan_only.zip_code != '938xx')]



loan_only['median_est_income_hh'] = np.NAN
loan_only['work_prop'] = np.NAN

acs_join[acs_join.index == '772xx'].iloc[0,[1,5]]
loan_only.iloc[0][['median_est_income_hh', 'work_prop']]

acs_join[acs_join.index ==loan_only.zip_code[0]].iloc[0][]
#Median income lookup. Has to be a better way.
for index1, row in loan_only.iterrows():
    try:
        if row.year <= 2011 and np.isnan(row.median_est_income_hh)==True:
            loan_only.set_value(index1, 'median_est_income_hh', acs_join[acs_join.index == row.zip_code].iloc[0,0])
            #print index1
        if (row.year == 2012) and np.isnan(row.median_est_income_hh)==True:
            loan_only.set_value(index1, ['median_est_income_hh', 'work_prop'], acs_join[acs_join.index == row.zip_code].iloc[0,1])
            #print index1
        if (row.year == 2013) and np.isnan(row.median_est_income_hh)==True:
            loan_only.set_value(index1, ['median_est_income_hh', 'work_prop'], acs_join[acs_join.index == row.zip_code].iloc[0,2])
            #print index1
        if (row.year >= 2014) and np.isnan(row.median_est_income_hh)==True:
            loan_only.set_value(index1, ['median_est_income_hh', 'work_prop'], acs_join[acs_join.index == row.zip_code].iloc[0,3])
            #print index1
    except IndexError:
        print index1
        pass
    except ValueError:
        print index1
        pass

#Work proportion. Has to be better way,
for index1, row in loan_only.iterrows():
    try:
        if row.year <= 2011 and np.isnan(row.work_prop)==True:
            loan_only.set_value(index1, 'work_prop', acs_join[acs_join.index == row.zip_code].iloc[0,4])
            #print index1
        if (row.year == 2012) and np.isnan(row.work_prop)==True:
            loan_only.set_value(index1, 'work_prop', acs_join[acs_join.index == row.zip_code].iloc[0,5])
            #print index1
        if (row.year == 2013) and np.isnan(row.work_prop)==True:
            loan_only.set_value(index1, 'work_prop', acs_join[acs_join.index == row.zip_code].iloc[0,6])
            #print index1
        if (row.year >= 2014) and np.isnan(row.work_prop)==True:
            loan_only.set_value(index1,'work_prop', acs_join[acs_join.index == row.zip_code].iloc[0,7])
            #print index1
    except IndexError:
        print index1
        pass
    except ValueError:
        print index1
        pass


loan_only[loan_only.year<=2011][['year', 'zip_code', 'median_est_income_hh', 'work_prop']]
acs_join[acs_join.index ==loan_only.loc[4].zip_code]
#loan_only.set_value(349999, 'median_est_income_hh', acs_1014_df[acs_1014_df.index == loan_only.ix[349999].zip_code].iloc[0,0])
#loan_only.set_value(431886, 'median_est_income_hh', acs_1014_df[acs_1014_df.index == loan_only.ix[431886].zip_code].iloc[0,0])
#loan_only.set_value(492724, 'median_est_income_hh', acs_1014_df[acs_1014_df.index == loan_only.ix[492724].zip_code].iloc[0,0])
#loan_only.set_value(558590, 'median_est_income_hh', acs_1014_df[acs_1014_df.index == loan_only.ix[558590].zip_code].iloc[0,0])
#loan_only.set_value(620956, 'median_est_income_hh', acs_1014_df[acs_1014_df.index == loan_only.ix[620956].zip_code].iloc[0,0])
#loan_only.set_value(874148, 'median_est_income_hh', acs_1014_df[acs_1014_df.index == loan_only.ix[874148].zip_code].iloc[0,0])
#loan_only[loan_only.median_est_income_hh.isnull()==True]

#Engineer the magnitude diff between annual income and median.
loan_only['magnitude_ann_inc_to_median'] = loan_only['annual_inc'].divide(loan_only['median_est_income_hh'])
loan_only.work_prop.isnull().value_counts()
loan_only.median_est_income_hh.isnull().value_counts()
loan_only.median_est_income_hh.head()


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
loan_only_no_outliers = loan_only[~loan_only.id.isin(loan_only_outliers.id)]

loan_only_no_outliers.columns.values

'''Saving whole loan file.'''
loan_only.to_csv('../../Data/Loans_stage2.csv', sep = ',', index = True, index_label = 'row_num')

'''Loan file where max date is 2011 and term is 60 months.'''
loans_5yr_until_2011 = loan_only_no_outliers[(loan_only_no_outliers.date <= '2011-12-31') & (loan_only_no_outliers.term == '60 months')]

'''Loan file where max date is 2013-12-31 and term is 36 months.'''
loans_3yr_until_2013 = loan_only_no_outliers[(loan_only_no_outliers.date <= '2013-12-31') & (loan_only_no_outliers.term == '36 months')]

'''Loan file where min date is 2012-01-01 and term is 60 months'''
loans_5yr_2012on = loan_only_no_outliers[(loan_only_no_outliers.date > '2011-12-31') & (loan_only_no_outliers.term == '60 months')]

'''Loan file where min date is 2014-01-01 and term is 36 months'''
loans_3yr_2014on = loan_only_no_outliers[(loan_only_no_outliers.date > '2013-12-31') & (loan_only_no_outliers.term == '36 months')]

'''Outliers file for 3 year loans.'''
loan_outliers_3yr = loan_only_outliers[loan_only_outliers.term == '36 months']

'''Outliers file for 5 year loans'''
loan_outliers_5yr = loan_only_outliers[loan_only_outliers.term == '60 months']

'''Saving files... Next step is to train_test_split, Robust Scale, TFIDF (only for 2013 data on), and build models
Will also have to separate out the joint applications.'''

loans_3yr_2014on.to_csv('../../Data/second_run/loanbook_3yr_2014on.csv', sep = ',', index = True, index_label = 'row_num')
loans_3yr_until_2013.to_csv('../../Data/second_run/loanbook_3yr_until_2013.csv', sep = ',', index = True, index_label = 'row_num')
loans_5yr_2012on.to_csv('../../Data/second_run/loanbook_5yr_2012on.csv', sep = ',', index = True, index_label = 'row_num')
loans_5yr_until_2011.to_csv('../../Data/second_run/loanbook_5yr_until_2011.csv', sep = ',', index = True, index_label = 'row_num')

loan_outliers_3yr.to_csv('../../Data/second_run/outliers_3yr.csv', sep = ',', index = True, index_label = 'row_num')
loan_outliers_5yr.to_csv('../../Data/second_run/outliers_5yr.csv', sep = ',', index = True, index_label = 'row_num')
