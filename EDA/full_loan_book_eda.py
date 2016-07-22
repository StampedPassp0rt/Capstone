'''Exploratory Data Analysis

In this file, I'll create the full loan book for Lending Club. I'll probably do
some of the following:

    1) Create year and month variables to separate data into monthly or annual tranches
    for comparison;

    2) Collate some basic descriptive statistics;

'''
#libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

'''Obtain the datasets...'''

#rejected loans
reject_combined = pd.read_csv('../../Data/RejectStatsCombined.csv')

reject_combined.columns.values

loan.head()

reject_combined.head()

#approved loans
loan = pd.read_csv('../../Data/LoanStats.csv', usecols = ['loan_amnt', 'issue_d', 'title', \
'FICO_mid', 'dti', 'emp_length', 'policy_code', 'approved', 'zip_code', 'addr_state'], \
dtype = {'id': str, 'member_id': str, 'sub_grade': str, 'desc': str, \
'verification_status_joint': str})

loan.rename(columns = {'loan_amnt': 'amnt_request', 'issue_d': 'date', 'FICO_mid': 'Risk_Score', \
'addr_state': 'State'}, inplace = True)

reject_combined.rename(columns = {'Amount Requested': 'amnt_request', 'Application Date': 'date',\
'app_date': 'date',
'Loan Title': 'title', 'Debt-To-Income Ratio': 'dti', 'Zip Code': 'zip_code', \
'Employment Length': 'emp_length', 'Policy Code': 'policy_code'}, inplace = True)

reject_combined.info()

#get the dates into datetime format...
loan['date'] = pd.to_datetime(loan['date'], infer_datetime_format = True)
reject_combined['date'] = pd.to_datetime(reject_combined['date'], infer_datetime_format = True)

#get rid of row_num column
reject_combined.drop('row_num', axis = 1, inplace = True)
loan.columns.values

reject_combined.columns.values

fullbook = pd.concat([loan, reject_combined], ignore_index = True)

fullbook.dti.describe()

#Adding month and year.
#Adding in year and month columns...
#for index1, row in fullbook.iterrows():
#    fullbook.set_value(index1, 'year', row.date.year)
#    fullbook.set_value(index1, 'month', row.date.month)


#Exporting full loan book
fullbook.to_csv('../../Data/full_loan_book.csv', sep = ',', index = True, index_label = 'row_num')
