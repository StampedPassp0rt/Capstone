'''Step 1 of the Overall Workflow: Cleaning the Data (now that we have it).

Despite what the Lending Club data dictionary says, there are only two datasets,
not three.

    1) Loan Statistics for loans grants. These are installment loans.
    2) Rejection Statistics - the data on loan applications that were rejected.

    https://www.lendingclub.com/info/download-data.action

While I explored the data initially from Kaggle, the data directly from Lending
Club is just better. The Kaggle Data is missing several important credit variables
that Lending Club has in its data dictionary for Loan Stats, such as FICO.

There are at least four types of loans:
    1) 36 month loans: Single borrower, and joint borrower (co-signer)
    2) 60 month loans: Same as above.

From my analysis below, there are 509 joint approved loans. This feels too small, but
I might produce a model.

This is my second go-around looking at the data. I am going to do more EDA, but to
do that, I need to join the data together. That is this file.

EDA I want to do includes:
1) separating outliers from the majority when looking at DTI, annual_inc.
2) Checking the distributions of outliers and rest.
3) Dimensionality reduction and clustering. See if Segments of Customers, like
by sub-grade, emerge.
    -Also may want to separate out the ones where they do not have any delinq, etc.
    from those that do. Different data, different models.
    -Also check to see if that data looks separable in any way after reduction.

###############################################################################
'''

#Loading libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns
import os
import string
from sklearn.preprocessing import LabelEncoder

%matplotlib inline

#reject_combined.columns.values

'''Dataset 1: Loan Statistics 2007-15'''
#Importing the loan data book...
loanstatsa = pd.read_csv('../../LoanStats/LoanStats3a_securev1.csv', skiprows = 1, dtype = \
{'id': str, 'member_id': str, 'sub_grade': str, 'desc': str, 'verification_status_joint': str})
loanstatsb = pd.read_csv('../../LoanStats/LoanStats3b_securev1.csv', skiprows = 1, dtype = \
{'id': str, 'member_id': str, 'sub_grade': str, 'desc': str, 'verification_status_joint': str})
loanstatsc = pd.read_csv('../../LoanStats/LoanStats3c_securev1.csv', skiprows = 1, dtype = \
{'id': str, 'member_id': str, 'sub_grade': str, 'desc': str, 'verification_status_joint': str})
loanstatsd = pd.read_csv('../../LoanStats/LoanStats3d_securev1.csv', skiprows = 1, dtype = \
{'id': str, 'member_id': str, 'sub_grade': str, 'desc': str, 'verification_status_joint': str})

loanstatsa.head(2)
loanstatsb.head(2)
loanstatsc.head(2)
loanstatsd.head(2)

loanstatsd.fico_range_high.isnull().value_counts()

loan = pd.concat([loanstatsa, loanstatsb, loanstatsc, loanstatsd], ignore_index = True)
#Note this only includes the loan stats itself, not BrowseNotes or the Rejected Loans.

'''Pre-Data Cleaning:'''

#head of csv.
loan.head(10)
loan.info()

loan['issue_d'] = pd.to_datetime(loan['issue_d'], infer_datetime_format = True)

'''Quickly exploring if loan description could be a useful feature.'''
desc_df = loan[['issue_d', 'desc']]
desc_df = desc_df.assign(desc_ind = desc_df.desc.isnull())
desc_df.drop('desc', axis = 1, inplace = True)
desc_df.set_index('issue_d', inplace = True)
plt.figure(figsize = (10, 5))
desc_df.resample('M').mean().plot(kind = 'line', fontsize = 12)
plt.title('Proportion of Loans without User-Provided Description, Over Time', fontsize = 15)
plt.ylabel('Proportion, bounded between 0 and 1')
plt.xlabel('Year of issuance')
plt.savefig('../EDA/loan_description_absence_by_year.png')
'''Short answer is that loan description is not a useful feature because it practically disappears.'''

purpose_df = loan[['issue_d', 'purpose']]
purpose_df = purpose_df.assign(purpose_ind = purpose_df.purpose.isnull())
purpose_df.drop('purpose', axis = 1, inplace = True)
purpose_df.set_index('issue_d', inplace = True)
plt.figure(figsize = (10, 5))
purpose_df.resample('M').mean().plot(kind = 'line', fontsize = 12)
plt.title('Proportion of Loans without Purpose, Over Time', fontsize = 15)
plt.ylabel('Proportion, bounded between 0 and 1')
plt.xlabel('Year of issuance')

'''Purpose, on the other hand, has always been present.'''


#Column 55
loan.verification_status_joint.value_counts() #verification of joint income if a co-borrow. Null for so many b/c
                #no co-borrower. Not sure if useful

#dti - Debt Payments to Income. What does the distribution look like for first 75%, and last quartile?
loan.dti.describe()

#Null DTIs look like they are the total securitization amounts...
loan.dti.isnull().value_counts()
pd.DataFrame((loan[loan.dti.isnull()==True][['id', 'issue_d', 'term']]))

#Subsetting out null DTIs.
loan = loan[loan.dti.notnull()==True]
loan.dti.isnull().value_counts() #Checking that subsetted out.

loan[loan.dti == loan.dti.max()]

def plot_hist(data,feat):
    plt.figure(figsize = (10,5))
    ax = data[feat].plot(kind = 'hist', alpha = 0.3, normed = True)
    plt.title(' '.join([feat.capitalize(), 'Distribution']))
    return ax

#Let's see what the distribution of debt to income looks when stripping out a good portion of the top quartile of values.
#Looking for a cutoff... 100 looks good. 50 even better!


plot_hist(loan[loan.dti < 100], 'dti')

plot_hist(loan[loan.dti < 50], 'dti')

loan[loan.dti>=50].dti.describe() #Note, only 44 loans above a dti of 50.

#What's the time dist of these loans?
loan[loan.dti>=150].plot(kind = 'scatter', x= 'dti', y = 'fico_range_low')

#1) What we see here is that dti outliers are present Oct to Dec 2015 only.
#As I did not pull 2016 data, that means they could be there too.
#ALso, I have a feeling that this ratio is in percentage points. So 50 really is 50%.
#9999 is 9999%, or 99.99x.
loan[(loan.dti>= 50) & (loan.dti<=150)][['issue_d', 'dti', 'term', 'emp_title', 'purpose', 'loan_status']].set_index('issue_d')

loan[loan.dti>=150][['issue_d', 'dti', 'term', 'emp_title', 'purpose', 'loan_status']].set_index('issue_d')
plot_hist(loan[(loan.dti <= 150) & (loan.dti >= 50)], 'dti') #Not normal, also like tail of normal dist.
'''For DTI, since DTI <= 50 for most of period of study, if implementing model on
website, just put in an if/then that causes it to err if it gets a DTI > 50.
'''

#emp_length
loan.emp_length.value_counts()

loan.emp_length.unique()

def hist_empl_length(data, groupcol, group1, group2, feat):
    plt.figure(figsize = (10,5))
    data[data[groupcol] == group1][feat].plot(kind = 'hist', alpha = 0.3, \
    normed = True)
    data[data[groupcol] == group2][feat].plot(kind = 'hist', alpha = 0.3, \
    normed = True)
    plt.legend([group1, group2])
    plt.xlabel(feat.capitalize())
    plt.title(' '.join([feat.capitalize(), 'Distribution by', group1, 'and', group2]))


for fico in ['fico_range_low', 'fico_range_high']:
    hist_empl_length(loan, 'emp_length','< 1 year', 'n/a', fico)

#Risk_Score - this variable doesn't exist. Rather we're given a FICO range. Frustrating!!!
for i in loan[loan.fico_range_high.isnull()].index:
    print loan.ix[i]


#null values appear to not be for actual loans but referring to total loans funded under a policy code.

#Saving mid-way through
loan.to_csv('../../Data/LoanStats.csv', sep = ',', index = True, index_label = 'row_num')

#loan = pd.read_csv('../../Data/LoanStats.csv', dtype = {'id': str, \
#'member_id': str, 'sub_grade': str, 'desc': str, 'verification_status_joint': str})
#loan.head()

loan.rename(columns = {'loan_amnt': 'amnt_request', 'issue_d': 'date', \
'addr_state': 'State'}, inplace = True)

#loan.drop(['row_num', 'row_num.1', 'row_num.1'], axis = 1, inplace = True)



'''Data Cleaning - Joint Accounts
1) dti_joint: 2 missing values. DTI is there. I can use dti.
'''

'''Data Cleaning....

Features to include:
    1) annual_inc
    2) earliest_cr_line - looks like this is from time of application.
    3) home_ownership - at time of app. No missing values...
    4) installment - the payment at time of approval. Definitely interesting. No missings.
    5) int_rate - probably useful.
    6) verification_status - was the income verified. Probably good to know...
    7) FICO Range (hence FICO_mid or Risk_Score is a good feature)
    8) Lending Club Grade and SubGrade - map to numerical, or label encode it.
    9) Term of Loan (36 or 60 month)
    10) Loan amount - the funded amount column.
    11) Purpose
    12) Employment length
    13) Title (assuming this is shown)
    14) dti
    15) Job Title
    16) Credit History: so these stats are at time of underwriting:
        -open credit lines
        -total credit lines
        -rev credit Balance
        -rev line utilization
        -inquiries in last 6 months
        -accounts now delinquent
        -delinquent amount
        -delinquencies last 2 years
        -months since last delinquency - construct a categorical from this.
        -public records on file
        -months since last record (n/a if zero records; might only be useful for those
            w/o records - so might first try to classify without this variable)
        -months since last derogatory  (significant - construst a derog/no derog var)
        -collections excluding medical

1) Loan term (length) should probably be recoded numerically. Actually, I will
keep it as is for now since it is a categorical.

2) Features from application time with missing values:

    annual_inc: Four missing ones, and they do not appear to be joint borrowers.
                They all also have excellent Risk Scores, but also all employed
                less than one year. The titles are related to moving, buying a car,
                education. Employee title missing:
                    rows: 42449, 42450, 42480, and 42533 for future reference.
                My gut tells me that these are students or people just starting out in life.
                I could KNN imput, or just put zero for income.


2) Debt to Income (dti): Odd that the max value is 9999. Distribution looks more
familiar (Gaussian) when cutoff is 100% debt payments per month to income per month.

    Further exploration showed that above 50 for dti, there are outliers. Between
    50 and 150, it looks like the tail of a normal distribution, but just very far
    out. Beyond that, there are two genuine outliers.

    Note that all of these occur in 2015 - Oct through Dec. For now, have logic
    that if DTI > 50, can not decision it. Besides, I likely will not use 2015 data
    for training, but purely for testing.

    So I might want to see if those loans cluster with other ones based on Risk_Score,
    or another credit metric. Do a DBSCAN in clustering script.

3) Employment Length: FICO scores are relatively high (prime and super-prime) for both
                    < 1 year and n/a. Keep n/a as its own value.
'''
#date
#loan['date'] = pd.to_datetime(loan['date'], infer_datetime_format = True)

#FICO data
    ##Let's get rid of the null FICOs...
#mask_fico = loan['fico_range_low'].notnull().value_counts()
#loan = loan[mask_fico]

    ##Let's create a variable that is the average of the FICO range for each row.
loan = loan.assign(Risk_Score = np.mean([loan.fico_range_low, loan.fico_range_high], axis = 0))

#Employment Length
#Get rid of years string.
yearstrip = lambda x: string.rstrip(x, 'year[s]')
loan['emp_length'] = loan['emp_length'].apply(yearstrip)
loan['emp_length'] = loan['emp_length'].str.strip()
loan['emp_length'] = loan['emp_length'].str.strip('+')


#Convert '< 1 year' to zero...
for index1, row in loan.iterrows():
    if row['emp_length'] == '< 1':
        loan.set_value(index1, 'emp_length', '0')
    elif row['emp_length'] == 'n/':
        loan.set_value(index1, 'emp_length', 'n/a')
#    elif row['Employment Length'] == '10+':
#        reject_combined.set_value(index1, 'Employment Length', '10')

loan.emp_length.unique()
#Loan Title
#replace _ with space...
replace_underscore = lambda x: string.replace(x, '_', ' ')

loan['title'] = loan['title'].astype(str)

loan['title'] = loan['title'].apply(replace_underscore)
#reject_combined['Loan Title'] = reject_combined.str.lower()

loan = loan.assign(approved = 1)

loan.columns.values

#earliest_cr_line. convert to datetime.
loan['earliest_cr_line'] = pd.to_datetime(loan.earliest_cr_line, infer_datetime_format = True)

#annual_inc - data provided at time of application. Filling in with zero.
print loan[loan.annual_inc.isnull() == True]['emp_title']
loan.annual_inc.fillna(0, axis = 0, inplace = True)
#Current data - chargeoff within 12 months... 145 nulls.
loan.chargeoff_within_12_mths.isnull().value_counts()


#All of the missing chargeoffs have missing mths since last major derog, meaning no derog.
#Hence, likely no chargeoff, so encode this as 0
loan[loan.chargeoff_within_12_mths.isnull()==True]['mths_since_last_major_derog'].isnull().value_counts()
loan.chargeoff_within_12_mths.fillna(0, axis = 0, inplace = True)

#The variable is a number. Interestingly, for the majority of LC approved loans, no chargeoffs in past 12 months.
loan[loan.chargeoff_within_12_mths.isnull()==False]['chargeoff_within_12_mths'].describe()

#collections 12 months current. # of collections in past 12 months.
loan.collections_12_mths_ex_med.isnull().value_counts()

#All of the missing values for collections are those where there were no derog records, so impute as zero collections...
loan[loan.collections_12_mths_ex_med.isnull()==True]['mths_since_last_major_derog'].isnull().value_counts()
loan.collections_12_mths_ex_med.fillna(0, axis = 0, inplace = True)

#Earliest Credit Line - no data from others, other than zero revol balance. Probably impute this as LC app date.
loan.earliest_cr_line.isnull().value_counts()
#Interesting that this missing data is from LC's early days.
loan[loan.earliest_cr_line.isnull()==True]['date']
for index1, row in loan[loan.earliest_cr_line.isnull()== True].iterrows():
    loan.set_value(index1, 'earliest_cr_line', row['date'])

loan.home_ownership.isnull().value_counts()

loan.installment.isnull().value_counts()
loan.int_rate.isnull().value_counts()
loan.verification_status.value_counts()


#months since last delinquency - will try using as is...
loan.mths_since_last_delinq.isnull().value_counts()
#mnths_since_last_delinq also tells me effectively for the n/a that they never were delinquent.
#could encode as binary. Should.
#If this becomes a valuable variable, then will have to build into the module
#logic to check if mths_since_last_delinq is null or not, and build dummy var accord.
loan = loan.assign(ever_delinq_dummy = loan.mths_since_last_delinq.notnull())

#nulls for delinq_2yrs should be zero since never delinquent. Ditto this logic for a predictor cleaner.
for index1, row in loan[loan.delinq_2yrs.isnull()== True].iterrows():
    loan.set_value(index1, 'delinq_2yrs', 0)

#Nulls for inquiries last six months are likely the same people who had no revol balance, etc.
#Code as zero.
loan.inq_last_6mths.isnull().value_counts()
loan.inq_last_6mths.fillna(0, axis = 0, inplace = True)

plot_hist(loan[loan.inq_last_6mths<2], 'inq_last_6mths')

plot_hist(loan[loan.inq_last_6mths<=10], 'inq_last_6mths')
loan_no_ru_nulls[loan_no_ru_nulls.inq_last_6mths>=8].year.value_counts()

#open credit lines - the same 29 w/o credit histories... Assume LC will be first account.
loan.open_acc.isnull().value_counts()

for index1, row in loan[loan.open_acc.isnull()==True].iterrows():
    loan.set_value(index1, 'open_acc', 0)
#total credit lines - same as above. I suppose using fillna as logic is easier.
loan.total_acc.fillna(0, axis = 0, inplace = True)
loan.total_acc.isnull().value_counts()

#Same 29 have no delinq_amnt. Fill with zero.
loan.delinq_amnt.fillna(0, axis = 0, inplace = True)

#rev line utilization: 1) Strip percentage; 2) convert to numeric; 3) divide by 100 to scale.
loan['revol_util'] = loan['revol_util'].str.rstrip('%')
divide_100 = lambda x: float(x)/100
loan['revol_util'] = pd.to_numeric(loan['revol_util'])
loan['revol_util'] = loan['revol_util'].apply(divide_100)

#Another variable with outliers...
loan[loan.revol_util.isnull()==False]['revol_util'].describe()
#the ones where revol_bal = 0 can have util be zero.
for index1, row in loan[loan.revol_util.isnull()==True].iterrows():
    if row['revol_bal']==0:
        loan.set_value(index1, 'revol_util', 0)

#1.1 to 1.2 make sense as cutoffs, because CC companies allow a certain amount of over util.
plot_hist(loan[loan.revol_util <= 1.2], 'revol_util')

loan[loan.revol_util>1.2][['revol_util', 'dti', 'date', 'purpose', 'title', 'term', 'loan_status']].sort_values('date')
'''What we see is that revol_util went above 1.2 for loans approved in 2013.
Makes you wonder if LC was stretching things... I would be considering using 2013 data for training
and validation among itself. And if it worked well, and better than the 2012 model on it,
it would be what I would use for production.

revol_util <2 is a logical cutoff. But why don't I take a look and see how it ends up cluster-wise
with the other credit data? Just like I was going to do with the high dtis? Especially since most of these
seem to have dtis within reason.

Tells me that when doing PCA on 2013 loans, check with a 1.2 cutoff and a 2 cutoff.'''


#Drop the others for revol_util = NaN. Not sure of a rule, and knn imputation
#could be too slow.
loan[loan.revol_util.isnull()]

loan_no_ru_nulls = loan[loan.revol_util.notnull()==True]

loan_no_ru_nulls.revol_util.isnull().value_counts()

#LC sub_grade. Essentially a continuous variable, in how I can treat it. Label Encode.
label_subgrade = LabelEncoder()
loan_no_ru_nulls = loan_no_ru_nulls.assign(sub_grade_encoded = label_subgrade.fit_transform(loan_no_ru_nulls.sub_grade))
label_subgrade.classes_

#Fill with zero for NaNs
loan_no_ru_nulls.acc_now_delinq.fillna(0, axis = 0, inplace = True)

#Convert interest rate to numeric.
loan_no_ru_nulls['int_rate'] = loan_no_ru_nulls['int_rate'].str.rstrip('%')
loan_no_ru_nulls['int_rate'] = pd.to_numeric(loan_no_ru_nulls['int_rate'])
loan_no_ru_nulls['int_rate'] = loan_no_ru_nulls['int_rate'].apply(divide_100)


#public records on file - same as the ones with no prior credit
loan_no_ru_nulls.pub_rec.isnull().value_counts()
loan_no_ru_nulls.pub_rec.fillna(0, axis = 0, inplace = True)

#I could assume that public record = pub rec bankruptcies. But since not public facing, not going to adjust for now.
loan_no_ru_nulls[loan_no_ru_nulls.pub_rec_bankruptcies.isnull()==True]['pub_rec']

#Months since last record...All the N/As have zero public records.
loan_no_ru_nulls[loan_no_ru_nulls.mths_since_last_record.isnull()==True]['pub_rec'].describe()

#public_record_present dummy.
public_record_ind = loan_no_ru_nulls.mths_since_last_record.notnull()
loan_no_ru_nulls = loan_no_ru_nulls.assign(public_record_ind = public_record_ind)

#months since last derogatory  (significant - construst a derog/no derog var)
loan_no_ru_nulls.mths_since_last_major_derog.isnull().value_counts()

#derog_record_present
deroq_record_ind = loan_no_ru_nulls.mths_since_last_major_derog.notnull()
loan_no_ru_nulls = loan_no_ru_nulls.assign(derog_record_present = deroq_record_ind)


#Term of Loan. Get rid of leading white space.
loan_no_ru_nulls['term'] = loan_no_ru_nulls.term.str.lstrip()

#purpose of loan. No nulls. Categorical variable. Dummify.
#for giving a fresh prediction array, will have to create logic to check the text
#in purpose column against the dummies, and if match, then it's that dummy, and the rest are zero.
purpose_dummies = pd.get_dummies(loan_no_ru_nulls.purpose, drop_first = True)
loan_no_ru_nulls = pd.merge(loan_no_ru_nulls, purpose_dummies, how = 'inner', right_index = True, left_index = True)

#home_ownership - categorical. Dummify
home_dummies = pd.get_dummies(loan_no_ru_nulls.home_ownership, drop_first = True)
loan_no_ru_nulls = pd.merge(loan_no_ru_nulls, home_dummies, how = 'inner', right_index = True, left_index = True)
loan_no_ru_nulls.info()
'''emp_title. Pre 2013, this was the company where they worked. Post 2013, a job title.
It's different data, and unless it's guaranteed no defaults occur in late stage of a loan (worth checking out)
data from 2013, would not have run the full 36 month term (shorter) to judge default.
In short, keep as is for now. I can separate out 2013-09-23 data anyway and
try to process the text data  with a simple bag of words tf-idf. But look and see
what is possible with Spacy

I would prefer to do this in another file.
'''
loan[loan.date<'2013-09-23']['emp_title'].value_counts()



'''Looking at the Joint Loans pre Cleaning'''
#annual_inc joint.
loan_no_ru_nulls.annual_inc_joint.isnull().value_counts()
#All else equal, two missing dtis...
loan_no_ru_nulls.dti_joint.isnull().value_counts()
loan_no_ru_nulls.verification_status_joint.isnull().value_counts()
for index1, row in loan_no_ru_nulls[loan_no_ru_nulls.annual_inc_joint.isnull() == False][loan_no_ru_nulls.dti_joint.isnull() == True].iterrows():
    loan_no_ru_nulls.set_value(index1, 'dti_joint', 0)


#Engineering some outcome vars.

#Difference b/t requested and paid. Essentially the gross profit/return.
loan_no_ru_nulls['gross_return'] = loan_no_ru_nulls['total_pymnt'] - loan_no_ru_nulls['funded_amnt']
loan_no_ru_nulls['gross_return_scaled'] = loan_no_ru_nulls['gross_return'].divide(loan_no_ru_nulls['funded_amnt'])

loan_no_ru_nulls.title.fillna('none', axis = 0, inplace = True)
loan_no_ru_nulls.emp_title.fillna('none', axis = 0, inplace = True)
#Engineering the NaNs as high values. Fillna
loan_no_ru_nulls['mths_since_last_delinq'].fillna(984, axis = 0, inplace = True)
loan_no_ru_nulls['mths_since_last_record'].fillna(984, axis = 0, inplace = True)
loan_no_ru_nulls['mths_since_last_major_derog'].fillna(984, axis = 0, inplace = True)

'''Engineer variable for difference between earliest_cr_line and application date'''
loan_no_ru_nulls['diff_earliest_app_date'] = loan_no_ru_nulls['date'] - loan_no_ru_nulls['earliest_cr_line']
convert_to_dayint = lambda x: x.days
loan_no_ru_nulls['diff_earliest_app_date'] = loan_no_ru_nulls['diff_earliest_app_date'].apply(convert_to_dayint)
loan_no_ru_nulls['diff_earliest_app_date'].dtype

'''Create year and month filters'''
year_filter = lambda x: x.year
month_filter = lambda x: x.month

loan_no_ru_nulls['year'] = loan_no_ru_nulls['date'].apply(year_filter)
loan_no_ru_nulls['month'] = loan_no_ru_nulls['date'].apply(month_filter)

#Dummifying other vars.
loan_no_ru_nulls.emp_length = loan_no_ru_nulls.emp_length.astype(str)

dummies_stateetc = pd.get_dummies(loan_no_ru_nulls[['emp_length','State', 'verification_status_joint', 'verification_status']], drop_first = True)
dummies_stateetc.columns.unique()


'''Engineer Loan Status to have the classes I want to predict.
Default/Chargeoff grouped together as 1, Fully Paid as 0. Simplest model, first looking at fully run loans.'''


loan_no_ru_nulls['loan_status_encoded'] = loan_no_ru_nulls.loan_status.replace(to_replace = {'Fully Paid': 0, 'Current' : 1, 'In Grace Period' : 1,\
'Late (31-120 days)': 2, 'Late (16-30 days)': 2, 'Default': 3, 'Charged Off': 3,\
'Does not meet the credit policy. Status:Fully Paid': 0,\
 'Does not meet the credit policy. Status:Charged Off': 3})

#Annual income. Check outliers. By date, they are all throughout the data.
plot_hist(loan_no_ru_nulls, 'annual_inc')
ax = plot_hist(loan_no_ru_nulls[(loan_no_ru_nulls.annual_inc>200000)&(loan_no_ru_nulls.annual_inc<1000000)], 'annual_inc')
plt.title('Distribution of Annual Incomes, $200K-$1MM', fontsize = 15)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: ''.join(['$',format(int(x), ',')])))
plt.savefig('../EDA/annual_income_dist_outliers_200k_1MM.png')

ax = plot_hist(loan_no_ru_nulls[(loan_no_ru_nulls.annual_inc<200000)], 'annual_inc')
plt.title('Distribution of Annual Incomes, Cut Off at $200K', fontsize = 15)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: ''.join(['$',format(int(x), ',')])))
plt.savefig('../EDA/annual_income_dist_without_sig_outliers.png')

ax = plot_hist(loan_no_ru_nulls[(loan_no_ru_nulls.annual_inc>1000000)], 'annual_inc')
plt.title('Distribution of Annual Incomes, $1MM+', fontsize = 15)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: ''.join(['$',format(int(x), ',')])))
plt.savefig('../EDA/annual_income_dist_outliers_1MM.png')


#skewed left dist. Don't cut out the outliers,
plot_hist(loan_no_ru_nulls[loan_no_ru_nulls.annual_inc<=200000], 'annual_inc')

#open accounts - outliers here probably matter a lot too. This will be a cutoff.
plot_hist(loan_no_ru_nulls[loan_no_ru_nulls.open_acc <=25], 'open_acc')
loan_no_ru_nulls.open_acc.describe()
#This is an interesting cluster of people...
loan_no_ru_nulls[loan_no_ru_nulls.open_acc >30][['open_acc', 'annual_inc', 'dti', 'loan_status_encoded']].describe()

#Given how few there are, wonder if even gradient boosting will help.
loan_no_ru_nulls[loan_no_ru_nulls.open_acc>25].year.value_counts()

plot_hist(loan_no_ru_nulls[loan_no_ru_nulls.total_acc <80], 'total_acc')
loan_no_ru_nulls.total_acc.describe()
loan_no_ru_nulls[loan_no_ru_nulls.total_acc >80][['total_acc', 'open_acc']]
plot_hist(loan_no_ru_nulls[loan_no_ru_nulls.total_acc >100], 'total_acc')
loan_no_ru_nulls[loan_no_ru_nulls.total_acc >60]['year'].value_counts()

loan_no_ru_nulls.revol_bal.describe()
plot_hist(loan_no_ru_nulls[loan_no_ru_nulls.revol_bal<=80000], 'revol_bal')

loan_no_ru_nulls[loan_no_ru_nulls.revol_bal>80000].year.value_counts()

plot_hist(loan_no_ru_nulls[loan_no_ru_nulls.revol_util<1.2], 'revol_util')

plot_hist(loan_no_ru_nulls[loan_no_ru_nulls.acc_now_delinq<2], 'acc_now_delinq')
loan_no_ru_nulls[loan_no_ru_nulls.acc_now_delinq>1].year.value_counts()

plot_hist(loan_no_ru_nulls[loan_no_ru_nulls.delinq_amnt<.1], 'delinq_amnt')

plot_hist(loan_no_ru_nulls[loan_no_ru_nulls.delinq_2yrs<=3], 'delinq_2yrs')

loan_no_ru_nulls[loan_no_ru_nulls.delinq_2yrs>3].delinq_2yrs.describe()
loan_no_ru_nulls[loan_no_ru_nulls.delinq_2yrs>3].year.value_counts()

loan_no_ru_nulls[loan_no_ru_nulls.pub_rec>1].pub_rec.describe()
plot_hist(loan_no_ru_nulls[loan_no_ru_nulls.pub_rec<=1], 'pub_rec')
loan_no_ru_nulls[loan_no_ru_nulls.pub_rec>2].year.value_counts()
loan_no_ru_nulls.pub_rec.isnull().value_counts()

plot_hist(loan_no_ru_nulls[loan_no_ru_nulls.collections_12_mths_ex_med<2], 'collections_12_mths_ex_med')
loan_no_ru_nulls[loan_no_ru_nulls.collections_12_mths_ex_med>2].collections_12_mths_ex_med.describe()


#Note, categoricals that were dummified are still in this. So you would want to strip these out for analysis.
#This will actually be done in the file where we add in ACS median income information.
loan_merged_whole = pd.merge(loan_no_ru_nulls, dummies_stateetc, how = 'left', right_index = True, left_index = True)
loan_merged_whole.head()
loan_merged_whole.columns.values
#Every single one with a null last payment date charged off or defaulted. However, not realy a value I'll use?
loan_merged_whole[loan_merged_whole.last_pymnt_d.isnull()==True].total_pymnt_inv.value_counts()

#Cleaned file... but not scaled...for combining with rejected loans
#loan.to_csv('../../Data/LoanStats.csv', sep = ',', index = True, index_label = 'row_num')

'''Filtering out inq_last_6mths > 8.'''
loan_stage1= loan_merged_whole[loan_merged_whole.inq_last_6mths<=8]
loan_stage1.info()
#Cleaned file.
loan_stage1.to_csv('../../Data/LoanStatsstage1.csv', sep = ',', index = True, index_label = 'row_num')


#Remember, this is being cleaned to see if we can predict probability of default
#from the initial provided data (and if the model carries over to future years,
#or if it needs to be refit periodically.)



'''Annual Income:
Given the outliers, I think I will do DBSCAN with annual income cutoff at 200000,
and then see for those above 200K, how they cluster.

Might turn into three income groups - below 200K, 200K to 1MM, and 1MM up.

Open Accounts: will look to creating a model for those with more than 25 accounts, and less than.
    Will also have to see how linearly separable things look if filtering on 25 accounts or not.

And make sure for 2013 model, to train on emp_title and compare to non-emp title model

Total Accounts: Cutting off at 100 leaves only 70 some loans. Too little.
    Cutting off at 80 gives too few also when looking by year.
    Cutting off at 60 gives a number, not a lot. However, could be worth to look
    at how it clusters and separability.

Revol_bal = See how it separates at 80K and 100K for subsetting.

Revol_util - when doing PCA on 2013 loans, check with a 1.2 cutoff and a 2 cutoff.

inq_last_6mths. Cut off at 8. Also, no loans in the period of interest for future prediction (2015 on)
                have inquiries beyond this, so alright with just cutting them off, period.

acc_now_delinq. 1) Not a good var for PCA given that it is practically binary. 2) The ones with more than 1 delinq are small,
                and mostly 2013 on. In short, not enough in my dataset?

delinq_amnt: practically no loans above .2 for the training period. Can not really
decision as part of an outlier group.

delinq_2yrs: definitely filter out the ones > 3 into an outlier group.

pub_rec: definitely not for PCA. Also, might drop out in a random forest filtering on
        feature importance, since it also is practically captured with my ind variable.
        But can consider having it for the outlier group.

collections_12_mths_ex_med: definitely not in PCA.

So I have credit and continuous variables to do PCA on, and see if classes are separable.
I also have DBSCANing to do.
And also double checking with SelectKBest or the like, and random forest feature importance.
And I have an outlier group to check if it clusters separably.

Outlier group for modelling would be (a series of ors):
    delinq_amnt > .2
    revol_util > 2
    acc_now_delinq >= 1
    revol_bal > 80k or 100K
    total_acc > 60
    open_acc >25
    annual_inc > 200k?
    delinq_2yrs > 3
    pub_rec > 2
    collections_12_mths_ex_med >=2
'''
