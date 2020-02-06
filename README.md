# Bank-Marketing-Data-Analysis:
Data Analysis of Banking Marketing campaigns (over phone) of a Portuguese banking institution.
Source of the Data : https://archive.ics.uci.edu/ml/datasets/bank+marketing

# Main Objective:
The goal is to predict if the client will subscribe a term deposit (indicated in the y variable). Create a model that will help this banking institution determine, in advance, clients who will be receptive to such marketing campaigns.

# Dataset Information:
The dataset consist of 41188 records and 17 attributes.

# Dataset features information:

### bank client data:

1. age (numeric)
2. job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3. marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4. education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5. default: has credit in default? (categorical: 'no','yes','unknown')
6. housing: has housing loan? (categorical: 'no','yes','unknown')
7. loan: has personal loan? (categorical: 'no','yes','unknown')

### related with the last contact of the current campaign:

8. contact: contact communication type (categorical: 'cellular','telephone')
9. month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10. day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11. duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

### other attributes:

12. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14. previous: number of contacts performed before this campaign and for this client (numeric)
15. poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

### social and economic context attributes

16. emp.var.rate: employment variation rate - quarterly indicator (numeric)
17. cons.price.idx: consumer price index - monthly indicator (numeric)
18. cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19. euribor3m: euribor 3 month rate - daily indicator (numeric)
20. nr.employed: number of employees - quarterly indicator (numeric)

### Output variable (desired target):

21. y - has the client subscribed a term deposit? (binary: 'yes','no')

## Instructions to run the code:
1. Make sure the data file ("bank-additional-full.csv") is in the 'dataset' directory.
2. This process is implemented using Notebook in python 3 environment and make sure all the dependencies used in the notebook are installed in the local machine before run the code blocks.
3. Notebook is commented to give explanations and clarification about the code blocks and findings.


## Data Science Process


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```

Loading dataset


```python
df=pd.read_csv("./dataset/bank-additional-full.csv",sep=';')
```

Task is to create a model that will help this banking institution determine, in advance, clients who will be receptive to such marketing campaigns. "Duration" is determined after a call is made and and highly corelated with the target value. To create a realistic model "Duration" should be droped. "campaign" is the number of contact made during the current marketing campaign but the task is to determine the targeted clients befor the campaign so we drop this variables from the dataset.


```python
df=df.drop(['duration', 'campaign'],axis=1)
df.shape
```




    (41188, 19)




```python
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>56</td>
      <td>housemaid</td>
      <td>married</td>
      <td>basic.4y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <td>1</td>
      <td>57</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>unknown</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <td>2</td>
      <td>37</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <td>3</td>
      <td>40</td>
      <td>admin.</td>
      <td>married</td>
      <td>basic.6y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <td>4</td>
      <td>56</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>



### Checking for missing values

Checking columns with Null/ Nan values


```python
df.isnull().sum()
```




    age               0
    job               0
    marital           0
    education         0
    default           0
    housing           0
    loan              0
    contact           0
    month             0
    day_of_week       0
    pdays             0
    previous          0
    poutcome          0
    emp.var.rate      0
    cons.price.idx    0
    cons.conf.idx     0
    euribor3m         0
    nr.employed       0
    y                 0
    dtype: int64




Above checking for missing values indicates that there are no Null, Nan values present in any feature. Further missing value analysis for categorical and nuemrical variables are performed below and impuation is performed for certain variables.

## Target variable Analysis


```python
sns.barplot(df['y'].value_counts().values,df['y'].value_counts().index)

df.y.value_counts()
```




    no     36548
    yes     4640
    Name: y, dtype: int64




![png](Case%20Studies_files/Case%20Studies_11_1.png)


Above barplot indicates that given dataset is having two classes in the target variable hence it is a binary classification problem and the dataset is **highly class imbalance** with 'no' class with 88.73% and 'yes' class with 11.27%

## Analysis of categorical variables


```python
categorcial_variables = ['job', 'marital', 'education', 'default', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y']
fig, axis = plt.subplots(5, 2, figsize=(13,20))
axis = axis.flatten()
index =0
for col in categorcial_variables:
    axis[index].set_title(col)
    sns.barplot(x=df[col].value_counts().values,y=df[col].value_counts().index , ax=axis[index])
    index +=1
fig.tight_layout()
```


![png](Case%20Studies_files/Case%20Studies_14_0.png)


Categorical features "job", "marital", "education", "default", "loan" are having the value 'unknown' which can be considered as missing value but in a real setting these values may not able to obtain so we treat them as a separate value.

"marital" is having a very low number of 'unknown' values count.

"default" is has credit in default feature which is argueably have a value "unknown" as a recorded value.

Hence , we do not remove or impute "unknown" values for the features and we treat 'unknown' as another value for the variables.

## Helper function

Here we analyze the success rate with categorical variables and analyze the success rate of each value in that category.


```python
def cat_analysis(feature):
    df_2 = pd.DataFrame()
    df_2 = pd.crosstab(df[feature], df['y'])
    df_2['success_rate'] = df_2['yes']/(df_2['yes']+df_2['no'])
    df_2 = df_2.sort_values(by=['success_rate'], ascending=False)
    print(df_2)
    df_2 = df_2[['yes','no']]
    df_2.plot.bar(stacked=True, figsize=(15,6), color=['green','orange'])
```

### Job distribution

Here we analyze the job distribution with categorical values and sorted by success percentage in decending order


```python
cat_analysis('job')
```

    y                no   yes  success_rate
    job                                    
    student         600   275      0.314286
    retired        1286   434      0.252326
    unemployed      870   144      0.142012
    admin.         9070  1352      0.129726
    management     2596   328      0.112175
    unknown         293    37      0.112121
    technician     6013   730      0.108260
    self-employed  1272   149      0.104856
    housemaid       954   106      0.100000
    entrepreneur   1332   124      0.085165
    services       3646   323      0.081381
    blue-collar    8616   638      0.068943
    


![png](Case%20Studies_files/Case%20Studies_19_1.png)


Here we can see that 'students' and 'retired' peoples reposnded positively to the campaign

### Marital distribution

Here we analyze the marital distribution with categorical values and sorted by success percentage in decending order


```python
cat_analysis('marital')
```

    y            no   yes  success_rate
    marital                            
    unknown      68    12      0.150000
    single     9948  1620      0.140041
    divorced   4136   476      0.103209
    married   22396  2532      0.101573
    


![png](Case%20Studies_files/Case%20Studies_22_1.png)


Here we can see that 'single' people responded positively than other values 

### Previous campaign outcome distribution

Here we analyze the previous campaign outcome distribution with categorical values and sorted by success percentage in decending order


```python
cat_analysis('poutcome')
```

    y               no   yes  success_rate
    poutcome                              
    success        479   894      0.651129
    failure       3647   605      0.142286
    nonexistent  32422  3141      0.088322
    


![png](Case%20Studies_files/Case%20Studies_25_1.png)


success rate illustrates that previously subscribed people responded more positively to the current campaign

### Contact method distribution

Here we analyze the contact method distribution with categorical values and sorted by success percentage in decending order


```python
cat_analysis('contact')
```

    y             no   yes  success_rate
    contact                             
    cellular   22291  3853      0.147376
    telephone  14257   787      0.052313
    


![png](Case%20Studies_files/Case%20Studies_28_1.png)


data shows that cellular is the more preferred method of communication

## Correlation


```python
f, ax = plt.subplots(figsize=(9, 9))
sns.heatmap(df.corr(method='spearman'), annot=False, cmap='coolwarm')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x205f89a56d8>




![png](Case%20Studies_files/Case%20Studies_31_1.png)


social and economic context attributes have high correlation among them.
'euribor3m', 'nr.employed' and 'euribor3m', 'emp.var.rate' have high correlation among them

## Missing values and outliers in numerical variables


```python
df.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>pdays</th>
      <th>previous</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>41188.00000</td>
      <td>41188.000000</td>
      <td>41188.000000</td>
      <td>41188.000000</td>
      <td>41188.000000</td>
      <td>41188.000000</td>
      <td>41188.000000</td>
      <td>41188.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>40.02406</td>
      <td>962.475454</td>
      <td>0.172963</td>
      <td>0.081886</td>
      <td>93.575664</td>
      <td>-40.502600</td>
      <td>3.621291</td>
      <td>5167.035911</td>
    </tr>
    <tr>
      <td>std</td>
      <td>10.42125</td>
      <td>186.910907</td>
      <td>0.494901</td>
      <td>1.570960</td>
      <td>0.578840</td>
      <td>4.628198</td>
      <td>1.734447</td>
      <td>72.251528</td>
    </tr>
    <tr>
      <td>min</td>
      <td>17.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-3.400000</td>
      <td>92.201000</td>
      <td>-50.800000</td>
      <td>0.634000</td>
      <td>4963.600000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>32.00000</td>
      <td>999.000000</td>
      <td>0.000000</td>
      <td>-1.800000</td>
      <td>93.075000</td>
      <td>-42.700000</td>
      <td>1.344000</td>
      <td>5099.100000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>38.00000</td>
      <td>999.000000</td>
      <td>0.000000</td>
      <td>1.100000</td>
      <td>93.749000</td>
      <td>-41.800000</td>
      <td>4.857000</td>
      <td>5191.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>47.00000</td>
      <td>999.000000</td>
      <td>0.000000</td>
      <td>1.400000</td>
      <td>93.994000</td>
      <td>-36.400000</td>
      <td>4.961000</td>
      <td>5228.100000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>98.00000</td>
      <td>999.000000</td>
      <td>7.000000</td>
      <td>1.400000</td>
      <td>94.767000</td>
      <td>-26.900000</td>
      <td>5.045000</td>
      <td>5228.100000</td>
    </tr>
  </tbody>
</table>
</div>



"pdays" have the value 999 which is indicated as the customers are never been contacted previously.

"Age", "previous" have outliers since outliers can be defined as $values$ > $Q_3 + 1.5 \times IQR$ or $values$ < $Q_1 - 1.5 \times IQR$. From the analysis of numerical variables we can observe that  max('age') = 98 , max('previous')=7 respectively which are having outliers


```python
fig, axes = plt.subplots(1, 2)
fig.tight_layout()
sns.boxplot(y=df['age'], ax=axes[0])
sns.boxplot(y=df['previous'], ax=axes[1])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x205fb2a35c0>




![png](Case%20Studies_files/Case%20Studies_36_1.png)


Age is having maximum of 98 and previous is having maximum of 7 which are acceptable values and in a real life setting the values can be exist and we do not remove them.


```python
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
fig.tight_layout()
sns.distplot(df['pdays'],kde=False, ax=axes[0])
axes[1].set_title("pdays without 999")
sns.distplot(df.loc[df.pdays != 999, 'pdays'],kde=False, ax=axes[1])
plt.show()
```


![png](Case%20Studies_files/Case%20Studies_38_0.png)


very high percentage of 'pdays' is having the value 999 which indicated that the clients never been contacted before. We can handle this by changing it as categorical variable.


```python
df['pdays_missing'] = 0
df['pdays_less_5'] = 0
df['pdays_greater_15'] = 0
df['pdays_bet_5_15'] = 0
df['pdays_missing'][df['pdays']==999] = 1
df['pdays_less_5'][df['pdays']<5] = 1
df['pdays_greater_15'][(df['pdays']>15) & (df['pdays']<999)] = 1
df['pdays_bet_5_15'][(df['pdays']>=5)&(df['pdays']<=15)]= 1
df =df.drop('pdays', axis=1);
```


## Age distribution
Age distribution with success rate


```python
df.pivot(columns='y').age.plot(kind = 'hist', stacked=True,figsize=(15,6), color=['green','orange'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x205f8927fd0>




![png](Case%20Studies_files/Case%20Studies_42_1.png)



```python
adult = (df[(df.age < 60) & (df.y=='yes')]['age'].count()/sum(df.age < 60))*100
senior = (df[(df.age > 60) & (df.y=='yes')]['age'].count()/sum(df.age > 60))*100

print('Clients with age less than 60 :',adult,"% subscribed")
print('Clients with age greater than 60 :',senior,"% subscribed")
```

    Clients with age less than 60 : 10.421302662832854 % subscribed
    Clients with age greater than 60 : 45.494505494505496 % subscribed
    

## Categorical variables One-hot encoding


```python
df_encoded = pd.get_dummies(df,columns=['housing','loan','job','marital', \
                                     'education','contact','month','day_of_week', \
                                     'poutcome','default'], drop_first=True)
```


```python
df_encoded.y.replace({'yes': 1, 'no': 0}, inplace=True)
X = df_encoded.drop('y',axis = 1)
y = df_encoded['y'].values
```

## Scaling 


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## Train Test Split


```python
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.25, stratify=y, random_state=0)
```


```python
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
fig.tight_layout()
axes[0].set_title("Train data class distribution")
axes[1].set_title("Test data class distribution")
sns.countplot(Y_test, ax=axes[1])
sns.countplot(Y_train, ax=axes[0])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x205fb34d390>




![png](Case%20Studies_files/Case%20Studies_51_1.png)


## Evaluation helper functions


```python
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def print_evaluation(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_score = auc(fpr, tpr)
    print(classification_report(y_test, y_pred))
    print('AUC: {}'.format(auc_score))

def model_score(model , test_data, actual_label):
    model_name = type(model).__name__
    y_pred = model.predict(test_data)
    model_f1_score = f1_score(y_true=actual_label, y_pred=y_pred)
    
    return {'classifier':model_name, 'f1-score':model_f1_score}
    
```

# Build and evaluate Models

I'm creating models with default parameters for initial train and evaluation

## Logistic Regression model


```python
from sklearn.linear_model import LogisticRegression
lgr_model = LogisticRegression()
```


```python
lgr_model.fit(X_train, Y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)



## Decision Tree Model


```python
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier() 
```


```python
dt.fit(X_train, Y_train)
```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best')



## Random forrest Model


```python
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier()
```


```python
random_forest.fit(X_train, Y_train)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)



## Gradient Boost model


```python
from sklearn.ensemble import GradientBoostingClassifier
gdboost = GradientBoostingClassifier()
```


```python
gdboost.fit(X_train, Y_train)
```




    GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                               learning_rate=0.1, loss='deviance', max_depth=3,
                               max_features=None, max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=100,
                               n_iter_no_change=None, presort='deprecated',
                               random_state=None, subsample=1.0, tol=0.0001,
                               validation_fraction=0.1, verbose=0,
                               warm_start=False)



## AdaBoost Model


```python
from sklearn.ensemble import AdaBoostClassifier
adaBC = AdaBoostClassifier()
```


```python
adaBC.fit(X_train, Y_train)
```




    AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                       n_estimators=50, random_state=None)



## XGBoost Model


```python
from xgboost import XGBClassifier
xgb = XGBClassifier()
```


```python
xgb.fit(X_train, Y_train)
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0,
                  learning_rate=0.1, max_delta_step=0, max_depth=3,
                  min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
                  nthread=None, objective='binary:logistic', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                  silent=None, subsample=1, verbosity=1)



## KNN classifier Model


```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
```


```python
knn.fit(X_train, Y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                         weights='uniform')



## Evaluate Models based on unbalanced data

Evaluate the models with default parameters and record their f1-score based on unbalanced data


```python
models = [lgr_model, dt, random_forest, gdboost, adaBC, xgb, knn]
f1_scores_model = []
for ml_model in models:
    score_data = model_score(model=ml_model, test_data=X_test, actual_label=Y_test)
    f1_scores_model.append(score_data)

df_scores = pd.DataFrame(f1_scores_model)
df_scores.sort_values('f1-score', ascending=False)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>classifier</th>
      <th>f1-score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2</td>
      <td>RandomForestClassifier</td>
      <td>0.362153</td>
    </tr>
    <tr>
      <td>3</td>
      <td>GradientBoostingClassifier</td>
      <td>0.358079</td>
    </tr>
    <tr>
      <td>5</td>
      <td>XGBClassifier</td>
      <td>0.353016</td>
    </tr>
    <tr>
      <td>6</td>
      <td>KNeighborsClassifier</td>
      <td>0.351893</td>
    </tr>
    <tr>
      <td>1</td>
      <td>DecisionTreeClassifier</td>
      <td>0.323810</td>
    </tr>
    <tr>
      <td>4</td>
      <td>AdaBoostClassifier</td>
      <td>0.322122</td>
    </tr>
    <tr>
      <td>0</td>
      <td>LogisticRegression</td>
      <td>0.314767</td>
    </tr>
  </tbody>
</table>
</div>



## Sampling to balance dataset (SMOTE)

Since this is highly imbalanced data, we use oversampling technique by aplying SMOTE to balance the class.


```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=0)
X_balanced, y_balanced = smote.fit_resample(X_train, Y_train)
```


```python
sns.countplot(y_balanced)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2059b22e390>




![png](Case%20Studies_files/Case%20Studies_81_1.png)



```python
f1_scores_balanced_model = []
for ml_model in models:
    ml_model.fit(X_balanced, y_balanced)
    score_data = model_score(model=ml_model, test_data=X_test, actual_label=Y_test)
    f1_scores_balanced_model.append(score_data)

df_scores_balanced = pd.DataFrame(f1_scores_balanced_model)
df_scores_balanced.sort_values('f1-score', ascending=False)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>classifier</th>
      <th>f1-score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3</td>
      <td>GradientBoostingClassifier</td>
      <td>0.475578</td>
    </tr>
    <tr>
      <td>5</td>
      <td>XGBClassifier</td>
      <td>0.473210</td>
    </tr>
    <tr>
      <td>4</td>
      <td>AdaBoostClassifier</td>
      <td>0.462191</td>
    </tr>
    <tr>
      <td>0</td>
      <td>LogisticRegression</td>
      <td>0.439927</td>
    </tr>
    <tr>
      <td>2</td>
      <td>RandomForestClassifier</td>
      <td>0.404377</td>
    </tr>
    <tr>
      <td>6</td>
      <td>KNeighborsClassifier</td>
      <td>0.349087</td>
    </tr>
    <tr>
      <td>1</td>
      <td>DecisionTreeClassifier</td>
      <td>0.322822</td>
    </tr>
  </tbody>
</table>
</div>



Balanced data improved the overall performance of the models. XGBoost and Gradient Boosting models shows best performance among tested models.

## Hyperparameter Tuning

I'm tuning hyper parameters for the best two models to improve their performance


```python
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
```

## Gradient Boost Hyper parameter tuning


```python
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [6,8, 10, 12]
}
```


```python
grid = GridSearchCV(gdboost, param_grid, n_jobs=-1, scoring='f1')
```


```python
grid.fit(X_balanced, y_balanced)
```




    GridSearchCV(cv=None, error_score=nan,
                 estimator=GradientBoostingClassifier(ccp_alpha=0.0,
                                                      criterion='friedman_mse',
                                                      init=None, learning_rate=0.1,
                                                      loss='deviance', max_depth=3,
                                                      max_features=None,
                                                      max_leaf_nodes=None,
                                                      min_impurity_decrease=0.0,
                                                      min_impurity_split=None,
                                                      min_samples_leaf=1,
                                                      min_samples_split=2,
                                                      min_weight_fraction_leaf=0.0,
                                                      n_estimators=100,
                                                      n_iter_no_change=None,
                                                      presort='deprecated',
                                                      random_state=None,
                                                      subsample=1.0, tol=0.0001,
                                                      validation_fraction=0.1,
                                                      verbose=0, warm_start=False),
                 iid='deprecated', n_jobs=-1,
                 param_grid={'max_depth': [6, 8, 10, 12],
                             'n_estimators': [100, 150, 200]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='f1', verbose=0)




```python
grid.best_params_
```




    {'max_depth': 12, 'n_estimators': 100}




```python
gdboost_tuned = GradientBoostingClassifier(max_depth=12, n_estimators=100)
gdboost_tuned.fit(X_balanced, y_balanced)
y_predicted = gdboost_tuned.predict(X_test)
f1_score(y_true=Y_test, y_pred=y_predicted)
```




    0.38782877772047447




```python
print_evaluation(Y_test, y_predicted)
```

                  precision    recall  f1-score   support
    
               0       0.92      0.96      0.94      9137
               1       0.48      0.32      0.39      1160
    
        accuracy                           0.88     10297
       macro avg       0.70      0.64      0.66     10297
    weighted avg       0.87      0.88      0.87     10297
    
    AUC: 0.6400157751921893
    

## XGBoost Hyper parameter tuning


```python
param_grid_xgb = {
    'max_depth': [7, 10, 11], 
    'n_estimators': [15, 20, 50, 70], 
    'min_child_weight': [2, 3]
}
```


```python
gridsearch = GridSearchCV(xgb, param_grid_xgb, n_jobs=-1)
```


```python
gridsearch.fit(X_balanced, y_balanced)
```




    GridSearchCV(cv=None, error_score=nan,
                 estimator=XGBClassifier(base_score=0.5, booster='gbtree',
                                         colsample_bylevel=1, colsample_bynode=1,
                                         colsample_bytree=1, gamma=0,
                                         learning_rate=0.1, max_delta_step=0,
                                         max_depth=3, min_child_weight=1,
                                         missing=None, n_estimators=100, n_jobs=1,
                                         nthread=None, objective='binary:logistic',
                                         random_state=0, reg_alpha=0, reg_lambda=1,
                                         scale_pos_weight=1, seed=None, silent=None,
                                         subsample=1, verbosity=1),
                 iid='deprecated', n_jobs=-1,
                 param_grid={'max_depth': [7, 10, 11], 'min_child_weight': [2, 3],
                             'n_estimators': [15, 20, 50, 70]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=0)




```python
gridsearch.best_params_
```




    {'max_depth': 11, 'min_child_weight': 2, 'n_estimators': 70}




```python
xgboost_tuned = XGBClassifier(max_depth=11, 
                            n_estimators=70, 
                            min_child_weight=2, 
                            n_jobs=-1)
```


```python
xgboost_tuned.fit(X_balanced, y_balanced)
y_predicted = xgboost_tuned.predict(X_test)
f1_score(y_true=Y_test, y_pred=y_predicted)
```




    0.4113110539845759




```python
print_evaluation(Y_test, y_predicted)
```

                  precision    recall  f1-score   support
    
               0       0.92      0.96      0.94      9137
               1       0.51      0.34      0.41      1160
    
        accuracy                           0.89     10297
       macro avg       0.71      0.65      0.67     10297
    weighted avg       0.87      0.89      0.88     10297
    
    AUC: 0.6513456087978775
    


```python
n_features = X_train.shape[1]
plt.figure(figsize=(8,12))
plt.barh(range(n_features), xgboost_tuned.feature_importances_, align='center') 
plt.yticks(np.arange(n_features), X.columns.values) 
plt.xlabel("Feature importance")
plt.ylabel("Feature")
```




    Text(0, 0.5, 'Feature')




![png](Case%20Studies_files/Case%20Studies_101_1.png)


## Findings

- XGBoost model showed best performance among the tested models
- Most important feature that impact on client's decision is **nr.employed**: number of employees
- Previously subscibed clients are more positive to the following campaign (65.1%)
- In the current campaign 11.3% of clients subscribed to the term deposit
- Old people tend to have higher success rate (clients who are over 60 years old subcribed 45.5% of the time)


```python

```

