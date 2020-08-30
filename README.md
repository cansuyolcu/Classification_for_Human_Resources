# Classification_for_Human_Resources
Using Deep Learning for a Human Resources Department in order to predict which employees are more likely to quit.
I will be exploring and analyzing the dataset from [kaggle.com](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset).


Education
1 'Below College'
2 'College'
3 'Bachelor'
4 'Master'
5 'Doctor'

EnvironmentSatisfaction
1 'Low'
2 'Medium'
3 'High'
4 'Very High'

JobInvolvement
1 'Low'
2 'Medium'
3 'High'
4 'Very High'

JobSatisfaction
1 'Low'
2 'Medium'
3 'High'
4 'Very High'

PerformanceRating
1 'Low'
2 'Good'
3 'Excellent'
4 'Outstanding'

RelationshipSatisfaction
1 'Low'
2 'Medium'
3 'High'
4 'Very High'

WorkLifeBalance
1 'Bad'
2 'Good'
3 'Better'
4 'Best'

## Importing the libraries

```python

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

```
```python
employee_df = pd.read_csv('Human_Resources.csv')
employee_df.head()
```

<img src= "https://user-images.githubusercontent.com/66487971/91656964-e9314a80-eac5-11ea-979e-2c9d442775a1.png" width = 800>

```python
employee_df.info()
```
<img src= "https://user-images.githubusercontent.com/66487971/91657009-5a70fd80-eac6-11ea-9f79-8d514de362d9.png" width = 400>

```python
employee_df.describe()
```

<img src= "https://user-images.githubusercontent.com/66487971/91657029-8a200580-eac6-11ea-8f38-f4a2fe41437e.png" width = 1000>

**Replacing the 'Attritition', 'overtime' and 'Over18' columns with integers before performing any visualizations**

```python
employee_df['Attrition'] = employee_df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['OverTime'] = employee_df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['Over18'] = employee_df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)

```

**Looking at the missing Data**

```python

 sns.heatmap(employee_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")
 
 ```
 
 <img src= "https://user-images.githubusercontent.com/66487971/91657072-e2ef9e00-eac6-11ea-99a5-e4f552345038.png" width = 400>

```python

employee_df.hist(bins = 30, figsize = (20,20), color = 'r')

```

 <img src= "https://user-images.githubusercontent.com/66487971/91657173-b7b97e80-eac7-11ea-920c-2fd87394afca.png" width = 1000>


**It makes sense to drop 'EmployeeCount' , 'Standardhours' and 'Over18' since they do not change from one employee to the other**
**I'll drop 'EmployeeNumber' as well**

```python
employee_df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, inplace=True)
```
**I'm dividing the dataframes in order to get a better grasp of the trends**

```python
left_df        = employee_df[employee_df['Attrition'] == 1]
stayed_df      = employee_df[employee_df['Attrition'] == 0]
```

```python
print("Total =", len(employee_df))

print("Number of employees who left the company =", len(left_df))
print("Percentage of employees who left the company =", 1.*len(left_df)/len(employee_df)*100.0, "%")
 
print("Number of employees who did not leave the company (stayed) =", len(stayed_df))
print("Percentage of employees who did not leave the company (stayed) =", 1.*len(stayed_df)/len(employee_df)*100.0, "%")
```

 <img src= "https://user-images.githubusercontent.com/66487971/91657281-6cec3680-eac8-11ea-83d0-2cf9bca582f3.png" width = 600>


```python
left_df.describe()
```

 <img src= "https://user-images.githubusercontent.com/66487971/91657325-c3597500-eac8-11ea-9a8a-2392c1584be9.png" width = 800>


```python
right_df.describe()
```


 <img src= "https://user-images.githubusercontent.com/66487971/91657330-da986280-eac8-11ea-8842-341fef4c8f1b.png" width = 800>


**From here it can be deducted that mean age of the employees who stayed is higher compared to who left, Rate of employees who stayed is higher, Employees who stayed live closer to home, Employees who stayed are generally more satisifed with their jobs and Employees who stayed tend to have higher stock option level.**



```python
correlations = employee_df.corr()
f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(correlations, annot = True)
```

 <img src= "https://user-images.githubusercontent.com/66487971/91658596-387d7800-ead2-11ea-9d12-d960d298aa06.png" width = 700>
 
 **Job level is strongly correlated with total working hours. Monthly income is strongly correlated with Job level. Monthly income is strongly correlated with total working hours. Age is stongly correlated with monthly income**
 
 ```python
 plt.figure(figsize=[25, 12])
sns.countplot(x = 'Age', hue = 'Attrition', data = employee_df)
```

 <img src= "https://user-images.githubusercontent.com/66487971/91658710-4bdd1300-ead3-11ea-916c-0560285f4a4c.png" width = 700>

 
 ```python
 
 plt.figure(figsize=[20,20])
plt.subplot(411)
sns.countplot(x = 'JobRole', hue = 'Attrition', data = employee_df)
plt.subplot(412)
sns.countplot(x = 'MaritalStatus', hue = 'Attrition', data = employee_df)
plt.subplot(413)
sns.countplot(x = 'JobInvolvement', hue = 'Attrition', data = employee_df)
plt.subplot(414)
sns.countplot(x = 'JobLevel', hue = 'Attrition', data = employee_df)

```

 <img src= "https://user-images.githubusercontent.com/66487971/91658744-98c0e980-ead3-11ea-91ac-1fe1159b0483.png" width = 700>
 
 **Single employees tend to leave compared to married and divorced. Sales Representitives tend to leave compared to any other job. Less involved employees tend to leave the company. Less experienced (low job level) tend to leave the company**
 
 ```python
 plt.figure(figsize=(12,7))

sns.kdeplot(left_df['DistanceFromHome'], label = 'Employees who left', shade = True, color = 'r')
sns.kdeplot(stayed_df['DistanceFromHome'], label = 'Employees who Stayed', shade = True, color = 'b')

plt.xlabel('Distance From Home')
```

 <img src= "https://user-images.githubusercontent.com/66487971/91658775-e0477580-ead3-11ea-8023-227cc8206553.png" width = 700>




```python
plt.figure(figsize=(12,7))

sns.kdeplot(left_df['YearsWithCurrManager'], label = 'Employees who left', shade = True, color = 'r')
sns.kdeplot(stayed_df['YearsWithCurrManager'], label = 'Employees who Stayed', shade = True, color = 'b')

plt.xlabel('Years With Current Manager')
```

 <img src= "https://user-images.githubusercontent.com/66487971/91658795-0e2cba00-ead4-11ea-9d24-86c307189581.png" width = 700>
 
 ```python
 
 plt.figure(figsize=(12,7))

sns.kdeplot(left_df['TotalWorkingYears'], shade = True, label = 'Employees who left', color = 'r')
sns.kdeplot(stayed_df['TotalWorkingYears'], shade = True, label = 'Employees who Stayed', color = 'b')

plt.xlabel('Total Working Years')

```

 <img src= "https://user-images.githubusercontent.com/66487971/91658807-2ac8f200-ead4-11ea-9342-130f1836253b.png" width = 700>
 
 
 ```python
 
 plt.figure(figsize=(15, 10))
sns.boxplot(x = 'MonthlyIncome', y = 'Gender', data = employee_df)
 ```
 
 
 
 <img src= "https://user-images.githubusercontent.com/66487971/91658820-47fdc080-ead4-11ea-9f35-2898a1a4589c.png" width = 700>
 
 
 ```python
 
 plt.figure(figsize=(15, 10))
sns.boxplot(x = 'MonthlyIncome', y = 'JobRole', data = employee_df)
 ```



 <img src= "https://user-images.githubusercontent.com/66487971/91658835-682d7f80-ead4-11ea-8e6b-35a2e68d39da.png" width = 700>


```python
X_cat = employee_df[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]
X_cat
```

 <img src= "https://user-images.githubusercontent.com/66487971/91658900-c0648180-ead4-11ea-882e-28881c71068e.png" width = 600>
 
 ```python
 
 from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat)

```


```python
X_numerical = employee_df[['Age', 'DailyRate', 'DistanceFromHome',	'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',	'JobLevel',	'JobSatisfaction',	'MonthlyIncome',	'MonthlyRate',	'NumCompaniesWorked',	'OverTime',	'PercentSalaryHike', 'PerformanceRating',	'RelationshipSatisfaction',	'StockOptionLevel',	'TotalWorkingYears'	,'TrainingTimesLastYear'	, 'WorkLifeBalance',	'YearsAtCompany'	,'YearsInCurrentRole', 'YearsSinceLastPromotion',	'YearsWithCurrManager']]

```

```python
X_all = pd.concat([X_cat, X_numerical], axis = 1)

```

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X_all)
```

```python
y = employee_df['Attrition']


```
## TRAINING AND EVALUATING A LOGISTIC REGRESSION CLASSIFIER


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


```
 
 ```python
 from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

```python
from sklearn.metrics import confusion_matrix, classification_report

print("Accuracy {} %".format( 100 * accuracy_score(y_pred, y_test)))

```

Accuracy 89.94565217391305 %

```python
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True)
```

<img src= "https://user-images.githubusercontent.com/66487971/91659036-b727e480-ead5-11ea-8b23-8441a8d80f23.png" width = 400>




























