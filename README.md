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

 <img src= "https://user-images.githubusercontent.com/66487971/91658596-387d7800-ead2-11ea-9d12-d960d298aa06.png" width = 1000>












































