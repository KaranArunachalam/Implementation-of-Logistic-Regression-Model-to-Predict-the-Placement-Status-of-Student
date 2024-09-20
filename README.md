# Implementation of Logistic Regression Model to Predict the Placement Status of Student

## Aim:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Get the data and use label encoder to change all the values to numeric.
2. Drop the unwanted values,Check for NULL values, Duplicate values.
3. Classify the training data and the test data.
4. Calculate the accuracy score, confusion matrix and classification report. 

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Karan A
RegisterNumber: 212223230099
```

```python
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear") # A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("Predicted values : ")
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy : ")
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("Confusion matrix:\n",confusion)

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("Classification Report : ")
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
#### Dataset
![image](https://github.com/user-attachments/assets/05d9b94c-37ab-44b0-8101-45acf68877d3)


#### Transformed Data
![image](https://github.com/user-attachments/assets/b848cf09-77d2-4ea7-b240-3eeeb6dfd026)

#### Null values
![image](https://github.com/user-attachments/assets/f6a15ac8-81b9-45ae-bdd9-8ad5815d87ec)


#### X values
![image](https://github.com/user-attachments/assets/0879ca89-d138-4061-ab0c-e4f9e3460a70)

#### Y values
![image](https://github.com/user-attachments/assets/8fa8c51a-11fb-4614-ba03-6b94e7c31f96)


![image](https://github.com/user-attachments/assets/7547ccb3-a788-4aec-8a32-0e5ae5324f36)

![image](https://github.com/user-attachments/assets/c34157e0-5285-4cae-be51-4e1ad6ea8f55)


![image](https://github.com/user-attachments/assets/9f68e769-7e23-45f7-bf42-d4c886bb46a4)

![image](https://github.com/user-attachments/assets/2b3854e4-4cf4-4e7c-bb46-e5011f2b3c2e)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
