import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error,r2_score 
df = pd.read_csv("Student_Marks.csv")
print("Dataset:\n")
print(df)
X= df[['StudyHrs']].values
y= df[['Marks']].values
X_mean = np.mean(X)
X_std = np.std(X)
X= X-X_mean/X_std
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("R2 score:",r2_score(y_test,y_pred))
print("Mean Squared Error:",mean_squared_error(y_test,y_pred))
print("slope:",model.coef_)
print("Intercept:",model.intercept_)
plt.scatter(X,y)
plt.plot(X,model.predict(X),color="red")
plt.title("Linear Regression")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()
hrs=float(input("Enter study hours:"))
hrs_n= hrs - X_mean / X_std
predicted_marks=model.predict([[hrs_n]])
predicted_marks=np.minimum(predicted_marks,100)
print("Predicted marks:",predicted_marks[0])