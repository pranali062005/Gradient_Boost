# %%
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder


# %%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("prince7489/employee-salary-dataset")

print("Path to dataset files:", path)

# %%
df = pd.read_csv(f'{path}/employee_salary_dataset.csv')

# %%
df.dtypes

# %%
df.isnull().sum()

# %%
df.shape

# %%
df.duplicated().sum()

# %%
df.columns

# %%
df = df.drop(columns='EmployeeID')

# %%
df.dtypes

# %%
OBJ = LabelEncoder()

df['Name'] = OBJ.fit_transform(df['Name'])
df['Department'] = OBJ.fit_transform(df['Department'])
df['Education_Level'] = OBJ.fit_transform(df['Education_Level'])
df['Gender'] = OBJ.fit_transform(df['Gender'])
df['City'] = OBJ.fit_transform(df['City'])

# %%
df.columns

# %%
x = df[['Name', 'Department', 'Experience_Years', 'Education_Level', 'Age',
       'Gender', 'City',]]

y = df['Monthly_Salary']

# %%
x_train , x_test ,y_train , y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42)

# %%
model = GradientBoostingRegressor(
    n_estimators=50
)
model.fit(x_train,y_train)

# %%
predict = model.predict(x_test)
predict

# %%
predict = model.predict(x_test)

score = accuracy_score(y_test, predict)

print("Accuracy:", score)


