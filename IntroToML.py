import pandas as pd

data = pd.read_csv("./test.csv")
data.describe()
data.head()
data.columns

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)


# Step 1: Specify Prediction Target
# print the list of columns in the dataset to find the name of the prediction target
home_data.columns
y = home_data.SalePrice

# Step 2: Create X
# Create the list of features below
feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

# Select data corresponding to features in feature_names
X = home_data[feature_names]

# Review data
# print description or statistics from X
X.describe()

# print the top few lines
X.head()

# Step 3: Specify and Fit Model
from sklearn.tree import DecisionTreeRegressor
#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state = 1)

# Fit the model
iowa_model.fit(X, y)

# Step 4: Make Predictions
predictions = iowa_model.predict(X)
print(predictions)

# Think About Your Results
print(predictions)
print(y)