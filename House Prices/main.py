# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# Path of the file to read
file_path = './home-data-for-ml-course/train.csv'

data = pd.read_csv(file_path)
# Create target object and call it y
y = data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[features]

# Split into validation and training data
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
model = RandomForestRegressor(random_state = 1)
# Fit Model
model.fit(X, y)

# Make validation predictions and calculate mean absolute error
val_predictions = model.predict(X)
val_mae = mean_absolute_error(val_predictions, y)
print("Validation MAE: {:,.0f}".format(val_mae))

# print(len(val_predictions))
# print(val_y.columns)
# print("******\n", val_X.columns)
# print(type(val_y))

# # Appying Test Datas
test_data_path = "./home-data-for-ml-course/test.csv"
test_data = pd.read_csv(test_data_path)

test_X = test_data[features]

val_test_predictions = model.predict(test_X)
# val_test_mae = mean_absolute_error(val_test_predictions, test_y)
# print("Validation MAE: {:,.0f}".format(val_test_mae))
# # Run the code to save predictions in the format used for competition scoring

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': val_test_predictions})
output.to_csv('submission.csv', index=False)