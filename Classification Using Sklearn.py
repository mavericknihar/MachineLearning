# Import necessary libraries
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# Classification using Iris dataset
# Load the Iris dataset
iris = load_iris()
X_class = iris.data
y_class = iris.target

# Split the data into training and testing sets
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Create a K-Nearest Neighbors classifier
classifier = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
classifier.fit(X_train_class, y_train_class)

# Make predictions on the test set
y_pred_class = classifier.predict(X_test_class)

# Evaluate accuracy
accuracy_class = accuracy_score(y_test_class, y_pred_class)
print(f'Classification Accuracy: {accuracy_class}')


# Regression using Boston Housing dataset
# Load the Boston Housing dataset
boston = load_boston()
X_reg = boston.data
y_reg = boston.target

# Split the data into training and testing sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Create a Linear Regression model
regressor = LinearRegression()

# Train the model
regressor.fit(X_train_reg, y_train_reg)

# Make predictions on the test set
y_pred_reg = regressor.predict(X_test_reg)

# Evaluate mean squared error
mse_reg = mean_squared_error(y_test_reg, y_pred_reg)
print(f'Regression Mean Squared Error: {mse_reg}')
