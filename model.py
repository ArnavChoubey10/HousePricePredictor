import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("data/data.csv")

# Features and target
X = data[['area', 'bedrooms', 'age']]
y = data['price']

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create & train model
model = LinearRegression()
model.fit(X_train, y_train)

# Score on test set
score = model.score(X_test, y_test)
print(f"Model R^2 (test): {score:.2f}")

# Example prediction
new_house = [[1500, 3, 2]]  # area, bedrooms, age
predicted_price = model.predict(new_house)
print(f"Predicted price for 1500 sqft, 3 BHK, 2 years old: {predicted_price[0]:.2f} (units)")

