!pip install pandas numpy scikit-learn
import pandas as pd                                 
import numpy as np                                             
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Sample dataset
data = {  
    'Size (sq ft)': [750, 800, 850, 900, 950],
    'Bedrooms': [2, 3, 2, 3, 4],
    'Price (in $1000s)': [150, 200, 180, 220, 250]
}
         
df = pd.DataFrame(data)

# Features and target
X = df[['Size (sq ft)', 'Bedrooms']]
y = df['Price (in $1000s)']
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model         
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Display predictions
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(predictions)
import joblib

# Save model
joblib.dump(model, 'house_price_model.pkl')

# Load model
loaded_model = joblib.load('house_price_model.pkl')

                
