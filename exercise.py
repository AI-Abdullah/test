from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
isri = load_iris()
x = isri.data
y = isri.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
model = LogisticRegression(max_iter=5000)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
# Display the names of the flower categories
print("Flower Categories:", isri.target_names)
# Print the predicted species for the entire test dataset
print("Predictions on the test set:")
print(y_pred)
# Print the actual species for comparison
print("\nActual species in the test set:")
print(y_test)
# Predict the species for the first 5 samples in the test set
print("\nPrediction for the first 5 samples:")
first_five_predictions = model.predict(x_test[:5])
print(first_five_predictions)