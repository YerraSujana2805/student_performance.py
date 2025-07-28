# student_performance.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample student dataset (Hours studied vs Scores)
data = {
    'Hours_Studied': [1, 2, 3, 4.5, 5.5, 6, 7, 8, 9, 10],
    'Scores': [35, 45, 50, 60, 68, 70, 75, 85, 88, 95]
}

df = pd.DataFrame(data)

# 📊 Visualize the data
plt.scatter(df['Hours_Studied'], df['Scores'], color='blue')
plt.title('Study Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.grid(True)
plt.show()

# 🧪 Prepare the data
X = df[['Hours_Studied']]
y = df['Scores']

# 🔀 Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🎯 Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 🔍 Predict on test set
predictions = model.predict(X_test)

# 🧾 Show results
results = pd.DataFrame({'Hours': X_test['Hours_Studied'], 'Actual': y_test, 'Predicted': predictions})
print(results)

# 🎓 Predict for custom input
hours = 7.5
predicted_score = model.predict([[hours]])
print(f"\nA student who studies {hours} hours is predicted to score {predicted_score[0]:.2f} marks.")
