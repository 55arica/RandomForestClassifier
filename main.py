import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('dataset.csv')


encoder = LabelEncoder()

# Encode the 'meter' column
data['meter'] = encoder.fit_transform(data['meter'])

# Separate the features and target variable
x = data.drop(columns=['condition'])
y = data['condition']

# Encode the 'condition' column (if needed)
y = encoder.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

model = RandomForestClassifier()

# Train the model
model.fit(x_train, y_train)


output = model.predict(x_test)

print(f"Output of the random forest classifier: {output}")


accuracy = accuracy_score(y_test, output)
print(f"Accuracy: {accuracy}")
