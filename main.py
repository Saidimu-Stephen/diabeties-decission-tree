# import pandas as pd
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.tree import export_graphviz
#
# # Load data
# df = pd.read_csv(r'/home/saidimu/Desktop/programing for data science/diabetes.csv')
# print(df)
# # Define target and feature variables
# X = df.drop("Outcome", axis=1)
# y = df["Outcome"]
#
# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
#
# # Build the Decision Tree Model
# dtc = DecisionTreeClassifier(random_state=42)
# dtc.fit(X_train, y_train)
#
# # Make predictions on test set
# y_pred = dtc.predict(X_test)
#
# # Compute accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
#
#
# # Export the decision tree as a dot file
# export_graphviz(dtc, out_file="tree.dot", feature_names=X.columns, class_names=["No Diabetes", "Diabetes"], filled=True)
#
# # Convert the dot file to a PNG image
# # !dot -Tpng tree.dot -o tree.png
#
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data


diabetes_df = pd.read_csv(r'/home/saidimu/Desktop/programing for data science/diabetes.csv')
print(diabetes_df)
target_variable = 'Outcome'
feature_variables = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X_train, X_test, y_train, y_test = train_test_split(diabetes_df[feature_variables], diabetes_df[target_variable], test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(clf, out_file=None,
                           feature_names=feature_variables,
                           class_names=['No Diabetes', 'Diabetes'],
                           filled=True, rounded=True,
                           special_characters=True)

graph = graphviz.Source(dot_data)
graph.render('diabetes_tree', view=True)

