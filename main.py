import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load iris dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
target_names = iris.target_names

# Prepare dataset dataframe including target for display and download
iris_df = X.copy()
iris_df['target'] = y

st.title("🌸 Iris Flower ML Demo with Advanced Features")

# CSV download button
csv = iris_df.to_csv(index=False)
st.download_button(
    label="Download Iris Dataset as CSV",
    data=csv,
    file_name='iris_data.csv',
    mime='text/csv',
    key='download_csv'
)

# Section: Data overview
st.header("Data Overview")
st.dataframe(iris_df.head())
st.write(iris_df.describe())

# Section: Data visualization
st.header("Feature Visualizations")

# Pairplot colored by species
st.subheader("Pairplot of features colored by Iris species")
pairplot_fig = sns.pairplot(iris_df, hue='target', vars=iris.feature_names, palette='bright').fig
st.pyplot(pairplot_fig)

# Boxplot
st.subheader("Boxplot of features")
fig, ax = plt.subplots()
sns.boxplot(data=X, orient='h', palette='pastel', ax=ax)
st.pyplot(fig)

# Section: Model training and comparison
st.header("Model Training and Accuracy Comparison")

# Split data for training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifiers = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42)
}

accuracies = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    st.write(f"Accuracy of {name}: {acc:.2f}")

# Accuracy barplot
fig2, ax2 = plt.subplots()
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette='Set2', ax=ax2)
ax2.set_ylim(0, 1)
ax2.set_ylabel("Accuracy")
ax2.set_title("Model Accuracy Comparison on Test Set")
st.pyplot(fig2)

# Feature importances from Random Forest
st.header("Feature Importances from Random Forest")
rf = classifiers["Random Forest"]
importances = rf.feature_importances_
fig3, ax3 = plt.subplots()
sns.barplot(x=X.columns, y=importances, palette='viridis', ax=ax3)
plt.xticks(rotation=45)
st.pyplot(fig3)

# Section: Make a prediction with user inputs
st.header("Make a Prediction")

input_data = []
cols = st.columns(len(X.columns))
for i, feature in enumerate(X.columns):
    with cols[i]:
        val = st.number_input(f"{feature}", value=float(X[feature].mean()))
        input_data.append(val)

if st.button("Predict"):
    pred = rf.predict([input_data])[0]
    st.success(f"Prediction: {target_names[pred]}")

# Optional CSS for colorful backgrounds and text
st.markdown(
    """
    <style>
    .css-1v3fvcr {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
    }
    .css-ffhzg2 {
        color: #005f73;
    }
    </style>
    """,
    unsafe_allow_html=True
)
