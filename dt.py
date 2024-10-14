import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from os import system
from graphviz import Source
from sklearn import tree
import dtreeviz
import io

light_blue_sidebar_css = """
<style>
.stSidebar {
    background-color: #B0E0E6;  /* Light blue background */
}
.stApp {
    background-color: #FFFFFF;  /* White background for the main area */
}
</style>
"""

# Inject custom CSS
st.markdown(light_blue_sidebar_css, unsafe_allow_html=True)

def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)

    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array

# Sidebar for dataset selection
st.sidebar.markdown("# Choose Dataset")
dataset_choice = st.sidebar.selectbox(
    'Select dataset type',
    ('Moons', 'Circles', 'Classification')
)

# Generate dataset based on user choice
if dataset_choice == 'Moons':
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
elif dataset_choice == 'Circles':
    X, y = make_circles(n_samples=500, noise=0.20, factor=0.5, random_state=42)
else:  # 'Classification'
    X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Decision Tree Classifier")

criterion = st.sidebar.selectbox(
    'Criterion',
    ('gini', 'entropy')
)

splitter = st.sidebar.selectbox(
    'Splitter',
    ('best', 'random')
)

max_depth = int(st.sidebar.number_input('Max Depth'))

min_samples_split = st.sidebar.slider('Min Samples Split', 1, X_train.shape[0], 2,key=1234)

min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, X_train.shape[0], 1,key=1235)

max_features = st.sidebar.slider('Max Features', 1, 2, 2,key=1236)

max_leaf_nodes = int(st.sidebar.number_input('Max Leaf Nodes'))

min_impurity_decrease = st.sidebar.number_input('Min Impurity Decrease')

# Load initial graph
fig, ax = plt.subplots()

# Plot initial graph
ax.scatter(X.T[0], X.T[1], c=y, cmap='tab20c')
orig = st.pyplot(fig)

if st.sidebar.button('Run Algorithm'):

    orig.empty()

    if max_depth == 0:
        max_depth = None

    if max_leaf_nodes == 0:
        max_leaf_nodes = None

    clf = DecisionTreeClassifier(criterion=criterion,splitter=splitter,max_depth=max_depth,random_state=42,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_features=max_features,max_leaf_nodes=max_leaf_nodes,min_impurity_decrease=min_impurity_decrease)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    XX, YY, input_array = draw_meshgrid()
    labels = clf.predict(input_array)

    ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='Set3')
    plt.xlabel("Col1")
    plt.ylabel("Col2")
    orig = st.pyplot(fig)
    st.markdown(
        f"""
        <h2 style='color: black;'>Accuracy for Decision Tree: {str(round(accuracy_score(y_test, y_pred), 2))}</h2>
        """,
        unsafe_allow_html=True
    )
    tree = export_graphviz(
        clf,
        out_file=None,  # output as string for Streamlit
        feature_names=["Col1", "Col2"],
        class_names=["Class 0", "Class 1"],
        filled=True,  # Fill nodes with colors
        rounded=True,  # Round node corners

    )

    st.graphviz_chart(tree)

    # Visualize the decision tree using dtreeviz
    viz = dtreeviz.model(
        clf,
        X_train,
        y_train,
        feature_names=["Col1", "Col2"],
        class_names=["Class 0", "Class 1"]
    )
    st.markdown(
        f"""
            <h2 style='color: black;'>DTreeViz Visualisation </h2>
            """,
        unsafe_allow_html=True
    )
    # Save the visualization to a file (SVG format)
    v = viz.view()
    v.save("tree_viz.svg")

    # Display the SVG in Streamlit
    st.image("tree_viz.svg", use_column_width = True)