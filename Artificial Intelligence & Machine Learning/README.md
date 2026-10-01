# AI & Machine Learning Python Programs 

## 1. Project Overview

This project contains practical Python programs covering fundamental concepts in:

1. **Data Exploration and Unsupervised Learning**
2. **Data Preprocessing and Visualization**
3. **Supervised Machine Learning**
4. **Classical AI Search and Logic-Based Algorithms**
5. **Simple Neural Networks and Model Evaluation**

The programs use real-world-style examples so that each concept can be understood through a practical application rather than only through theory.

---

# 2. Technologies Used

| Technology / Library | Purpose                        |
| -------------------- | ------------------------------ |
| Python               | Programming language           |
| NumPy                | Numerical computation          |
| Pandas               | Data manipulation and analysis |
| Matplotlib           | Data visualization             |
| Seaborn              | Statistical visualization      |
| Scikit-learn         | Machine learning               |
| TensorFlow/Keras     | Neural networks                |

### Recommended Python Version

```text
Python 3.10+
```

---

# 3. Project Structure

```text
AI-ML-Python-Programs/
│
├── 01_data_exploration_clustering/
│   ├── data_exploration.py
│   └── kmeans_clustering.py
│
├── 02_data_preprocessing_visualization/
│   ├── preprocessing.py
│   └── visualization.py
│
├── 03_supervised_learning/
│   ├── train_test_split.py
│   ├── classification.py
│   └── regression.py
│
├── 04_ai_search_logic/
│   ├── bfs_dfs.py
│   ├── a_star.py
│   └── logic_based.py
│
├── 05_neural_network/
│   └── simple_neural_network.py
│
├── requirements.txt
└── README.md
```

---

# 4. Installation

## Step 1: Install Python

Download Python from:

```text
https://www.python.org/
```

Check the installed version:

```bash
python --version
```

Example:

```text
Python 3.11.5
```

---

# 5. Install Required Libraries

Run:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
```

You can also install the packages individually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

---

# 6. Area 1 — Data Exploration and Unsupervised Learning

## Objective

The first stage is to understand a dataset before building a machine-learning model.

Data exploration helps us answer questions such as:

* How many records are present?
* What columns are available?
* Are there missing values?
* What are the average values?
* Are there relationships between variables?
* Are there natural groups in the data?

After exploration, an **unsupervised learning algorithm** can be used to discover patterns without predefined labels.

---

## Program 1.1 — Data Exploration

### Real-Time Example

Suppose a shopping company has customer information:

| Customer | Age | Annual Income | Spending Score |
| -------- | --: | ------------: | -------------: |
| A        |  19 |            15 |             39 |
| B        |  21 |            16 |             81 |
| C        |  20 |            17 |              6 |
| D        |  23 |            25 |             77 |
| E        |  31 |            30 |             40 |

The company wants to understand its customers.

### Basic Program

```python
import pandas as pd

data = {
    "Customer": ["A", "B", "C", "D", "E"],
    "Age": [19, 21, 20, 23, 31],
    "Annual_Income": [15, 16, 17, 25, 30],
    "Spending_Score": [39, 81, 6, 77, 40]
}

df = pd.DataFrame(data)

print("First five records:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nCorrelation:")
print(df.corr(numeric_only=True))
```

### What We Learn

Important Pandas functions:

```python
df.head()
df.info()
df.describe()
df.isnull().sum()
df.corr()
```

---

# 7. Program 1.2 — K-Means Clustering

## Real-Time Example

The shopping company does not know the customer categories in advance.

We can use **K-Means clustering** to automatically group customers according to:

* Annual income
* Spending score

The algorithm may discover groups such as:

```text
Customer Group
       |
       +---- High income / High spending
       |
       +---- High income / Low spending
       |
       +---- Low income / High spending
       |
       +---- Low income / Low spending
```

The groups are discovered from the data rather than supplied beforehand.

### Program

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    "Customer": ["A", "B", "C", "D", "E", "F", "G", "H"],
    "Annual_Income": [15, 16, 17, 80, 85, 82, 30, 35],
    "Spending_Score": [39, 81, 6, 90, 85, 20, 70, 65]
}

df = pd.DataFrame(data)

X = df[["Annual_Income", "Spending_Score"]]

model = KMeans(n_clusters=3, random_state=42, n_init=10)

df["Cluster"] = model.fit_predict(X)

print(df)

plt.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=df["Cluster"]
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.show()
```

### Algorithm

```text
Customer Data
      ↓
Select Features
      ↓
Choose Number of Clusters
      ↓
Apply K-Means
      ↓
Assign Customers to Clusters
      ↓
Visualize Groups
```

---

# 8. Area 2 — Data Preprocessing and Visualization

## Objective

Real-world data is often incomplete or inconsistent.

Before training a machine-learning model, we usually need to:

* Handle missing values
* Remove duplicates
* Encode categorical values
* Scale numerical features
* Detect outliers
* Visualize the data

---

# 9. Program 2.1 — Data Preprocessing

## Real-Time Example

Consider student data:

| Name  | Gender | Attendance |   Marks |
| ----- | ------ | ---------: | ------: |
| Arun  | Male   |         85 |      78 |
| Priya | Female |         92 |      88 |
| Ravi  | Male   |    Missing |      65 |
| Anu   | Female |         78 | Missing |

### Program

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    "Name": ["Arun", "Priya", "Ravi", "Anu"],
    "Gender": ["Male", "Female", "Male", "Female"],
    "Attendance": [85, 92, None, 78],
    "Marks": [78, 88, 65, None]
}

df = pd.DataFrame(data)

print("Original data:")
print(df)

# Fill missing numerical values
df["Attendance"] = df["Attendance"].fillna(
    df["Attendance"].mean()
)

df["Marks"] = df["Marks"].fillna(
    df["Marks"].mean()
)

# Encode Gender
df["Gender"] = df["Gender"].map({
    "Male": 0,
    "Female": 1
})

# Scale numerical features
scaler = StandardScaler()

df[["Attendance", "Marks"]] = scaler.fit_transform(
    df[["Attendance", "Marks"]]
)

print("\nPreprocessed data:")
print(df)
```

### Preprocessing Flow

```text
Raw Data
   ↓
Find Missing Values
   ↓
Fill Missing Values
   ↓
Encode Categories
   ↓
Scale Features
   ↓
Clean Dataset
```

---

# 10. Program 2.2 — Data Visualization

Visualization helps us understand data before applying machine-learning algorithms.

### Example: Student Marks

```python
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Student": ["A", "B", "C", "D", "E"],
    "Marks": [78, 85, 65, 92, 88]
}

df = pd.DataFrame(data)

plt.bar(df["Student"], df["Marks"])

plt.xlabel("Student")
plt.ylabel("Marks")
plt.title("Student Marks")

plt.show()
```

### Other Useful Visualizations

```python
plt.hist(df["Marks"])
```

```python
plt.boxplot(df["Marks"])
```

```python
plt.scatter(df["Student"], df["Marks"])
```

Visualization helps identify:

* Trends
* Outliers
* Distributions
* Relationships
* Differences between groups

---

# 11. Area 3 — Train, Test and Evaluate Supervised Models

## Objective

In supervised learning, the model learns from data where the target/output is known.

The general process is:

```text
Dataset
   ↓
Features + Target
   ↓
Train/Test Split
   ↓
Train Model
   ↓
Make Predictions
   ↓
Evaluate Model
```

---

# 12. Program 3.1 — Train/Test Split

## Real-Time Example

Suppose we want to predict whether a student will pass based on:

* Study hours
* Attendance

```python
from sklearn.model_selection import train_test_split

X = [
    [2, 60],
    [3, 65],
    [5, 75],
    [6, 80],
    [8, 90],
    [9, 95]
]

y = [0, 0, 1, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

print("Training data:")
print(X_train)

print("\nTesting data:")
print(X_test)
```

Here:

```text
X = Input features
y = Target/output
```

---

# 13. Program 3.2 — Classification

## Example: Student Pass/Fail Prediction

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = [
    [2, 60],
    [3, 65],
    [5, 75],
    [6, 80],
    [8, 90],
    [9, 95]
]

y = [0, 0, 1, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Actual values:", y_test)
print("Predicted values:", predictions)
print("Accuracy:", accuracy)
```

### Evaluation Metrics

For classification, commonly used metrics include:

```text
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
```

---

# 14. Program 3.3 — Regression

## Real-Time Example

Predict house prices based on house size.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = [[500], [750], [1000], [1250], [1500], [1750]]
y = [100000, 150000, 200000, 250000, 300000, 350000]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predicted prices:")
print(predictions)

print("Mean Squared Error:",
      mean_squared_error(y_test, predictions))

print("R2 Score:",
      r2_score(y_test, predictions))
```

### Regression Metrics

```text
Mean Absolute Error
Mean Squared Error
Root Mean Squared Error
R² Score
```

---

# 15. Area 4 — Classical AI Search Techniques and Logic-Based Algorithms

## Objective

Classical AI techniques solve problems by searching through possible states or applying logical rules.

Topics covered:

* Breadth-First Search
* Depth-First Search
* A* Search
* Rule-based reasoning
* Logic-based decision making

---

# 16. Program 4.1 — Breadth-First Search

## Real-Time Example

Imagine a map of cities:

```text
A --- B --- D
|     |
C --- E
```

We want to find a path from A to D.

### Program

```python
from collections import deque

graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "E"],
    "D": ["B"],
    "E": ["B", "C"]
}

def bfs(graph, start, goal):

    queue = deque([[start]])
    visited = set()

    while queue:

        path = queue.popleft()
        node = path[-1]

        if node == goal:
            return path

        if node not in visited:

            visited.add(node)

            for neighbor in graph[node]:
                new_path = path + [neighbor]
                queue.append(new_path)

print("Path:", bfs(graph, "A", "D"))
```

BFS explores nodes level by level.

---

# 17. Program 4.2 — Depth-First Search

DFS explores one branch as deeply as possible before backtracking.

```python
graph = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F"],
    "D": [],
    "E": [],
    "F": []
}

visited = set()

def dfs(node):

    if node in visited:
        return

    visited.add(node)

    print(node)

    for neighbor in graph[node]:
        dfs(neighbor)

dfs("A")
```

---

# 18. Program 4.3 — A* Search

A* is commonly used for pathfinding.

A* uses:

```text
f(n) = g(n) + h(n)
```

Where:

* `g(n)` = cost from the starting point
* `h(n)` = estimated cost to the goal
* `f(n)` = total estimated cost

### Real-Time Example

A robot needs to find a path through a grid.

```python
import heapq

def a_star(graph, start, goal, heuristic):

    queue = [(0, start)]
    cost = {start: 0}
    parent = {start: None}

    while queue:

        _, current = heapq.heappop(queue)

        if current == goal:
            path = []

            while current is not None:
                path.append(current)
                current = parent[current]

            return path[::-1]

        for neighbor, distance in graph[current]:

            new_cost = cost[current] + distance

            if neighbor not in cost or new_cost < cost[neighbor]:

                cost[neighbor] = new_cost

                priority = new_cost + heuristic[neighbor]

                heapq.heappush(
                    queue,
                    (priority, neighbor)
                )

                parent[neighbor] = current

    return None


graph = {
    "A": [("B", 1), ("C", 4)],
    "B": [("D", 2)],
    "C": [("D", 1)],
    "D": []
}

heuristic = {
    "A": 3,
    "B": 2,
    "C": 1,
    "D": 0
}

print(a_star(graph, "A", "D", heuristic))
```

---

# 19. Program 4.4 — Logic-Based Algorithm

## Real-Time Example

A simple medical decision-support example:

```text
IF fever AND cough
THEN possible respiratory infection
```

### Program

```python
fever = True
cough = True
headache = False

if fever and cough:
    print("Possible respiratory infection")
elif fever:
    print("Fever detected")
elif cough:
    print("Cough detected")
else:
    print("No matching rule")
```

This demonstrates simple rule-based reasoning.

> This is only a programming example and is not a medical diagnostic system.

---

# 20. Area 5 — Simple Neural Networks

## Objective

A neural network learns patterns from examples.

Basic structure:

```text
Input Layer
     ↓
Hidden Layer
     ↓
Output Layer
```

A simple neural network can be used for classification problems.

---

# 21. Program 5.1 — Simple Neural Network

## Real-Time Example

Predict whether a student passes based on:

* Study hours
* Attendance

### Program

```python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.array([
    [2, 60],
    [3, 65],
    [4, 70],
    [5, 75],
    [6, 80],
    [7, 85],
    [8, 90],
    [9, 95]
])

y = np.array([
    0, 0, 0, 1, 1, 1, 1, 1
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model = MLPClassifier(
    hidden_layer_sizes=(5,),
    max_iter=2000,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Actual:", y_test)
print("Predicted:", predictions)
print("Accuracy:", accuracy)
```

---

# 22. Assessing Neural Network Performance

The model can be evaluated using:

```text
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
Loss
```

Example:

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        predictions
    )
)
```

---

# 23. Complete Machine Learning Workflow

The overall workflow of this project is:

```text
                 Raw Dataset
                      |
                      ↓
             Data Exploration
                      |
                      ↓
             Data Preprocessing
                      |
             +--------+--------+
             |                 |
             ↓                 ↓
       Visualization     Unsupervised
                           Learning
             |                 |
             +--------+--------+
                      |
                      ↓
               Feature / Target
                      |
                      ↓
                Train / Test
                      |
                      ↓
             Supervised Model
                      |
                      ↓
                 Prediction
                      |
                      ↓
                Evaluation
                      |
                      ↓
              Neural Network
```

Classical AI search algorithms operate separately on problems represented as states, graphs, or rules:

```text
Problem
   ↓
State Representation
   ↓
Search / Logic
   ↓
Solution
```

---

# 24. How to Run the Programs

Navigate to the required folder.

### Data Exploration

```bash
python 01_data_exploration_clustering/data_exploration.py
```

### K-Means

```bash
python 01_data_exploration_clustering/kmeans_clustering.py
```

### Preprocessing

```bash
python 02_data_preprocessing_visualization/preprocessing.py
```

### Visualization

```bash
python 02_data_preprocessing_visualization/visualization.py
```

### Classification

```bash
python 03_supervised_learning/classification.py
```

### Regression

```bash
python 03_supervised_learning/regression.py
```

### BFS / DFS

```bash
python 04_ai_search_logic/bfs_dfs.py
```

### A* Search

```bash
python 04_ai_search_logic/a_star.py
```

### Logic-Based Algorithm

```bash
python 04_ai_search_logic/logic_based.py
```

### Neural Network

```bash
python 05_neural_network/simple_neural_network.py
```

---

# 25. Summary of Programs

| Area                  | Program             | Real-Time Example     | Main Technology     |
| --------------------- | ------------------- | --------------------- | ------------------- |
| Data Exploration      | Dataset analysis    | Customer data         | Pandas              |
| Unsupervised Learning | K-Means             | Customer segmentation | Scikit-learn        |
| Preprocessing         | Cleaning/scaling    | Student data          | Pandas/Scikit-learn |
| Visualization         | Charts              | Student performance   | Matplotlib          |
| Classification        | Logistic Regression | Pass/Fail             | Scikit-learn        |
| Regression            | Linear Regression   | House price           | Scikit-learn        |
| Search                | BFS                 | City route            | Python              |
| Search                | DFS                 | Graph traversal       | Python              |
| Search                | A*                  | Pathfinding           | Python              |
| Logic                 | Rule-based system   | Decision rules        | Python              |
| Neural Network        | MLP                 | Student prediction    | Scikit-learn        |

---

# 26. Learning Outcomes

After completing these programs, you should be able to:

* Explore datasets using Pandas.
* Calculate descriptive statistics.
* Identify patterns in data.
* Apply unsupervised clustering.
* Clean and preprocess datasets.
* Visualize data using Python.
* Split datasets into training and testing sets.
* Build classification models.
* Build regression models.
* Evaluate machine-learning models.
* Implement BFS and DFS.
* Implement A* search.
* Create simple rule-based systems.
* Build a simple neural network.
* Measure model performance using appropriate evaluation metrics.

---

# 27. Recommended Learning Order

The programs are designed to be completed in this order:

```text
1. Data Exploration
        ↓
2. Data Preprocessing
        ↓
3. Data Visualization
        ↓
4. Unsupervised Learning
        ↓
5. Train/Test Split
        ↓
6. Classification
        ↓
7. Regression
        ↓
8. Model Evaluation
        ↓
9. BFS / DFS
        ↓
10. A* Search
        ↓
11. Logic-Based Algorithms
        ↓
12. Neural Networks
        ↓
13. Neural Network Evaluation
```

---

# 28. Final Project Goal

The purpose of this project is to move from **raw data to intelligent decision-making**:

```text
             RAW DATA
                 ↓
        ┌────────────────┐
        │ Data Exploration│
        └────────────────
                 ↓
        ┌────────────────┐
        │ Preprocessing  │
        └────────────────┘
                 ↓
        ┌────────────────┐
        │ Visualization  │
        └────────────────┘
                 ↓
        ┌────────────────┐
        │ Machine Learning│
        └────────────────┘
                 ↓
        ┌────────────────┐
        │ AI Algorithms  │
        └────────────────┘
                 ↓
        ┌────────────────┐
        │ Neural Network │
        └────────────────┘
                 ↓
             PREDICTION

Each program is intentionally small and independent so it can be run, understood, modified, and demonstrated separately.
