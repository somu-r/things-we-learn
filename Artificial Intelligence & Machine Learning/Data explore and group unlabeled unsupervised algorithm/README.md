# NumPy and Pandas: Matrix Operations, Descriptive Statistics & Data Cleaning 

## 1. Project Overview 

This project demonstrates three common data-analysis tasks using Python:

1. **Matrix Operations using NumPy**
2. **Descriptive Statistics using Pandas**
3. **Data Cleaning using Pandas**

### Real-Time Example

Imagine a college wants to analyze student examination data.

The college has information such as:

* Student name
* Age
* Marks
* Attendance
* Other numerical information

Before analyzing the data, we need to:

* Perform mathematical operations on numerical data.
* Calculate statistics such as mean, median, minimum, and maximum.
* Identify and fix missing or duplicate data.

This project demonstrates these processes step by step.

---

# 2. Technologies Used

| Technology | Purpose                         |
| ---------- | ------------------------------- |
| Python     | Programming language            |
| NumPy      | Numerical and matrix operations |
| Pandas     | Data manipulation and analysis  |

### Python Version

Recommended:

```text
Python 3.9+
```

---

# 3. Project Structure

A simple project structure can be:

```text
numpy-pandas-project/
│
├── matrix_operations.py
├── descriptive_statistics.py
├── data_cleaning.py
├── README.md
└── requirements.txt
```

---

# 4. Installation

## Step 1: Install Python

Download and install Python from:

```text
https://www.python.org/
```

Verify the installation:

```bash
python --version
```

or:

```bash
python3 --version
```

---

## Step 2: Install NumPy and Pandas

Open Command Prompt, Terminal, or VS Code Terminal and execute:

```bash
pip install numpy pandas
```

To verify:

```bash
pip show numpy
pip show pandas
```

---

# 5. Program 1 — Matrix Operations Using NumPy

## Real-Time Scenario

Suppose two departments have student marks represented as matrices.

We can use NumPy to perform:

* Addition
* Subtraction
* Matrix multiplication
* Transpose
* Determinant
* Inverse

## Step 1: Import NumPy

```python
import numpy as np
```

NumPy provides efficient tools for numerical calculations.

---

## Step 2: Create Matrices

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
```

The matrices are:

```text
A = | 1  2 |
    | 3  4 |

B = | 5  6 |
    | 7  8 |
```

---

## Step 3: Matrix Addition

```python
print(A + B)
```

Output:

```text
[[ 6  8]
 [10 12]]
```

The corresponding elements are added:

```text
1 + 5 = 6
2 + 6 = 8
3 + 7 = 10
4 + 8 = 12
```

---

## Step 4: Matrix Subtraction

```python
print(A - B)
```

Output:

```text
[[-4 -4]
 [-4 -4]]
```

---

## Step 5: Matrix Multiplication

```python
print(np.dot(A, B))
```

Output:

```text
[[19 22]
 [43 50]]
```

Matrix multiplication is different from element-by-element multiplication.

---

## Step 6: Transpose

```python
print(A.T)
```

Output:

```text
[[1 3]
 [2 4]]
```

The rows become columns.

---

## Step 7: Determinant

```python
print(np.linalg.det(A))
```

For matrix A:

```text
|1 2|
|3 4|
```

The determinant is:

```text
(1 × 4) - (2 × 3)
= 4 - 6
= -2
```

---

## Step 8: Inverse

```python
print(np.linalg.inv(A))
```

NumPy calculates the inverse matrix.

> Note: A matrix has an inverse only when its determinant is non-zero.

---

# 6. Program 2 — Descriptive Statistics Using Pandas

## Real-Time Scenario

Suppose a teacher wants to analyze student examination marks.

The dataset is:

| Name    | Age | Marks |
| ------- | --: | ----: |
| Alice   |  20 |    85 |
| Bob     |  21 |    78 |
| Charlie |  19 |    92 |
| David   |  22 |    88 |
| Eva     |  20 |    76 |

We can use Pandas to calculate statistical information.

---

## Step 1: Import Pandas

```python
import pandas as pd
```

---

## Step 2: Create the Dataset

```python
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "Age": [20, 21, 19, 22, 20],
    "Marks": [85, 78, 92, 88, 76]
}

df = pd.DataFrame(data)
```

`DataFrame` is Pandas' table-like data structure.

---

## Step 3: Display the Data

```python
print(df)
```

Output:

```text
      Name  Age  Marks
0    Alice   20     85
1      Bob   21     78
2  Charlie   19     92
3    David   22     88
4      Eva   20     76
```

---

## Step 4: Generate Descriptive Statistics

```python
print(df.describe())
```

This provides information such as:

* Count
* Mean
* Standard deviation
* Minimum
* 25th percentile
* Median
* 75th percentile
* Maximum

---

## Step 5: Calculate Mean

```python
print(df["Marks"].mean())
```

The mean is calculated as:

```text
(85 + 78 + 92 + 88 + 76) / 5
= 419 / 5
= 83.8
```

Therefore, the average mark is:

```text
83.8
```

---

## Step 6: Calculate Median

```python
print(df["Marks"].median())
```

Sorted marks:

```text
76, 78, 85, 88, 92
```

The middle value is:

```text
85
```

Therefore, the median is **85**.

---

## Step 7: Find Minimum and Maximum

```python
print(df["Marks"].min())
print(df["Marks"].max())
```

Output:

```text
76
92
```

This tells us that:

* Lowest mark = 76
* Highest mark = 92

---

# 7. Program 3 — Data Cleaning Using Pandas

## Real-Time Scenario

In real-world applications, data is rarely perfect.

For example, a college database might contain:

* Missing age
* Missing marks
* Duplicate student records
* Incorrect data types

Consider this dataset:

| Name    |     Age |   Marks |
| ------- | ------: | ------: |
| Alice   |      20 |      85 |
| Bob     |      21 |      78 |
| Charlie | Missing |      92 |
| David   |      22 | Missing |
| Alice   |      20 |      85 |

There are two problems:

1. Charlie's age is missing.
2. David's marks are missing.
3. Alice appears twice.

---

# 8. Step-by-Step Data Cleaning

## Step 1: Import Libraries

```python
import pandas as pd
import numpy as np
```

---

## Step 2: Create the Dataset

```python
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Alice"],
    "Age": [20, 21, np.nan, 22, 20],
    "Marks": [85, 78, 92, np.nan, 85]
}

df = pd.DataFrame(data)
```

`np.nan` represents a missing numerical value.

---

## Step 3: Identify Missing Values

```python
print(df.isnull())
```

To count missing values in each column:

```python
print(df.isnull().sum())
```

Example output:

```text
Name      0
Age       1
Marks     1
dtype: int64
```

This tells us:

* Name has 0 missing values.
* Age has 1 missing value.
* Marks has 1 missing value.

---

# 9. Fill Missing Values

## Fill Missing Age

We can replace the missing age with the average age:

```python
df["Age"] = df["Age"].fillna(df["Age"].mean())
```

The available ages are:

```text
20, 21, 22, 20
```

Mean:

```text
83 / 4 = 20.75
```

The missing age is therefore replaced with approximately:

```text
20.75
```

---

## Fill Missing Marks

```python
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())
```

Available marks:

```text
85, 78, 92, 85
```

Mean:

```text
340 / 4 = 85
```

Therefore, David's missing mark is replaced with:

```text
85
```

---

# 10. Remove Duplicate Records

Alice appears twice in the dataset.

We can remove duplicate rows using:

```python
df = df.drop_duplicates()
```

The duplicate Alice record is removed.

---

# 11. Convert Data Type

Age may contain decimal values after filling the missing value.

If whole-number ages are required:

```python
df["Age"] = df["Age"].astype(int)
```

This converts the Age column to an integer type.

---

# 12. Final Cleaned Dataset

After cleaning, the data looks approximately like:

```text
      Name  Age  Marks
0    Alice   20   85.0
1      Bob   21   78.0
2  Charlie   20   92.0
3    David   22   85.0
```

The dataset is now ready for further analysis.

---

# 13. Important Pandas Functions

| Function               | Purpose                               |
| ---------------------- | ------------------------------------- |
| `pd.DataFrame()`       | Create a DataFrame                    |
| `df.head()`            | Display first rows                    |
| `df.tail()`            | Display last rows                     |
| `df.describe()`        | Generate descriptive statistics       |
| `df.isnull()`          | Find missing values                   |
| `df.isnull().sum()`    | Count missing values                  |
| `df.fillna()`          | Fill missing values                   |
| `df.dropna()`          | Remove rows containing missing values |
| `df.drop_duplicates()` | Remove duplicate records              |
| `df.astype()`          | Change data type                      |

---

# 14. Important NumPy Functions

| Function          | Purpose                          |
| ----------------- | -------------------------------- |
| `np.array()`      | Create an array                  |
| `np.dot()`        | Matrix multiplication            |
| `np.linalg.det()` | Calculate determinant            |
| `np.linalg.inv()` | Calculate matrix inverse         |
| `.T`              | Transpose a matrix               |
| `np.nan`          | Represent missing numerical data |

---

# 15. How to Run the Programs

Open the project folder in VS Code or a terminal.

### Run Matrix Operations

```bash
python matrix_operations.py
```

### Run Descriptive Statistics

```bash
python descriptive_statistics.py
```

### Run Data Cleaning

```bash
python data_cleaning.py
```

---

# 16. Learning Flow

The recommended learning order is:

```text
              Python Basics
                   |
                   ↓
              NumPy Arrays
                   |
                   ↓
          Matrix Operations
                   |
                   ↓
             Pandas DataFrame
                   |
                   ↓
       Descriptive Statistics
                   |
                   ↓
             Data Cleaning
                   |
                   ↓
          Data Analysis Project
```

---

# 17. Real-World Applications

These concepts are commonly used in:

### Education

Analyzing:

* Student marks
* Attendance
* Examination results
* Class performance

### Business

Analyzing:

* Sales
* Revenue
* Customers
* Products

### Healthcare

Analyzing:

* Patient records
* Test results
* Medical measurements

### Finance

Analyzing:

* Stock prices
* Transactions
* Revenue
* Expenses

### Data Science

NumPy and Pandas are fundamental tools for:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Data Analysis
      ↓
Statistics
      ↓
Visualization
      ↓
Machine Learning
```

---

# 18. Summary

In this project, we learned how to:

* Create and manipulate matrices using **NumPy**.
* Perform matrix addition and subtraction.
* Perform matrix multiplication.
* Calculate transpose, determinant, and inverse.
* Create datasets using **Pandas DataFrame**.
* Calculate mean, median, minimum, maximum, variance, and standard deviation.
* Generate descriptive statistics using `describe()`.
* Identify missing data.
* Fill missing values.
* Remove duplicate records.
* Convert column data types.

These are essential foundations for **Python data analysis and data science**.
