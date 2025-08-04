
# 1. INTRODUCTION:

Pandas is an open-source Python library designed specifically for data analysis and manipulation. Known for its speed, power, flexibility, and ease of use, Pandas is built on the **NumPy** library. It is highly effective for working with tabular data, similar to tables in SQL or spreadsheets in Excel. Pandas offers a wide range of functions for cleaning, analyzing, and modeling data, helping you discover key insights within datasets.

### Key Data Structures

Pandas provides two fundamental data structures:

* **Series**: A one-dimensional array with a labeled index, similar to a single column in a dataset.
* **DataFrame**: The most important and widely used two-dimensional table-like structure. It consists of multiple rows and columns, where each column is essentially a **Series**.

### Key Functions and Data Processing Capabilities

Pandas provides a powerful toolkit for various data analysis tasks:

* **Data Manipulation and Exploration**: It supports reading and writing data from common formats like CSV, Excel, JSON, and SQL. Functions such as `head()`, `info()`, `describe()`, and `dtypes` help quickly inspect data structure, basic statistics, and data types.

* **Data Selection and Filtering**: It offers flexible methods to access data based on labels (`.loc[]`, `.at[]`) or integer positions (`.iloc[]`, `.iat[]`).

* **Data Transformation and Aggregation**: It allows you to group data (`groupby()`) to compute aggregate statistics, sort data (`sort_values()`), and apply custom functions (`apply()`) to rows or columns.

* **Data Cleaning**: It has flexible capabilities for handling missing data by filling values (`fillna()`) or dropping them (`dropna()`). It also supports modifying data formats (e.g., converting to numbers, dates) and removing duplicate records (`drop_duplicates()`).

* **Time Series Analysis**: It integrates advanced techniques like `rolling()` for calculating statistics on moving windows and `resample()` for changing the data frequency. It also includes time-based indexing to access data by specific dates and times.

### Common Applications

Pandas is widely used in various fields:

* **Data Science**: Used for data preprocessing, exploratory data analysis (EDA), and feature extraction.

* **Machine Learning (ML)**: Supports normalizing and transforming input data for machine learning models.

* **Business and Financial Analysis**: Used to analyze revenue, costs, profits, and customer segments.

* **Big Data / Log Data Processing**: Helps clean, filter, and transform data from large systems.

* **Data Visualization**: Combines with other libraries like Matplotlib and Seaborn to create charts and visual representations of data.

* **Office Task Automation**: Automates reading, updating, and writing reports from Excel/CSV files, streamlining day-to-day office tasks.

# 2. Pandas

Pandas is a important library in **Data Science**, **Data Analysis**, and
**Machine Learning**.

## 2.1 Installation and Import

To use Pandas, first install it:

``` {caption="Install Pandas"}
pip install pandas
```

Then, import it into your Python script:

``` {caption="Import Libraries"}
import pandas as pd
```

## 2.2 Data Loading

Pandas supports multiple common data formats.

### Read CSV File

    df = pd.read_csv('data.csv')

### Read Excel File

    df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

### Read from JSON

    df = pd.read_json('data.json')

### Read from URL (public data)

    url = 'https://pandas/data/aio.csv'
    df = pd.read_csv(url)


## 2.3 Series: One-Dimensional Data Structure

A `Series` is a labeled one-dimensional array, representing a single
column of data.

### Create a Series

    ages = pd.Series([23, 25, 22, 24], name='Age')
    print(ages)

**Output:**

    0    23
    1    25
    2    22
    3    24
    Name: Age, dtype: int64

### Common Series Operations

\- `series.head(n)`: Display first $n$ elements

\- `series.tail(n)`: Display last $n$ elements

\- `series.mean()`: Compute mean

\- `series.isnull()`: Check for missing values

\- `series.fillna(value)`: Fill missing values

\- `series.astype(dtype)`: Change data type

## 2.4 DataFrame: Two-Dimensional Table

The `DataFrame` is Pandas' primary structure --- a 2D labeled data
table, similar to a spreadsheet.

## Create a DataFrame

    df = pd.DataFrame(data)
    print(df)

**Output:**

       Name  Age 
    0    An   23     
    1  Binh   25      
    2   Chi   22      
    3 Duong   24 

### Common DataFrame Operations

\- `df.head()`: First 5 rows

\- `df.tail()`: Last 5 rows

\- `df.shape`: Dimensions (rows, columns)

\- `df.columns`: Column names

\- `df.dtypes`: Data types of columns

\- `df.info()`: Summary of the DataFrame

\- `df.describe()`: Descriptive statistics

\- `df.drop_duplicates()`: Remove duplicate rows

\- `df.reset_index()`: Reset index after filtering

## 2.5 Data Retrieval

### Access Columns

    print(df['Name'])           # Series
    print(df[['Name', 'Score']]) # DataFrame

### Access Rows: `.loc[]` (by label/index)

    print(df.loc[0])                    # row 0
    print(df.loc[1:2, ['Name', 'Major']]) # rows 1–2, columns 'Name', 'Major'

### Access Rows: `.iloc[]` (by position)

    print(df.iloc[0])           # first row
    print(df.iloc[0:2, 0:2])    # first 2 rows, first 2 columns

### Filtering Data

    print(df[df['Age'] > 23])
    print(df[df['Major'] == 'CS'])
    print(df[(df['Major'] == 'CS') & (df['Score'] > 8.0)])

## 2.6 Data Cleaning

### Check for Missing Values

    print(df.isnull())        # True/False matrix
    print(df.isnull().sum())  # Count NaN per column

### Fill Missing Values

    df['Score'] = df['Score'].fillna(df['Score'].mean())

### Remove Duplicates

    df.drop_duplicates(inplace=True)

### Change Data Types

    df['Age'] = df['Age'].astype('int32')
    df['Major'] = df['Major'].astype('category')

### Reset Index After Filtering

    filtered_df = df[df['Score'] > 8.0]
    filtered_df = filtered_df.reset_index(drop=True)


**Pandas** is an indispensable tool when working with data in Python.
Mastering basic operations helps you confidently handle real-world data.



