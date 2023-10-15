# Libraries
import pandas as pd

"""
Python Native Data Structures
"""
# List
# Create a list
list = ["Jan", "Feb", "Mar", "Apr", "May"]
list.append("Jun")
print(list)

# Tuple
# Create a tuple
tuple = ("Jan", "Feb", "Mar", "Apr", "May")
# tuple.append("Jun") # This will throw an error
print(tuple)

# Dictionary
# Create a dictionary
dictionary = {
    "Jan": 31,
    "Feb": 28,
    "Mar": 31,
    "Apr": 30,
    "May": 31
}
dictionary["Jun"] = 30
print(dictionary)

# Set
set = {"Jan", "Feb", "Mar", "Apr", "May"}
set.add("Dec")
print(set)

set.add("Jan")  # This will not add Jan to the set
print(set)

# print(set[0]) # This will throw an error

# Check for membership
print("Jan" in set)


"""
Control Flow
"""
# If-Else
# Check if x is greater than 5
x = 5
if x > 3:
    print("x is greater than 3")

# Check if num is in nums
num = 10
nums = [1, 2, 3, 4, 5]
if num not in nums:
    print(f"{num} is not in {nums}")

# For Loop
# Print each element in nums
for num in nums:
    print(num)

# For loop with range (right exclusive)
# Print each number from 1 to 10
for index in range(1, 11):
    print(index)

# List comprehension
# Create a list of even numbers from 1 to 10
"""
Syntax: [expression for item in iterable if condition]
Syntax Nested: [expression for outer_loop_variable in outer_iterable for inner_loop_variable in inner_iterable]
"""
# Example 1
# Without list comprehension
numbers = [1, 2, 3, 4, 5]
doubled = []
for num in numbers:
    doubled.append(num * 2)
print(doubled)  # Output: [2, 4, 6, 8, 10]

# With list comprehension
numbers = [1, 2, 3, 4, 5]
doubled = [num * 2 for num in numbers]
print(doubled)  # Output: [2, 4, 6, 8, 10]

# Example 2
# Without list comprehension
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
doubled_matrix = []

for row in matrix:
    doubled_row = []
    for num in row:
        doubled_row.append(num * 2)
    doubled_matrix.append(doubled_row)

print(doubled_matrix)
# Output: [[2, 4, 6], [8, 10, 12], [14, 16, 18]]

# With list comprehension
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

doubled_matrix = [[num * 2 for num in row] for row in matrix]
print(doubled_matrix)

"""
Pandas Data Structures
"""
# Pandas Series
# Create a Pandas Series
series = pd.Series(["Jan", "Feb", "Mar", "Apr", "May"])
print(series)

# Create a Pandas Series with custom index
series = pd.Series(
    data=[100, 86, 50],
    index=["Student1", "Student2", "Student3"]
)
print(series)


# Pandas DataFrame
# Create a Pandas DataFrame
df = pd.DataFrame(
    {
        "Name": ["John", "Jane", "Jack", "Jill"],
        "Age": [23, 21, 22, 24]
    }
)
print(df)

# Accessing 1st column by index
print(df.iloc[:, 0])

# Accessing 2nd column by name
print(df.loc[:, "Age"])


"""
Vectorized Operations
"""
# List vs Pandas Series
list = [1, 2, 3, 4, 5]
print(list/2)  # This will throw an error

ps = pd.Series(list)
print(ps/2)

# Panda DataFrame
df = pd.DataFrame(
    {
        "Revenue": [500, 600, 550, 450],
        "Cost": [23, 21, 22, 24]
    }
)
print(df/2)
print(df["Revenue"] - df["Cost"])
