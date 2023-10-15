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

"""
Control Flow
"""
# If-Else
# Check if x is greater than 5
x = 5
if x > 5:
    print("x is greater than 5")

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
