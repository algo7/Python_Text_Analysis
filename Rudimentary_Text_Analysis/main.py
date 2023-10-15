# Libraries
import pandas as pd
import nltk
from nltk.stem import snowball
from nltk.tokenize import word_tokenize as wt
from nltk.corpus import stopwords

# Download nltk components
nltk.download('punkt')
nltk.download('stopwords')

# Read the data from a csv file
data = pd.read_csv(
    'https://raw.githubusercontent.com/algo7/Python_Text_Analysis/main/Data/Hotel_Alexander.csv')

"""
Data Exploration
"""
# Print the column names
data.columns

# List the first 5 rows
data.head(n=5)

# Print the shape/dimension of the data
data.shape

# Get basic information about the data
data.info()


"""
Data Preprocessing
"""
