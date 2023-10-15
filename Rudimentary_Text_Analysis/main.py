# Libraries
import pandas as pd
import nltk
from nltk.stem import snowball
from nltk.tokenize import word_tokenize as wt
from nltk.corpus import stopwords

# Download nltk components
nltk.download('punkt')
nltk.download('stopwords')

# Read the data from a csv file into a pandas dataframe
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
# Extract the review content into a pandas series
reviews = data['content']
print(reviews)
print(type(reviews))

# Convert all reviews to lowercase
reviews = reviews.str.lower()
print(reviews)

# Tokenize the reviews
reviews = reviews.apply(wt)
print(reviews)

# Remove stopwords
sw = stopwords.words('english')


# reviews = reviews.apply(remove_stopwords)
# print(reviews)
