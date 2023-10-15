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

"""
Stage 1: Tokenization and Normalization
"""
# Convert all reviews to lowercase
reviews = reviews.str.lower()
print(reviews)

# Tokenize the reviews
reviews = reviews.apply(wt)
print(reviews)

"""
Stage 2 : Stopwords and Punctuation Removal
"""
# Remove stopwords
sws = set(stopwords.words('english'))

# Define a function to remove stopwords


def remove_stopwords(tokens):
    return [token for token in tokens if token not in sws]


reviews = reviews.apply(remove_stopwords)
print(reviews)

# Remove punctuation and non-alphabetic characters

# Define a function to remove punctuation and non-alphabetic characters


def remove_punctuation(tokens):
    return [token for token in tokens if token.isalpha()]


# Apply the function to the reviews
reviews = reviews.apply(remove_punctuation)
print(reviews)


"""
Stage 3 : Stemming
"""
# Define a function to stem tokens


def stem_tokens(tokens):
    stemmer = snowball.SnowballStemmer('english')
    return [stemmer.stem(token) for token in tokens]

# Apply the function to the reviews
reviews = reviews.apply(stem_tokens)

print(reviews)