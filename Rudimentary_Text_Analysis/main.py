# Libraries
import wordcloud as wc
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

print(type(reviews))

"""
Stage 1: Tokenization and Normalization
"""
# Convert all reviews to lowercase
reviews = reviews.str.lower()


# Tokenize the reviews
reviews = reviews.apply(wt)


"""
Stage 2 : Stopwords and Punctuation Removal
"""
# Remove stopwords
sws = set(stopwords.words('english'))

# Define a function to remove stopwords


def remove_stopwords(tokens):
    return [token for token in tokens if token not in sws]


reviews = reviews.apply(remove_stopwords)

# Remove punctuation and non-alphabetic characters

# Define a function to remove punctuation and non-alphabetic characters


def remove_punctuation(tokens):
    return [token for token in tokens if token.isalpha()]


# Apply the function to the reviews
reviews = reviews.apply(remove_punctuation)


# Combine all tokens into a single list
reviews_all = reviews.sum()

# Create word frequency distribution
freq_dist = nltk.FreqDist(reviews_all)

# Plot the frequency distribution
max = 10
freq_dist_plot = freq_dist.plot(max, title=(f"Top {max} Most Frequent Words"))

# Using bar plot
# Create document term matrix
dtm = pd.DataFrame(
    freq_dist.most_common(max),
    columns=['Word', 'Frequency']
)

dtm.plot.bar(x='Word', y='Frequency', title=(f"Top {max} Most Frequent Words"))

# Wordcloud circle
wordcloud = wc.WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate_from_frequencies(freq_dist)

wordcloud.to_image()

# Save the wordcloud as an image
wordcloud.to_file('wordcloud.png')


# Generate wc from dtm
# Convert dtm to a dictionary
dtmd = dict(dtm.values.tolist())
wcdtm = wc.WordCloud().generate_from_frequencies(dtmd)
wcdtm.to_image()

"""
Stage 3 : Stemming
"""
# Define a function to stem tokens


def stem_tokens(tokens):
    stemmer = snowball.SnowballStemmer('english')
    return [stemmer.stem(token) for token in tokens]


# Apply the function to the reviews
reviews = reviews.apply(stem_tokens)
