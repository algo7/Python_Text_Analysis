# Libraries
import wordcloud as wc
import pandas as pd
import nltk
from nltk.stem import snowball
from nltk.tokenize import word_tokenize as wt
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

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
# Stopwords list
sws = set(stopwords.words('english'))

# Custom stopwords list
custom_sws = {'hotel', 'room', 'rooms', 'staff',
              'stay', 'stayed', 'night', 'alexander', 'lausanne', 'beau-rivage', 'beau', 'rivage', 'zurich'}
all_sws = sws.union(custom_sws)

# Define a function to remove stopwords


def remove_stopwords(tokens, sw_list):
    return [token for token in tokens if token not in sw_list]


reviews = reviews.apply(remove_stopwords, args=(all_sws,))

# Remove punctuation and non-alphabetic characters

# Define a function to remove punctuation and non-alphabetic characters


def remove_punctuation(tokens):
    return [token for token in tokens if token.isalpha()]


# Apply the function to the reviews
reviews = reviews.apply(remove_punctuation)


"""
Visualization and Wordcloud
"""
# Combine all tokens into a single list
reviews_all = reviews.sum()

# Create word frequency distribution
freq_dist = nltk.FreqDist(reviews_all)

# Plot the frequency distribution
max = 20
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
Bigrams and Trigrams
"""
triGramsFinder = nltk.TrigramCollocationFinder.from_words(reviews_all)
biGramsFinder = nltk.BigramCollocationFinder.from_words(reviews_all)

# Get the 10 most common bigrams
top_bigrams = biGramsFinder.ngram_fd.most_common(10)
top_trigrams = triGramsFinder.ngram_fd.most_common(10)

# Extracting bigram names and their respective frequencies
# [Expression for item in iteratble condition]
bigram_names = [str(bigram[0]) for bigram in top_bigrams]
bigram_freqs = [bigram[1] for bigram in top_bigrams]

# Extracting trigram names and their respective frequencies
trigram_names = [str(trigram[0]) for trigram in top_trigrams]
trigram_freqs = [trigram[1] for trigram in top_trigrams]

# Plotting
plt.figure(figsize=(12, 8))

# Bigram subplot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
plt.barh(bigram_names, bigram_freqs, color='skyblue')
plt.xlabel('Frequency')
plt.ylabel('Bigrams')
plt.title('Top 10 Most Frequent Bigrams in Reviews')
plt.gca().invert_yaxis()

# Trigram subplot
plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 2
plt.barh(trigram_names, trigram_freqs, color='salmon')
plt.xlabel('Frequency')
plt.ylabel('Trigrams')
plt.title('Top 10 Most Frequent Trigrams in Reviews')
plt.gca().invert_yaxis()

plt.tight_layout(pad=5.0)  # To ensure that the subplots do not overlap


"""
Stage 3 : Stemming (Optional)
"""
# Define a function to stem tokens


# def stem_tokens(tokens):
#     stemmer = snowball.SnowballStemmer('english')
#     return [stemmer.stem(token) for token in tokens]


# # Apply the function to the reviews
# reviews = reviews.apply(stem_tokens)


"""
Exporting Pre-processed Data
"""
pd.DataFrame(reviews)

# Joining the tokens back into strings
reviews_joined = reviews.apply(' '.join)

# Save the preprocessed reviews to a CSV
reviews_joined.to_csv('preprocessed_reviews.csv',
                      index=False, header=["review"])
