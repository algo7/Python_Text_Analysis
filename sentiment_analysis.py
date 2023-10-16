# Libraries
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import matplotlib.pyplot as plt
from nrclex import NRCLex
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize as wt
import seaborn as sns
nltk.download('averaged_perceptron_tagger')

# Read the preprocessed reviews
preprocessed_reviews = pd.read_csv('preprocessed_reviews.csv')

sample_review = preprocessed_reviews['review']


# Tokenize the reviews
reviews = sample_review.apply(wt)

# Define a function to extract adjectives and adverbs


def extract_adj_adv(tokens):
    pos_tags = nltk.pos_tag(tokens)
    adjectives_adverbs = [
        word for word, tag in pos_tags if tag in ('JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS')
    ]
    """"
    JJ: Adjective (e.g., "happy")

    JJR: Adjective, comparative (e.g., "happier")

    JJS: Adjective, superlative (e.g., "happiest")

    RB: Adverb (e.g., "happily")

    RBR: Adverb, comparative (e.g., "more happily")

    RBS: Adverb, superlative (e.g., "most happily")
    """
    return adjectives_adverbs


# Apply the function to extract adjectives and adverbs
reviews_adj_adv = reviews.apply(extract_adj_adv)


# Define a function to perform sentiment analysis
def analyze_emotion(words_list, type):
    text = ' '.join(words_list)
    text_object = NRCLex(text)
    if type == 'raw':
        return text_object.raw_emotion_scores
    else:
        return text_object.affect_frequencies


# Apply the function to analyze sentiments
sentiments = reviews_adj_adv.apply(analyze_emotion, args=('freq',))


"""
Saving the data
"""
# Convert the Series of dictionaries into a DataFrame
emotions_df = sentiments.apply(pd.Series)

# You might want to concatenate this DataFrame with your original reviews for comprehensive data.
final_data = pd.concat([preprocessed_reviews, emotions_df], axis=1)

# Write the final data to a csv file
final_data.to_csv('final_data.csv', index=False)


"""
Visualization
"""
# Choose a specific review (the 1st one in this case)
specific_emotion = sentiments.iloc[0]

# Plot the sentiment frequencies for the specific review
sns.barplot(x=list(specific_emotion.keys()), y=list(specific_emotion.values()))
plt.title("Emotion Distribution for a Specific Review")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

"""
Note:
Emotions like "joy" and "positive" have higher median frequencies, suggesting these emotions are more commonly expressed across reviews.
Emotions such as "fear" and "anger" show a lot of outliers towards the lower end, indicating that these emotions might not be frequently expressed in most reviews. However, when they are expressed, it can be in a pronounced manner in some reviews.

From these observations, one could infer that when reviews are written, they often reflect strong positive feelings (e.g., "joy", "positive"). However, strong negative feelings (e.g., "fear", "anger") might be less consistently expressed, but when they do appear, they can be quite pronounced.

To draw a parallel with the frequency-based analysis: The raw scores reinforce the idea that reviews tend to be polarized, either expressing strong positive sentiments or intense negative emotions. Neutral emotions or emotions with ambiguous valence, like "anticipation", exhibit a broader range of expression, indicating that their presence is more varied across reviews.
"""

# Plot emotions_df as a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=emotions_df)
plt.title("Distributions of Emotions Across All Reviews")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()


# Calculate the mean frequency for each sentiment across all reviews
emotion_means = emotions_df.mean()

# Plot the average sentiment frequencies across all reviews
plt.figure(figsize=(10, 6))
sns.barplot(x=emotion_means.index, y=emotion_means.values)
plt.title("Average Emotion Distributions Across All Reviews")
plt.ylabel("Average Frequency")
plt.xticks(rotation=45)
plt.show()


# Topic Modeling
# Latent Dirichlet Allocation (LDA)
# This creates a mapping from words to their integer IDs
dictionary = Dictionary(reviews)

# creates a "bag-of-words" representation for a review. It returns a list of (word_id, word_frequency) tuples.
corpus = [dictionary.doc2bow(review) for review in reviews]

lda_model = LdaModel(
    # The bag-of-words representation of the reviews.
    corpus=corpus,
    # A mapping from word IDs to words, which helps interpret the topics.
    id2word=dictionary,
    # The number of topics the model should discover.
    num_topics=5,
    # This ensures reproducibility (the seed)
    random_state=42,
    # The number of times the algorithm should traverse the corpus
    passes=15,
    # Per-word topic assignments should be computed, not just per-document topic distributions
    per_word_topics=True
)

# Print the topics. -1 - all topics will be in result
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}\n")


# Prepare the LDA model visualization
lda_vis = gensimvis.prepare(lda_model, corpus, dictionary)

# Display the visualization
pyLDAvis.display(lda_vis)
