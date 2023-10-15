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
def analyze_emotion(words_list):
    text = ' '.join(words_list)
    text_object = NRCLex(text)
    return text_object.affect_frequencies


# Apply the function to analyze sentiments
sentiments = reviews_adj_adv.apply(analyze_emotion)

# Convert the Series of dictionaries into a DataFrame
emotions_df = sentiments.apply(pd.Series)

# You might want to concatenate this DataFrame with your original reviews for comprehensive data.
final_data = pd.concat([preprocessed_reviews, emotions_df], axis=1)


# Choose a specific review (the 1st one in this case)
specific_emotion = sentiments.iloc[0]

# Plot the sentiment frequencies for the specific review
sns.barplot(x=list(specific_emotion.keys()), y=list(specific_emotion.values()))
plt.title("Emotion Distribution for a Specific Review")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()


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


dictionary = Dictionary(reviews)
corpus = [dictionary.doc2bow(review)
          for review in reviews]

lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5,
                     random_state=42, passes=15, per_word_topics=True)
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}\n")


# Prepare the LDA model visualization
lda_vis = gensimvis.prepare(lda_model, corpus, dictionary)

# Display the visualization
pyLDAvis.display(lda_vis)


def plot_top_words(lda_model, topic_num, top_n=10):
    """
    Plot the top `top_n` words from a given topic number `topic_num`
    from the LDA model `lda_model`.
    """
    terms = lda_model.show_topic(topic_num, topn=top_n)
    term_names = [term[0] for term in terms]
    term_weights = [term[1] for term in terms]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(term_names, term_weights, align='center', color='blue')
    ax.invert_yaxis()  # display the top word at the top
    ax.set_xlabel('Weight')
    ax.set_title(f"Top {top_n} words for topic {topic_num + 1}")
    plt.show()


# Loop over all topics and plot the top words for each topic
for i in range(lda_model.num_topics):
    plot_top_words(lda_model, i)
