# Libraries
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pandas as pd
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize as wt

# Read the preprocessed reviews
preprocessed_reviews = pd.read_csv('preprocessed_reviews.csv')

sample_review = preprocessed_reviews['review']


# Tokenize the reviews
reviews = sample_review.apply(wt)

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
    per_word_topics=False
)

# Print the topics. -1 - all topics will be in result
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}\n")


# Prepare the LDA model visualization
lda_vis = gensimvis.prepare(lda_model, corpus, dictionary)

# Display the visualization
pyLDAvis.display(lda_vis)