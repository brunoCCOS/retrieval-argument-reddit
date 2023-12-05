#%%
from utils.generative import Generator
from utils.scrap import RedditDataset
from utils.tokenizer import Tokenizer
from utils.embedding import Embedding
from utils.similarity import *
import pandas as pd

def main():

    comments = pd.read_csv('../data/the-reddit-dataset-dataset-comments.csv')
    posts = pd.read_csv('../data/the-reddit-dataset-dataset-posts.csv')

    print('Extracting IDs')
    comments['post_id'] = comments['permalink'].apply(lambda x: extract_post_id(x))
    posts['post_id'] = posts['permalink'].apply(lambda x: extract_post_id(x))


    posts['selftext'].fillna(value="",inplace=True)
    posts['text'] = posts['title'] + posts['selftext']

    # Create a tokenizer
    tokenizer = Tokenizer()
    # # Tokenize the dataset
    print('Tokenizing sentences')
    posts['tokenized_text'] = posts['text'].apply(lambda x: tokenizer(x))
    comments['tokenized_text'] = comments['body'].apply(lambda x: tokenizer(x))

    # Create an Embedding object
    embedding = Embedding()

    print('Creating word representation')
    # # Perform the embeddings
    posts['word_representation'] = posts['text'].apply(lambda x: embedding(x))
    comments['word_representation'] = comments['body'].apply(lambda x: embedding(x))

    print('Creating embeddings representation')

    posts['embedding_representation'] = posts['text'].apply(lambda x: embedding(x,'embedding'))
    comments['embedding_representation'] = comments['body'].apply(lambda x: embedding(x,'embedding'))

if __name__ == "__main__":
    main()
# %%

