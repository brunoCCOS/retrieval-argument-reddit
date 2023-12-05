import gensim
from gensim.models import KeyedVectors
import gensim.downloader as api
import os
import numpy as np

class Embedding():
    def __init__(self,pre_vocab = []):
        self.vocab = pre_vocab
        self.model_path = 'model/numberbatch-en.txt'
        self.model = KeyedVectors.load_word2vec_format(self.model_path, binary=False)
        return

    def __call__(self,corpus,embedding_type='word'):
        '''The function takes a corpus and an embedding type as input, creates a vocabulary based on the
        corpus, and then vectorizes the corpus either using bag-of-words or word embeddings depending on
        the specified embedding type.
        
        Parameters
        ----------
        corpus
            The `corpus` parameter is a list of sequences. Each sequence represents a document or a
        sentence in the corpus.
        embedding_type, optional
            The `embedding_type` parameter specifies the type of embedding to use for vectorizing the
        corpus. It can take two values:
        
        Returns
        -------
            a vectorized version of the input corpus. The specific type of vectorization depends on the
        value of the `embedding_type` parameter. If `embedding_type` is set to 'word', the function will
        return a bag-of-words representation of each sequence in the corpus. If `embedding_type` is set
        to 'embedding', the function will return a word embedding representation of each
        
        '''
        self.vocab = self.create_vocab(corpus)
        vectorized_corpus = []
        if embedding_type == 'word':
            for sequence in corpus:
                vectorized_corpus.append(self.vectorize_bow(sequence))
        elif embedding_type == 'embedding':
            for sequence in corpus:
                vectorized_corpus.append(self.vectorize_w2v(self.model,sequence))
        return vectorized_corpus
    

    def create_vocab(self,texts):
        """Create a combined vocabulary from multiple texts."""
        vocabulary = set()
        for text in texts:
            vocabulary.update((text))
        return sorted(vocabulary)

    def vectorize_bow(self,text):
        """Convert text to a vector based on the given vocabulary."""
        return np.array([text.count(word) for word in self.vocab])

    def vectorize_w2v(self,model,preprocessed_text):
        """ Convert preprocessed text into its embedding representation using the given model. """
        embeddings = []
        for word in preprocessed_text:
            if word in model:  # Check if the word is in the model's vocabulary
                embeddings.append(model[word])
        return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)