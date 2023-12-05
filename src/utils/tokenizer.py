from nltk.stem import PorterStemmer
import nltk

class Tokenizer:
    def __init__(self):
        self.tokenizer = nltk.word_tokenize
        self.stemmer = PorterStemmer()
        
    def __call__ (self, text):
        '''The above function takes in a text and returns a list of stemmed tokens.
        
        Parameters
        ----------
        text
            The "text" parameter is a string that represents the input text that needs to be tokenized and
        stemmed.
        
        Returns
        -------
            A list of stemmed tokens.
        
        '''
        return [self.stemmer.stem(t) for t in self.tokenizer(text)]

    def __repr__ (self):
        return "Tokenizer()"
