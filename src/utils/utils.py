import numpy as np
import re
import pandas as pd
from scipy.stats import norm

def extract_post_id(link):
    match = re.search(r"comments/([^/]+)/", link)

    if match:
        result = match.group(1)
        return result
    else:
        return np.nan


def compute_wmd(embedding1, embedding2, model):
    '''The function `compute_wmd` calculates the Word Mover's Distance between two sets of word embeddings
    using a given model.
    
    Parameters
    ----------
    embedding1
        The first set of word embeddings.
    embedding2
        The parameter "embedding2" represents the second set of word embeddings. It is a numerical
    representation of words or phrases in a vector space. These embeddings capture the semantic meaning
    of the words or phrases.
    model
        The model parameter refers to the Word2Vec model that has been trained on a corpus of text. This
    model is used to compute the Word Mover's Distance (WMD) between two sets of word embeddings.
    
    Returns
    -------
        the Word Mover's Distance between two sets of embeddings.
    
    '''
    
    return model.wmdistance(embedding1, embedding2)

def print_full(x):
    '''The function `print_full` sets the maximum number of columns to display in a pandas DataFrame and
    then prints the DataFrame.
    
    Parameters
    ----------
    x
        The parameter `x` is the variable or object that you want to print. It can be any data type such as
    a string, list, dictionary, or pandas DataFrame.
    
    '''
    pd.set_option('display.max_columns', 10000)  # or 1000
    print(x)
    pd.reset_option('display.max_rows')

def calc_word_sim(arg,countarg,embedding):
    '''The function calculates the word similarity between two given words using an embedding model.
    
    Parameters
    ----------
    arg
        The "arg" parameter is a string representing the first word or phrase.
    countarg
        The `countarg` parameter is a string representing the counter argument.
    embedding
        The "embedding" parameter is a function that takes a list of words as input and returns their
    corresponding word embeddings. Word embeddings are vector representations of words in a
    high-dimensional space, where words with similar meanings are closer to each other. The function
    should return the word embeddings for the input words.
    
    Returns
    -------
        the sum of the absolute differences between the embeddings of the "arg" word and the "countarg"
    word.
    
    '''
    arg_word, counterarg_word = embedding([arg,countarg])

    return np.sum(np.abs(arg_word - counterarg_word))

def extract_post_id(link):
    '''The function `extract_post_id` takes a link as input and extracts the post ID from the link.
    
    Parameters
    ----------
    link
        The `link` parameter is a string that represents a URL or link to a post on a website.
    
    Returns
    -------
        the post ID extracted from the given link. If a match is found, the post ID is returned as a
    string. If no match is found, it returns np.nan, which is a value representing "Not a Number" from
    the numpy library.
    
    '''
    match = re.search(r"comments/([^/]+)/", link)

    if match:
        result = match.group(1)
        return result
    else:
        return np.nan
    
def calc_sim_dissim(wc,wp,ec,ep,alpha=0.8):
    '''The function calculates the similarity and dissimilarity between two sets of values based on
    weighted counts and proportions, using a specified alpha value.
    
    Parameters
    ----------
    wc
        The parameter wc represents the word count of the current document.
    wp
        The parameter "wp" represents the number of words in the positive class.
    ec
        The parameter "ec" represents the error count for a certain task or calculation.
    ep
        The parameter "ep" represents the number of errors in the predicted output.
    alpha
        The alpha parameter is a weighting factor that determines the balance between similarity and
    dissimilarity in the calculation. It ranges from 0 to 1, where 0 means only dissimilarity is
    considered and 1 means only similarity is considered.
    
    Returns
    -------
        the calculated similarity and dissimilarity score based on the given inputs.
    
    '''
    w_plus = wc + wp
    e_plus = ec + ep
    w_up = np.max([wc,wp])
    e_up = np.max([ec,ep])
    return alpha*(w_plus + e_plus) - (1-alpha)*(w_up+e_up)

def compute_t_test(series1,series2):
    var1 = np.var(series1)/len(series1)
    var2 = np.var(series2)/len(series2)

    mean1 = np.mean(series1)
    mean2 = np.mean(series2)
    crit_value = (mean1-mean2)/np.sqrt(var1 + var2)

    return norm.cdf(crit_value)