import pickle
import sys
from collections import defaultdict
import re
import string
from tweetokenize import *
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import bigrams
import nltk
from gensim import corpora, models
import gensim
from tqdm import tqdm
from scipy import spatial
from urllib.parse import urlparse
import pandas as pd
from os.path import join
import text_processing
#from gensim.models import CoherenceModel
from gensim.models.coherencemodel import CoherenceModel

#nltk.data.path.append("/nfs/juhu/data/rakhasan/nltk-downloads/")

tokenizer = Tokenizer()



class TextIterator(object):
    def __init__(self, fname):
        self.filename = fname
    def __iter__(self):
        for doc in open(self.filename):
            # assume there's one document per line and words are separated by space
            yield doc.strip().split()


class CorpusIterator(object):
    def __init__(self, fname, dictionary):
        self.filename = fname
        self.dictionary = dictionary
    def __iter__(self):
       for doc in open(self.filename): # assume there's one document per line
            yield self.dictionary.doc2bow(doc.strip().split())

def create_ngram_models(data_words, min_count=5, threshold=100):
    # higher threshold -> fewer phrases.
    bigram = gensim.models.Phrases(data_words, min_count=min_count, threshold=threshold) 
    trigram = gensim.models.Phrases(bigram[data_words], threshold=threshold)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return bigram_mod, trigram_mod

def train_lda_model(dictionary, corpus, num_of_topics):
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, 
        num_topics=num_of_topics, id2word = dictionary,passes=5)
    return ldamodel

# def create_bigrams(data_words, bigram_mod):
#     '''Takes a list of list-of-words and returns a list of list-of-words    '''
#     return [bigram_mod[tokens] for tokens in data_words]

def make_trigrams(data_words, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in data_words]

def create_dictionary_and_corpus(text_corpus_file, outpath):
    text_corpus=[line.split() for line in open(text_corpus_file)]
    dictionary= corpora.Dictionary(text_corpus) 
    pickle.dump(dictionary,open(join(outpath,'dictionary.pkl'),'wb'))
    corpus = [dictionary.doc2bow(text) for text in text_corpus]
    pickle.dump(corpus,open(join(outpath,'corpus.pkl'),'wb'))
    return dictionary,corpus

 
def create_trigrams(textfile, outpath):
    '''
        Create trigrams (and bigrams as a byproduct)
        params
            textfile: a file containing (tweet) text in each line
            outpath: path where the models and trigrams will be stored
    '''
    bigram_mod, trigram_mod = create_ngram_models(TextIterator(textfile))
    trigrams = make_trigrams(TextIterator(textfile), bigram_mod, trigram_mod)
    pickle.dump((bigram_mod,trigram_mod), open(join(outpath,'bigram-trigram-models.pkl'), 'wb'))
    f = open(join(outpath,'trigrams.txt'),'w') # trigrams will be used for later processing
    for t in trigrams:
        if len(t)>0:
            f.write(' '.join(t)+'\n')
    f.close()

def calculate_topic_similarities(reference_tweet_words, tweets_words, model, dictionary, dim=10):
    '''
    Returns list of similarity scores between the reference tweet and all other tweets. Each tweet is sent 
    as a list of words after preprocessing.
    '''
    ref_topic = model.get_document_topics(dictionary.doc2bow(reference_tweet_words))
    ref_topic = sorted(ref_topic, key=lambda x: (x[1]), reverse=True)[:dim]
    ref_topic = [t[1] for t in ref_topic]
    scores = []
    for t in tweets_words:
        topic = model.get_document_topics(dictionary.doc2bow(t))
        topic = sorted(topic, key=lambda x: (x[1]), reverse=True)[:dim]
        topic = [t[1] for t in topic]
        scores.append(1 - spatial.distance.cosine(ref_topic, topic))
    return scores


def compute_coherence_values(preprocessed_text_file, dictionary_file, corpus_file,
                         out_dir, limit=500, start=100, step=100):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    preprocessed_text_file : file containing preprocessed text
    dictionary : Gensim dictionary
    corpus : Gensim corpus (not text)
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    print('loading data...')
    texts = [line.split() for line in open(preprocessed_text_file)]
    dictionary = pickle.load(open(dictionary_file,'rb'))
    corpus = pickle.load(open(corpus_file,'rb'))

    coherence_values = []
    model_list = []
    print('training models...')
    for num_topics in range(start, limit, step ):
        print('\tbuilding model with num-topics: ', num_topics)
        model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, 
            id2word = dictionary, passes=10)
        model_list.append(model)
        model.save(out_dir+'/model_'+str(num_topics))
        try:
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            cv = coherencemodel.get_coherence()
            coherence_values.append(cv)
            print('\tcoherence value:{}'.format(cv))
        except Exception as e:
            print(e)
        print()        
#
    return model_list, coherence_values