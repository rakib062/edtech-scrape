#!/usr/bin/python3

# Standard imports
#import random
import numpy as np
#import pdb
import math
import os, sys
from sklearn.metrics.pairwise import cosine_similarity
# argparser
import argparse
#from distutils.util import str2bool

# Custom imports

def topic_similarity_vec(topic_words, ntopics, ft_model):
    '''
        Computes mean of the pairwise cosine similarity in top-k words in a topic.
        Final score is the average across all topics.
    '''
    all_topics_score = []
    for k in range(ntopics):
        if len(topic_words[k])<3:
            print('insufficient words: ',  ' '.join(topic_words[k]))
            continue
        ntopw = len(topic_words[k]) #number of top words in the current topic
        topic_score = []
        for i in range(ntopw-1):
            for j in range(i+1, ntopw):
                w1 = topic_words[k][i] #first top word
                w2 = topic_words[k][j] #next top word
                sim = cosine_similarity([ft_model.get_word_vector(w1)], [ft_model.get_word_vector(w2)])
                topic_score.append(sim)
        
        print('{:.3f} : {}'.format(np.mean(topic_score), ' '.join(topic_words[k])))
        all_topics_score.append(np.mean(topic_score))

    print('Mean: {:.3f}, var:{:.3f}'.format(np.mean(all_topics_score), np.var(all_topics_score)))

    return np.mean(all_topics_score)


def average_npmi_topics(topic_words, ntopics, word_doc_counts, nfiles):

    eps = 10**(-12)

    all_topics = []
    for k in range(ntopics): #for each topic
        word_pair_counts = 0
        topic_score = []

        ntopw = len(topic_words[k]) #number of top words in the current topic

        for i in range(ntopw-1): 
            for j in range(i+1, ntopw):
                w1 = topic_words[k][i] #first top word
                w2 = topic_words[k][j] #next top word

                w1w2_dc = len(word_doc_counts.get(w1, set()) & word_doc_counts.get(w2, set())) #count how many times both w1 and w2' present  in word_doc_counts
                w1_dc = len(word_doc_counts.get(w1, set())) #how many times only w1 is present
                w2_dc = len(word_doc_counts.get(w2, set())) #how many times only w2 is present

                # what we had previously:
                #pmi_w1w2 = np.log(((w1w2_dc * nfiles) + eps) / ((w1_dc * w2_dc) + eps))

                # Correct eps:
                pmi_w1w2 = np.log((w1w2_dc * nfiles) / ((w1_dc * w2_dc) + eps) + eps)
                npmi_w1w2 = pmi_w1w2 / (- np.log( (w1w2_dc)/nfiles + eps)) #NPMI for w1,w2

                # Sanity check Which is equivalent to this:
                #if w1w2_dc ==0:
                #    npmi_w1w2 = -1
                #else:
                    #pmi_w1w2 = np.log( (w1w2_dc * nfiles)/ (w1_dc*w2_dc))
                    #npmi_w1w2 = pmi_w1w2 / (-np.log(w1w2_dc/nfiles))

                #if npmi_w1w2>1 or npmi_w1w2<-1:
                #    print("NPMI score not bounded for:", w1, w2)
                #    print(npmi_w1w2)
                #    sys.exit(1)

                topic_score.append(npmi_w1w2)

        all_topics.append(np.mean(topic_score))

    for k in range(ntopics):
        print(np.around(all_topics[k],5), " ".join(topic_words[k]))

    avg_score = np.around(np.mean(all_topics), 5)
    #print(f"\nAverage NPMI for {ntopics} topics: {avg_score}")

    return avg_score
