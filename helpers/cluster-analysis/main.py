import clustering 
import preprocess
import embedding

from sklearn.metrics import pairwise_distances_argmin_min
import sys, os
import npmi
import argparse
import string
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import gensim
# import pdb
import math
import random
import fasttext
import faulthandler 
import pickle

faulthandler.enable() #for stacktrace

NSEEDS = 1

def trace(frame, event, arg):
    print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace

# sys.settrace(trace)

def check_paths(paths):
    for path in paths:
        if not os.path.exists(path):
            print('Path not exists: ', path)
            return False
    return True

def main():
    args = parse_args()

    check_paths([args.ftmodel, args.vocab])

    stopwords = set([line.strip() for line in open(args.stopwords_file)])
    vocab = []#preprocess.create_global_vocab(args.vocab)or line in open(vocab_file):
    data = []
    for line in open(args.vocab):
        data.append(line)
        vocab+=line.split()
        
    vocab=set(vocab)


    train_word_to_file, train_w_to_f_mult, files = preprocess.create_vocab_and_files(stopwords, args.dataset, args.preprocess, "train", vocab)
    # train_word_to_file, train_w_to_f_mult, files = preprocess.create_vocab_preprocess_(stopwords, data, vocab, args.preprocess)
    files_num = len(files)
    print("len vocab size:", len(train_word_to_file.keys()))

    
    intersection = None
    words_index_intersect = None

    tf_idf = embedding.get_tfidf_score(files, train_word_to_file)

    # if False: #args.entities == "word2vec":
        # model = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
        # intersection, words_index_intersect  = embedding.find_intersect(model.vocab,  train_w_to_f_mult, model, files_num, args.entities, args.weight_type)
    # elif args.entities == "fasttext":
        # ft = fasttext.load_model(args.ftmodel)
        # intersection, words_index_intersect = embedding.create_entities_ft(ft, train_w_to_f_mult, args.weight_type)
        # print(intersection.shape)
    # elif args.entities == "KG" or args.entities == "glove" :
    #     elmomix = [float(a) for a in args.elmomix.split(";")] if args.elmomix != "" else None
    #     data, word_index = embedding.read_entity_file(args.entities_file, args.id2name, train_word_to_file, args.entities, elmomix=elmomix)
    #     intersection, words_index_intersect = embedding.find_intersect(word_index, train_w_to_f_mult, data, files_num, args.entities, args.weight_type)
    print('ftmodel: ', args.ftmodel)
    ft_model = fasttext.load_model(args.ftmodel)
    intersection, words_index_intersect = embedding.create_entities_ft(ft_model, train_w_to_f_mult, args.weight_type)
    print(intersection.shape)

    if args.use_dims:
        intersection = clustering.PCA_dim_reduction(intersection, args.use_dims)

    weights = None
    tfdf = None
    if os.path.isfile(args.weight_file):
        weights = pickle.load(open(args.weight_file, 'rb'))
        if os.path.isfile(args.tfidf_file):
            tfdf = pickle.load(open(args.tfidf_file, 'rb'))
    else:
        if args.weight_type == "WGT":
            weights = embedding.get_weights_tf(words_index_intersect, train_w_to_f_mult)

        if args.weight_type == "robust":
            weights = embedding.get_rs_weights_tf(words_index_intersect, train_w_to_f_mult)

        if args.weight_type == "tfdf":
            weights , tfdf = embedding.get_weights_tfdf(words_index_intersect, train_w_to_f_mult, files_num)

        pickle.dump(weights, open(args.weight_file, 'wb'))
        pickle.dump(tfdf, open(args.tfidf_file, 'wb'))

    if weights is not None and args.scale == "sigmoid":
        print("scaling.. sigmoid")
        weights = 1 / (1 + np.exp(weights))
    elif weights is not None and args.scale == "log":
        print("scaling.. log")
        weights = np.log(weights)





    # dev_word_to_file, dev_word_to_file_mult, dev_files = preprocess.create_vocab_and_files(stopwords, args.dataset,args.preprocess, "valid", vocab)
    # dev_files_num = len(dev_files)


    # test_word_to_file, test_word_to_file_mult, test_files = preprocess.create_vocab_and_files(stopwords, args.dataset,args.preprocess, "test", vocab)
    # test_files_num = len(test_files)




    topics_npmi = []

    for topics in args.num_topics:
        npmis = []

        print("Number of Clusters:" + str(topics))
        rand = 0
        global NSEEDS
        while rand < NSEEDS:

            model_outfile = None
            if args.model_outdir:
                model_outfile = '{}/model-{}-{}-{}.model'.format(args.model_outdir, args.clustering_algo, topics, rand)

            try:
                top_k_words, topk_indices = cluster(args.clustering_algo, intersection, words_index_intersect, topics, args.rerank, weights, args.topics_file, rand, model_outfile=model_outfile)
            except:
                print("Warning: failed, try diff random seed.")
                new_rand = random.randint(5,1000)
                top_k_words, topk_indices = cluster(args.clustering_algo, intersection, \
                        words_index_intersect, topics, args.rerank, weights, args.topics_file, new_rand,  model_outfile=model_outfile)



            top_k_words = rerank(args.rerank, top_k_words, topk_indices, train_w_to_f_mult, train_word_to_file, tf_idf, tfdf)

            print(type(top_k_words))
            print(top_k_words)
            #evaluate through NMPI, currently the train set is used instead of any test/validation set
            val = npmi.average_npmi_topics(top_k_words, len(top_k_words), train_word_to_file, files_num) 

            if np.isnan(val):
                NSEEDS +=1
                rand += 1
                continue

            npmi_score = np.around(val, 5)
            print("NPMI:" + str(npmi_score))
            npmis.append(npmi_score)

            rand += 1

        topics_npmi.append(np.mean(npmis))
        print("NPMI Mean:" + str(np.around(topics_npmi[-1], 5)))
        print("NPMI Var:" + str(np.around(np.var(npmis), 5)))

        print('\n******************************************\n')

    best_topic = args.num_topics[np.argmax(topics_npmi)]







def cluster(clustering_algo, intersection, words_index_intersect, num_topics, rerank, weights, topics_file, rand, model_outfile=None):
    if clustering_algo == "KMeans":  
        predicted_labels, topk_indices, model= clustering.KMeans_model(intersection, words_index_intersect, num_topics, rerank, rand, weights)
        if model_outfile:
            pickle.dump(model, open(model_outfile, 'wb'))
    # elif clustering_algo == "SPKMeans":
    #     labels, top_k  = clustering.SphericalKMeans_model(intersection, words_index_intersect, num_topics, rerank, rand, weights)
    # elif clustering_algo == "GMM":
    #     labels, top_k = clustering.GMM_model(intersection, words_index_intersect, num_topics, rerank, rand)
    # elif clustering_algo == "KMedoids":
    #     labels, top_k  = clustering.KMedoids_model(intersection,  words_index_intersect,  num_topics, rand)
    # elif clustering_algo == "VMFM":
    #     labels, top_k = clustering.VonMisesFisherMixture_Model(intersection, words_index_intersect, num_topics, rerank, rand)

    # #Affinity matrix based
    # elif clustering_algo == "DBSCAN":
    #     k=6
    #     labels, top_k  = clustering.DBSCAN_model(intersection,k)
    # elif clustering_algo == "Agglo":
    #     labels, top_k  = clustering.Agglo_model(intersecticlustering_algoon, num_topics, rand)
    # elif clustering_algo == "Spectral":
    #     labels, top_k  = clustering.SpectralClustering_Model(intersection,num_topics, rand,  weights)

    # if clustering_algo == 'from_file':
    #     with open('bert_topics.txt', 'r') as f:
    #         top_k_words = f.readlines()
    #     top_k_words = [tw.strip().replace(',', '').split() for tw in top_k_words]

    # elif clustering_algo == 'LDA':
    #     with open(topics_file, 'r') as f:
    #         top_k_words = f.readlines()
    #     top_k_words = [tw.strip().replace(',', '').split() for tw in top_k_words]
    #     for i, top_k in enumerate(top_k_words):
    #         top_k_words[i] = top_k_words[i][2:12]
    # else:
    #     bins, top_k_words = sort(labels, top_k,  words_index_intersect)
    bins, top_k_words = sort(predicted_labels, topk_indices,  words_index_intersect)
    return top_k_words, np.array(topk_indices)


def rerank(rerank, top_k_words, top_k, train_w_to_f_mult, train_w_to_f, tf_idf, tfdf):
    if rerank=="tf":
        top_k_words =  clustering.rank_freq(top_k_words, train_w_to_f_mult)
    elif rerank=="tfidf":
        top_k_words = clustering.rank_td_idf(top_k_words, tf_idf)

    elif rerank=="tfdf":
        top_k_words = clustering.rank_td_idf(top_k_words, tfdf)

    elif rerank=="graph":
        #doc_matrix = npmi.calc_coo_matrix(words_index_intersect, train_word_to_file)
        top_k_words = clustering.rank_centrality(top_k_words, top_k, train_w_to_f)
    return top_k_words






def sort(labels, indices, word_index):
    bins = {}
    index = 0
    top_k_bins = []
    for label in labels:
        if label not in bins:
            bins[label] = [word_index[index]]
        else:
            bins[label].append(word_index[index])
        index += 1;
    for i in range(0, len(indices)):
        ind = indices[i]
        top_k = []
        for word_ind in ind:
            top_k.append(word_index[word_ind])
        top_k_bins.append(top_k)
    return bins, top_k_bins

def print_bins(bins, name, type):
    f = open(name + "_" + type + "_corpus_bins.txt","w+")
    for i in range(0, 20):
        f.write("Bin " + str(i) + ":\n")
        for word in bins[i]:
            f.write(word + ", ")
        f.write("\n\n")

    f.close()

def print_top_k(top_k_bins, name, type):
    f = open(name + "_" + type + "_corpus_top_k.txt","w+")
    for i in range(0, 20):
        f.write("Bin " + str(i) + ":\n")
        top_k = top_k_bins[i]
        for word in top_k:
            f.write(word + ", ")
        f.write("\n\n")
    f.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--entities", type=str, choices=["word2vec", "fasttext", "glove", "KG"], required=True)
    parser.add_argument("--ftmodel", type=str, help='fasttext model file', required=True)
    parser.add_argument("--vocab", required=True,  type=str)#, nargs='+', default=[])
    parser.add_argument( "--stopwords_file", type=str, required=True, help="file containing stopwords")
    parser.add_argument( "--entities_file", type=str, help="entity file")
    parser.add_argument( "--weight_file", type=str, required=True, help="file to save to/load from vocab weights")
    parser.add_argument( "--tfidf_file", type=str, required=True, help="file to save to/load from vocab tf-idf weights")
    parser.add_argument("--model_outdir",  type=str)#, nargs='+', default=[])
    
    parser.add_argument("--clustering_algo", type=str, required=True, choices=["KMeans", "SPKMeans", "GMM", "KMedoids","Agglo","DBSCAN","Spectral","VMFM",
        'from_file', 'LDA'])

    parser.add_argument( "--topics_file", type=str, help="topics file")

    parser.add_argument('--use_dims', type=int)
    parser.add_argument('--num_topics',  nargs='+', type=int, default=[20])
    parser.add_argument("--weight_type", type=str, choices=["SVD", "DUP", "WGT", "robust", \
    "logtfdf"])
    parser.add_argument("--rerank", type=str, choices=["tf", "tfidf", "tfdf", "graph"]) \

    parser.add_argument('--id2name', type=Path, help="id2name file")

    parser.add_argument("--dataset", type=str, default ="20NG", choices=["20NG", "children", "reuters"])

    parser.add_argument("--preprocess", type=int, default=5, help='cuttoff threshold for words to keep in the vocab based on frequency')

    parser.add_argument( "--elmomix", type=str, default="", help="elmomix coefficients, separated by ';', should sum to 1")

    parser.add_argument("--scale", type=str, required=False)


    args = parser.parse_args()
    return args



if __name__ == "__main__":
    main()
