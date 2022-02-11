import os
import sys
from os.path import join
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


sys.path.append('helpers/')

import text_processing, topic_modeling

data_root ='./data/'
preprocessed_tweet_file = join(data_root,'tweet-text-en.csv')
tweet_df_file = join(data_root,'en-tweets-sample.pkl')
# tweet_df_file = join(data_root,'ed-tweets-all-en.pkl')
topic_modeling_dir = join(data_root, 'topic_modeling')

if not os.path.exists(topic_modeling_dir):
	os.makedirs(topic_modeling_dir)
if not os.path.exists(join(topic_modeling_dir, 'ldamodels')):
	os.makedirs(join(topic_modeling_dir, 'ldamodels'))


print('#################### Identifying security/privacy related tweets ####################\n')
sec_priv_kws = [line.strip() for line in  open('privacy_kws')]

def contains_any_keyword(text, kws):
    for kw in kws:
        if kw in text:
            return True
    return False

tweet_df = pd.read_pickle(tweet_df_file)
sec_priv_df = tweet_df.progress_apply(lambda tweet: contains_any_keyword(tweet.text, sec_priv_kws), axis=1)
sec_priv_df = sec_priv_df[sec_priv_df==True]
sec_priv_df = tweet_df.loc[sec_priv_df.index]
sec_priv_df.to_pickle('data/sec_priv_df.pkl')

#################### Preprocessing Tweet Texts ####################
print('#################### Preprocessing Tweet Texts ####################\n')
text_processing.create_preprocessed_tweet_data(outfile=preprocessed_tweet_file, data_frame_file=tweet_df_file)

# print('creating trigrams...')
# topic_modeling.create_trigrams(textfile=preprocessed_tweet_file, outpath=topic_modeling_dir)

# print('creating dictionary and corpus..')
# _,_ = topic_modeling.create_dictionary_and_corpus(text_corpus_file=join(topic_modeling_dir,'trigrams.txt'), outpath=topic_modeling_dir)

# print('training and evaluating lda models..')
# topic_modeling.compute_coherence_values(
#                 preprocessed_text_file = join(topic_modeling_dir,'trigrams.txt'), 
#                 dictionary_file=join(topic_modeling_dir,'dictionary.pkl'),  
#                 corpus_file=join(topic_modeling_dir,'corpus.pkl'), 
#                 out_dir = join(topic_modeling_dir,'ldamodels'), 
#                 limit=500, start=50, step=50)

