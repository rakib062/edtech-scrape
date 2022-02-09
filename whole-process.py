import os
import sys
from os.path import join

sys.path.append('helpers/')

import text_processing, topic_modeling

data_root ='./data/'
preprocessed_tweet_file = join(data_root,'tweet-text-en.txt')
tweet_df_file = join(data_root,'ed-tweets-all-en.pkl')
topic_modeling_dir = join(data_root, 'topic_modeling')

if not os.path.exists(topic_modeling_dir):
	os.makedirs(topic_modeling_dir)
if not os.path.exists(join(topic_modeling_dir, 'ldamodels')):
	os.makedirs(join(topic_modeling_dir, 'ldamodels'))


#################### Preprocessing Tweet Texts ####################
print('#################### Preprocessing Tweet Texts ####################\n')
text_processing.create_preprocessed_tweet_data(outfile=preprocessed_tweet_file, data_frame_file=tweet_df_file, append=False)

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

