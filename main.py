import sys, csv,os
import collect_tweets
import json
import argparse
import combine_dataframes



def collect_tweets(outdir, tagfile, stat_dir):
    if not os.path.exists(outdir):
            os.makedirs(outdir)
            
    with open(tagfile) as csv_file:
        reader = csv.reader(csv_file)
        tags = set(list(reader)[0])

    tags =  set([t.strip().lower() for t in tags])
    print("tags: ", len(tags))
        
    donetags = set([])
    if os.path.isfile('donetags.csv'): 
        with open('donetags.csv') as csv_file:
            reader = csv.reader(csv_file)
            donetags = set(list(reader)[0])

    donetags =  set([t.strip().lower() for t in donetags])
    tags = tags.difference(donetags)
    print("donetags:{}, tags:{} ".format(len(donetags), len(tags)))

    i=1
    for tag in tags:
        tag = tag.strip()
        print("Starting search for tag no:{} of {}, tag:{}".format(i, len(tags),tag))
        i+=1
        '''
        Retrieve the latest tweet collected before
        '''
        start_time = '2006-03-21T00:00:00Z'
        since_id = None
        if os.path.isfile('{}/tweet-stat-{}.json'.format(stat_dir, tag)):
            with open('{}/tweet-stat-{}.json'.format(stat_dir, tag), 'r') as fp:
                latest_tweet = json.loads(json.load( fp))
                since_id = latest_tweet['tweetid']


        success = collect_tweets.search_tweets(tag, 
                                    start_time, since_id,outdir, i)

        if success:
            donetags.add(tag)
            with open('donetags.csv', 'w') as csv_file:  
                writer = csv.writer(csv_file)
                writer.writerow(donetags)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--task", type=str, choices=["collect-tweet", "combine-csv-dfs", "merge-dfs"], help='what do you want to do?')
    parser.add_argument("--outdir", type=str, default='', help='out put directory')
    parser.add_argument( "--indir", type=str, default='', help="input directory")
    # parser.add_argument( "--outfile", type=str, default='', help="output file for merged data frames")
    parser.add_argument( "--kwfile", type=str, help="files containig search keywords/hashtags/...")
    parser.add_argument( "--statdir", type=str, default='', help="tweet stat dir")
    parser.add_argument("--rerank", type=str, choices=["tf", "tfidf", "tfdf", "graph"])

    # parser.add_argument("--clustering_algo", type=str, required=True, choices=["KMeans", "SPKMeans", "GMM", "KMedoids","Agglo","DBSCAN","Spectral","VMFM",
    #     'from_file', 'LDA'])

    # parser.add_argument( "--topics_file", type=str, help="topics file")

    # parser.add_argument('--use_dims', type=int)
    # parser.add_argument('--num_topics',  nargs='+', type=int, default=[20])
    # parser.add_argument("--doc_info", type=str, choices=["SVD", "DUP", "WGT", "robust", \
    # "logtfdf"])
     

    # parser.add_argument('--id2name', type=Path, help="id2name file")

    # parser.add_argument("--dataset", type=str, default ="20NG", choices=["20NG", "children", "reuters"])

    # parser.add_argument("--preprocess", type=int, default=5)
    
    parser.add_argument("--vocab", required=True,  type=str, nargs='+', default=[])
    # parser.add_argument("--scale", type=str, required=False)


    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()

    if args.task=='merge-dfs':
        combine_dataframes.merge_tweet_dfs(indir=args.indir, tagfile=args.kwfile, outdir=args.outdir)

