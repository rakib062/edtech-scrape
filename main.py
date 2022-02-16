import sys, csv,os
import collect_tweets
import json
import argparse
# import combine_dataframes



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
    parser.add_argument("--task", type=str, required=True, choices=["collect-tweet", "combine-csv-dfs", "merge-dfs"], help='what do you want to do?')
    parser.add_argument("--out_dir", required=True, type=str, default='', help='out put directory')
    parser.add_argument( "--in_dir", type=str, default='', help="input directory")
    parser.add_argument( "--kw_file", type=str, help="files containig search keywords/hashtags/...")
    parser.add_argument( "--stat_dir", type=str, default='', help="tweet stat dir")
    
  
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()

    # if  args.task=='merge-dfs':
    #     combine_dataframes.merge_tweet_dfs(indir=args.indir, tagfile=args.kwfile, outdir=args.outdir)
    if args.task=='collect-tweet':
        collect_tweets(args.out_dir, args.kw_file, args.stat_dir)

