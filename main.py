import sys, csv,os
import collect_tweets
import json
import argparse
# import combine_dataframes


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
        collect_tweets.collect_tweets(args.out_dir, args.kw_file, args.stat_dir)

