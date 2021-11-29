import sys, csv,os
import collect_tweets

outdir = sys.argv[1]
tagfile=sys.argv[2]
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
            tweet = json.loads(json.load((latest_tweet.to_json(), fp)))
            since_id = tweet.tweetid


    collect_tweets.search_tweets(tag+" -is:retweet", 
                                start_time, since_id,outdir, i)

    donetags.add(tag)
    with open('donetags.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        writer.writerow(donetags)
