import sys, csv,os
import collect_tweets

outdir = sys.argv[1]
if not os.path.exists(outdir):
        os.makedirs(outdir)
        
with open('tags.csv') as csv_file:
    reader = csv.reader(csv_file)
    tags = set(list(reader)[0])

tags =  set([t.lower() for t in tags])
print("tags: ", len(tags))
    
donetags = set([])
if os.path.isfile('donetags.csv'): 
    with open('donetags.csv') as csv_file:
        reader = csv.reader(csv_file)
        donetags = set(list(reader)[0])

donetags =  set([t.lower() for t in donetags])
tags = tags.difference(donetags)
print("donetags:{}, tags:{} ".format(len(donetags), len(tags)))

i=1
for tag in tags:
    tag = tag.strip()
    print("Starting search for tag no:{} of {}, tag:{}".format(i, len(tags),tag))
    collect_tweets.search_tweets(tag, outdir, i)
    donetags.add(tag)
    with open('donetags.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        writer.writerow(donetags)
