import os
import re
import csv

import twitter
from tqdm import tqdm
import numpy as np


BASE_DIR = os.path.abspath(os.path.dirname(__file__))



class MSRManager:
    DEFAULT_DIR = 'MSRSocialMediaConversationCorpus'
    TURNING_FILENAME = 'twitter_ids.tuning.txt'
    VALIDATION_FILENAME = 'twitter_ids.validation.txt'
    TWITTER_CONSUMER_KEY = os.getenv('TWITTER_CONSUMER_KEY')
    TWITTER_CONSUMER_SECRET = os.getenv('TWITTER_CONSUMER_SECRET')
    TWITTER_ACCESS_TOKEN_KEY = os.getenv('TWITTER_ACCESS_TOKEN_KEY')
    TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

    def download_tweets(self, validation=False, dist='.data', reload=False):
        class load_tweets:
            def __init__(self, validation=False):
                filename = self.VALIDATION_FILENAME if validation else self.TURNING_FILENAME
                filepath = os.path.join(BASE_DIR, self.DEFAULT_DIR, filename)
                self.filepath = filepath
                self.f = open(filepath, 'r+')
                self.api = twitter.Api(consumer_key=self.TWITTER_CONSUMER_KEY,
                                       consumer_secret=self.TWITTER_CONSUMER_SECRET,
                                       access_token_key=self.TWITTER_ACCESS_TOKEN_KEY,
                                       access_token_secret=self.TWITTER_ACCESS_TOKEN_SECRET)

            def __iter__(self):
                return self

            def __next__(self):
                line = self.f.readline()
                if not line:
                    self.f.close()
                    raise StopIteration()
                tweet_ids = line.split('\t')
                tweet_ids = tweet_ids[:-1] + [tweet_ids[-1].replace('\n', '')]
                tweets = [self.get_tweet(tid) for tid in tweet_ids]
                tweets = [tweet for tweet in tweets if tweet is not None]
                return tweets

            def __len__(self):
                return sum(1 for line in open(self.filepath))

            def get_tweet(self, tweet_id):
                try:
                    data = self.api.GetStatus(tweet_id)
                except twitter.error.TwitterError:
                    return None
                text = data.text
                return re.sub(r'@\w+ ', '', text)

        filename = 'validation.tsv' if validation else 'turning.tsv'
        dirpath = os.path.join(BASE_DIR, dist)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filepath = os.path.join(dirpath, filename)

        if os.path.exists(filepath) and not reload:
            return filepath

        print('Downloading corpus datasets...')
        with open(filepath, 'w+') as f:
            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            for tweets in tqdm(load_tweets(validation)):
                if len(tweets) > 1:
                    for _ in range(len(tweets)-3):
                        tweets.append('')
                    writer.writerow(tweets)
        return filepath

    def fix_tsv(self, validation=False, dist='.data'):
        filename = 'validation.tmp.tsv' if validation else 'turning.tmp.tsv'
        dirpath = os.path.join(BASE_DIR, dist)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filepath = os.path.join(dirpath, filename)

        with open(filepath, 'w+') as wf:
            writer = csv.writer(wf, delimiter='\t', lineterminator='\n')
            with open(filepath.replace('.tmp', ''), 'r') as rf:
                reader = csv.reader(rf, delimiter='\t')
                for row in reader:
                    print(len(row))
                    for _ in range(3-len(row)):
                        row.append('')
                    writer.writerow(row)

def msr_corpus(dist='.data', reload=False):
    validation_tsv = download_tweets(validation=False, dist=dist, reload=reload)
    turning_tsv = download_tweets(validation=True, dist=dist, reload=reload)
    dtype = {
        'names':('s1', 's2', 's3'),
        'formats':('S280', 'S280', 'S280')
    }
    return dict(
        validation=np.genfromtxt(validation_tsv, delimiter='\t', dtype=dtype, invalid_raise=False),
        turning=np.genfromtxt(turning_tsv, delimiter='\t', dtype=dtype, invalid_raise=False),
    )
