import os
import sys
import time

import numpy as np
import pandas as pd
import yaml
import tweepy


class TweepyDownloader():
    """Dowloader of Tweets using Tweepy library API. Refreshes rate limit statuses
    depending on the current rate limit status' reset time rather than default 
    hard-coded Tweepy behaviour (15 minutes after every excess of rate limit)
    """

    def __init__(self, api_keys_path):
        self.api = tweepy.API(self.make_auth_handler(api_keys_path))

    def make_auth_handler(self, api_keys_path):
        with open(api_keys_path, 'r') as fp:
            api_keys = yaml.safe_load(fp)
            consumer_key = api_keys['consumer_key']
            consumer_secret = api_keys['consumer_secret']
            auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
        return auth 

    def pull_tweets(self, tweet_ids, **lookup_kwargs):
        while True:
            try:
                statuses = self.api.statuses_lookup(tweet_ids, **lookup_kwargs)
            except tweepy.RateLimitError as err:
                epsilon = 5  # seconds of additional time.sleep()
                rls = self.api.rate_limit_status('statuses')
                sleep_for = np.ceil(rls['resources']['statuses']
                                       ['/statuses/lookup']['reset']
                                    - time.time()) + epsilon
                msg = 'Error {0}, sleeping for {1:.0f}s'
                print(msg.format(err.reason, sleep_for))
                time.sleep(sleep_for)
            else:
                return statuses

def yield_chunks(df, chunk_size=100):
    return (list(df.iloc[i:i+chunk_size, df.columns.get_loc('postID')]) 
            for i in range(0, len(df), chunk_size))


def main():
    tweet_labels_path = 'data/raw/labels/TRECIS_2018_2019-labels.json'
    api_keys_path = 'api_keys.yml'

    tweet_labels = pd.read_json(tweet_labels_path, orient='records', 
                                dtype=dict(postID=np.str_))
    loader = TweepyDownloader(api_keys_path)

    tweet_json = []
    kwargs = dict(tweet_mode='extended', map_=True)
    for id_chunk in yield_chunks(tweet_labels):
        tweet_json += [s._json for s in loader.pull_tweets(id_chunk, **kwargs)]
        print(f'Got {len(tweet_json)} tweets...')

    tweets = pd.DataFrame(tweet_json)

    target_path = 'data/raw/tweets'
    target_filename = 'TRECIS_2018_2019-tweets.jsonl'

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    tweets.to_json(os.path.join(target_path, target_filename), 
                                orient='records', lines=True)
    msg = 'Saved to... {0}'
    print(msg.format(os.path.join(os.path.join(target_path, target_filename))))


if __name__  == '__main__':
    main()