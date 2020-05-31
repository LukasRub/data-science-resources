import pandas as pd
import numpy as np

import tweepy
import time
import yaml
import os


class TweetViewer(object):

    def __init__(self, api_keys_path: str):
        self.api = tweepy.API(self.__create_auth_handler(api_keys_path))
        self.__rate_limit_status = None

    def pull_tweets(self, tweet_ids: list, **lookup_kwargs):
        if self.rate_limit_reached():
            print('Rate limit reached')
            self.__timeout()

        if not hasattr(lookup_kwargs, 'tweet_mode'):
            lookup_kwargs['tweet_mode'] = 'extended'

        if not hasattr(lookup_kwargs, 'map_'):
            lookup_kwargs['map_'] = True

        statuses = self.api.statuses_lookup(tweet_ids, **lookup_kwargs)
        self.__decrement_remaining_request_count()
        
        return statuses

    def rate_limit_reached(self):
        return not (self.rate_limit_status['remaining'] > 0)

    @property
    def rate_limit_status(self):
        if ((self.__rate_limit_status is None) or 
                self.__rate_limit_status['reset'] < time.time()):
            print('Fetching rate limit status... ')
            rls = self.api.rate_limit_status()
            status_lookup_rls = rls['resources']['statuses']['/statuses/lookup']
            print(status_lookup_rls)
            self.__rate_limit_status = status_lookup_rls
        
        return self.__rate_limit_status

    def __decrement_remaining_request_count(self):
        self.__rate_limit_status['remaining'] -= 1

    def __create_auth_handler(self, api_keys_path: str):
        auth = None
        with open(api_keys_path, 'r') as f:
            api_keys = yaml.safe_load(f)
            consumer_key = api_keys['consumer_key']
            consumer_secret = api_keys['consumer_secret']
            auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
        return auth
    
    def __timeout(self):
        epsilon = 5  # Addtional seconds to sleep before requesting rate limit status agin
        sleep_for = (self.rate_limit_status['reset'] 
                     - int(np.floor(time.time()))
                     + epsilon)
        print(f'Sleeping for {sleep_for}...')
        time.sleep(sleep_for) 
        # Reset rate limit status
        self.__rate_limit_status = self.rate_limit_status
    

def make_chunks(df:pd.DataFrame, chunk_size=100):
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    return chunks


if __name__  == '__main__':

    tweet_info_path = "data/utils/tweet_ids.jsonl"
    api_keys_path = "api_keys.yml"

    tweet_info = pd.read_json(tweet_info_path, lines=True)
    tweet_scraper = TweetScraper(api_keys_path=api_keys_path, timeout_secs=600)

    tweets = []
    for chunk in make_chunks(tweet_info):
        _statuses = tweet_scraper.pull_tweets(list(chunk.postID))
        tweets += [s._json for s  in _statuses]
        print(f'Got {len(tweets)} tweets...')
    tweets = pd.DataFrame(tweets)

    # Save to file for use in label preparation pipeline 
    unavail_tweet_ids_path = "data/utils/unavailable_tweets.csv"
    unavailable_tweet = tweets.full_text.isna()
    tweets[unavailable_tweet].id.to_csv(unavail_tweet_ids_path, 
                                        header='id', index=False)
    tweets.drop(tweets[unavailable_tweet].index, axis='index', inplace=True)

    # Join tweet dfs with tweet info dfs for splitting by event id
    info_cols = tweet_info.columns
    data_set = pd.merge(tweets, tweet_info, how='left', left_on='id',  
                        right_on='postID', validate='one_to_one')
    
    # Group by event type and save to appropriate folders
    raw_tweets_path= "data/raw/train/tweets"

    by_event = data_set.groupby('eventID')
    for event_id, index in by_event.groups.items():
        event = by_event.get_group(event_id)
        event.to_json(os.path.join(raw_tweets_path, f'{event_id}.jsonl'), 
                                   orient='records', lines=True)