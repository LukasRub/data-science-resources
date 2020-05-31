import numpy as np
import pandas as pd
import re
import os

from xml_parsing import XMLTopicsParser
from get_tweets import TweetViewer
from get_tweets import make_chunks


def simplify_event_ids(labels: pd.DataFrame, pattern=r'\w+\d{4}') -> pd.DataFrame:
    """Remove segmented data set indicators from eventID column
    """
    # Save orginal event ids (with segment indicators)
    labels['datasetID'] = labels.eventID 

    event_ids = labels.eventID.apply(lambda x: re.match(pattern, x).group())
    labels.eventID = event_ids

    return labels


def join_with_event_types(labels: pd.DataFrame, topics_path: str) -> pd.DataFrame:
    """Merge labels with event types from the proper topics XML file
    """
    xml_parser = XMLTopicsParser(topics_path)
    event_types = xml_parser.parse_attribs(['dataset', 'type'])

    labels = labels.copy()
    labels = pd.merge(labels, event_types, left_on='eventID', 
                      right_on='dataset', copy=False)
    
    
    labels.eventType = labels.type
    labels = labels.drop(columns=['type', 'dataset'])  # Clean-up

    return labels


def encode_post_priority(labels: pd.DataFrame, mapping='default') -> pd.DataFrame:
    if mapping is 'default':
        mapping = {
            'Critical': 1.0,
            'High': 0.75,
            'Medium': 0.5,
            'Low': 0.25
        }
    
    labels = labels.copy()

    # Dropping postPriority values that are not in mapping keys
    priority_cat = labels.postPriority
    not_in_mapping = priority_cat.apply(lambda x: x not in mapping.keys())
    labels = labels.drop(labels.index[not_in_mapping], axis='index')

    numerical_priority = labels.postPriority.map(mapping)
    labels.postPriority = numerical_priority

    return labels


def encode_post_categories(labels: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    _ = labels.postCategories.apply(pd.Series).stack()
    one_hot_categories = pd.get_dummies(_).sum(level=0)

    labels = labels.join(one_hot_categories)    
    labels = labels.drop(columns=['postCategories']) # Clean-up

    return labels, one_hot_categories


def download_tweets(labels: pd.DataFrame, api_keys_path: str) -> pd.DataFrame():
    tweet_viewer = TweetViewer(api_keys_path=api_keys_path)
    tweets = []
    for labels_chunk in make_chunks(labels):
        status_objs = tweet_viewer.pull_tweets(list(labels_chunk.postID))
        tweets += [s._json for s in status_objs]
        print(f'Got {len(tweets)} tweets...')
    tweets = pd.DataFrame(tweets)

    return tweets


if __name__  == '__main__':
    print('Importing and cleaning raw labels...')
    raw_labels_path = 'data/raw/train/labels/TRECIS_2018_2019-labels.json'
    raw_labels = pd.read_json(raw_labels_path)
    raw_labels = raw_labels.drop_duplicates('postID', keep='last')
    raw_labels = simplify_event_ids(raw_labels)

    print('Joining with topics file...')
    topics_path = 'data/raw/TRECIS-2018-2019.topics.xml'
    labels = join_with_event_types(raw_labels, topics_path)

    # Encode categorical labels
    print('Encoding categorical labels...')
    labels = encode_post_priority(labels)
    labels, categories = encode_post_categories(labels)

    # Download tweets
    print('Downloading tweets...')
    api_keys_path = 'api_keys.yml'
    raw_tweets = download_tweets(labels, api_keys_path)

    # Drop no-longer-available tweets from both raw tweets and label dfs
    print('Dropping no longer available tweets')
    unavailable_tweets_idx = raw_tweets[raw_tweets.id_str.isna()].index
    labels = labels.drop(unavailable_tweets_idx, axis='index')
    raw_tweets = raw_tweets.drop(unavailable_tweets_idx, axis='index')

    # Save processed label data
    print('Saving labels...')
    processed_labels_dir = 'data/processed/train/labels'
    os.makedirs(processed_labels_dir, exist_ok=True)
    processed_labels_path = os.path.join(processed_labels_dir, 
                                         'TRECIS_2018_2019-labels.jsonl')
    cols = (['eventType', 'eventID', 'postID']
            + list(categories.columns)
            + ['postPriority'])
    processed_labels = labels[cols]
    processed_labels.to_json(processed_labels_path, 
                             orient='records', lines=True)

    # Save raw tweet data
    print('Saving tweets...')
    raw_tweets_dir = 'data/raw/train/tweets'
    os.makedirs(raw_tweets_dir, exist_ok=True)
    raw_tweets_path = os.path.join(raw_tweets_dir, 
                                   'TRECIS_2018_2019-tweets.jsonl')
    raw_tweets.to_json(raw_tweets_path, orient='records', lines=True)
        
    
    # Exporting to separate datasets and folders
    # datasets = labels.groupby(['eventType', 'eventID'])

    # for (event_type, event_id), indices in datasets.items()
    #     path = os.path.join(path, event_type)
    #     os.makedirs(path, exist_ok=True)
        
    #     data = datasets.get_group((event_type, event_id)).copy()
    #     pd.to_json(os.path.join(path, f'{tweet_id}.jsonl'))