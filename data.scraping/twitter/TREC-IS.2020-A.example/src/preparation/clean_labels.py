import numpy as np
import pandas as pd
import re
import os

from xml_parsing import XMLTopicsParser
from get_tweets import TweetViewer, yeld_chunks
from get_tweets import make_chunks


def simplify_event_ids(labels: pd.DataFrame, pattern=r'\w+\d{4}') -> pd.DataFrame:
    """Remove segmented data set indicators from eventID column
    """
    # Save orginal event ids (with segment indicators)
    labels = labels.copy()
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
    labels['Priority'] = numerical_priority
    labels = labels.drop(columns=['postPriority']) # Clean-up

    return labels


def encode_post_categories(labels: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # Get rid of annotations without assigned categories
    labels = labels[labels.postCategories.apply(len) > 0] 

    _ = labels.postCategories.apply(pd.Series).stack()
    one_hot_categories = pd.get_dummies(_).sum(level=0).astype(np.int8)

    labels = labels.join(one_hot_categories)    
    labels = labels.drop(columns=['postCategories']) # Clean-up

    return labels, one_hot_categories


def download_tweets(labels, api_keys_path):
    tweet_viewer = TweetViewer(api_keys_path=api_keys_path)
    tweets = []
    for labels_chunk in make_chunks(labels):
        status_objs = tweet_viewer.pull_tweets(
            tweet_ids=list(labels_chunk.postID.astype(str)),
            tweet_mode='extended', map_=True)
        tweets += [s._json for s in status_objs]
        print(f'Got {len(tweets)} tweets...')
    tweets = pd.DataFrame(tweets)

    return tweets


if __name__  == '__main__':
    print('Importing and cleaning raw labels...')
    raw_labels_path = 'data/raw/train/labels/TRECIS_2018_2019-labels.json'
    raw_labels = pd.read_json(raw_labels_path, orient='records',
                              dtype=dict(postID=np.str_))
    print(raw_labels.info())
    print('Annotations before dropping duplicates...', len(raw_labels))
    raw_labels = raw_labels.drop_duplicates('postID', keep='last')
    print('Annotations after dropping duplicates...', len(raw_labels))
    raw_labels = simplify_event_ids(raw_labels)

    # Copy un-processed categories
    _cols = ['eventType', 'eventID',  'postID', 'postCategories']
    post_categories = raw_labels.copy()[_cols]

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
    print(raw_tweets.info())

    # Drop no-longer-available tweets from both raw tweets and label dfs
    print('Dropping no longer available tweets...')
    unavailable_tweets_ids = raw_tweets[raw_tweets.id_str.isna()]['id']
    print(f'{len(unavailable_tweets_ids)} tweets are no longer available')
    raw_tweets = raw_tweets[~raw_tweets.id.isin(unavailable_tweets_ids)]
    labels = labels[~labels.postID.isin(unavailable_tweets_ids.astype(np.str_))]
    print(f'{len(raw_tweets)} tweets available')
    print(f'{len(labels)} annotations available')

    # Drop non-english tweets
    # print('Dropping non-english tweets...')
    # lang_en = raw_tweets.lang == 'en'
    # print(f'{len(raw_tweets[~lang_en])} non-english tweets found')
    # raw_tweets = raw_tweets[lang_en]
    # labels = labels[labels.postID.isin(raw_tweets.id.astype(np.str_))]

    # Check if no irrelevant tweets were downloaded
    print('Dropping unannotated tweets...')
    print(f'All tweets have annotations... {set(raw_tweets.id_str) == set(labels.postID)}')
    raw_tweets = raw_tweets[raw_tweets.id_str.isin(labels.postID)]

    # Save processed label data
    print('Saving labels...')
    processed_labels_dir = 'data/processed/train/labels'
    os.makedirs(processed_labels_dir, exist_ok=True)
    processed_labels_path = os.path.join(processed_labels_dir, 
                                         'TRECIS_2018_2019-labels.jsonl')
    cols = (['eventType', 'eventID', 'postID']
            + list(categories.columns)
            + ['Priority'])
    processed_labels = labels[cols]
    processed_labels.to_json(processed_labels_path, 
                             orient='records', lines=True)

    # Saving un-processed categories
    print('Saving un-processed categories...')
    post_categories_path = os.path.join(processed_labels_dir, 
                                        'TRECIS_2018_2019-categories.jsonl')
    post_categories.to_json(post_categories_path, orient='records', lines=True)

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