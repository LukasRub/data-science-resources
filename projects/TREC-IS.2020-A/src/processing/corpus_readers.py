from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.corpus.reader.twitter import TwitterCorpusReader

import os

CAT_PATTERN = r'(\w+)/.*'
DOC_PATTERN = r'.*.jsonl'
ATTRIBS = ['full_text']


class TweepyRawCorpusReader(CategorizedCorpusReader, TwitterCorpusReader):
    """
    A corpus reader for raw line-delimited JSON documents (Tweets)
    to enable preprocessing.
    
    Examples drawn and adapted from Applied Text Analysis with Python
    by Benjamin Bengfort, Rebecca Bilbro, and Tony Ojeda (O'Reilly).
    978-1-491-96304-3.
    """

    
    def __init__(self, root, fileids=DOC_PATTERN, encoding='utf8', 
                 attribs=ATTRIBS, single_cat='Advice', **kwargs):
        """
        Initialize the corpus reader.  Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor.  The remaining
        arguments are passed to the ``TwitterCorpusReader`` constructor.
        
        ``single_cat`` is the singled-out category from which we draw
        Tweets since they are duplicated across categories with only
        ``target`` attribute values as different.
        """
        
        # Add the default category pattern if not passed into the class.
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN
        
        # Initialize the NLTK corpus reader object
        CategorizedCorpusReader.__init__(self, kwargs)
        TwitterCorpusReader.__init__(self, root, fileids=fileids, encoding=encoding) 
        
        # Save the Tweet attributes that we specifically want to extract.
        self.attribs = attribs
        
        # Store fileids of a singled-out category since Tweets are duplicated 
        # across categories (with only ``target`` attribute values as different)
        # self.single_cat = self.fileids(categories='Advice')
      
    
    def strings(self, fileids=None, categories=None):
        """
        Returns only the text content of Tweets in the file(s)

        Overrides TwitterCorpusReader.strings method to return 
        'full_text' attribute of a Tweet instead of ``text`` to
        accommodate changes to raw data retriewed by Tweepy library

        :return: the given file(s) as a list of Tweet text strings.
        :rtype: list(str)
        """
        fileids = self.resolve(fileids, categories)
            
        full_tweets = self.docs(fileids)
        tweets = []
        
        for jsono in full_tweets:
            try:
                # If a retweet, remove the 'RT @<handle>' string literal
                text = jsono['full_text']
                if isinstance(text, bytes):
                    text = text.decode(self.encoding)
                tweets.append(text)
            except KeyError:
                tweets.append('') 
            
        return tweets


    def target(self, fileids=None, categories=None):
        """
        Returns a list of only the binary target value of Tweets, to be interpreted
        as do they belong to the particular category in the context of docs supplied
        in the fileids or categories parameters.

        :return: a list of binary values to determine if the Tweets belong in their 
                 particular category given the fileids or categories.
        :rtype: list
        """
        
        fileids = self.resolve(fileids, categories)
            
        full_tweets = self.docs(fileids)
        target_labels = []
        
        for jsono in full_tweets:
            try:
                target = jsono['target']
                if not isinstance(target, int):
                    target = int(target)
                target_labels.append(target)
            except KeyError:
                pass
            
        return target_labels
    
    
    def labels(self, fileids=None, categories=None):
        """
        Returns a list of all target labels assigned Tweets in docs that are 
        supplied by the the fileids or categories parameters.

        :return: all labels (categories) assigned to Tweets in particular fileids
                 or categories.
        :rtype: list
        """
        
        fileids = self.resolve(fileids, categories)
            
        full_tweets = self.docs(fileids)
        target_labels = []
        
        for jsono in full_tweets:
            try:
                target = jsono['labels']
                if not isinstance(target, list):
                    target = list(target)
                target_labels.append(target)
            except KeyError:
                pass
            
        return target_labels


    def quoted_status_indicators(self, fileids=None, categories=None):
        fileids = self.resolve(fileids, categories or self.single_cat)

        full_tweets = self.docs(fileids)
        quoted_status_indicators = []
        
        for jsono in full_tweets:
            try:
                is_quoted_status = jsono['is_quoted_status']
                quoted_status = jsono['quoted_status']
                target_labels.append(is_quoted_status and quoted_status)
            except KeyError:
                pass
            
        return quoted_status_indicators

    
    def resolve(self, fileids=None, categories=None):
        """
        Returns a list of fileids or categories depending on what is passed
        to each internal corpus reader function. Implemented similarly to
        the NLTK ``CategorizedPlaintextCorpusReader``.
        """
        
        if fileids is not None and categories is not None:
            raise ValueError('Specify fileids or categories, not both')
        
        if categories is not None:
            return self.fileids(categories)
        
        return fileids
    
     
    def sizes(self, fileids=None, categories=None):
        """
        Returns a list of tuples, the fileid and size on disk of the file.
        This function is used to detect oddly large files in the corpus.
        """
        
        # Resolve fileids and the categories
        fileids = self.resolve(fileids, categories)
        
        # Create a generator, getting every path and compute filesize
        for path in self.abspaths(fileids):
            yield os.path.getsize(path)