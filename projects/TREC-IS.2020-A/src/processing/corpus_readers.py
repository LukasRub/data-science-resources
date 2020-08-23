from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.corpus.reader.twitter import TwitterCorpusReader

import os

CAT_PATTERN = r'(\w+)/.*'
DOC_PATTERN = r'.*.jsonl'
ATTRIBS = ['full_text']


class TRECISTweetCorpusReader(CategorizedCorpusReader, TwitterCorpusReader):
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
        self.single_cat = self.fileids(categories='Advice')
        
    
    def strings(self, fileids=None):
        """
        Returns only the text content of Tweets in the file(s)

        Overrides TwitterCorpusReader.strings method to return 
        'full_text' attribute of a Tweet instead of ``text``

        :return: the given file(s) as a list of Tweets.
        :rtype: list(str)
        """
        if fileids is None:
            fileids = self.single_cat
            
        fulltweets = self.docs(fileids)
        tweets = []
        
        for jsono in fulltweets:
            try:
                text = jsono['full_text']
                if isinstance(text, bytes):
                    text = text.decode(self.encoding)
                tweets.append(text)
            except KeyError:
                pass
            
        return tweets
    
    
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