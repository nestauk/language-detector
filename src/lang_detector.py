import re
import sys
import pickle

import pandas as pd
from nltk.tokenize import sent_tokenize

import langid
from guess_language import guess_language
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0
REGEX = "http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),](?:%[0-9a-f][0-9a-f]))+"


def remove_links(document, REGEX=REGEX):
    """Remove URLs from document.

    Args:
        document (:obj:`str`): Raw text string.

    Return:
        :obj:`str` document without URLs.

    """
    return re.sub(REGEX, '', document)


def tokenize_sent(document):
    """Tokenize document to sentences.

    Args:
        document (:obj:`str`): Raw text string.

    Return:
        :obj:`list` of sentences.

    """
    return sent_tokenize(document)


def max_chars(documents):
    """Choose the document with the maximum length.

    Args:
        documents (:obj:`list` of :obj:`str`): List of documents.

    Return:
        :obj:`str` with the maximum length in documents.

    """
    return max(documents, key=len)


def detector(document, length=50):
    """Detect the language of a string.

    Args:
        documents (:obj:`list` of :obj:`str`): List of documents.
        length (:obj:`int`): Number of characters in a string.

    Return:
        """
    if isinstance(document, str) and len(remove_links(document)) > length:
        document = remove_links(document)
        document = max_chars(tokenize_sent(document))
        detectA = guess_language(document)
        try:
            detectB = detect(document)
        except LangDetectException:
            detectB = 'unknown'
        detectC = langid.classify(document)[0]
        eng = sum(1 if d == 'en' else 0 for d in [detectA, detectB, detectC])
        if eng >= 2:
            return 1
        else:
            return 0
    else:
        return 0


def main():
    # Read the organization_descriptions.csv
    org_descs = pd.read_csv(sys.argv[1])
    lang = []
    for desc in org_descs.description:
        try:
            lang.append(detector(desc))
        except Exception as e:
            print(e)

    with open('../data/lang.pickle', 'wb') as h:
        pickle.dump(lang, h)


if __name__ == '__main__':
    main()
