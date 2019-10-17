from __future__ import unicode_literals
from functools import reduce
from typing import Optional

from extractor import Function
import re
import nltk

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Precompile often used regex
first_cap_regex = re.compile('(.)([A-Z][a-z]+)')
all_cap_regex = re.compile('([a-z0-9])([A-Z])')


class NLPreprocessor:
    def preprocess(self, function: Function) -> Function:
        """
        Preprocess a function's comments and identifiers by removing punctuating, removing stopwords and lemmatization
        """
        return Function(
            name=self.process_identifier(function.name),
            docstring=self.process_sentence(function.docstring),
            func_descr=self.process_sentence(function.func_descr),
            arg_names=[self.process_identifier(arg_name) for arg_name in function.arg_names],
            arg_types=function.arg_types,
            arg_descrs=[self.process_sentence(arg_descr) for arg_descr in function.arg_descrs],
            return_type=function.return_type,
            return_expr=[self.process_identifier(expr.replace('return ', '')) for expr in function.return_expr],
            return_descr=self.process_sentence(function.return_descr)
        )

    def process_sentence(self, sentence: str) -> Optional[str]:
        """
        Process a natural language sentence
        """
        if sentence is None:
            return None

        pipeline = [
            SentenceProcessor.replace_digits_with_space,
            SentenceProcessor.remove_punctuation_and_linebreaks,
            SentenceProcessor.tokenize,
            SentenceProcessor.lemmatize,
            SentenceProcessor.remove_stop_words
        ]

        return reduce(lambda s, action: action(s), pipeline, sentence)

    def process_identifier(self, sentence: str) -> str:
        """
        Process a sentence mainly consisting of identifiers

        Similar to process_sentence, but does not remove stop words.
        """
        pipeline = [
            SentenceProcessor.replace_digits_with_space,
            SentenceProcessor.remove_punctuation_and_linebreaks,
            SentenceProcessor.tokenize,
            SentenceProcessor.lemmatize
        ]

        return reduce(lambda s, action: action(s), pipeline, sentence)


class SentenceProcessor:
    """
    A collection of static functions to process a natural language sentence

    Roughly based on https://github.com/sola-da/NL2Type/blob/master/scripts/preprocess_raw_data.py
    """

    @staticmethod
    def replace_digits_with_space(sentence: str) -> str:
        """
        Replaces digits with a space
        """
        return re.sub('[0-9]+', ' ', sentence)

    @staticmethod
    def remove_punctuation_and_linebreaks(sentence: str) -> str:
        """
        Removes and replaces non-textual elements

        Removes whitespace and all punctuation except questions marks. Question marks are replaced with full stops,
        as we do not want to care about questions. Full stops not followed by a space are replace with a space, e.g.
        object.property -> object property.
        """
        sentence = re.sub('[^A-Za-z0-9. ?]+', ' ', sentence) \
            .replace('?', '.') \
            .replace('\n', '') \
            .replace('\r', '')

        return re.sub('\.(?! )', ' ', sentence)

    @staticmethod
    def tokenize(sentence: str) -> str:
        """
        Tokenize camel case and snake case in a sentence and convert the sentence to lower case
        """
        sentence = sentence.replace("_", " ")
        sentence = SentenceProcessor.convert_camelcase(sentence)

        return sentence.lower()

    @staticmethod
    def lemmatize(sentence: str) -> str:
        """
        Lemmatize a sentence (e.g. running -> run)
        """
        words = [word for word in sentence.split(' ') if word != '']

        lemmatized = []
        for token, tag in nltk.pos_tag(words):
            word_pos = SentenceProcessor.get_wordnet_pos(tag)
            lemmatizer = nltk.WordNetLemmatizer()
            try:
                if word_pos != '':
                    lemmatized.append(lemmatizer.lemmatize(token, pos=word_pos))
                else:
                    lemmatized.append(lemmatizer.lemmatize(token))
            except UnicodeDecodeError:
                print(f'Lemmatization failed for {token}, tag: {tag}, word pos: {word_pos}')

        return ' '.join(lemmatized)

    @staticmethod
    def remove_stop_words(sentence: str) -> str:
        """
        Remove stop words from a sentence
        """
        return ' '.join([word for word in sentence.split(' ') if word not in nltk.corpus.stopwords.words('english')])

    @staticmethod
    def get_wordnet_pos(treebank_tag: str) -> str:
        """
        Get the WordNet part-of-speech constant for the treebank tag
        """
        if treebank_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif treebank_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return ''

    @staticmethod
    def convert_camelcase(sentence: str) -> str:
        """
        Convert `camelCase` to `camel case`.
        """
        words = [all_cap_regex.sub(r'\1 \2', first_cap_regex.sub(r'\1 \2', word)) for word in sentence.split(" ")]

        return ' '.join(words)
