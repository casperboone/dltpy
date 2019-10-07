from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import datapath
import pandas as pd
import multiprocessing
import itertools

import time
import logging
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)



class Embedder:
    """
    Create a word embeddings using Word2Vec.
    """

    def __init__(self, file):
        self.df = pd.read_csv(file)

    def __init__(self, df):
        self.df = df

    def getLanguageIterator(self):
        """
        Get an iterator over the whole collection of descriptions language.
        :return: iterator over the language data
        """
        func_descr_sentences = (row.split() for row in self.df['func_descr'] if not isinstance(row, float))
        arg_descr_sentences = (arg_descr.split() for arg_descrs_method in self.df['arg_descrs'] for arg_descr in eval(arg_descrs_method))
        return_descr_sentences = (row.split() for row in self.df['return_descr'] if not isinstance(row, float))
        all_comment_sentences = itertools.chain(func_descr_sentences, arg_descr_sentences, return_descr_sentences)

        return all_comment_sentences

    def getCodeIterator(self):
        """
        Get an iterator over the whole collection of the code expressions.
        :return: iterator over the language data
        """
        #return_expr_sentences = (row.split() for row in self.df['return_expr'])
        func_name_sentences = (row.split(" ") for row in self.df['name'] if not isinstance(row, float))
        arg_names_sentences = (row.split(" ") for row in self.df['arg_names'])
        #all_comment_sentences = itertools.chain(return_expr_sentences, func_name_sentences, arg_names_sentences)
        all_comment_sentences = itertools.chain(func_name_sentences, arg_names_sentences)

        return all_comment_sentences

    def trainModel(self, corpusIteratorFunction, modelPathName):
        """
        Train a Word2Vec model and save the output to a file.
        :param corpusIteratorFunction: function expression that returns a iterator for the corpus
        :param modelPathName: path name of the output file
        """

        corpusIterator = corpusIteratorFunction()

        cores = multiprocessing.cpu_count()

        w2v_model = Word2Vec(min_count=5,           #Specified in NL2Type
                             #window=2,
                             size=100,              #Specified in NL2Type
                             #sample=6e-5,
                             #alpha=0.03,
                             #min_alpha=0.0007,
                             #negative=20,
                             workers=cores-1)

        #t = time.time()

        w2v_model.build_vocab(sentences=corpusIterator,
                              progress_per=10000)

        #print('Time to build vocab: {} mins'.format(round((time.time() - t) / 60, 2)))

        #The iterator is reset here to the beginning again
        corpusIterator = corpusIteratorFunction()
        #t = time.time()

        w2v_model.train(sentences=corpusIterator,
                        total_examples=w2v_model.corpus_count,
                        epochs=1,
                        report_delay=1)

        #print('Time to train the model: {} mins'.format(round((time.time() - t) / 60, 2)))

        w2v_model.save(modelPathName)

    def trainLanguageModel(self):
        """
        Train a Word2Vec model for the descriptions and save to file.
        """
        self.trainModel(self.getLanguageIterator, 'output/temp_language.bin')

    def trainCodeModel(self):
        """
        Train a Word2Vec model for the code expressions and save to file.
        """
        self.trainModel(self.getCodeIterator, 'output/temp_code.bin')

    def useModel(self):
        wv_from_bin = Word2Vec.load("output/temp_code.bin")
        print(wv_from_bin.wv.index2entity[:100])
       # print(wv_from_bin.wv.most_similar(positive=["string"]))


if __name__ == '__main__':
    embedder = Embedder("output/_temp.csv")
    embedder.useModel()
