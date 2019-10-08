from gensim.models import Word2Vec
import pandas as pd
import multiprocessing
import itertools
import os
from typing import Iterator, List, Callable

#CONFIG
OUTPUT_DIRECTORY = os.path.join('./output')
PARAM_DF_PATH = os.path.join(OUTPUT_DIRECTORY, 'ml_inputs', '_ml_param.csv')
RETURN_DF_PATH = os.path.join(OUTPUT_DIRECTORY, 'ml_inputs', '_ml_return.csv')
OUTPUT_EMBEDDINGS_DIRECTORY = os.path.join('./output', 'embeddings')
LANGUAGE_EMBEDDING_OUTPUT_PATH = os.path.join(OUTPUT_EMBEDDINGS_DIRECTORY, 'w2v_language_model.bin')
CODE_EMBEDDING_OUTPUT_PATH = os.path.join(OUTPUT_EMBEDDINGS_DIRECTORY, 'w2v_code_model.bin')


class Embedder:
    """
    Create embeddings for the code names and docstring names using Word2Vec.
    """

    def __init__(self, param_df: pd.DataFrame, return_df: pd.DataFrame) -> None:
        self.param_df = param_df
        self.return_df = return_df

    def getLanguageIterator(self) -> Iterator[List[str]]:
        """
        Get an iterator over the whole collection of descriptions language.
        :return: iterator over the language data
        """
        func_descr_sentences = (row.split() for row in self.return_df['func_descr'])
        arg_descr_sentences = (arg_descr.split() for arg_descr in param_df['arg_comment'])
        return_descr_sentences = (row.split() for row in self.return_df['return_descr'])
        all_comment_sentences = itertools.chain(func_descr_sentences, arg_descr_sentences, return_descr_sentences)

        return all_comment_sentences

    def getCodeIterator(self) -> Iterator[List[str]]:
        """
        Get an iterator over the whole collection of the code expressions.
        :return: iterator over the language data
        """
        return_expr_sentences = (row.split() for row in self.return_df['return_expr_str'])
        func_name_sentences = (row.split() for row in self.return_df['name'])
        arg_names_sentences = (row.split() for row in self.return_df['arg_names_str'])

        all_comment_sentences = itertools.chain(return_expr_sentences, func_name_sentences, arg_names_sentences)

        return all_comment_sentences

    def trainModel(self, corpus_iterator_function: Callable[[], Iterator[List[str]]], model_path_name: str) -> None:
        """
        Train a Word2Vec model and save the output to a file.
        :param corpus_iterator_function: function expression that returns a iterator for the corpus
        :param model_path_name: path name of the output file
        """

        corpus_iterator = corpus_iterator_function()

        cores = multiprocessing.cpu_count()

        w2v_model = Word2Vec(min_count=5,           #Specified in NL2Type
                             #window=2,
                             size=100,              #Specified in NL2Type
                             #sample=6e-5,
                             #alpha=0.03,
                             #min_alpha=0.0007,
                             #negative=20,
                             workers=cores-1)

        w2v_model.build_vocab(sentences=corpus_iterator,
                              progress_per=10000)


        #The iterator is reset here to the beginning again
        corpus_iterator = corpus_iterator_function()

        w2v_model.train(sentences=corpus_iterator,
                        total_examples=w2v_model.corpus_count,
                        epochs=20,
                        report_delay=1)

        w2v_model.save(model_path_name)

    def trainLanguageModel(self) -> None:
        """
        Train a Word2Vec model for the descriptions and save to file.
        """
        self.trainModel(self.getLanguageIterator, LANGUAGE_EMBEDDING_OUTPUT_PATH)

    def trainCodeModel(self) -> None:
        """
        Train a Word2Vec model for the code expressions and save to file.
        """
        self.trainModel(self.getCodeIterator, CODE_EMBEDDING_OUTPUT_PATH)


if __name__ == '__main__':
    param_df = pd.read_csv(PARAM_DF_PATH)
    param_df = param_df.dropna()

    return_df = pd.read_csv(RETURN_DF_PATH)
    return_df = return_df.dropna()

    if not os.path.isdir(OUTPUT_EMBEDDINGS_DIRECTORY):
        os.mkdir(OUTPUT_EMBEDDINGS_DIRECTORY)

    embedder = Embedder(param_df, return_df)
    embedder.trainCodeModel()
    embedder.trainLanguageModel()

    w2v_language_model = Word2Vec.load(LANGUAGE_EMBEDDING_OUTPUT_PATH)
    w2v_code_model = Word2Vec.load(CODE_EMBEDDING_OUTPUT_PATH)

    print("W2V statistics: ")
    print("W2V language model total amount of words : " + str(w2v_language_model.corpus_total_words))
    print("W2V code model total amount of words : " + str(w2v_code_model.corpus_total_words))
    print(" ")
    print("Top 20 words for language model:")
    print(w2v_language_model.wv.index2entity[:20])
    print("\n Top 20 words for code model:")
    print(w2v_code_model.wv.index2entity[:20])

