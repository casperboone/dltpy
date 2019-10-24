from gensim.models import Word2Vec
import pandas as pd
import multiprocessing
import os
from time import time
import config


#CONFIG

#LANGUAGE_EMBEDDING_OUTPUT_PATH = os.path.join(OUTPUT_EMBEDDINGS_DIRECTORY, 'w2v_language_model.bin')
#CODE_EMBEDDING_OUTPUT_PATH = os.path.join(OUTPUT_EMBEDDINGS_DIRECTORY, 'w2v_code_model.bin')



class HelperIterator:
    """
    Subclass for type Hinting the iterators listed below
    """
    pass


class LanguageIterator(HelperIterator):
    """
    Helper Iterator that iterates over the whole collection of descriptions language.
    """
    def __init__(self, param_df: pd.DataFrame, return_df: pd.DataFrame) -> None:
        self.param_df = param_df
        self.return_df = return_df

    def __iter__(self):
        for func_descr_sentence in self.return_df['func_descr']:
            yield func_descr_sentence.split()

        for param_descr_sentence in self.param_df['arg_comment']:
            yield  param_descr_sentence.split()

        for return_descr_sentence in self.return_df['return_descr']:
            yield return_descr_sentence.split()


class CodeIterator(HelperIterator):
    """
    Helper Iterator that iterates over the whole collection of the code expressions.
    """
    def __init__(self, param_df: pd.DataFrame, return_df: pd.DataFrame) -> None:
        self.param_df = param_df
        self.return_df = return_df

    def __iter__(self):
        for return_expr_sentences in self.return_df['return_expr_str']:
            yield return_expr_sentences.split()

        for func_name_sentences in self.return_df['name']:
            yield func_name_sentences.split()

        for arg_names_sentences in self.return_df['arg_names_str']:
            yield arg_names_sentences.split()


class Embedder:
    """
    Create embeddings for the code names and docstring names using Word2Vec.
    """

    def __init__(self, param_df: pd.DataFrame, return_df: pd.DataFrame) -> None:
        self.param_df = param_df
        self.return_df = return_df

    def train_model(self, corpus_iterator: HelperIterator, model_path_name: str) -> None:
        """
        Train a Word2Vec model and save the output to a file.
        :param corpus_iterator: class that can provide an iterator that goes through the corpus
        :param model_path_name: path name of the output file
        """

        cores = multiprocessing.cpu_count()

        w2v_model = Word2Vec(min_count=5,
                             window=5,
                             size=config.W2V_VEC_LENGTH,
                             workers=cores-1)

        t = time()

        w2v_model.build_vocab(sentences=corpus_iterator)

        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

        t = time()

        w2v_model.train(sentences=corpus_iterator,
                        total_examples=w2v_model.corpus_count,
                        epochs=20,
                        report_delay=1)

        print('Time to train model: {} mins'.format(round((time() - t) / 60, 2)))

        w2v_model.save(model_path_name)

    def train_language_model(self) -> None:
        """
        Train a Word2Vec model for the descriptions and save to file.
        """
        self.train_model(LanguageIterator(self.param_df, self.return_df), config.W2V_MODEL_LANGUAGE_DIR)

    def train_code_model(self) -> None:
        """
        Train a Word2Vec model for the code expressions and save to file.
        """
        self.train_model(CodeIterator(self.param_df, self.return_df), config.W2V_MODEL_CODE_DIR)


if __name__ == '__main__':
    param_df = pd.read_csv(config.ML_PARAM_DF_PATH)
    param_df = param_df.dropna()

    return_df = pd.read_csv(config.ML_RETURN_DF_PATH)
    return_df = return_df.dropna()

    if not os.path.isdir(config.OUTPUT_EMBEDDINGS_DIRECTORY):
        os.mkdir(config.OUTPUT_EMBEDDINGS_DIRECTORY)

    embedder = Embedder(param_df, return_df)
    embedder.train_code_model()
    embedder.train_language_model()

    w2v_language_model = Word2Vec.load(config.W2V_MODEL_LANGUAGE_DIR)
    w2v_code_model = Word2Vec.load(config.W2V_MODEL_CODE_DIR)

    print("W2V statistics: ")
    print("W2V language model total amount of words : " + str(w2v_language_model.corpus_total_words))
    print("W2V code model total amount of words : " + str(w2v_code_model.corpus_total_words))
    print(" ")
    print("Top 20 words for language model:")
    print(w2v_language_model.wv.index2entity[:20])
    print("\n Top 20 words for code model:")
    print(w2v_code_model.wv.index2entity[:20])


