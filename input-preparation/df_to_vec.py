from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple

import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import os

from pandas import Series

import config

w2v_models = {
    'code': Word2Vec.load(config.W2V_MODEL_CODE_DIR),
    'language': Word2Vec.load(config.W2V_MODEL_LANGUAGE_DIR)
}


def vectorize_string(sentence: str, feature_length: int, w2v_model: Word2Vec) -> np.ndarray:
    """
    Vectorize a sentence to a multi-dimensial NumPy array

    Roughly based on https://github.com/sola-da/NL2Type/blob/master/scripts/csv_to_vecs.py
    """
    vector = np.zeros((feature_length, config.W2V_VEC_LENGTH))

    for i, word in enumerate(sentence.split()):
        if i >= feature_length:
            break
        try:
            vector[i] = w2v_model.wv[word]
        except KeyError:
            pass

    return vector


class Datapoint(ABC):
    """
    Abstract class to represent a datapoint
    """

    @property
    @abstractmethod
    def feature_lengths(self) -> Dict[str, int]:
        """
        The lengths (number of vectors) of the features
        """
        pass

    @property
    @abstractmethod
    def feature_types(self) -> Dict[str, str]:
        """
        The types (datapoint_type, code or language) of the features
        """
        pass

    def vector_length(self) -> int:
        """
        The length of the whole vector for this datapoint
        """
        return sum(self.feature_lengths.values()) + len(self.feature_lengths.values()) - 1

    def to_vec(self) -> np.ndarray:
        """
        The vector for this datapoint

        The vector contains all the features specified in the subclass. Natural language features are converted to
        vectors using the word2vec models.
        """
        datapoint = np.zeros((self.vector_length(), config.W2V_VEC_LENGTH))

        separator = np.ones(config.W2V_VEC_LENGTH)

        position = 0
        for feature, feature_length in self.feature_lengths.items():

            if self.feature_types[feature] == 'datapoint_type':
                datapoint[position] = self.datapoint_type_vector()
                position += 1

            if self.feature_types[feature] == 'code' or self.feature_types[feature] == 'language':
                vectorized_feature = vectorize_string(
                    self.__dict__[feature] if isinstance(self.__dict__[feature], str) else '',
                    feature_length,
                    w2v_models[self.feature_types[feature]]
                )

                for word in vectorized_feature:
                    datapoint[position] = word
                    position += 1

            if self.feature_types[feature] == 'padding':
                for i in range(0, feature_length):
                    datapoint[position] = np.zeros(config.W2V_VEC_LENGTH)
                    position += 1

            # Add separator after each feature
            if position < len(datapoint):
                datapoint[position] = separator
                position += 1

        return datapoint

    def to_be_predicted_to_vec(self) -> np.ndarray:
        """
        A vector representation of what needs to be predicted, in this case the type
        """
        vector = np.zeros(config.NUMBER_OF_TYPES)
        vector[self.type] = 1
        return vector

    @abstractmethod
    def datapoint_type_vector(self) -> np.ndarray:
        """
        The vector corresponding to the type
        """
        pass

    def __repr__(self) -> str:
        values = list(map(lambda kv: kv[0] + ': ' + repr(kv[1]), self.__dict__.items()))
        values = "\n\t" + ",\n\t".join(values) + "\n"
        return type(self).__name__ + "(%s)" % values


class ParameterDatapoint(Datapoint):
    """
    A parameter data point representing the tuple (n_p_i, c_p_i)
    """
    @property
    def feature_lengths(self) -> Dict[str, int]:
        return {
            'datapoint_type': 1,
            'name': 6,
            'comment': 15,
            'padding_0': 6,
            'padding_1': 12,
            'padding_2': 10
        }

    @property
    def feature_types(self) -> Dict[str, str]:
        return {
            'datapoint_type': 'datapoint_type',
            'name': 'code',
            'comment': 'language',
            'padding_0': 'padding',
            'padding_1': 'padding',
            'padding_2': 'padding'
        }

    def __init__(self, name: str, comment: str, type: int, lineno: int, file: str, full_name: str):
        self.name = name
        self.comment = comment
        self.type = type
        self.lineno = lineno
        self.file = file
        self.full_name = full_name

    def datapoint_type_vector(self) -> np.ndarray:
        datapoint_type = np.zeros((1, config.W2V_VEC_LENGTH))
        datapoint_type[0][0] = 1
        return datapoint_type

    def to_be_predicted_src(self) -> np.ndarray:
        vector = np.array([self.file, str(self.lineno), self.full_name, "PARAMETER"])
        return vector


class ReturnDatapoint(Datapoint):
    """
    A return data point representing the tuple (n_f, c_f, r_c, r_e, n_p)
    """
    @property
    def feature_lengths(self) -> Dict[str, int]:
        return {
            'datapoint_type': 1,
            'name': 6,
            'function_comment': 15,
            'return_comment': 6,
            'return_expressions': 12,
            'parameter_names': 10
        }

    @property
    def feature_types(self) -> Dict[str, str]:
        return {
            'datapoint_type': 'datapoint_type',
            'name': 'code',
            'function_comment': 'language',
            'return_comment': 'language',
            'return_expressions': 'code',
            'parameter_names': 'code'
        }

    def __init__(self, name: str, function_comment: str, return_comment: str, return_expressions: list,
                 parameter_names: list, type: int, lineno: int, file: str, full_name: str):
        self.name = name
        self.function_comment = function_comment
        self.return_comment = return_comment
        self.return_expressions = return_expressions
        self.parameter_names = parameter_names
        self.type = type
        self.lineno = lineno
        self.file = file
        self.full_name = full_name

    def datapoint_type_vector(self) -> np.ndarray:
        datapoint_type = np.zeros((1, config.W2V_VEC_LENGTH))
        datapoint_type[0][1] = 1
        return datapoint_type

    def to_be_predicted_src(self) -> np.ndarray:
        vector = np.array([self.file, str(self.lineno), self.full_name, "FUNCTION"])
        return vector


def process_datapoints(filename: str, type: str, transformation: Callable[[Series], Datapoint]) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read dataframe, generate vectors for each row, and write them as multidimensional array to disk
    """
    print(f'Generating input vectors for {type} datapoints')

    df = pd.read_csv(filename)

    datapoints = df.apply(transformation, axis=1)

    datapoints_result_x = np.stack(datapoints.apply(lambda x: x.to_vec()), axis=0)
    np.save(os.path.join(config.VECTOR_OUTPUT_DIRECTORY, type + '_datapoints_x'), datapoints_result_x)
    datapoints_result_y = np.stack(datapoints.apply(lambda x: x.to_be_predicted_to_vec()), axis=0)
    np.save(os.path.join(config.VECTOR_OUTPUT_DIRECTORY, type + '_datapoints_y'), datapoints_result_y)

    datapoints_result_y_src = np.stack(datapoints.apply(lambda x: x.to_be_predicted_src()), axis=0)
    np.save(os.path.join(config.VECTOR_OUTPUT_DIRECTORY, type + '_datapoints_y_src'), datapoints_result_y_src)
    return datapoints_result_x, datapoints_result_y, datapoints_result_y_src


if __name__ == '__main__':
    if not os.path.isdir(config.VECTOR_OUTPUT_DIRECTORY):
        os.mkdir(config.VECTOR_OUTPUT_DIRECTORY)

    # Process parameter datapoints
    param_datapoints_result_x, param_datapoints_result_y, param_datapoints_result_y_src = process_datapoints(
        config.ML_PARAM_DF_PATH,
        'param',
        lambda row: ParameterDatapoint(row.arg_name, row.arg_comment, row.arg_type_enc, row.lineno, row.file, row.full_name)
    )

    return_datapoints_result_x, return_datapoints_result_y, return_datapoints_result_y_src = process_datapoints(
        config.ML_RETURN_DF_PATH,
        'return',
        lambda row: ReturnDatapoint(row['name'], row.func_descr if row.func_descr is str else row.docstring,
                                    row.return_descr, row.return_expr_str, row.arg_names_str, row.return_type_enc, row.lineno, row.file, row.full_name),
    )

    assert param_datapoints_result_x.shape[1] == return_datapoints_result_x.shape[1], \
        "Param datapoints and return datapoints must have the same length, thus padding must be added."
