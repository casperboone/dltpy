from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple

import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import os

from pandas import Series

OUTPUT_DIRECTORY = './output/vectors'
PARAM_DATAPOINTS_DATAFRAME = './output/ml_inputs/_ml_param.csv'
RETURN_DATAPOINTS_DATAFRAME = './output/ml_inputs/_ml_return.csv'

VEC_LENGTH = 100
NUMBER_OF_TYPES = 1000

w2v_models = {
    'code': Word2Vec.load('resources/w2v_code_model.bin'),
    'language': Word2Vec.load('resources/w2v_language_model.bin')
}


def vectorize_string(sentence: str, feature_length: int, w2v_model: Word2Vec) -> np.ndarray:
    """
    Vectorize a sentence to a multi-dimensial NumPy array

    Roughly based on https://github.com/sola-da/NL2Type/blob/master/scripts/csv_to_vecs.py
    """
    vector = np.zeros((feature_length, VEC_LENGTH))

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
        datapoint = np.zeros((self.vector_length(), VEC_LENGTH))

        separator = np.ones(VEC_LENGTH)

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
                    datapoint[position] = np.zeros(VEC_LENGTH)
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
        vector = np.zeros(NUMBER_OF_TYPES)
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

    def __init__(self, name: str, comment: str, type: int):
        self.name = name
        self.comment = comment
        self.type = type

    def datapoint_type_vector(self) -> np.ndarray:
        datapoint_type = np.zeros((1, VEC_LENGTH))
        datapoint_type[0][0] = 1
        return datapoint_type


class ReturnDatapoint(Datapoint):
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
                 parameter_names: list, type: int):
        self.name = name
        self.function_comment = function_comment
        self.return_comment = return_comment
        self.return_expressions = return_expressions
        self.parameter_names = parameter_names
        self.type = type

    def datapoint_type_vector(self) -> np.ndarray:
        datapoint_type = np.zeros((1, VEC_LENGTH))
        datapoint_type[0][1] = 1
        return datapoint_type


def process_datapoints(filename: str, type: str, transformation: Callable[[Series], Datapoint]) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Read dataframe, generate vectors for each row, and write them as multidimensional array to disk
    """
    print(f'Generating input vectors for {type} datapoints')

    df = pd.read_csv(filename)

    datapoints = df.apply(transformation, axis=1)

    datapoints_result_x = np.stack(datapoints.apply(lambda x: x.to_vec()), axis=0)
    np.save(os.path.join(OUTPUT_DIRECTORY, type + '_datapoints_x'), datapoints_result_x)
    datapoints_result_y = np.stack(datapoints.apply(lambda x: x.to_be_predicted_to_vec()), axis=0)
    np.save(os.path.join(OUTPUT_DIRECTORY, type + '_datapoints_y'), datapoints_result_y)

    return datapoints_result_x, datapoints_result_y


if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_DIRECTORY):
        os.mkdir(OUTPUT_DIRECTORY)

    # Process parameter datapoints
    param_datapoints_result_x, param_datapoints_result_y = process_datapoints(
        PARAM_DATAPOINTS_DATAFRAME,
        'param',
        lambda row: ParameterDatapoint(row.arg_name, row.arg_comment, row.arg_type_enc)
    )

    return_datapoints_result_x, return_datapoints_result_y = process_datapoints(
        RETURN_DATAPOINTS_DATAFRAME,
        'return',
        lambda row: ReturnDatapoint(row['name'], row.func_descr if row.func_descr is str else row.docstring,
                                    row.return_descr, row.return_expr_str, row.arg_names_str, row.return_type_enc),
    )

    assert param_datapoints_result_x.shape[1] == return_datapoints_result_x.shape[1], \
        "Param datapoints and return datapoints must have the same length, thus padding must be added."
