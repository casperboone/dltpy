import random
import time

import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import os

WORD_VEC_LENGTH = 100
NUMBER_OF_TYPES = 1000

w2v_models = {
    'code': Word2Vec.load('resources/word2vec_model_code.bin'),
    'language': Word2Vec.load('resources/word2vec_model_language.bin')
}


# NL2Type code
def vectorize_string(text, feature_length, w2v_model):
    text_vec = np.zeros((feature_length, WORD_VEC_LENGTH))
    if text == 'unknown':
        return text_vec
    count = 0
    for word in text.split():
        if count >= feature_length:
            return text_vec
        try:
            text_vec[count] = w2v_model.wv[word]
        except KeyError:
            pass
        count += 1

    return text_vec
# /NL2Type code

class Datapoint:
    def __repr__(self) -> str:
        values = list(map(lambda kv: kv[0] + ': ' + repr(kv[1]), self.__dict__.items()))
        values = "\n\t" + ",\n\t".join(values) + "\n"
        return type(self).__name__ + "(%s)" % values

    def vector_length(self):
        return sum(self.features.values()) + len(self.features.values()) - 1

    def to_vec(self):
        datapoint = np.zeros((self.vector_length(), WORD_VEC_LENGTH))

        separator = np.ones(WORD_VEC_LENGTH)

        position = 0
        for feature, feature_length in self.features.items():

            if self.feature_types[feature] == 'datapoint_type':
                datapoint[position] = self.datapoint_type_vector()
                position += 1

            if self.feature_types[feature] == 'code' or self.feature_types[feature] == 'language':
                vectorized_feature = vectorize_string(
                    self.__dict__[feature],
                    feature_length,
                    w2v_models[self.feature_types[feature]]
                )

                for word in vectorized_feature:
                    datapoint[position] = word
                    position += 1

            # Add separator after each feature
            if position < len(datapoint):
                datapoint[position] = separator
                position += 1

        return datapoint

    def to_be_predicted_to_vec(self):
        vector = np.zeros(NUMBER_OF_TYPES)
        # for now a random number
        vector[random.randint(0, 3)] = 1
        # vector[self.type] = 1
        return vector


class ParameterDatapoint(Datapoint):
    def __init__(self, name: str, comment: str, type: int):
        self.name = name
        self.comment = comment
        self.type = type

        self.features = {
            'datapoint_type': 1,
            'name': 6,
            'comment': 12
        }

        self.feature_types = {
            'datapoint_type': 'datapoint_type',
            'name': 'code',
            'comment': 'language'
        }

    def datapoint_type_vector(self):
        datapoint_type = np.zeros((1, WORD_VEC_LENGTH))
        datapoint_type[0][0] = 1
        return datapoint_type


class ReturnDatapoint(Datapoint):
    def __init__(self, name: str, function_comment: str, return_comment: str, return_expressions: list,
                 parameter_names: list, type: int):
        self.name = name
        self.function_comment = function_comment
        self.return_comment = return_comment
        self.return_expressions = ' '.join(return_expressions)
        self.parameter_names = ' '.join(parameter_names)
        self.type = type

        self.features = {
            'datapoint_type': 1,
            'name': 6,
            'function_comment': 12,
            'return_comment': 10,
            'return_expressions': 10,  # check what this should be
            'parameter_names': 10  # check what this should be
        }

        self.feature_types = {
            'datapoint_type': 'datapoint_type',
            'name': 'code',
            'function_comment': 'language',
            'return_comment': 'language',
            'return_expressions': 'code',
            'parameter_names': 'code'
        }

    def datapoint_type_vector(self):
        datapoint_type = np.zeros((1, WORD_VEC_LENGTH))
        datapoint_type[0][1] = 1
        return datapoint_type


if not os.path.isdir('./output'):
    os.mkdir('./output')
output_directory = os.path.join('./output', str(int(time.time())))
os.mkdir(output_directory)

df = pd.read_csv('resources/df_limited.csv')  # assumption: only functions with types

df['arg_names'] = df['arg_names'].apply(lambda x: np.asarray(eval(x)))  # might not be necessary
df['arg_descrs'] = df['arg_descrs'].apply(lambda x: np.asarray(eval(x)))
df['return_expr'] = df['return_expr'].apply(lambda x: np.asarray(eval(x)))

count = 0

parameter_datapoints = []
return_datapoints = []

df = df[:2]

for index, row in df.iterrows():
    # Set function comment to be docstring if function comment is empty
    function_comment = row.func_descr if row.func_descr is str else row.docstring

    # Generate data points
    for parameter_index, parameter_name in enumerate(row.arg_names):
        parameter_datapoints.append(
            ParameterDatapoint(parameter_name, row.arg_descrs[parameter_index], row.arg_types[parameter_index])
        )

    return_datapoints.append(
        ReturnDatapoint(row['name'], function_comment, row.return_descr, row.return_expr, row.arg_names,
                        row.return_type))

# Write parameter datapoints to disk
parameter_datapoints_result_x = np.zeros(
    (len(parameter_datapoints), parameter_datapoints[0].vector_length(), WORD_VEC_LENGTH)
)
parameter_datapoints_result_y = np.zeros(
    (len(parameter_datapoints), NUMBER_OF_TYPES)
)
for i, datapoint in enumerate(parameter_datapoints):
    parameter_datapoints_result_x[i] = datapoint.to_vec()
    parameter_datapoints_result_y[i] = datapoint.to_be_predicted_to_vec()

np.save(os.path.join(output_directory, 'parameter_datapoints_x'), parameter_datapoints_result_x)
np.save(os.path.join(output_directory, 'parameter_datapoints_y'), parameter_datapoints_result_y)

# Write return datapoints to disk
return_datapoints_result_x = np.zeros(
    (len(return_datapoints), return_datapoints[0].vector_length(), WORD_VEC_LENGTH)
)
return_datapoints_result_y = np.zeros(
    (len(return_datapoints), NUMBER_OF_TYPES)
)
for i, datapoint in enumerate(return_datapoints):
    return_datapoints_result_x[i] = datapoint.to_vec()
    return_datapoints_result_y[i] = datapoint.to_be_predicted_to_vec()

np.save(os.path.join(output_directory, 'return_datapoints_result_x'), return_datapoints_result_x)
np.save(os.path.join(output_directory, 'return_datapoints_result_y'), return_datapoints_result_y)
