# DLTPy
Deep Learning Type Inference of Python Function Signatures using their Natural Language Context

DLTPy makes type predictions based on comments, on the semantic elements of the function name and argument names,
and on the semantic elements of identifiers in the return expressions.  Using the natural language of these 
different elements, we have trained a classifier that predicts types. We use a recurrent neural network (RNN)
with a Long Short-Term Memory (LSTM) architecture.

_Read our [paper](paper.pdf) for the full details._

## Components

![DLTPy flow](https://user-images.githubusercontent.com/15815208/67791371-98049480-fa77-11e9-95ed-bb94e7b06eeb.png)

### `preprocessing/` Preprocessing Pipeline (a-d)
Downloads projects, extracts comments and typesm and gives a csv file per project containing all functions.

Start using:
``` bash
$ python preprocessing/pipeline.py
```
Optional arguments:
```
  -h, --help            show this help message and exit
  --projects_file PROJECTS_FILE
                        json file containing GitHub projects
  --limit LIMIT         limit the number of projects for which the pipeline
                        should run
  --jobs JOBS           number of jobs to use for pipeline.
  --output_dir OUTPUT_DIR
                        output dir for the pipeline
  --start START         start position within projects list
```

### `input-preparation/` Input Preparation (e-f)
`input-preparation/generate_df.py` can be used to combine all the separate csv files per project into one big file
while applying filtering.

`input-preparation/df_to_vec.py` can be used to convert this generated csv to vectors.

`input-preparation/embedder.py` can be used to train word embeddings for `input-preparation/df_to_vec.py`.

### `learning/` Learning (g)
The different RNN models we evaluated can be found in `learning/learn.py`.

## Testing
``` bash
$ pytest
```

## Credits
- [Casper Boone](https://github.com/casperboone)
- [Niels de Bruin](https://github.com/nielsdebruin)
- [Arjan Langerak](https://github.com/alangerak)
- [Fabian Stelmach](https://github.com/fabianstelmach)
- [All contributors](../../contributors)

## License
The MIT License (MIT). Please see the [license file](LICENSE) for more information.
