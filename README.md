# Tabular Data Science Project – AutomaticCategoricalFeatureBinning

Yarin Shaked 207234196

Eden Shuker 208991406

### Installation

Make sure your current directory is the assignment folder.

Install requirements:
`pip install requirements.txt`

Add the project directory to your PYTHONPATH:

`export PYTHONPATH="${PYTHONPATH}:/your/source/root"`

Or if you open the project from pycharm - mark project directory as source root

The behavior controlled with a commandline switch.
For help in using cli use: `python main.py --help`

### Running our Auto-Binning algorithm on a dataset

```text
python main.py --dataset_name titanic --target_column_name Survived 
```