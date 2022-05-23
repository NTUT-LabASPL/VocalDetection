import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, './data')
DATASET_DIR = os.path.join(DATA_DIR, './dataset')
WEIGHT_DIR = os.path.join(DATA_DIR, './weight')
MODEL_DIR = os.path.join(DATA_DIR, './model')
LOG_DIR = os.path.join(DATA_DIR, './logs')
VOTE_DIR = os.path.join(DATA_DIR, './voting')
MERGE_DIR = os.path.join(DATA_DIR, './merge')
DIAGRAM_DIR = os.path.join(DATA_DIR, './diagram')