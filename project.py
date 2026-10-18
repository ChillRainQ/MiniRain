import os
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION = "v0.1.0-Vision"
AUTHOR = ["Chill_Rain"]
ARCHITECTURES = "architectures"
CLI = "cli"
COMPONENTS = "components"
TRAIN = "train"
DATASET = "dataset"
TRAIN_OUTPUT_DIR = "outputs"
MODEL_CODE_DIR_AND_FILE = [os.path.join(PROJECT_ROOT_DIR, ARCHITECTURES),
                  os.path.join(PROJECT_ROOT_DIR, CLI),
                  os.path.join(PROJECT_ROOT_DIR, COMPONENTS),
                  os.path.join(PROJECT_ROOT_DIR, TRAIN),
                  os.path.join(PROJECT_ROOT_DIR, DATASET),
                os.path.join(PROJECT_ROOT_DIR, "project.py")]