import os
import pandas as pd

def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_default_path():
    return os.path.join(project_root(), "data", "global_cars_enhanced.csv")

def load_dataset(path):
    return pd.read_csv(path)