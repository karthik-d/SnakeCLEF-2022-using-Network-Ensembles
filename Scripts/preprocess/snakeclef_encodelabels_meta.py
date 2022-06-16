import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd

data = pd.read_csv("SnakeCLEF2022-TrainMetadata.csv")
meta_cols = [ 'code', 'endemic']

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metalabels')

for col in meta_cols:
	encoder = LabelEncoder()
	encoder.fit(data[col])
	file_path = os.path.join(base_path, col+'_classes.npy')
	np.save(file_path, encoder.classes_)
	print("Saved", file_path)

