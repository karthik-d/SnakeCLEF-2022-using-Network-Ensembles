# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
"""

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dainCb6W8mG_A2A0VIE7j_lSAIbDP921
"""

"""
! mkdir ~/.kaggle

!cp /content/drive/MyDrive/ML/kaggle.json ~/.kaggle/kaggle.json

!kaggle competitions download -c snakeclef2022

!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6

! mkdir ~/.kaggle

!cp /content/drive/MyDrive/ML/kaggle.json ~/.kaggle/kaggle.json

!kaggle competitions download -c snakeclef2022

! rm /content/*.jpg
! rm /content/*.jpeg
! rm /content/*.JPG

!unzip "/content/snakeclef2022.zip" -d "/content/drive/MyDrive/ML"

!pwd
import os
os.chdir('/content/drive/MyDrive/ML')
!pwd

'''
!pwd
import os
os.chdir(os.path.join('/', 'content', 'drive', 'MyDrive', 'Research', 'LifeCLEF\'22', 'SnakeCLEF-2022', 'Dataset', 'SNAKE_CLEF'))
!pwd
# - Karthik
'''
"""

from efficientnet.efficientnet.model import EfficientNetB4

import tensorflow as tf
import keras.utils
import numpy as np
import os
#import cv2
from skimage import io, transform, color
from PIL import Image
import pandas as pd 

import numpy as np
from sklearn.preprocessing import LabelEncoder

print(tf.config.list_physical_devices('GPU'))
print("GPU Count: ", len(tf.config.list_physical_devices('GPU')))



BATCH_SIZE=8
IMG_SIZE=(224,224)
check=[]

class InputSequencer(tf.keras.utils.Sequence):

	def __init__(self, base_path=None, shuffle=True):
		self.label_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metalabels')
		self.BATCH_SIZE = BATCH_SIZE
		self.IMG_SIZE = IMG_SIZE
		self.shuffle = shuffle
		self.csv_filename = "SnakeCLEF2022-TrainMetadata.csv"
		self.x_col_name = "file_path"
		self.y_col_name = "class_id"
		self.check = []
		print(os.getcwd())
		self.data_file = pd.read_csv(self.csv_filename)
		print(self.data_file.columns)
		self.data_file.head()
		print("Classes:", max(self.data_file.class_id.unique())+1)
		self.num_data_pts = len(self.data_file)
		print(self.num_data_pts)
		self.base_path = base_path
		self.meta_cols = [ 'code', 'endemic']
		#self.indexes = np.arange(len(self.image_paths))
		self.on_epoch_end()
		self.meta_encoders = []
		for col in self.meta_cols:
			encoder = LabelEncoder()
			encoder.classes_ = np.load(os.path.join(self.label_path, col+'_classes.npy'), allow_pickle=True)
			self.meta_encoders.append(encoder)	

	def on_epoch_end(self, *args):
		self.check = []
		pass
		"""
		if(self.shuffle):
			np.random.shuffle(self.indexes)
		"""

	def __len__(self):
		return self.num_data_pts // self.BATCH_SIZE
		pass
		
	def encode_metadata(self, df_part):
		encoded = []
		for encoder, col in zip(self.meta_encoders, self.meta_cols):
			encoded.append(encoder.transform(df_part[col]))
		return encoded
			

	def __getitem__(self, idx):
		"""Returns tuple (input, target) correspond to batch #idx."""
		#
		# data_rows = self.data_file.sample(n=self.BATCH_SIZE,replace=False)
		# batch_paths = data_rows[self.x_col_name].to_list()
		# batch_labels = data_rows[self.y_col_name].to_list()
		# batch_labels = list(data_rows.loc[:, [self.y_col_name]])
		if self.base_path is None:
			base_path="/usr/home/bharathi/snake_clef2022"
		else:
			base_path = self.base_path
		'''
		# Karthik
		base_path = os.path.join('/', 'content', 'drive', 'MyDrive', 'Research', 'LifeCLEF\'22', 'SnakeCLEF-2022', 'Dataset', 'SNAKE_CLEF', 'SnakeCLEF2022-small_size', 'SnakeCLEF2022-small_size')
		'''

		batch_images = []
		batch_labels = []
		batch_meta = []
		# The resize error may be occuring because the file is not found and `img` holds None
		# Adding file existence check
		while len(batch_images)<self.BATCH_SIZE:
			
			new_row = self.data_file.sample(n=1, replace=False)
	 		
			path = new_row[self.x_col_name].to_list()[0]
			label = new_row[self.y_col_name].to_list()[0]
			
			if(path in self.check):
				#print("check")
				continue
			else:
				#print("append")
				self.check.append(path)
			
			try:
				img = Image.open(os.path.join(base_path, path)).convert('RGB')
			except FileNotFoundError:
				print("file not found")
				continue
			except Exception:
				print("Image corrupt")
				continue

			# Resize
			img_res = img.resize(self.IMG_SIZE)			
			
			# print(os.path.join(base_path,path))
			image_data = np.array(np.asarray(img_res), dtype='uint8')
			batch_meta.append(self.encode_metadata(new_row))
			batch_images.append(image_data)
			batch_labels.append(label)
			# print(f"{len(batch_images)} of {self.BATCH_SIZE} images prepared")
			#print(path)
	 	
		#print(np.array(batch_images).shape)
		return ([np.array(batch_images), np.squeeze(np.array(batch_meta))], np.array(batch_labels))

data_path = os.path.join('./Datasets/SnakeCLEF2022-large_size/')
data_reader = InputSequencer(base_path=data_path)
inps, labels = data_reader[5]
print(inps[0])
print(inps[1])
print(labels)
print("TESTED")

"""
check_img = imgs[0]
check_label = labels[0]
import matplotlib.pyplot as plot
print(check_img.shape)
plot.imshow(check_img)
plot.show()
print(check_label)
"""

"""Randomize or shuffle training data and ensure that all images are fed to the model

Upload images to the drive
"""

#from tensorflow.keras.applications import EfficientNetB0
from keras.utils.vis_utils import plot_model

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# load the model
#model = resnet.ResNet101()

# load an image from file
#image = load_img('mug.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
#image = img_to_array(image)
# reshape data for the model
#image = imgs[0].reshape((1, imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2]))
# prepare the image for the VGG model
#image = preprocess_input(image)
# predict the probability across all output classes
#yhat = model.predict(image)
# convert the probabilities to class labels
#label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
#label = label[0][0]
# print the classification
#print('%s (%.2f%%)' % (label[1], label[2]*100))

print(keras.__version__)
print(tf.__version__)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate
from keras import backend as K

#model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
model = EfficientNetB4(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3)
       )
# model.summary(line_length=150)

flatten = Flatten()
new_layer2 = Dense(1572, activation='softmax', name='my_dense_2')

meta_in = Input(shape=(len(data_reader.meta_cols),))
img_in = model.input

print(model.output.shape)
print(meta_in.shape)
out = new_layer2(Concatenate(axis=-1)([flatten(model.output), meta_in]))
#out2 = new_layer2(flatten(model.output))


opt = keras.optimizers.Adam(learning_rate=1e-04)
	
model2 = Model((img_in, meta_in), out)
model2.summary()
model2.compile(
    optimizer=opt,
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

#need to create weight folder
weight_save = keras.callbacks.ModelCheckpoint('weights/weights-efficientnetb4-wmeta/weights-epoch-1_{epoch:03d}.h5', save_weights_only=True, period=1)
on_epoch_end_call = keras.callbacks.LambdaCallback(on_epoch_end=data_reader.on_epoch_end())

# model2.load_weights('/home/miruna/LifeCLEF/SnakeCLEF/weights/weights-efficientnetb4-wmeta/weights-epoch-2_005.h5')
model2.fit(data_reader,
    epochs=10,
    verbose=1,
    steps_per_epoch=1,
    callbacks=[weight_save, on_epoch_end_call]
)
