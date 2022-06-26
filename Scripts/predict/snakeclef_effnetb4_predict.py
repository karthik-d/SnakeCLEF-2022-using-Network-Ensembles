from efficientnet.efficientnet.model import EfficientNetB4

import tensorflow as tf
import keras.utils
import numpy as np
import pandas as pd
import os
import csv

IMG_SIZE=(224,224)
check=[]

#from tensorflow.keras.applications import EfficientNetB0
from keras.utils.vis_utils import plot_model

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

print(keras.__version__)
print(tf.__version__)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from keras import backend as K

#model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
model = EfficientNetB4(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3)
       )
# model.summary(line_length=150)

flatten = Flatten()
new_layer2 = Dense(1604, activation='softmax', name='my_dense_2')

inp2 = model.input
out2 = new_layer2(flatten(model.output))

opt = keras.optimizers.Adam(learning_rate=1e-05)

model2 = Model(inp2, out2)
model2.summary()
model2.compile(
    optimizer=opt,
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

#work from here

model2.load_weights('weights/weights-efficientnetb4/weights-epoch-010.h5')

'''modelPredictionsArray = []
outputDF = pd.DataFrame(modelPredictionsArray, columns = ['observation_id', 'class_id'])
outputDF.to_csv('/kaggle/working/predictions.csv', index=False) # making an empty csv file

#iterating through array'''

#download and change
os.chdir('Datasets/SnakeCLEF2022-test_images/SnakeCLEF2022-large_size')



op_file = open('effpredictions_dup_1.csv', mode='w')
op_writer = csv.writer(op_file, delimiter=',', quotechar='"')
op_writer.writerow(['observation_id', 'ClassId'])  
skipping = 0
i = 0

filename = "/home/miruna/LifeCLEF/SnakeCLEF/SnakeCLEF2022-TestMetadata.csv"
df=pd.read_csv(filename)

while not df.empty:
    i += 1
    observation_id = df.iloc[0]['observation_id']
    all_imgs = df.loc[df['observation_id']==observation_id, ['file_path']]
    num_rows = len(all_imgs)
    predict_probs = np.zeros((1604,))
    for idx, row in all_imgs.iterrows():
        path = row['file_path']
        try:
            image = load_img(path, target_size=(224, 224))
            image = img_to_array(image)
        except OSError:
            skipping += 1
            print(skipping," skipping ", path)
            continue
        images = np.expand_dims(image, axis=0)
        predict_probs += model2.predict(images)[0]
    predicted_class = np.argmax(predict_probs)
    op_writer.writerow([observation_id, predicted_class])
    df.drop(df[df['observation_id']==observation_id].index, inplace = True)
    if(i%50==0):
        print(num_rows, '/', len(df))
        print([observation_id, predicted_class])
        #outputDF = pd.DataFrame(modelPredictionsArray, columns = ['observation_id', 'class_id'])
        #outputDF.to_csv('/kaggle/working/outputdata.csv', index=False, mode='a', header=False)
        #print(outputDF)
        #modelPredictionsArray = []
        print("SAVING DATA")
op_file.close()

