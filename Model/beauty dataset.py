#Importing Libraries

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten  
from tensorflow.keras.models import Sequential 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Model Network

model=Sequential([Conv2D(128,(3,3),input_shape=(224,224,3)),
                  MaxPooling2D(2,2),
                  Conv2D(64,(3,3),activation='relu'),
                  MaxPooling2D(2,2),
                  Dropout(0.2),   
                  Conv2D(32,(3,3),activation='relu'),
                  MaxPooling2D(2,2),
                  Flatten(),
                  Dense(128,activation='relu'),
                  Dropout(0.2),
                  Dense(64,activation='relu'),
                  Dense(1,activation='sigmoid')
                  ])

#Compiling

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#Preparing ImageDataGenerator

train_dir="/content/drive/My Drive/beauty/beauty/train"
val_dir="/content/drive/My Drive/beauty/beauty/valid"
image_generator=ImageDataGenerator(rescale=1./255,horizontal_flip=True,rotation_range=30,shear_range=0.2,
                                   zoom_range=0.2,width_shift_range=0.2,height_shift_range=0.2,fill_mode='nearest')
train_gen=image_generator.flow_from_directory(train_dir,target_size=(224,224),class_mode='binary',batch_size=64)
val_gen=image_generator.flow_from_directory(val_dir,target_size=(224,224),class_mode='binary',batch_size=64)



test_dir="/content/drive/My Drive/beauty/beauty/test"
test_gen=image_generator.flow_from_directory(test_dir,target_size=(224,224),class_mode='binary',batch_size=64)

#Model Training

history=model.fit(train_gen,epochs=30,validation_data=val_gen)

#Plotting Graphs
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy graph')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','vaidation'])
plt.grid()

#Testing on Dataset

model.evaluate(test_gen)