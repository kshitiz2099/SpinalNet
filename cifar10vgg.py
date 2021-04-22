import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers, Input, Sequential, applications
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import random

"""Loading and Preprocessing"""

cifar10=tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test)= cifar10.load_data()
y_train = OneHotEncoder().fit_transform(np.array(y_train).reshape(-1,1)).toarray()
y_test = OneHotEncoder().fit_transform(np.array(y_test).reshape(-1,1)).toarray()
labels=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

X_train, X_test=X_train/255.0, X_test/255.0
HALF_WIDTH=256
WIDTH=512
batch_size=100
input_tensor= Input(shape=(32,32,3))


"""Models and Utils"""

#VGG19 from Keras
def VGG19(input_tensor, trainable=False):
  vgg= applications.VGG19(weights='imagenet',
                               include_top=False,
                               input_tensor=input_tensor)
  for layers in vgg.layers:
    layers.trainable=trainable
  return vgg

#Defining Spinal Net Architecture
def SpinalNetFC(vgg):
  x=Flatten()(vgg.output)

  x=Dropout(0.1)(x[:, 0:HALF_WIDTH])
  x1=Dense(WIDTH, activation='relu')(x)

  x2=Dropout(0.1)(tf.concat([x1,x[:,HALF_WIDTH: HALF_WIDTH*2]], axis=1))
  x2=Dense(WIDTH, activation='relu')(x2)
  
  x3=Dropout(0.1)(tf.concat([x2,x[:,0: HALF_WIDTH]], axis=1))
  x3=Dense(WIDTH, activation='relu')(x3)
  
  x4=Dropout(0.1)(tf.concat([x3,x[:,HALF_WIDTH: HALF_WIDTH*2]], axis=1))
  x4=Dense(WIDTH, activation='relu')(x4)
  
  x=tf.concat([x1,x2], axis=1)
  x=tf.concat([x,x3], axis=1)
  x=tf.concat([x,x4], axis=1)

  x=Dropout(0.1)(x)
  output=Dense(10, activation='softmax')(x)

  return Model(vgg.input,outputs=output)

#Defining the Normal NN Architecture
def normalModel(vgg):
  x=Flatten()(vgg.output)

  x=Dropout(0.25)(x)
  x=Dense(4096, activation='relu')(x)

  x=Dropout(0.25)(x)
  x=Dense(4096, activation='relu')(x)

  x=Dropout(0.25)(x)
  x=Dense(10, activation='softmax')(x)

  return Model(vgg.input, outputs=x)

#Generate plots
def plot_graph(history={}, metrics=[], width=10, height=4):
  f=plt.figure()
  f.set_figwidth(width)
  f.set_figheight(height)
  for metric in metrics:
    plt.plot(history[metric], label=metric)
  plt.legend()
  plt.grid()
  plt.show()


#Data Augmentation, Training, and Validation
def data_augmentation(X_train, y_train, batch_size, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=False, vertical_flip=False, shear_range=0):
  data_generator=tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=width_shift_range, height_shift_range=height_shift_range, horizontal_flip=horizontal_flip, shear_range=shear_range)
  return data_generator.flow(X_train, y_train, batch_size)

#Train the Model
def train_model(model, X_train, y_train, epochs, callbacks=[]):
  model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

  steps_per_epoch=X_train.shape[0]//batch_size

  training_data=data_augmentation(X_train, y_train, batch_size=100, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, shear_range=0.2)
  
  return model.fit(training_data, validation_data=(X_test, y_test), steps_per_epoch=steps_per_epoch, callbacks=callbacks, epochs=epochs)

#Predict a single example
def predict(model, X, labels):
  return model.predict(X.reshape(-1,32,32,3)).argmax()

def lr_schedule(epoch):
  if epoch<5:
    lr=0.0001
  else:
    lr=0.00001
  return lr

EPOCHS=200
"""First Training Cycle (100 Epochs)"""

#Spinal Net
spinal_net_model=SpinalNetFC(VGG19(input_tensor, trainable=True))
spinal_net_model.summary()
spinal_r=train_model(spinal_net_model, X_train, y_train, epochs=EPOCHS//2)

plot_graph(spinal_r.history, ['accuracy', 'val_accuracy'])
plot_graph(spinal_r.history, ['loss', 'val_loss'])

spinal_net_history_full={}
for key in spinal_r.history.keys():
    spinal_net_history_full[key]=spinal_r.history[key]

#Normal NN
normal_model=normalModel(VGG19(input_tensor, trainable=True))
normal_model.summary()
normal_r=train_model(normal_model, X_train, y_train, epochs=EPOCHS//2)

plot_graph(normal_r.history, ['accuracy', 'val_accuracy'])
plot_graph(normal_r.history, ['loss', 'val_loss'])

normal_nn_history_full={}
for key in normal_r.history.keys():
    normal_nn_history_full[key]=normal_r.history[key]

#Spinal Net vs Normal NN
plot_graph({'spinal_net_accuracy': spinal_r.history['val_accuracy'], 'normal_nn_accuracy':normal_r.history['val_accuracy']}, ['spinal_net_accuracy','normal_nn_accuracy'])
plot_graph({'spinal_net_loss': spinal_r.history['val_loss'], 'normal_nn_loss':normal_r.history['val_loss']}, ['spinal_net_loss','normal_nn_loss'])

"""Second Training Cycle (100 Epochs)"""

#Learning Rate Scheduling
lr=LearningRateScheduler(lr_schedule)
callbackList=[lr]

#Spinal Net
spinal_r=train_model(spinal_net_model, X_train, y_train, epochs=EPOCHS//2, callbacks=callbackList)

plot_graph(spinal_r.history, ['accuracy', 'val_accuracy'])
plot_graph(spinal_r.history, ['loss', 'val_loss'])

for key in spinal_net_history_full.keys():
    spinal_net_history_full[key]+=spinal_r.history[key]

"""Save Model and History to files"""
# spinal_net_model.save('./SpinalNetCifar10.hdf5')
# SNH=open('SpinalNetHistory.txt', 'w')
# SNH.write(str(spinal_net_history_full))
# SNH.close()

#Normal NN
normal_r=train_model(normal_model, X_train, y_train, epochs=EPOCHS//2,callbacks=callbackList)

plot_graph(normal_r.history, ['accuracy', 'val_accuracy'])
plot_graph(normal_r.history, ['loss', 'val_loss'])

for key in normal_nn_history_full.keys():
    normal_nn_history_full[key]+=normal_r.history[key]

"""Save Model and History to files"""
# normal_model.save('./NormalNNCifar10.hdf5')
# nn=open('NormalNNHistory.txt', 'w')
# nn.write(str(normal_nn_history_full))
# nn.close()


"""Compare Spinal Net vs Normal NN validation loss and accuracy"""
plot_graph({'spinal_net_accuracy': spinal_net_history_full['val_accuracy'], 'normal_nn_accuracy':normal_nn_history_full['val_accuracy']}, ['spinal_net_accuracy','normal_nn_accuracy'])
plot_graph({'spinal_net_loss': spinal_net_history_full['val_loss'], 'normal_nn_loss':normal_nn_history_full['val_loss']}, ['spinal_net_loss','normal_nn_loss'])


"""Evaluation and Prediction"""
[sn_loss,sn_accuracy]=spinal_net_model.evaluate(X_test, y_test)
[normalnn_loss, normalnn_accuracy]=normal_model.evaluate(X_test, y_test)

print("SpinalNet Loss: "+ str(sn_loss) +", Normal NN Loss: "+ str(normalnn_loss))
print("SpinalNet Accuracy: "+ str(sn_accuracy*100)+"%"+", Normal NN Accuracy: "+str(normalnn_accuracy*100)+"%")

noOfPredictions=10
for i in range(noOfPredictions):
  i=random.randrange(10000)
  print("\nSample",i)
  sn_prediction=predict(spinal_net_model, X_test[i], labels)
  nn_prediction=predict(normal_model, X_test[i], labels)
  true_label=labels[y_test[i].argmax()]

  print("True Label:",true_label)
  print("SpinalNet Prediction:", labels[sn_prediction])
  print("Normal NN Prediction:", labels[nn_prediction])

  plt.imshow(X_test[i])
  plt.show()