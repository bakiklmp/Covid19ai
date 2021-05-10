import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Model,Sequential, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.applications.densenet import DenseNet121
import seaborn as sns


'''CREATE DATASET'''

labels =['COVID', 'non-COVID']
data_dir = 'C:\\Users\\rojha\\Desktop\\literature\\covid19'
train_dir = os.path.join(data_dir)

train_data = []
for defects_id, sp in enumerate(labels):
    for file in os.listdir(os.path.join(train_dir, sp)):
        train_data.append(['{}/{}'.format(sp, file), defects_id, sp])
        
train = pd.DataFrame(train_data, columns=['File', 'DiseaseID','Disease Type'])
print(train.head())

''' Eğitim Görüntülerinin Okunması  '''
def read_image(filepath):
    return cv2.imread(os.path.join(data_dir, filepath))
# Görüntülerin yeniden boyutlandırılması
def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)


''' Eğitme '''
IMAGE_SIZE = 64
X_train = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
for i, file in tqdm(enumerate(train['File'].values)):
    image = read_image(file)
    if image is not None:
        X_train[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))
        
# Verileri Normalleştirme
X_Train = X_train / 255.
print('Train Shape: {}'.format(X_Train.shape))

Y_train = train['DiseaseID'].values
Y_train = to_categorical(Y_train, num_classes=2)

BATCH_SIZE = 64

X_train,X_val,Y_train,Y_val = train_test_split(X_Train, Y_train, 
                                               test_size = 0.2, random_state=42)

EPOCHS = 50
SIZE=64
N_ch=3

def build_densenet():
    densenet = DenseNet121(weights='imagenet', include_top=False)
    input = Input(shape=(SIZE, SIZE, N_ch))
    x = Conv2D(3, (3, 3), padding='same')(input) 
    x = densenet(x)
    
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # multi output
    output = Dense(2,activation = 'softmax', name='root')(x)
 
    # model
    model = Model(input,output)
    
    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()
    
    return model

model = build_densenet()
annealer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)

#Data Augmentation ile görüntüler oluşturma
datagen = ImageDataGenerator(rotation_range=360, # rasgele döndürme aralığı
                        width_shift_range=0.2, # Yatayda kaydırma aralığı
                        height_shift_range=0.2, # Dikeyde kaydırma aralığı
                        zoom_range=0.2, #Zoom değer aralığı
                        horizontal_flip=True, # Yatayda döndürme 
                        vertical_flip=True) # Dikeyde döndürme 
datagen.fit(X_train)
# Modeli, gerçek zamanlı veri artırımı(data augmentation) ile fit etme
hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
               steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
               epochs=EPOCHS,
               verbose=2,
               callbacks=[annealer, checkpoint],
               validation_data=(X_val, Y_val))



print(hist.history.keys())
# summarize history for accuracy
plt.plot(hist.history['accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

#karmaşıklık matrisi
import seaborn as sns
loadModel = load_model("model.h5") #daha önce eğitilen ağırlıklı modelin import edilmesi

#Y_pred = model.predict(X_val)
Y_pred = loadModel.predict(X_val) #predict fonksiyonu sayesinde tahmin yapılması

Y_pred = np.argmax(Y_pred, axis=1) 
Y_true = np.argmax(Y_val, axis=1)

cm = confusion_matrix(Y_true, Y_pred)  #karmaşıklık matrisinin hesabı
plt.figure(figsize=(12,12))

# karmaşıklık matrisi sonuçlarının görselleştirilmesi
ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, xticklabels=labels, yticklabels=labels, fmt=".1f")
ax.set_ylabel('Actual', fontsize=40)
ax.set_xlabel('Predicted', fontsize=40)
print(Y_pred.size, Y_true.size)