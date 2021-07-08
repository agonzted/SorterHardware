import numpy as np # linear algebra
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import random
import matplotlib.pyplot as plt
#Dependencias
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
#CNN
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import warnings
import os
import shutil
from PIL import ImageFile
warnings.simplefilter('error', Image.DecompressionBombWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

config = ConfigProto()
config.gpu_options.allow_growth = False
session = InteractiveSession(config=config)
#variables y arreglos
datasetFolderName='dataset'
MODEL_FILENAME="model_cv.h5"
WEIGHTS_FILENAME="weights_cv.h5"
sourceFiles=[]
classLabels=['abrazaderas','brocas','clavos','dados','desarmadores','grapas','rondanas','tabique','tornillos','tuercas']
#cargar imagenes de los folders del  dataset
def transferBetweenFolders(source, dest, splitRate):   
    global sourceFiles
    sourceFiles=os.listdir(source)
    if(len(sourceFiles)!=0):
        transferFileNumbers=int(len(sourceFiles)*splitRate)
        transferIndex=random.sample(range(0, len(sourceFiles)), transferFileNumbers)
        for eachIndex in transferIndex:
            shutil.move(source+str(sourceFiles[eachIndex]), dest+str(sourceFiles[eachIndex]))
    else:
        print("No hay archivos!")
        
def transferAllClassBetweenFolders(source, dest, splitRate):
    for label in classLabels:
        transferBetweenFolders(datasetFolderName+'/'+source+'/'+label+'/', 
                               datasetFolderName+'/'+dest+'/'+label+'/', 
                               splitRate)


#Primero checamos si los folders estan vacios o no, si estan transferimos todos los archivos existentes a la carpeta train.
transferAllClassBetweenFolders('test', 'train', 1.0)
# Enviamos unos cuantos archivos de la carpeta train a test para el testeo.
transferAllClassBetweenFolders('train', 'test', 0.20)

#Declaramos nuestro arreglos para las imagenes o inputs (X) y para las clases o labels (Y)
X=[]
Y=[]
#Este metodo obtiene mediante un foreach loop los archivos/imagenes en la carpeta y los mete al arreglo X, los labels se meten al arreglo
#Y dependiendo si el nombre del folder corresponde al valor en el arreglo de clases.
def prepareNameWithLabels(folderName):
    sourceFiles=os.listdir(datasetFolderName+'/train/'+folderName)
    for val in sourceFiles:
        X.append(val)
        if(folderName==classLabels[0]):
            Y.append(0)
        elif(folderName==classLabels[1]):
            Y.append(1)
        elif(folderName==classLabels[2]):
            Y.append(2)
        elif(folderName==classLabels[3]):
            Y.append(3)
        elif(folderName==classLabels[4]):
            Y.append(4)
        elif(folderName==classLabels[5]):
            Y.append(5)
        elif(folderName==classLabels[6]):
            Y.append(6)
        elif(folderName==classLabels[7]):
            Y.append(7)
        elif(folderName==classLabels[8]):
            Y.append(8)
        else:
            Y.append(9)

#Organizamos nuestras carpetas usando los metodos.
prepareNameWithLabels(classLabels[0])
prepareNameWithLabels(classLabels[1])
prepareNameWithLabels(classLabels[2])
prepareNameWithLabels(classLabels[3]) 
prepareNameWithLabels(classLabels[4])
prepareNameWithLabels(classLabels[5])
prepareNameWithLabels(classLabels[6])
prepareNameWithLabels(classLabels[7])
prepareNameWithLabels(classLabels[8])
prepareNameWithLabels(classLabels[9])                

#Convertimos nuestros vectores X y Y a arreglo numpy
X=np.asarray(X)
Y=np.asarray(Y)


# Determinamos el tamaño del batch, el numero de epocas y el metodo de activacion.
batch_size = 1
epoch=20
activationFunction='relu'
#Creamos el modelo de nuestra red neuronal, usamos dos capas, una de 32 neuronas y la otra de 16. Para el Flatten, usamos softmax como metodo
#de activacion, al compilar el modelos usamos el optimizador Adam.
def getModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation=activationFunction, input_shape=(img_rows, img_cols, 3)))
    model.add(Conv2D(32, (3, 3), activation=activationFunction))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Conv2D(16, (3, 3), padding='same', activation=activationFunction))
    model.add(Conv2D(16, (3, 3), activation=activationFunction))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Flatten())
    model.add(Dense(32, activation=activationFunction))
    model.add(Dense(16, activation=activationFunction))
    model.add(Dense(10, activation='softmax')) 
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return model

#Obtenemos la precision, exactitud y el puntaje F1, que es el promedio de la prescicion y la recuperacion del modelo.
def my_metrics(y_true, y_pred):
    accuracy=accuracy_score(y_true, y_pred)
    precision=precision_score(y_true, y_pred,average='weighted')
    f1Score=f1_score(y_true, y_pred, average='weighted') 
    print("Exactitud  : {}".format(accuracy))
    print("Precicion : {}".format(precision))
    print("Puntaje F1 : {}".format(f1Score))
    return accuracy, precision, f1Score

#determinamos el tamaño de las imagenes a 100 x 100 pixeles
img_rows, img_cols =  100, 100

train_path=datasetFolderName+'/train/'
validation_path=datasetFolderName+'/validation/'
test_path=datasetFolderName+'/test/'
model=getModel()


fold_cant = 3



# ===============Stratified K-Fold======================
skf = StratifiedKFold(n_splits=fold_cant, shuffle=True)
skf.get_n_splits(X, Y)
foldNum=0
for train_index, val_index in skf.split(X, Y):
    #Cortamos todas las imagenes de validation hacia train si alguna existe.
    transferAllClassBetweenFolders('validation', 'train', 1.0)
    foldNum+=1
    print("Results for fold",foldNum)
    X_train, X_val = X[train_index], X[val_index]
    Y_train, Y_val = Y[train_index], Y[val_index]

    

    #Movemos las imagenes a validar de este fold de la carpeta de train a la carpeta de validacion.
    for eachIndex in range(len(X_val)):
        classLabel=''
        if(Y_val[eachIndex]==0):
            classLabel=classLabels[0]
        elif(Y_val[eachIndex]==1):
            classLabel=classLabels[1]
        elif(Y_val[eachIndex]==2):
            classLabel=classLabels[2]
        elif(Y_val[eachIndex]==3):
            classLabel=classLabels[3]
        elif(Y_val[eachIndex]==4):
            classLabel=classLabels[4]
        elif(Y_val[eachIndex]==5):
            classLabel=classLabels[5]
        elif(Y_val[eachIndex]==6):
            classLabel=classLabels[6]
        elif(Y_val[eachIndex]==7):
            classLabel=classLabels[7]
        elif(Y_val[eachIndex]==8):
            classLabel=classLabels[8]
        else:
            classLabel=classLabels[9]   
        #Then, copy the validation images to the validation folder
        #Copiamos las imagenes a validar al folder de validation
        shutil.move(datasetFolderName+'/train/'+classLabel+'/'+X_val[eachIndex], 
                    datasetFolderName+'/validation/'+classLabel+'/'+X_val[eachIndex])
        
    train_datagen = ImageDataGenerator(
                rescale=1./255,
        		zoom_range=0.20,
            	fill_mode="nearest"
                )
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
        
    #Iniciamos con el modelo de ImageClassification
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

    validation_generator = validation_datagen.flow_from_directory(
            validation_path,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode=None, 
            shuffle=False)   
   
    # calculamos el fitness del modelo
    history=model.fit_generator(train_generator, 
                        epochs=epoch)
    

#Graficamos el error al final del proceso de aprendizaje con el historial de categorical-accuaracy
    if foldNum == fold_cant:
        plt.plot(history.history['categorical_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    predictions = model.predict_generator(validation_generator, verbose=1)
    yPredictions = np.argmax(predictions, axis=1)
    true_classes = validation_generator.classes
    # evaluamos el rendimiento de la validacion
    print("***Performance on Validation data***")    
    valAcc, valPrec, valFScore = my_metrics(true_classes, yPredictions)

 
# =============TESTEO=============
print("==============RESULTADOS DEL MODELO============")
test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False) 
predictions = model.predict(test_generator, verbose=1)
yPredictions = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
testAcc,testPrec, testFScore = my_metrics(true_classes, yPredictions)
#Guardamos el modelo y los pesos al final del aprendizaje.
model.save(MODEL_FILENAME)
model.save_weights(WEIGHTS_FILENAME)