import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import tkinter
from tkinter import *

longitud, altura = 100, 100
modelo = './model_cv.h5'
pesos_modelo = './weights_cv.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)



def predict(file):
  
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    prediccion.set("Abrazadera")
  elif answer == 1:
    prediccion.set("Broca")
  elif answer == 2:
    prediccion.set("Clavo")
  elif answer == 3:
    prediccion.set("Dado")
  elif answer == 4:
    prediccion.set("Desarmador")
  elif answer == 5:
    prediccion.set("Grapa")
  elif answer == 6:
    prediccion.set("Rondana")
  elif answer == 7:
    prediccion.set("Tabique")
  elif answer == 8:
    prediccion.set("Tornillo")
  elif answer == 9:
    prediccion.set("Tuerca")
    

  return answer

if __name__ == '__main__':
    ventana = tkinter.Tk()
    prediccion = StringVar()
    ventana.geometry("300x150")
    ventana.title("Ingresar n(eta)")
    etiquetan = tkinter.Label(ventana, text = "Nombre del archivo")
    etiquetan.pack()
    cajan = tkinter.Entry(ventana,font = "Helvetica 15")
    cajan.pack()
    botonEnviarInfo = tkinter.Button(ventana,text="Enviar",command = lambda: predict(cajan.get()+'.png'))
    etiquetan2 = tkinter.Label(ventana, text = "Predicción")
    etiquetan2.pack()
    etiquetan3 = tkinter.Label(ventana, textvariable= prediccion)
    etiquetan3.pack()
    botonEnviarInfo.pack()
    ventana.mainloop()