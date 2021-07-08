import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import tkinter

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
    print("pred: Abrazadera",answer)
  elif answer == 1:
    print("pred: Broca",answer)
  elif answer == 2:
    print("pred: Clavo",answer)
  elif answer == 3:
    print("pred: Dado",answer)
  elif answer == 4:
    print("pred: Desarmador",answer)
  elif answer == 5:
    print("pred: Grapa",answer)
  elif answer == 6:
    print("pred: Rondana",answer)
  elif answer == 7:
    print("pred: Tabique",answer)
  elif answer == 8:
    print("pred: Tornillo",answer)
  elif answer == 9:
    print("pred: Tuerca",answer)
    

  return answer

if __name__ == '__main__':
    ventana = tkinter.Tk()
    ventana.geometry("300x150")
    ventana.title("Ingresar n(eta)")
    etiquetan = tkinter.Label(ventana, text = "Nombre del archivo")
    etiquetan.pack()
    cajan = tkinter.Entry(ventana,font = "Helvetica 15")
    cajan.pack()
    botonEnviarInfo = tkinter.Button(ventana,text="Enviar",command = lambda: predict(cajan.get()+'.png'))
    botonEnviarInfo.pack()
    ventana.mainloop()