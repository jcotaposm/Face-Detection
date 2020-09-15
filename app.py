# IImportando las librerias requeridas
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os


# cargando modelos pre-entrenados
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
except Exception:
    st.write("Error en la carga")

def detect(image):
    '''
    Función para detectar rostros / ojos y sonrisas en la imagen pasada a esta función
    '''

    image = np.array(image.convert('RGB'))
    
    
    faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=5)

    # Dibuja un rectángulo alrededor de las caras
    for (x, y, w, h) in faces:
        
        # Los siguientes son los parámetros de cv2.rectangle ()
        cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        
        roi = image[y:y+h, x:x+w]
        
        # Detectar ojos
        eyes = eye_cascade.detectMultiScale(roi)
        
        # Detectar sonrisas
        smile = smile_cascade.detectMultiScale(roi, minNeighbors = 25)
        
        # Dibujar un rectángulo alrededor de los ojos
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
            
        # Dibujando un rectángulo alrededor de una sonrisa
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2)

    # Devolver la imagen con cuadros delimitadores dibujados en ella (en caso de objetos detectados) y matriz de caras
    return image, faces


def about():
	st.write(
		'''
		**Deteccion* de rostros fue pensado para implementar en sitios importantes de la empresa con el objetivo final es identificar quienes hacen ingreso a estas instalaciones. 
		''')


def main():
    st.title("Aplicacion Detecta Rostros:sunglasses: ")
    st.write("**Usando Haar cascade Classifiers**")

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Pick something fun", activities)

    if choice == "Home":

    	st.write("Vaya a la sección About de en la barra lateral para obtener más información al respecto.")
        
        # Puede especificar más tipos de archivos a continuación si lo desea
    	image_file = st.file_uploader("Sube tu imagen aqui", type=['jpeg', 'png', 'jpg', 'webp'])

    	if image_file is not None:

    		image = Image.open(image_file)

    		if st.button("Ejecutar"):
                
                # result_img es la imagen con un rectángulo dibujado en ella (en caso de que se detecten caras)
                # result_faces es la matriz con coordenadas de cuadro delimitador
    			result_img, result_faces = detect(image=image)
    			st.image(result_img, use_column_width = True)
    			st.success("Se ha encontrado {} caras\n".format(len(result_faces)))

    elif choice == "About":
    	about()




if __name__ == "__main__":
    main()
