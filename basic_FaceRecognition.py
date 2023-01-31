# Importamos las librerias
import cv2
import numpy as np
import face_recognition as fr
import os
import random
from datetime import datetime

# Accedemos a la carpeta
path='images'
images = []

clases = []
lista = os.listdir(path)
#print(lista)

# Variables
comp1 = 100

# Leemos los rostros del DB
for lis in lista:
    # Leemos las imagenes de los rostros
    imgdb = cv2.imread(f'{path}/{lis}')
    # Almacenamos imagen
    images.append(imgdb)
    # Almacenamos nombre
    clases.append(os.path.splitext(lis)[0])

print (clases)

# Funcion para codificar los rostros
def codrostros(images):
    listacod = []
    # Iteramos
    for img in images:
        # Correccion de color
        img = cv2.cvtColor (img, cv2.COLOR_BGR2RGB)
        # Codificamos la imagen
        cod= fr.face_encodings(img)[0]
        # Almacenamos
        listacod.append(cod)

    return listacod

# Hora de ingreso
def horario (nombre):
    #Abrimos el archivo en modo lectura y escritura
    with open('horario.csv', 'r+') as h:
        # Leemos la informacion
        data = h.readline()
        # Creanos lista de nombres
        listanombres = []

        # Iteranos cada linea del doc
        for line in data:
            # Buscamos la entrada y la diferenciamos con
            entrada = line.split(',')
            # Almacenamos los nombres
            listanombres.append(entrada [0])

        # Verificamos si ya hemos almacenado el nombre
        if nombre not in listanombres:
            # Extraemos informacion actual
            info = datetime.now()
            # Extraemos fecha
            fecha = info.strftime('%Y:%m:%d')
            # Extraemos hora
            hora = info.strftime('%H:%M:%S')

            # Guardamos la informacion
            h.writelines(f'\n{nombre}, {fecha}, {hora}')
            print (info)

# Llamanos la funcion
rostroscod=codrostros(images)

# Realizamos VidepCaptura
cap = cv2.VideoCapture (0)

# Empezamos
while True:
    # Leemos los fotogramas
    ret, frame = cap.read()

    # Reducimos las imagenes para mejor procesamiento
    frame2= cv2.resize(frame, (0,0), None, 0.25, 0.25)

    # Conversion de color de BGR (distrucion de canales en cv) a RGB (en face_recognition) 
    rgb = cv2.cvtColor (frame2, cv2.COLOR_BGR2RGB)

    # Buscamos los rostros
    faces = fr.face_locations (rgb)
    facescod = fr.face_encodings (rgb, faces)

    # Iteramos
    for facecod, faceloc in zip(facescod, faces):
    
        # Comparamos rostro de DB con rostro en tiempo real
        comparacion = fr.compare_faces (rostroscod, facecod)

        # Calculamos la similitud
        simi = fr.face_distance (rostroscod, facecod)
        #print (simi)
        # Buscamos el valor mas bajo
        min = np.argmin (simi)
        
        if comparacion[min]:
            nombre = clases[min].upper()
            print (nombre)
            # Extraemos coordenadas
            yi, xf, yf, xi = faceloc
            # Escalamos
            yi, xf, yf, xi = yi*4, xf*4, yf*4, xi*4
        
            indice = comparacion.index(True)
            
            # Comparamos
            if comp1 != indice:
                # Para dibujar cambiamos colores
                r = random.randrange (0, 255, 50)
                g = random.randrange (0, 255, 50)
                b = random.randrange (0, 255, 50)
                
                comp1 = indice

            if comp1 == indice:
                # Dibujamos Latest cheer: No data
                cv2.rectangle (frame, (xi, yi), (xf, yf), (r, g, b), 3)
                cv2.rectangle(frame, (xi, yf-35), (xf, yf), (r, g, b), cv2.FILLED)
                cv2.putText(frame, nombre, (xi+6, yf-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
                horario(nombre)
        
    # Mostramos Frames
    cv2.imshow("Reconocimiento Facial", frame)
    
    # Leemos el teclado
    t = cv2.waitKey(5)
    if t == 27:
        break

cv2.destroyAllWindows()
cap.release()
