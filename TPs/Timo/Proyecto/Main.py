import cv2
import mediapipe as mp
from fingers import thumbs_up
import data, time
import matplotlib.pyplot as plt


# Parámetros modificables
sleepTime = True #* Tiempo muerto antes de empezar utilizar la cámara
seeTime = 7  #* Tiempo durante el que se toman datos
online = False #* Enviar datos online
ip = "172.22.38.181" #* IP a la cual enviar datos
port = 1880 #* Puerto a la cual enviar datos


#! Código, no modificar

dni = input("Ingrese su DNI: ").strip().lower()
articulacion = input("Ingrese la articulacion: ").strip().lower()
if sleepTime:
  time.sleep(3)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands= mp.solutions.hands

allData = list()
started =""

sendData = False

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("sad :(")
      continue
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if sendData:
      datos= data.record(results, mp_pose, image, articulacion)
      allData.append(datos)
      dt = time.time() - started
      if dt > seeTime:
        break
      else:
        print(f"Time left: {seeTime-dt}")

    else:
      try:
        fingers = thumbs_up(image)
        if fingers != "_" and int(fingers) >= 4:
          sendData = True
          started = time.time()
      except ValueError:
        print("upsi")

    # Draw the pose annotation on the image.
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) == ord("q"):
      break
cap.release()

angulo = [allData[i][0] for i in range(len(allData))]
tiempo = [allData[i][1] for i in range(len(allData))]

exist = data.saveData(str(max(angulo)),str(min(angulo)), dni, articulacion)

if exist:
  old_data = data.recuperarData(dni, articulacion)

  fechas = list()
  maximos_old=list()
  minimos_old=list()

  for f, d in old_data:
    fechas.append(f)
    maximos_old.append(d[0])
    minimos_old.append(d[1])

plt.plot(tiempo, angulo)
plt.title("Resultados")
plt.show()

if exist:
  normal = data.recuperarNormal(articulacion)

  #*Evolucion en el tiempo
  plt.plot(fechas,maximos_old, color="red")
  plt.plot(fechas,minimos_old, color ="green")
  #*Normal maximo
  plt.axhline(normal[0], color = "yellow")
  plt.text(1,normal[0],"Valor máximo normal")

  #*Normal minimo
  plt.axhline(normal[1], color="yellow")
  plt.text(1,normal[1],"Valor mínimo normal")

  #*Plot setting
  plt.legend(["Máximos historicos", "Mínimos historicos"])
  plt.title("Evolución temporal")
  plt.show()

if online:
  data.sendData(str(max(angulo)),str(min(angulo)), dni, articulacion, ip)

