from moment import Moment
from datetime import datetime
import requests
from time import sleep

def __landmarks(results, mp_pose, image, articulacion) -> Moment:
  lm = results.pose_landmarks.landmark
  dim = image.shape
  if articulacion=="codo":
    try:
      left_shoulder = Moment(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x* dim[0], lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * dim[1])
      left_elbow = Moment(lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x* dim[0], lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y* dim[1])
      left_wrist = Moment(lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x* dim[0], lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y* dim[1])
      return left_shoulder, left_elbow ,left_wrist
    except:
      pass
  elif articulacion=="hombro":
    try:
      left_shoulder = Moment(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x* dim[0], lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * dim[1])
      left_elbow = Moment(lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x* dim[0], lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y* dim[1])
      left_hip = Moment(lm[mp_pose.PoseLandmark.LEFT_HIP.value].x* dim[0], lm[mp_pose.PoseLandmark.LEFT_HIP.value].y* dim[1])
      return left_hip, left_shoulder, left_elbow
    except:
      pass

def record(results, mp_pose, image, articulacion):
    start_point, middle_point, end_point = __landmarks(results, mp_pose, image, articulacion)
    angulo = start_point.angleBetween(middle_point, end_point)
    return (angulo, start_point.time)

def saveData(maximo, minimo, dni, articulacion):
  now = datetime.now().strftime("%b-%d-%Y")
  try:
    with open(f".\\data\\{dni}_{articulacion}.txt", "a") as f:
      f.writelines(f"{now}:{maximo},{minimo}\n")
      return True
  except FileNotFoundError:
    with open(f".\\data\\{dni}_{articulacion}.txt", "x") as f:
      f.writelines(f"{now}:{maximo},{minimo}\n")
      return False

def recuperarData(dni, articulacion):
  toReturn = list()
  with open(f".\\data\\{dni}_{articulacion}.txt", "r") as f:
    for l in f.readlines():
      datos = l.strip().split(":")
      fecha = datos[0]
      maximo = float(datos[1].split(",")[0])
      minimo = float(datos[1].split(",")[1])
      toReturn.append((fecha,[maximo,minimo]))
  return toReturn

def recuperarNormal(articulacion):
  with open(f".\\data\\rango_normal.txt", "r") as f:
    for l in f.readlines():
      datos = l.strip().split(":")
      arti = datos[0]
      if articulacion == arti:
        return (float(datos[1].split(",")[0]), float(datos[1].split(",")[1]))

def sendData(maximo, minimo, dni, articulacion, ip, port = 1880):
  url = f"http://{ip}:{port}/{dni}"
  dataToSend = {}
  dataToSend["valor"] = maximo
  dataToSend["type"] = "max"
  try:
      print("sending max")
      requests.post(url, data = dataToSend, timeout=1)
  except Exception as e:
    print(e)
  sleep(1)
  dataToSend["valor"] = minimo
  dataToSend["type"] = "min"
  try:
      print("sending min")
      requests.post(url, data = dataToSend, timeout=0.5)
  except Exception as e:
    print(e)