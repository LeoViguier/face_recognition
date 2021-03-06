print("Chargement...")
import os
import cv2
import face_recognition as fr
import numpy as np
import sys

# modules pour l'interface graphique
import entre_nom, init_graph, confirmation_capture, num_camera
import warnings
warnings.filterwarnings("ignore")

NUM_CAMERA = 0
CLICKED = False

#Liste des images de visages connus
NOM_CONNUS = []
VISAGE_CONNUS = os.listdir("./known_faces/")
for i in VISAGE_CONNUS:
	NOM_CONNUS.append(i.split('.jpg')[0])

# Chargement des images en mémoire et préparation à la reconnaissance
IMAGES_ENCODEES = []
for i in VISAGE_CONNUS:
	IMAGES_ENCODEES.append(fr.face_encodings(fr.load_image_file("known_faces/" + i))[0])
PATH = os.path.abspath(os.path.dirname(sys.argv[0]))


def affichagegraphique(noms, localisation):
	# button dimensions (y1,y2,x1,x2)
	button = [420,477,4,330]

	# function that handles the mousclicks
	def process_click(event, x, y,flags, params):
		# check if the click is within the dimensions of the button
		if event == cv2.EVENT_LBUTTONDOWN:
			if y > button[0] and y < button[1] and x > button[2] and x < button[3]:   
				quit()
	
	image = cv2.imread("gris.png")
	image[button[0]:button[1],button[2]:button[3]] = [63,72,204]
	a = 65
	for nom in noms:
		cv2.putText(image, 'Personne(s) trouve(s) : ', (10, 25),cv2.FONT_HERSHEY_DUPLEX, 0.8, (5, 5, 5), 1)
		# écriture du nom dans l'affichage graphique
		cv2.putText(image, (" - " + nom), (20, a),cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
		a += 30
	# create a window and attach a mousecallback and a trackbar
	cv2.setMouseCallback('Reconnaissance Faciale',process_click)
	# affichage du bouton pour quitter
	cv2.putText(image, 'Cliquez ici pour quitter',(18,458),cv2.FONT_HERSHEY_DUPLEX, 0.8,(255, 255, 255),1)
	# affichage
	cv2.imshow("Reconnaissance Faciale", image)



def capture_camera():
	# button dimensions (y1,y2,x1,x2)
	button = [0,58,0,364]

	# function that handles the mousclicks
	def prend_photo(event, x, y,flags, params):
		global CLICKED
		# check if the click is within the dimensions of the button
		if event == cv2.EVENT_LBUTTONDOWN:
			if y > button[0] and y < button[1] and x > button[2] and x < button[3]:   
				CLICKED = True
	
	image = cv2.imread("camera.jpg")
	image[button[0]:button[1],button[2]:button[3]] = [63,72,204]
	# création de la fenetre et association avec la fonction de click
	cv2.setMouseCallback('Reconnaissance Faciale',prend_photo)
	# affichage du bouton pour quitter
	cv2.putText(image, 'Cliquez ici pour enregistrer',(0,37),cv2.FONT_HERSHEY_DUPLEX, 0.8,(255, 255, 255),1)
	# affichage
	cv2.imshow("Reconnaissance Faciale", image)



def insert_personne(img):
	"""
	Cette fonction reçoit des informations de la part de la fonction principale d'ajout de personnes.
	Elle a pour but de demander le nom du visage capturé, étant le paramêtre img et aussi de vérifier que ce nom n'existe pas encore.
	Une fois cette chose faite, l'image est enregistrée dans le fichier ./known_faces
	"""
	entre_nom.main_entre_nom()
	with open('choix.txt', 'r') as f:
		nom = f.read()
	base = """<html><head/><body><p align="center"><span style=" font-size:10pt; font-weight:600;">Entrez le nom de la personne:</span></p><p align="center"><br/></p></body></html>"""

	faces = VISAGE_CONNUS
	if nom + ".jpg" in faces:
		nouveau_nom = nom
		while nouveau_nom + ".jpg" in faces:
			nouvelle_phrase = """<html><head/><body><p align="center"><span style=" font-size:10pt; font-weight:600;">Entrez le nom de la personne:</span></p><p align="center"><span style=" font-size:10pt; font-weight:600;">Permettant de distinguer de '"""+nouveau_nom+"""' déjà existant.</span></p></body></html>"""
			with open('txt.txt', 'w') as f:
				f.write(nouvelle_phrase)

			entre_nom.main_entre_nom()
			with open('choix.txt', 'r') as f:
				nouveau_nom = f.read()
		try:
			nom_complet = nouveau_nom.replace(".jpg", "") + ".jpg"
		except:
			pass
	else:
		nom_complet = nom + ".jpg"
	with open('txt.txt', 'w') as f:
		f.write(base)
	print(nom_complet, "enregistré.")

	# enregistrement de l'image dans le répertoire known_faces avec le nom nom_complet
	cv2.imwrite(PATH +"/known_faces/" + nom_complet, img)

	return None


def detection(frame):
	"""
	sert a détecter le ou les visages dans la frame reçue en argument de la fonction.
	"""

	faces = fr.face_locations(frame)
	if len(faces) > 0:
		return faces
	return False


def detection_cv2(frame):
	"""
	sert a détecter un seul visage a la fois dans la frame reçue en argument de la fonction. Le fonctionnement est le meme que la fonction
	detection(frame) mais elle est plus rapide.
	"""

	facecascade = cv2.CascadeClassifier("C:/Users/leovi/OneDrive/Documents/Python/OpenCV/cascades/haarcascade_frontalface_alt.xml")
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = facecascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,50))
	if len(faces)> 0:
		return faces
	return False


def crop(frame, localisations):
	"""
	utilisé uniquement pour l'ajout de personnes.
	découpe du visage localisé avec la fonction detection() et renvoi.
	On renvoit donc une liste de visage(s) sans information inutile autour.
	"""

	decoupes = []
	for localisation in localisations:
		x = localisation[0]
		y = localisation[1]
		w = localisation[2]
		h = localisation[3]
		decoupe = frame[y:y+h, x:x+w]
		decoupes.append(decoupe)
	return decoupes


def reconnaissance(frame, localisation):
	"""
	Cette fonction recoit comme argument la frame en cours de traitement et une liste contenant les (la) localisation de ce(s) visage(s).
	Le but est de faire la correspondance entre un visage est son nom, en comparant avec ceux déjà connus dans le dossier ./known_faces
	on retourne une liste d'un ou plusieurs noms, avec par défaut le nom "inconnu" si on n'a pas trouvé de correspondance.
	"""

	noms = []
	encodages = fr.face_encodings(frame, localisation)

	for i in encodages:
		correspondances = fr.compare_faces(IMAGES_ENCODEES, i)
		nom = "inconnu"

		face_distances = fr.face_distance(IMAGES_ENCODEES, i)
		best_match_index = np.argmin(face_distances)
		if correspondances[best_match_index]:
			nom = NOM_CONNUS[best_match_index]

		noms.append(nom)
	return noms


def affichage(frame, noms, localisation):
	"""
	Cette fonction recoit comme arguments la frame en cours de traitement, une liste de nom(s) et la (les) localisation(s) des visages.
	On dessine un rectangle autour du visage, puis un rectangle plein pour accueillir le nom et enfin, le nom.
	"""

	for (haut, droite, bas, gauche), nom in zip(localisation, noms):

		# Rectangle autour de la tete
		cv2.rectangle(frame, (gauche, haut), (droite, bas), (0, 0, 255), 2)

		# Rectangle pour écrire le nom
		cv2.rectangle(frame, (gauche, bas + 35), (droite, bas), (0, 0, 255), cv2.FILLED)
		# écriture du nom
		cv2.putText(frame, nom, (gauche + 6, bas + 29), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
	



def main():
	global NUM_CAMERA
	choix = init()

	if choix == "T":
		video_cap = cv2.VideoCapture(NUM_CAMERA, cv2.CAP_DSHOW)
		while True:
			ret, frame = video_cap.read()
			retouche = frame[:, :, ::-1]
			
			coordonees = detection(retouche)
			if coordonees != False:
				noms = reconnaissance(retouche, coordonees)
				affichage(frame, noms, coordonees)
				affichagegraphique(noms, coordonees)
			cv2.imshow("Video", frame)
			# Appuyer sur q pour arréter.
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	else:
		ajout_personne()
		
	video_cap.release()
	cv2.destroyAllWindows()
	
	exit()


def ajout_personne():
	global NUM_CAMERA
	global CLICKED
	path = os.path.abspath(os.path.dirname(sys.argv[0]))
	print("Appuyez sur C pour enregistrer une image. Une seule personne à la fois devant la caméra")
	a = ''
	capture_decoupes = []
	random_path = []
	video_cap = cv2.VideoCapture(NUM_CAMERA, cv2.CAP_DSHOW)
	while True:
		ret, frame = video_cap.read()
		retouche = frame[:, :, ::-1]
		capture_camera()
		coordonees = detection_cv2(frame)
		if type(coordonees) != bool:
			coordonees = coordonees.tolist()

		if coordonees != False:
			frame_good = frame
			if CLICKED == True:
				CLICKED = False
				decoupe = crop(frame_good, coordonees)
				cv2.imshow("Capture", decoupe[0])
				confirmation_capture.main_confirm_capture()
				with open('choix.txt', 'r') as f:
					choix = f.read()
				if choix == 'oui':
					insert_personne(decoupe[0])
					break
				else:
					cv2.destroyWindow("Capture")
			cv2.rectangle(frame, (coordonees[0][0],coordonees[0][1]), (coordonees[0][0]+coordonees[0][2], coordonees[0][1]+coordonees[0][3]), (0, 0, 255), 2)


		cv2.imshow("Video", frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	video_cap.release()
	cv2.destroyAllWindows()
	
	exit()




def init():
	print('Running...')
	global NUM_CAMERA
	num_camera.main_num_camera()
	with open('choix.txt', 'r') as f:
		tmp = f.read()
	if tmp == '':
		NUM_CAMERA = 0
	else:
		NUM_CAMERA = int(tmp)

	init_graph.main_init_graph()
	with open('choix.txt', 'r') as f:
		choix = f.read()
	
	return choix


if __name__ == '__main__':
	main()


