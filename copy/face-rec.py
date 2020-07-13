import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import platform
import pickle
import logging
import requests

known_face_encodings = []
known_face_metadata = []
opentime = datetime.now()
learn_mode = False  
number = "+61480019020"


def open_door_api():
	url = "http://192.168.0.97/api/devices/5/action/turnOn"

	payload = "{\"args\":[]}"
	headers = {
		'authorization': "Basic a2V2aW51ZGVsbEBpY2xvdWQuY29tOlRhbm5veTEx",
    		}

	response = requests.request("POST", url, data=payload, headers=headers)

	print(response.text)



def save_known_faces():
	with open("known_faces.dat", "wb") as face_data_file:
		face_data = [known_face_encodings, known_face_metadata]
		pickle.dump(face_data, face_data_file)
		print("Known faces backed up to disk.")

def load_known_faces():

	logging.basicConfig(filename="Face.log", level=logging.INFO)
	global known_face_encodings, known_face_metadata

	try:
		with open("known_faces.dat", "rb") as face_data_file:
			known_face_encodings, known_face_metadata = pickle.load(face_data_file)
			print("Known faces loaded from disk.")

	except FileNotFoundError as e:
		print("No previous face data found - starting with a blank known face list.")
		pass

def get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0):

	return (
		f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
		f'width=(int){capture_width}, height=(int){capture_height}, ' +
		f'format=(string)NV12, framerate=(fraction){framerate}/1 ! ' +
		f'nvvidconv flip-method={flip_method} ! ' +
		f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ' +
		'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
		)

def register_new_face(face_encoding, face_image):

	
	known_face_encodings.append(face_encoding)
	
	name = input("Name ")
	known_face_metadata.append({
		"name": name,
		"face_image": face_image,

	})
	logging.info("Face Saved as "+name+" at time/date "+ str(datetime.now()))
	print("Face saved")
	save_known_faces()

def face_match(face_encoding):
	
	metadata = None
	
	if len(known_face_encodings) == 0:
		return metadata
	
	face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
	best_match_index = np.argmin(face_distances)
	
	if face_distances[best_match_index] < 0.45:
		metadata = known_face_metadata[best_match_index]
		Name = metadata['name']
		unlock_door(Name)	

	return metadata

def unlock_door(name):
	
	global opentime, learn_mode

	if datetime.now() - opentime > timedelta(seconds=10) and learn_mode == False: 	
		open_door_api()
		print("open door for "+name+" at time/date "+ str(datetime.now()))
		logging.info("open door for "+name+" at time/date "+ str(datetime.now()))
		opentime = datetime.now()
		

def main_loop():
	
	#video_capture = cv2.VideoCapture(-1)
	video_capture = cv2.VideoCapture("rtsp://192.168.0.113/Streaming/Channels/101")	
	global learn_mode

	while True:

		ret, frame = video_capture.read()
		small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
		rgb_small_frame = small_frame[:, :, ::-1]
		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
		face_labels = []
		
		cv2.putText(frame, "Learn-mode = "+ str(learn_mode) , (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 197, 0), 1)

		for face_location, face_encoding in zip(face_locations, face_encodings):
			# See if this face is in our list of known faces.
			metadata = face_match(face_encoding)
			
			if metadata is not None:
				Name = metadata['name']
				face_label = f""+Name
			else:
				face_label = "Visitor"

				if learn_mode == True:

					top, right, bottom, left = face_location
					face_image = small_frame[top:bottom, left:right]
					face_image = cv2.resize(face_image, (150, 150))
					register_new_face(face_encoding, face_image)

			face_labels.append(face_label)

		# Draw a box around each face and label each face
		for (top, right, bottom, left), face_label in zip(face_locations, face_labels):
			# Scale back up face locations since the frame we detected in was scaled to 1/4 size
			top *= 4
			right *= 4
			bottom *= 4
			left *= 4

			# Draw a box around the face and label
			if face_label != "Visitor":
				cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
				cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
				cv2.putText(frame, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
			else:
				cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
				cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
				cv2.putText(frame, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
				
		# Display the final frame of video with boxes drawn around each detected fames
		cv2.imshow('Face-rec', frame)	
		
		key = cv2.waitKey(33)			
		
		if key == 61:
			save_known_faces()
			learn_mode = not learn_mode	
			logging.info("learn mode triggered ar time/date "+ str(datetime.now()))
			print("learn mode triggered into "+ str(learn_mode))
		if key == 45:
			save_known_faces()
			break
			


	# Release handle to the webcam
	video_capture.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	load_known_faces()
	main_loop()
























