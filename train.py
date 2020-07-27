from PIL import Image
import os
import numpy as np
import cv2 
import pickle
#import streamlit as st



def train_model(st):
	current_id=0
	label_ids={}
	y_labels=[]
	x_train=[]
	st.write("updating the data...")
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	recognizer=cv2.face.LBPHFaceRecognizer_create()
	root_path='C:/Users/dell/Documents/Face Recognition/train_images'
	for root,dirs, files in os.walk(root_path):
		for file in files:
			path=os.path.join(root,file)
			label=os.path.basename(os.path.dirname(path)).lower()
			if not label in label_ids:
				label_ids[label]=current_id
				current_id +=1
			idd=label_ids[label]
			img=Image.open(path).convert("L")
			img_array=np.array(img,"uint8")
			faces = face_cascade.detectMultiScale(img_array, 1.1, 6)
			for (x, y, w, h) in faces:
				roi=img_array[y:y+h,x:x+w]
				x_train.append(roi)
				y_labels.append(idd)
	with open("lables.pickle",'wb') as f:
		pickle.dump(label_ids,f)
	st.write("Training...")
	recognizer.train(x_train, np.array(y_labels))

	recognizer.save("training.yml")
	st.write("model saved")

