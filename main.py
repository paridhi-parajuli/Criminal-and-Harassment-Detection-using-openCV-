
import os
from PIL import Image
import numpy as np
import streamlit as st
import cv2
import pickle
from playsound import playsound
import alert_mail as m
import tensorflow 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import train as tt


roi_g=None
roi_c=None
def my_css():
	with open('style.css') as f:
		st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#st.sidebar.markdown("<div ><h2>this is an html string</h2><div>")
my_css()
html_string = "<h2>Criminal Detection</h2><br>"

st.markdown(html_string, unsafe_allow_html=True)

if st.button("Fetch Data from CBI") :
	tt.train_model(st)



html_string = "<h6>This allows you to turn on the CCTV and make you alert when suspicious person is seen.</h6><br>"

st.markdown(html_string, unsafe_allow_html=True)	
#st.markdown("This allows you to turn on the CCTV and make you alert when suspicious person is seen.")
if st.button("Start Monitoring", key='predict'):
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	recognizer=cv2.face.LBPHFaceRecognizer_create()
	recognizer.read("training.yml")
	with open("lables.pickle",'rb') as f:
		labels_=pickle.load(f)
		labels={v:k for k,v in labels_.items()}
	mail_img=False
	cap = cv2.VideoCapture(0)
	repeated=0
	n_repeated=0
	criminals=[]
	font=cv2.FONT_HERSHEY_SIMPLEX
	color=(255,0,0)
	stroke=2
	charles=0
	bin_laden=0
	while True:
		_, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.6, 8)
		for (x, y, w, h) in faces:
			roi_g=gray[y:y+h, x:x+w]
			roi_c=img[y:y+h, x:x+w]
			cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
			idd,confidence=recognizer.predict(roi_g)
			print(idd)
			print (confidence)
			if confidence>=50 and (idd==2) :
				
				cv2.putText(img,'Criminal',(x,y),font,1,color,stroke,cv2.LINE_AA)
				if repeated==0:
					criminals.append (roi_g)
					st.write("Alert! Criminal Suspected.")
					st.image(roi_c)
					mail_img=True
					bin_laden=1
					cv2.imwrite('bin_laden.jpg',roi_c)
					st.write("Mail has been sent")
					repeated=1
					playsound('beep.mp3')
			
			if confidence>=50 and (idd==1) :
				
				cv2.putText(img,'Criminal',(x,y),font,1,color,stroke,cv2.LINE_AA)
				if n_repeated==0:
					criminals.append (roi_g)
					st.write("Alert! Criminal Suspected.")
					st.image(roi_c)
					mail_img=True
					charles=1
					cv2.imwrite('charles.jpg',roi_c)
					st.write("Mail has been sent")
					n_repeated=1
					playsound('beep.mp3')
			if mail_img:
				m.do_this(charles, bin_laden)
				charles=0
				bin_laden=0
				mail_img=False
		cv2.imshow('CCTV', img)
		if  cv2.waitKey(1) & 0xFF==ord('q'):
			stop=True
			break
	cap.release()
	cv2.destroyAllWindows()


html_string = "<h2>Harassment Detection</h2><br>"

st.markdown(html_string, unsafe_allow_html=True)		

@st.cache(allow_output_mutation=True)
def load_harassment_model():
	model = tensorflow.keras.models.load_model('model.h5')
	return model



def detect_(uploaded):
	cv2.imwrite(str('C:/Users/dell/Documents/Face Recognition/')+'test5.jpg',uploaded)
	myimg = image.load_img('test5.jpg' , target_size=(150,150))
	x =image.img_to_array(myimg)
	x=np.expand_dims(x,axis=0)
	images = np.vstack([x])
	classes = model.predict(images,batch_size=10)
  
	if classes==0:
		st.write('HARASSMENT')
	else:
		st.write('NORMAL')	

uploaded_pic = st.file_uploader("Upload your picture", type=("png", "jpg","jpeg"))
if uploaded_pic is not None:
	st.write('Sucessfully Uploaded!')
	img = Image.open(uploaded_pic)
	st.image(img, caption='Your Image.', width=400)
	model=load_harassment_model()
	img=img.convert('RGB')
	newsize = (150, 150)
	img = img.resize(newsize) 
	x =image.img_to_array(img)
	x=np.expand_dims(x,axis=0)
	images = np.vstack([x])
	classes = model.predict(images,batch_size=10)
  
	if classes==0:
		st.write('The image is classified as: HARASSMENT')
	else:
		st.write('The image is classified as: NORMAL')
	

	
		
