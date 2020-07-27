import numpy as np
import tensorflow 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = tensorflow.keras.models.load_model('model.h5')
print("model loaded")
#uploaded = Image.open('test5.jpg')

img = image.load_img('C:/Users/dell/Documents/Face Recognition/demo test/test6.jpg' , target_size=(150,150))
print(img)
x =image.img_to_array(img)
x=np.expand_dims(x,axis=0)
images = np.vstack([x])
classes = model.predict(images,batch_size=10)
  
if classes==0:
	print('Harassment')
else:
	print('Normal')
