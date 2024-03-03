from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model= load_model('../Model/cat_dog_classification.h5')
image_path = 'Input_Data_to_predict/cat_1.jpg'

img = image.load_img(image_path, target_size=(150,150))
image_array= image.img_to_array(img)
print(image_array)
image_array= np.expand_dims(
                image_array,
                axis =0
)
print("expand")
print(image_array)

image_array /= 255.0
print(image_array)

prediction = model.predict(image_array)

if prediction[0][0]> 0.5:
    print('dog')
else:
    print('cat')