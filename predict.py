# -*- coding: utf-8 -*-
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import sys

with open('model_cnn.json', 'r') as f:
    model = model_from_json(f.read())

model.load_weights('model_cnn.h5')

test_image = image.load_img(sys.argv[1], target_size = (200, 200))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

val = model.predict(test_image)

result_label = {
    "cat": 0,
    "dog": 1
}

print("\n")
print("I think it's a {}".format(list(result_label.keys())[list(result_label.values()).index(int(val[0][0]))]))