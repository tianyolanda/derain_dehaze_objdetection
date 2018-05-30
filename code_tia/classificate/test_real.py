import numpy as np
from keras.models import load_model
from PIL import Image


model = load_model("./model_save/model_3_try.h5")
# path = "./varl/fog.jpg"
#path = "./varl/rain.jpg"
path = "./varl/sunny.jpg"
image_test = Image.open(path)
image_test = image_test.resize((150, 150))
image_test.show()

image_array = np.array(image_test)
image_array = image_array[np.newaxis,:,:,:]
result = model.predict(image_array)
print(result)
max_value = np.max(result)

if result[0, 0] == max_value:
    print("0", "sunny")
elif result[0, 1] == max_value:
    print("1", "Fog!")
elif result[0, 2] == max_value:
    print("2", "Rain!")


# if model.predict(image_array) >= 0.5:
#     print("the picture is rain!")
# else:
#     print("the picture is fog!")

