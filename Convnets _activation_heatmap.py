from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import cv2

#Load model
model = VGG16(weights='imagenet')

#Image preprocess
img_path = 'Datasets/DataVisualization/example_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

#Here must be selected the outputs regards the class of imagenet
#in this case the class 'tabby cat' (281)
image_output = model.output[:, 281]

#Extract the output of the specified layer of the model
last_conv_layer = model.get_layer('block5_conv3')

#Computes the gradient of the input picture with regard to this loss 
grads = K.gradients(image_output, last_conv_layer.output)[0]

#Vector of shape (512,), where each entryis the mean intensity of the gradient 
# over a specific feature-map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

#Lets you access the values of the quantities you just defined
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

#Multiplies each channel in the feature-map array by “how important this channel is” 
# with regard to the class inside imagenet model taht appears in our image
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)

#Visualize the heatmap
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

#Superposes the original image with the heatmap
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('Datasets/DataVisualization/seatmap_image.jpg', superimposed_img)