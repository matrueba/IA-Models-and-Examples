from keras.applications import VGG16
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size=150):

    #Extract the output of the specified layer of the model
    layer_output = model.get_layer(layer_name).output

    #Compute the loss of the specified layer and filter of the model
    loss = K.mean(layer_output[:, :, :, filter_index])

    #Computes the gradient of the input picture with regard to this loss 
    grads = K.gradients(loss, model.input)[0]
    
    #Normalize the gradient dividing by L2 Norm to ensure the magnitude of the 
    #updates done to the input image is always within the same range
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    #Returns a list of the loss values and the gradient values of te input picture
    iterate = K.function([model.input], [loss, grads])

    #Create a random noisy image
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.

    #Appky 40 iterations of gradient descent over the noisy image
    #The result is a image thaht need will be deprocessed
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]

    return deprocess_image(img)

#Load the model
model = VGG16(weights='imagenet', include_top=False)

layer_name = 'block4_conv1'
size = 64
margin = 5
results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

#iterates over the rows and columns to generate the filter patterns grid
for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
plt.figure(figsize=(20, 20))
plt.imshow(results)
plt.show()