from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
from scipy.misc import imsave, imresize
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
from keras.applications import vgg16
import matplotlib.pyplot as plt

origin_image_path = 'Datasets/Styletransfer/original_image_mir.jpg'
style_image_path = 'Datasets/Styletransfer/style_image_mir.jpg'
result_image_path = 'Final_Transfer_Image.jpg'
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 400

def preprocess_image(image):
    img = img_to_array(image)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img

def deprocess_image(x, img_rows, img_cols, img_channels):
    x = x.reshape((img_rows, img_cols, img_channels))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # BGR -> RGB
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination, img_rows,img_cols, img_channels):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    size = img_rows * img_cols
    return K.sum(K.square(S - C)) / (4. * (img_channels ** 2) * (size ** 2))
    

def content_loss(origin, combination):
    return K.sum(K.square(combination - origin))

def variation_loss(x, img_rows,img_cols):
    dx = K.square(x[:, :img_rows - 1, :img_cols - 1, :] -
    x[:, 1:, :img_cols - 1, :])
    dy = K.square(x[:, :img_rows - 1, :img_cols - 1, :] -
    x[:, :img_rows - 1, 1:, :])
    return K.sum(K.pow(dx + dy, 1.25))


class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_rows, img_cols, img_channels))
        outs = fetch_loss_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
    
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


#Load images
raw_origin_img = load_img(origin_image_path)
raw_style_img = load_img(style_image_path)

#Resize images
origin_img_r = imresize(raw_origin_img, (IMAGE_HEIGHT, IMAGE_WIDTH))
style_img_r = imresize(raw_style_img, (IMAGE_HEIGHT, IMAGE_WIDTH))
img_rows,img_cols, img_channels = origin_img_r.shape

#Plot original images
plt.subplot(1, 2, 1)
plt.title("original")
plt.imshow(origin_img_r)
plt.subplot(1, 2, 2)
plt.title("style")
plt.imshow(style_img_r)
plt.show()

#Create the input tensor to model
origin_img = K.constant(preprocess_image(origin_img_r))
style_img = K.constant(preprocess_image(style_img_r))
result_img = K.placeholder((1, img_rows, img_cols, img_channels))
input_tensor = K.concatenate([origin_img, style_img, result_img], axis=0)

#Import VGG19 model to retrain and create a dict with the layers of model
model = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
layers_dict = {layer.name : layer.output for layer in model.layers}

content_layer = 'block5_conv2'
style_layers = []
n_layers = 5
for i in range(n_layers):
    layer_name = "block{:d}_conv1".format(i+1)
    style_layers.append(layer_name)
    
#Define the hyperparameters
variation_weight = 0.01
style_weight = 1.0
content_weight = 0.025
iterations = 10

#apply functions to calculate the model loss
loss = K.variable(0.)
layer_features = layers_dict[content_layer]
target_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_features, combination_features)

for layer_name in style_layers:
    layer_features = layers_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features, img_rows,img_cols, img_channels)
    loss += (style_weight / len(style_layers)) * sl
loss += variation_weight * variation_loss(result_img, img_rows,img_cols)

#Train model
grads = K.gradients(loss, result_img)[0]
fetch_loss_grads = K.function([result_img], [loss, grads])

evaluator = Evaluator()

x = preprocess_image(origin_img_r)
x = x.flatten()
plot_every = False
for i in range(iterations):
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
    if plot_every:
        img = x.copy().reshape((img_rows, img_cols, img_channels))
        img = deprocess_image(img, img_rows, img_cols, img_channels)
        plt.imshow(img)
        plt.show()

img = x.copy().reshape((img_rows, img_cols, img_channels))
img = deprocess_image(img, img_rows, img_cols, img_channels)
imsave(result_image_path, img)
plt.imshow(img)
plt.show()

