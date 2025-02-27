import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, Activation
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.applications.xception import Xception, preprocess_input
# from keras.applications.resnet import ResNet50, preprocess_input
# # from keras.applications.inception_v3 import InceptionV3, preprocess_input
# from keras.applications.mobilenet import MobileNet, preprocess_input
# from keras.applications.convnext import ConvNeXtTiny, preprocess_input


import time
import scipy.ndimage as nd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
# import matplotlib.pyplot as plt
import random
from scipy.sparse.linalg import svds
from tensorflow.keras.losses import CategoricalCrossentropy
import cv2
import math
from fgm import *
import argparse


def uap_sgd(model, loader, nb_epoch, eps, model_name, beta=12, step_decay=0.8, y_target=None, layer_name=None, uap_init=None, path='/data/ImageNet/'):
    '''
    INPUT
    model       model
    loader      dataloader
    nb_epoch    number of optimization epochs
    eps         maximum perturbation value (L-infinity) norm
    beta        clamping value
    y_target    target class label for Targeted UAP variation
    loss_fn     custom loss function (default is CrossEntropyLoss)
    layer_name  target layer name for layer maximization attack
    uap_init    custom perturbation to start from (default is random vector with pixel values {-eps, eps})

    OUTPUT
    delta.data  adversarial perturbation
    losses      losses per iteration
    '''
    batch = loader.__getitem__(0)
    x_val = batch[0]
    y_val = batch[1]
    batch_size = len(x_val)
    if uap_init is None:
        batch_delta = tf.zeros_like(x_val)  # initialize as zero vector
    else:
        repeat = tf.constant([batch_size, 1, 1, 1], tf.int32)
        batch_delta = tf.tile(tf.expand_dims(uap_init, 0), repeat)
    delta = batch_delta[0]
    losses = []

    # loss function
    if layer_name is None:
        loss_fn = CategoricalCrossentropy()
        beta = tf.convert_to_tensor(beta, dtype=tf.float32)

        def clamped_loss(output, target):
            loss = tf.math.minimum(loss_fn(output, target), beta)
            return loss

    # layer maximization attack
    else:
        def get_norm(self, forward_input, forward_output):
            global main_value
            main_value = tf.norm(forward_output, ord='fro')

    mean_image = np.zeros((224, 224, 3))
    # mean_image = np.zeros((299, 299, 3))
    mean_image[:, :, 0] = 103.939
    mean_image[:, :, 1] = 116.779
    mean_image[:, :, 2] = 123.68

    per_image_file = open(path + model_name +'_sgd_deltas_' + str(args.epsilon) + '.npy', 'wb')
    print(path + model_name +'_sgd_deltas_' + str(args.epsilon) + '.npy')
    nb_epoch = 1
    corrects = 0
    for epoch in range(nb_epoch):
        print('epoch %i/%i' % (epoch + 1, nb_epoch))

        # perturbation step size with decay
        eps_step = eps * step_decay
        for batch_idx in range(loader.__len__()):
            print(batch_idx)
            # if batch_idx >= args.times:
            #     break
            batch = loader.__getitem__(batch_idx)
            x_val = tf.convert_to_tensor(batch[0], dtype=tf.float32)
            y_val = tf.convert_to_tensor(batch[1], dtype=tf.float32)

            repeat = tf.constant([x_val.shape[0], 1, 1, 1], tf.int32)
            # batch_delta = tf.tile(tf.expand_dims(delta, 0), repeat)
            batch_delta = tf.tile(tf.expand_dims(tf.zeros_like(x_val)[0], 0), repeat)
            with tf.GradientTape() as tape:
                tape.watch(batch_delta)

                # for targeted UAP, switch output labels to y_target
                if y_target is not None: y_val = tf.ones(shape=y_val.shape, dtype=tf.float32) * y_target
                X_p = x_val + batch_delta
                # X_p = x_val + tf.clip_by_value(X_p - x_val, -255, 255)
                # X_p = tf.clip_by_value(X_p, -mean_image, 255 - mean_image)
                # X_p = x_val + tf.clip_by_value(X_p - x_val, -1, 1)
                # X_p = tf.clip_by_value(X_p, -1, 1)
                # print(X_p)
                outputs = model(X_p)

                # loss function value
                if layer_name is None:
                    loss = clamped_loss(outputs, y_val)
                else:
                    loss = main_value

                if y_target is not None: loss = -loss  # minimize loss for targeted UAP
                losses.append(tf.reduce_mean(loss))
                grad = tape.gradient(loss, batch_delta)

            # batch update
            # grad_sign = tf.math.sign(tf.reduce_mean(grad, axis=0))
            # delta = delta + grad_sign * eps_step
            # grad = tf.reduce_mean(grad, axis=0)
            # # print(epoch, batch_idx, grad, tf.norm(grad, ord=2))
            # grad = grad / tf.norm(grad, ord=2)
            # delta = delta + grad * eps_step
            # delta = delta / tf.norm(delta, ord=2) * 1000
            # print(delta)
            preds = np.argmax(outputs, axis=1)
            gts = np.argmax(y_val, axis=1)
            for idx in range(len(grad)):
                if preds[idx] != gts[idx]:
                    continue
                print(batch_idx * batch_size + idx)
                corrects += 1
                delta = grad[idx]
                # delta = delta / tf.norm(delta, ord=2)
                # delta = np.clip(delta, -255, 255)
                np.save(per_image_file, delta)
    print(corrects)

    return delta.numpy(), losses


def apply_attack(model, test_generator, model_name):
    nb_epoch = args.epochs
    eps = args.epsilon
    beta = args.beta
    step_decay = args.step_decay
    uap_sgd(model, test_generator, nb_epoch, eps, model_name, beta, step_decay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=int, default=1500)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--beta', type=int, default=12)
    parser.add_argument('--step_decay', type=float, default=0.7)
    parser.add_argument('--times', type=int, default=np.inf)
    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--model', type=str, default='ResNet50')
    args = parser.parse_args()

    model = get_model(model_name=args.model, dataset=args.dataset)
    if args.dataset == "ImageNet":
        train_generator, val_generator, test_generator = load_ImageNet()
    elif args.dataset == "CIFAR10":
        train_generator, val_generator, test_generator = load_CIFAR10()
    apply_attack(model, val_generator, args.model)
