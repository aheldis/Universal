import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, Activation
from keras.applications.vgg16 import VGG16, preprocess_input
import time
import scipy.ndimage as nd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import glob as glob
import os
from livelossplot.inputs.keras import PlotLossesCallback
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from keras_contrib.callbacks import CyclicLR
import matplotlib.pyplot as plt
from tf_explain.callbacks.integrated_gradients import IntegratedGradients
import random
from scipy.sparse.linalg import svds
import argparse


def load_ImageNet(BATCH_SIZE=64):
    path = '../IamgeNet'

    # train_datagen = ImageDataGenerator(rotation_range=30,
    #                                    brightness_range=[0.3, 0.7],
    #                                    width_shift_range=0.2,
    #                                    height_shift_range=0.2,
    #                                    horizontal_flip=True,
    #                                    preprocessing_function=preprocess_input)

    valid_datagen = ImageDataGenerator(validation_split=0.5,
                                       preprocessing_function=preprocess_input)
    # train_generator = train_datagen.flow_from_directory(path + 'train/', batch_size=BATCH_SIZE, color_mode='rgb',
    #                                                     class_mode='categorical', target_size=(224, 224),
    #                                                     shuffle=True, seed=101)

    valid_generator = valid_datagen.flow_from_directory(path + 'val/', batch_size=BATCH_SIZE, color_mode='rgb',
                                                        class_mode='categorical', target_size=(224, 224), shuffle=True,
                                                        subset='training')
    test_generator = valid_datagen.flow_from_directory(path + 'val/', batch_size=BATCH_SIZE, color_mode='rgb',
                                                       class_mode='categorical', target_size=(224, 224), shuffle=False,
                                                       subset='validation')
    return valid_generator, test_generator


def get_model(model_name="VGG16", dataset="ImageNet"):
    if dataset == "ImageNet" or dataset == "Tiny ImageNet":
        input_shape = (224, 224, 3)
    if model_name == "VGG16":
        model = keras.models.load_model("../vgg_16.h5")
    elif model_name == "MobileNet":
        model = MobileNet(include_top=True, input_shape=input_shape, weights='imagenet')
    return model


class ModifiedIntegratedGradients(IntegratedGradients):

    def __init__(self, dataset="ImageNet"):
        self.dataset = dataset

    def generate_interpolations(self, images, n_steps):
        mean_image = np.zeros(images[0].shape)
        if self.dataset == "ImageNet" or self.dataset == "Tiny ImageNet":
            mean_image[:, :, 0] = -103.939
            mean_image[:, :, 1] = -116.779
            mean_image[:, :, 2] = -123.68
        elif self.dataset == "CIFAR-10":
            mean, std = 120.70748, 64.150024
            mean_image = (mean_image - mean) / std
        baseline = tf.convert_to_tensor(mean_image, dtype=tf.float32)

        return tf.concat([IntegratedGradients.generate_linear_path(baseline, image, n_steps) for image in images], 0)

    def set_num_steps(self, n_steps):
        self.n_steps = n_steps

    def get_integrated_gradients(interpolated_images, model, class_index, n_steps):
        with tf.GradientTape() as tape:
            inputs = tf.cast(interpolated_images, tf.float32)
            tape.watch(inputs)
            num_layers = len(model.layers)
            inter_output_model = keras.Model(model.input, model.get_layer(index=num_layers - 2).output)
            score = inter_output_model(inputs)
            score_l = score[:, class_index]

            grads = tape.gradient(score_l, inputs)
        grads_per_image = tf.reshape(grads, (-1, n_steps, *grads.shape[1:]))
        integrated_gradients = tf.reduce_mean(grads_per_image, axis=1)
        return integrated_gradients

    def explain(self, images, model, class_index):
        interpolated_images = self.generate_interpolations(images, self.n_steps)
        integrated_gradients = IntegratedGradients.get_integrated_gradients(interpolated_images, model, class_index,
                                                                            self.n_steps)
        grads = tf.reduce_sum(tf.abs(integrated_gradients), axis=-1)
        return grads / tf.norm(tf.reshape(grads, [-1]), ord=1)


class SimpleGradients(IntegratedGradients):

    def get_gradients(images, model, class_index):
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            predictions = model(inputs)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, inputs)
        return grads

    def explain(self, images, model, class_index):
        gradients = SimpleGradients.get_gradients(images, model, class_index)
        gradients = tf.reduce_sum(tf.abs(gradients), axis=-1)
        normalized_gradients = gradients / tf.norm(tf.reshape(gradients, [-1]), ord=1)
        return normalized_gradients


def attack(model, test_generator, epsilon, dataset="ImageNet"):
    p_range = 1
    alpha = epsilon
    if args.attack_type == 'integrated':
        M = 10
        explainer = ModifiedIntegratedGradients()
        explainer.set_num_steps(M)
    elif args.attack_type == 'simple':
        explainer = SimpleGradients()

    save_path = '../perturbations/VGG16_ImageNet_NEW/' + args.attack_type + '/'
    deltas_file = open(save_path + 'deltas_' + str(alpha) + '_without_power.npy', 'wb')

    if dataset == "ImageNet" or dataset == "Tiny ImageNet":
        mean, std = 0, 1
        mean_image = np.zeros((224, 224, 3))
        mean_image[:, :, 0] = 103.939
        mean_image[:, :, 1] = 116.779
        mean_image[:, :, 2] = 123.68
    if dataset == "CIFAR-10":
        mean, std = 120.70748, 64.150024
    num = 0

    for batch_idx in range(test_generator.__len__()):
        batch = test_generator.__getitem__(batch_idx)
        Xs = batch[0]
        ys = batch[1]
        batch_predict = model.predict(Xs)

        for idx in range(len(Xs)):
            X, y = Xs[idx], ys[idx]
            X_p = X.copy()
            max_d, max_xp, max_p = -np.inf, X_p, 0
            gt_class = np.where(y == 1)[0][0]
            p_class = np.argmax(batch_predict[idx])
            if gt_class != p_class:
                continue
            print('idx:', num)
            X_tensor = tf.convert_to_tensor([X], dtype=tf.float32)
            I = explainer.explain(X_tensor, model, gt_class)

            for p in range(p_range):
                noise = np.random.normal(0, 1, X_p.shape)
                X_p += noise / np.linalg.norm(noise) * 0.1 / std
                X_p_tensor = tf.convert_to_tensor([X_p], dtype=tf.float32)
                with tf.GradientTape() as tape:
                    tape.watch(X_p_tensor)
                    I_p = explainer.explain(X_p_tensor, model, gt_class)
                    # dissimilarity = tf.math.square(tf.norm(I[0] - I_p[0], ord=2))
                    dissimilarity = tf.norm(I[0] - I_p[0], ord=2)
                    grad = tape.gradient(dissimilarity, X_p_tensor)[0].numpy()

                if np.linalg.norm(grad) == 0:
                    print("grad is zero")
                    noise = np.random.normal(0, 1, X_p.shape)
                    X_p += noise / np.linalg.norm(noise) * 0.1 / std
                    p -= 1
                    continue
                X_p += grad
                # X_p = X_p + alpha * grad / (np.linalg.norm(grad) + 10 ** (-30))
                # X_p = X + np.clip(X_p - X, -255, 255)
                # X_p = np.clip(X_p, -mean_image, 255 - mean_image)

                if dissimilarity > max_d:
                    max_d = dissimilarity
                    max_xp = X_p.copy()
                    delta = max_xp - X
                    if np.linalg.norm(delta) == 0:
                        noise = np.random.normal(0, 1, X_p.shape)
                        X_p += noise / np.linalg.norm(noise) * 0.1 / std
                        p -= 1
                        continue

            np.save(deltas_file, delta)
            num += 1


def main():
    model = get_model(model_name="VGG16")
    val_generator, test_generator = load_ImageNet()

    ######## eval
    # (eval_loss, eval_accuracy) = model.evaluate(test_generator, batch_size=64, verbose=1)
    # print("eval_loss:", eval_loss, "eval_accuracy:", eval_accuracy)

    ######## attack
    attack(model, val_generator, args.epsilon)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=int)
    parser.add_argument('--attack_type', type=str, default='simple')
    args = parser.parse_args()

    main()
