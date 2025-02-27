import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, Activation
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.applications.xception import Xception, preprocess_input
# from keras.applications.resnet import ResNet50, preprocess_input
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
# from keras.applications.convnext import ConvNeXtTiny, preprocess_input
from keras.applications.convnext import LayerScale, StochasticDepth
# from vit_tensorflow import ViT
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from keras import backend as K
from tensorflow.keras.utils import img_to_array, array_to_img

import time
import scipy.ndimage as nd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import glob as glob
import os
# from livelossplot.inputs.keras import PlotLossesCallback
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import pandas as pd
from keras_contrib.callbacks import CyclicLR
# import matplotlib.pyplot as plt
from tf_explain.callbacks.integrated_gradients import IntegratedGradients
import random
from scipy.sparse.linalg import svds
import argparse
import tensorflow as tf
import tensorflow_hub as hub


def load_ImageNet(BATCH_SIZE=64):
    path = '/data/ImageNet/val'
    target_size = (224, 224)
    # target_size = (299, 299)

    train_datagen = ImageDataGenerator(rotation_range=30,
                                       brightness_range=[0.3, 0.7],
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       horizontal_flip=True,
                                       preprocessing_function=preprocess_input)

    valid_datagen = ImageDataGenerator(validation_split=0.5,
                                       preprocessing_function=preprocess_input)
    # train_generator = train_datagen.flow_from_directory(path + 'train/', batch_size=BATCH_SIZE, color_mode='rgb',
    #                                                     class_mode='categorical', target_size=(224, 224),
    #                                                     shuffle=True, seed=101)

    valid_generator = valid_datagen.flow_from_directory(path, batch_size=BATCH_SIZE, color_mode='rgb',
                                                        class_mode='categorical', target_size=target_size, shuffle=True,
                                                        subset='training')
    test_generator = valid_datagen.flow_from_directory(path, batch_size=BATCH_SIZE, color_mode='rgb',
                                                       class_mode='categorical', target_size=target_size, shuffle=False,
                                                       subset='validation')
    return None, valid_generator, test_generator


def normalize(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    print(mean, std)
    return X_train, X_test


def load_CIFAR10(BATCH_SIZE=64):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((32, 32))) for im in x_train])
    x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((32, 32))) for im in x_test])

    x_train, x_test = normalize(x_train, x_test)

    y_train = keras.utils.np_utils.to_categorical(y_train, 10)
    y_test = keras.utils.np_utils.to_categorical(y_test, 10)

    # data augmentation
    train_datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    # (std, mean, and principal components if ZCA whitening is applied).
    train_datagen.fit(x_train)

    test_datagen = ImageDataGenerator()
    test_datagen.fit(x_test[:5000])

    valid_datagen = ImageDataGenerator()
    valid_datagen.fit(x_test[5000:])

    train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=101)
    valid_generator = valid_datagen.flow(x_test[:5000], y_test[:5000], batch_size=BATCH_SIZE, shuffle=True, seed=101)
    test_generator = test_datagen.flow(x_test[5000:], y_test[5000:], batch_size=BATCH_SIZE, shuffle=False)
    return train_generator, valid_generator, test_generator


def get_model(model_name="VGG16", dataset="ImageNet"):
    if model_name == "VGG16":
        # model = keras.models.load_model("../vgg_16.h5")
        if dataset == "ImageNet":
            input_shape = (224, 224, 3)
        elif dataset == "CIFAR10":
            input_shape = (32, 32, 3)
        model = VGG16(include_top=dataset == "ImageNet", input_shape=input_shape, weights="imagenet")
        opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
        model.trainable = False
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    elif model_name == "Xception":
        if dataset == "ImageNet":
            input_shape = (299, 299, 3)
        elif dataset == "CIFAR10":
            input_shape = (75, 75, 3)
        model = Xception(include_top=dataset == "ImageNet", input_shape=input_shape, weights="imagenet")
    elif model_name == "ResNet50":
        if dataset == "ImageNet":
            input_shape = (224, 224, 3)
        elif dataset == "CIFAR10":
            input_shape = (75, 75, 3)
            input_shape = (32, 32, 3)
        model = ResNet50(include_top=dataset == "ImageNet", input_shape=input_shape, weights="imagenet")
    elif model_name == "InceptionV3":
        if dataset == "ImageNet":
            input_shape = (299, 299, 3)
        elif dataset == "CIFAR10":
            input_shape = (75, 75, 3)
        model = InceptionV3(include_top=dataset == "ImageNet", input_shape=input_shape, weights="imagenet")
    elif model_name == "MobileNet":
        if dataset == "ImageNet":
            input_shape = (224, 224, 3)
        elif dataset == "CIFAR10":
            input_shape = (32, 32, 3)
        model = MobileNet(include_top=dataset == "ImageNet", input_shape=input_shape, weights='imagenet')
    elif model_name == "ConvNeXt":
        if dataset == "ImageNet":
            input_shape = (224, 224, 3)
        elif dataset == "CIFAR10":
            input_shape = (32, 32, 3)
        model = ConvNeXtTiny(include_top=dataset == "ImageNet", input_shape=input_shape, weights='imagenet')
    elif model_name == "ViT":
        if dataset == "ImageNet":
            input_shape = (224, 224, 3)
        elif dataset == "CIFAR10":
            input_shape = (32, 32, 3)
        model = tf.keras.Sequential([
                hub.KerasLayer(
                "https://www.kaggle.com/models/spsayakpaul/vision-transformer/TensorFlow2/vit-b16-classification/1")
        ])
        opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
        model.trainable = False
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    elif model_name == "Swin":
        if dataset == "ImageNet":
            input_shape = (224, 224, 3)
        model = tf.keras.Sequential([
            hub.KerasLayer(
                "https://www.kaggle.com/models/spsayakpaul/swin/tensorFlow2/small-patch4-window7-224/1")
        ])
        opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
        model.trainable = False
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    if dataset == "CIFAR10":
        # prev_model = model
        # print(model.summary())
        # model = Sequential()
        #
        # for layer in prev_model.layers:
        #     print(layer)
        #     model.add(layer)
        #
        # model.add(Flatten(input_shape=(2, 2, 512)))
        #
        # model.add(Dense(512, activation='relu', name='FC1'))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.4))
        #
        # model.add(Dense(512, activation='relu', name='FC2'))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.4))
        #
        # # model.add(Dense(200, activation='softmax', name='FC3'))
        # model.add(Dense(10, name='FC3'))
        # model.add(Activation('softmax'))

        # input_layer = Input(shape=(75, 75, 3))
        # input_layer = Input(shape=(32, 32, 3))
        # model_base = model(input_layer, training=False)
        # #
        # x = tf.keras.layers.GlobalAveragePooling2D()(model_base)
        # dp2 = tf.keras.layers.Dropout(0.3)(x)
        # flatten = Flatten(input_shape=(2, 2, 512))(dp2)
        # dense1 = Dense(512, activation='relu')(flatten)
        # bn1 = BatchNormalization()(dense1)
        # dp1 = Dropout(0.4)(bn1)
        # dense2 = Dense(512, activation='relu')(dp1)
        # bn2 = BatchNormalization()(dense2)
        # dp2 = Dropout(0.4)(bn2)
        # # dense4 = Dense(200, activation='relu')(dp2)
        # # bn4 = BatchNormalization()(dense4)
        # # dp4 = Dropout(0.4)(bn4)
        # dense3 = Dense(10, activation='softmax')(dp2)
        # model = Model(inputs=input_layer, outputs=dense3)
        # i = 0
        # for layer in model.layers:
        #     if i == 1:
        #         layer.trainable = False
        #     i += 1
        # print(model.summary())

        # def classifier_module(inputs):
        #     '''
        #     Defines the final dense layers followed by the softmax layer for classification
        #     '''
        #     x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        #     x = tf.keras.layers.Dense(1024, activation="relu")(x)
        #     x = tf.keras.layers.Dense(512, activation="relu")(x)
        #     x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
        #     return x
        #
        # inputs = tf.keras.Input(shape=(32, 32, 3))
        # resize = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)
        #
        # features = ConvNeXtTiny(
        #     input_shape=(224, 224, 3),
        #     include_top=False,
        #     weights='imagenet'
        # )(resize)
        #
        # classification_output = classifier_module(features)
        #
        # model = tf.keras.Model(inputs=inputs, outputs=classification_output)
        #
        # # model.load_weights('resnet.h5')
        # model.compile(
        #     optimizer=tf.keras.optimizers.SGD(),
        #     loss='categorical_crossentropy',
        #     metrics=['accuracy']
        # )
        #
        # model = keras.models.load_model(model_name + '_CIFAR10_TEST.h5')
        # print(model.summary())
        # i = 0
        # for layer in model.layers:
        #     if i == 1:
        #         layer.trainable = False
        #     i += 1
        # print(model.summary())

        model = keras.models.load_model(model_name + '_CIFAR10_TEST.h5', compile=False, custom_objects=
        {"LayerScale": LayerScale, "StochasticDepth": StochasticDepth})

    return model


def train_CIFAR10(model_name, model, train_generator, valid_generator, BATCH_SIZE=64):
    model.trainable = True
    # sgd = optimizers.SGD(lr=0.0000001, momentum=0.9)
    n_steps = 50000 // BATCH_SIZE
    n_val_steps = 5000 // BATCH_SIZE
    # plot_loss = PlotLossesCallback()
    # opt = tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.0001)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(model_name + '_CIFAR10_TEST2.h5', monitor='val_accuracy', verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False, mode='auto')
    # cyclic = CyclicLR(base_lr=0.001, max_lr=0.001, step_size=1200., mode='triangular2')
    cyclic = CyclicLR(base_lr=0.0005, max_lr=0.006, step_size=1200., mode='triangular2')

    early = EarlyStopping(monitor='val_accuracy', patience=6, verbose=1, mode='auto')
    model.fit(train_generator, batch_size=BATCH_SIZE, steps_per_epoch=n_steps, validation_data=valid_generator,
              validation_steps=n_val_steps, epochs=100, callbacks=[checkpoint, early, cyclic], shuffle=True)
    # model.fit(
    #     train_generator,
    #     batch_size=BATCH_SIZE,
    #     epochs=100,
    #     validation_data=valid_generator,
    #     callbacks=[checkpoint, early]
    # )
    # model.save(model_name + '_CIFAR_10_final.h5')
    # model = keras.models.load_model(model_name + '_CIFAR10.h5')
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
        elif self.dataset == "CIFAR10":
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


def attack(model, test_generator, epsilon, dataset="ImageNet", path='/data/ImageNet/ResNet50_'):
    p_range = 1
    alpha = epsilon
    if args.attack_type == 'integrated':
        M = 10
        explainer = ModifiedIntegratedGradients()
        explainer.set_num_steps(M)
    elif args.attack_type == 'simple':
        explainer = SimpleGradients()

    save_path = path + args.attack_type + '_'
    deltas_file = open(save_path + 'deltas_' + str(alpha) + '_without_power.npy', 'wb')

    if dataset == "ImageNet" or dataset == "Tiny ImageNet":
        mean, std = 0, 1
        mean_image = np.zeros((224, 224, 3))
        # mean_image = np.zeros((299, 299, 3))

        mean_image[:, :, 0] = 103.939
        mean_image[:, :, 1] = 116.779
        mean_image[:, :, 2] = 123.68
    if dataset == "CIFAR10":
        mean, std = 120.70748, 64.150024
    num = 0

    for batch_idx in range(test_generator.__len__()):
        batch = test_generator.__getitem__(batch_idx)
        Xs = batch[0]
        ys = batch[1]
        batch_predict = model.predict(Xs)

        for idx in range(len(Xs)):
            X, y = Xs[idx], ys[idx]
            print(X)
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


def main(args):
    model = get_model(model_name=args.model, dataset=args.dataset)
    if args.dataset == 'ImageNet':
        train_generator, val_generator, test_generator = load_ImageNet(BATCH_SIZE=args.batch_size)
    elif args.dataset == 'CIFAR10':
        train_generator, val_generator, test_generator = load_CIFAR10(BATCH_SIZE=args.batch_size)
        ######## train
        # model = train_CIFAR10(args.model, model, train_generator, val_generator)

    ######## eval
    (eval_loss, eval_accuracy) = model.evaluate(val_generator, batch_size=args.batch_size, verbose=1)
    print("eval_loss:", eval_loss, "eval_accuracy:", eval_accuracy)

    ######## attack
    # path = '/data/' + args.dataset + '/' + args.model + '_'
    # attack(model, val_generator, args.epsilon, path=path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=int, default=1500)
    parser.add_argument('--attack_type', type=str, default='simple')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--model', type=str, default='Swin')
    args = parser.parse_args()

    main(args)
