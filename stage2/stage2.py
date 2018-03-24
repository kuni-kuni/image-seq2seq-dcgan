import glob
import math
import numpy as np
import os
import sys
import time

from keras.preprocessing.image import load_img, img_to_array, array_to_img, list_pictures
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Conv2D, BatchNormalization, Dropout, Concatenate, Activation, UpSampling2D, Dense, Flatten, Reshape, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import keras.backend as K

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils import load_scaled_img, save_img

def data_gen(input_path_root, target_path_root, batch_size):
    input_dir_names = os.listdir(input_path_root)
    target_dir_names = os.listdir(target_path_root)
    # 共通のディレクトリのみを対象とする
    dir_names = set(input_dir_names) & set(target_dir_names)

    path_pairs = []
    for dir_name in dir_names:
        input_dir = os.path.join(input_path_root, dir_name)
        target_dir = os.path.join(target_path_root, dir_name)

        i_pathes = sorted(list_pictures(input_dir))
        t_pathes = sorted(list_pictures(target_dir))

        path_pairs += zip(i_pathes, t_pathes)

    n_batch = math.ceil(len(path_pairs) / batch_size)
    
    while True:
        for i in range(n_batch):
            batch_path_pairs = path_pairs[i * batch_size:(i + 1) * batch_size]
            batch_size = len(batch_path_pairs)

            batch_input_img = np.empty(shape=(batch_size, 256, 256, 1))
            batch_target_img = np.empty(shape=(batch_size, 256, 256, 1))
            for i, path_pair in enumerate(batch_path_pairs):
                x = load_scaled_img(path_pair[0])
                x -= np.mean(x, keepdims=True)
                x /= (np.std(x, keepdims=True) + K.epsilon())
                batch_input_img[i] = x
                
                batch_target_img[i] = load_scaled_img(path_pair[1])

            yield batch_input_img, batch_target_img

def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def c(x, k, batch_norm=True, strides=2):
    x = Conv2D(k, kernel_size=(3, 3), strides=strides, padding="same")(x)
    
    if batch_norm:
        x = BatchNormalization()(x)

    x = LeakyReLU(0.2)(x)

    return x

def cd(x, e, k, unet=True):
    # UpSamplingは単に引き伸ばすだけなのでConvと組み合わせる
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(k, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Activation("relu")(x)
    if unet:
        x = Concatenate()([x, e])

    return x

def cd_last(x):
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(32, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Activation("relu")(x)
    x = Conv2D(16, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Activation("relu")(x)
    x = Conv2D(8, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Activation("relu")(x)
    x = Conv2D(4, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Activation("relu")(x)
    x = Conv2D(2, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Activation("relu")(x)
    x = Conv2D(1, kernel_size=(3, 3), padding="same")(x)
    x = Activation("sigmoid")(x)

    return x

def generator_model():
    g_in = Input(shape=(256, 256, 1))
    e1 = c(g_in, 64, batch_norm=False)
    e2 = c(e1, 128)
    e3 = c(e2, 256)
    e4 = c(e3, 512)
    e5 = c(e4, 512)
    e6 = c(e5, 512)
    e7 = c(e6, 512)
    e8 = c(e7, 512)
    d1 = cd(e8, e7, 512)
    d2 = cd(d1, e6, 512)
    d3 = cd(d2, e5, 512)
    d4 = cd(d3, e4, 512)
    d5 = cd(d4, e3, 256, unet=False)
    d6 = cd(d5, e2, 128, unet=False)
    d7 = cd(d6, e1, 64, unet=False)
    g_out = cd_last(d7)

    return Model(g_in, g_out)

def discriminator_model():
    d_in_x = Input(shape=(256, 256, 1))
    d_in_t = Input(shape=(256, 256, 1))

    x = Concatenate()([d_in_x, d_in_t])
    x = c(x, 64, batch_norm=False)
    x = c(x, 128, batch_norm=True)
    x = c(x, 256, batch_norm=True)
    x = c(x, 512, batch_norm=True)
    x = c(x, 512, batch_norm=True)
    x = c(x, 512, batch_norm=True)

    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)

    return Model([d_in_x, d_in_t], x)

def dcgan_model(generator, discriminator):
    g_in = Input(shape=(256, 256, 1))
    g_out = generator(g_in)
    d_out = discriminator([g_in, g_out])
    
    return Model(g_in, [g_out, d_out])

def train(input_path_root, target_path_root, iters, batch_size=1):
    data_g = data_gen(input_path_root, target_path_root, batch_size)

    discriminator = discriminator_model()
    discriminator.name = "discriminator"
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4, beta_1=0.5), metrics=['accuracy'])
    
    set_trainable(discriminator, False)
    
    generator = generator_model()
    generator.name = "generator"
    dcgan = dcgan_model(generator, discriminator)
    dcgan.compile(loss={'generator': 'mae', 'discriminator': 'binary_crossentropy'}, loss_weights={'generator': 1, 'discriminator': 1}, optimizer=Adam(lr=1e-3, beta_1=0.9), metrics=['accuracy'])
    dcgan.summary()
    print(dcgan.metrics_names)

    log_d = []
    log_gan = []
    start_time = time.time()
    for i in range(iters):
        print('iters {}'.format(i))
        batch_input_img, batch_target_img = next(data_g)
        batch_size = batch_input_img.shape[0]
        
        img = generator.predict(batch_input_img)[0]
        #print("gen:", img)
        if i % 10 == 0:
            batch_generated_img = generator.predict(batch_input_img)

            d_in_x = np.append(batch_input_img, batch_input_img, axis=0)
            d_in_t = np.append(batch_target_img, batch_generated_img, axis=0)
            batch_y = np.array([1] * batch_size + [0] * batch_size)
            loss_d, acc_d = discriminator.train_on_batch([d_in_x, d_in_t], batch_y)

            d1 = discriminator.predict([batch_input_img, batch_target_img])
            d2 = discriminator.predict([batch_input_img, batch_generated_img])
            print("d1:", d1)
            print("d2:", d2)

            log_d.append("{},{},{:.7f},{:.7f}\n".format(time.time() - start_time, i, loss_d, acc_d))
            print('loss_d:{:.7f} acc_d:{:7f}'.format(loss_d, acc_d))

        # Generatorが正しい画像を出力するよう学習
        batch_y = np.array([1] * batch_size)
        loss_gan, loss_gan_g, loss_gan_d, acc_gan_g, acc_gan_d = dcgan.train_on_batch(batch_input_img, [batch_target_img, batch_y])
        
        log_gan.append("{},{},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f}\n".format(time.time() - start_time, i, loss_gan, loss_gan_g, loss_gan_d, acc_gan_g, acc_gan_d))
        print('loss_gan:{:7f} loss_gan_g:{:7f} loss_gan_d:{:7f} acc_gan_g:{:7f} acc_gan_d:{:7f} '.format(loss_gan, loss_gan_g, loss_gan_d, acc_gan_g, acc_gan_d))
        
        # 途中経過確認
        if i > 0 and i % 100 == 0:
            img = generator.predict(batch_input_img)[0]
            save_img(img, "result/testgen{0:05d}.jpg".format(i))
                
        # 途中経過保存
        if i > 0 and i % 1000 == 0:
            generator.save("generator.model")
            discriminator.save("discriminator.model")
            dcgan.save("dcgan.model")
            
            with open("log_d.txt", "w") as f:
                f.write("time,iter,loss_d,acc_d\n")
                f.writelines(log_d)
            with open("log_gan.txt", "w") as f:
                f.write("time,iter,loss_gan,loss_gan_g,loss_gan_d,acc_gan_g,acc_gan_d\n")
                f.writelines(log_gan)
        
    generator.save("generator.model")
    discriminator.save("discriminator.model")
    dcgan.save("dcgan.model")
    
    with open("log_d.txt", "w") as f:
        f.write("time,iter,loss_d,acc_d\n")
        f.writelines(log_d)
    with open("log_gan.txt", "w") as f:
        f.write("time,iter,loss_gan,loss_gan_g,loss_gan_d,acc_gan_g,acc_gan_d\n")
        f.writelines(log_gan)

if __name__ == "__main__":
    input_path_root = os.path.join(os.path.dirname(__file__), "../data/stage2/train/")
    target_path_root = os.path.join(os.path.dirname(__file__), "../data/train/")
    train(input_path_root, target_path_root, 100000)