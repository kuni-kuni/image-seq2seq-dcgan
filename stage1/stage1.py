import numpy as np
import math
import os
import shutil
import sys

import keras.backend as K
from keras.preprocessing.image import load_img, list_pictures
from keras.models import Model, load_model
from keras.layers import GRU, Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation, Dropout, Add, AveragePooling2D, dot, concatenate, Reshape, UpSampling2D, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.applications.resnet50 import ResNet50
from keras.utils import plot_model

sys.path.append("../")
from utils import load_img_batch, save_img, BOS

HID_IMAGE_SIZE = 32
TRAIN_PATH = "../data/train/"
TEST_PATH = "../data/test/"

def data_gen(path_dir, batch_size):
    dir_names = sorted(os.listdir(path_dir))
    dir_pathes = [os.path.join(path_dir, d) for d in dir_names]

    n_pair = len(dir_names) - 1
    n_batches = math.ceil(n_pair / batch_size)

    while True:
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            if end > n_pair:
                end = n_pair

            file_pathes_list = [list_pictures(d) for d in dir_pathes[start:end]]
            max_page_len = max([len(ps) for ps in file_pathes_list])
            enc_inputs = [load_img_batch(p, max_page_len) for p in file_pathes_list]
            batch_enc_inputs = np.empty(shape=(len(enc_inputs), max_page_len, 256, 256, 1))
            for i, e in enumerate(enc_inputs):
                batch_enc_inputs[i] = e

            file_pathes_list = [list_pictures(d) for d in dir_pathes[start+1:end+1]]
            max_page_len = max([len(ps) for ps in file_pathes_list])
            
            # BOSあり
            reshaped_bos = BOS.reshape(1, 256, 256, 1)
            dec_inputs = [np.vstack((reshaped_bos, load_img_batch(p, max_page_len))) for p in file_pathes_list]
            
            batch_dec_inputs = np.empty(shape=(len(dec_inputs), max_page_len, 256, 256, 1))
            for i, d in enumerate(dec_inputs):
                batch_dec_inputs[i] = d[:-1]

            batch_dec_outputs = np.empty(shape=(len(dec_inputs), max_page_len, 256, 256, 1))
            for i, d in enumerate(dec_inputs):
                batch_dec_outputs[i] = d[1:]

            yield [batch_enc_inputs, batch_dec_inputs], batch_dec_outputs

def image_embedding_model():
    x_in = Input(shape=(256, 256, 1))
    
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    base_model.trainable = False
    for layer in base_model.layers:
        layer.trainable = False

    grayscale_triplet = Lambda(lambda x: K.repeat_elements(x, 3, axis=-1))
    x = grayscale_triplet(x_in)
    x = base_model(x)
    x = Flatten()(x) # 2048
    x = BatchNormalization()(x)

    return Model(x_in, x)

def upsample_model():
    x_in = Input(shape=(HID_IMAGE_SIZE, HID_IMAGE_SIZE, 1))
    
    #x = conv(x_in, 2)
    #x = conv(x, 4)
    #x = conv(x, 8)
    #x = conv(x, 16)
    
    x = upsample(x_in, 32) #64
    x = upsample(x, 32) #128
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(1, kernel_size=(3, 3), padding="same")(x)
    x = Activation("sigmoid")(x)
    
    return Model(x_in, x)
    
def upsample(x, filters):
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Activation("relu")(x)
    return x

def conv(x, filters):
    x = Conv2D(filters, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Activation("relu")(x)
    return x

def train(epochs=10):
    # GRUの隠れ層を大きくするとメモリが足りない
    hid_dim = HID_IMAGE_SIZE * HID_IMAGE_SIZE
    
    image_embedding = image_embedding_model()

    # Encoder
    encoder_input = Input(shape=(None, 256, 256, 1))
    encoder_embedded = TimeDistributed(image_embedding)(encoder_input)
    encoded_seq, encoder_state = GRU(hid_dim, return_sequences=True, return_state=True, recurrent_activation='tanh', recurrent_dropout=0.2, dropout=0.2)(encoder_embedded)

    # Decoder
    decoder_input = Input(shape=(None, 256, 256, 1))
    decoder_embedded = TimeDistributed(image_embedding)(decoder_input)
    decoder_gru = GRU(hid_dim, return_sequences=True, return_state=True, recurrent_activation='tanh', recurrent_dropout=0.2, dropout=0.2)
    decoded_seq, _ = decoder_gru(decoder_embedded, initial_state=encoder_state)

    # Attention
    score_dense = Dense(hid_dim)
    score = score_dense(decoded_seq)
    score = dot([score, encoded_seq], axes=(2,2))
    attention = Activation('softmax')(score)
    context = dot([attention, encoded_seq], axes=(2,1))
    concat = concatenate([context, decoded_seq], axis=2)
    attention_dense = Dense(hid_dim, activation='tanh')
    attentional = attention_dense(concat)
    
    upsample = upsample_model()
    latent_image = TimeDistributed(Reshape((HID_IMAGE_SIZE, HID_IMAGE_SIZE, 1)))(attentional)
    latent_image = TimeDistributed(upsample)(latent_image)

    model = Model(inputs=[encoder_input, decoder_input], outputs=latent_image)
    model.compile(loss="mae", optimizer="adam")
    
    train_data_gen = data_gen(TRAIN_PATH, 2)
    test_data_gen = data_gen(TEST_PATH, 1)
    history = model.fit_generator(train_data_gen, steps_per_epoch=59, epochs=epochs, validation_data=test_data_gen, validation_steps=3)
    model.save("model.h5")
    
    # モデル保存
    encoder_model = Model(encoder_input, [encoded_seq, encoder_state])
    encoder_model.save("encoder_model.h5")

    decoder_state_input = Input(shape=(hid_dim,))
    decoded_seq, decoder_state = decoder_gru(decoder_embedded, initial_state=decoder_state_input)
    
    encoded_seq_input = Input(shape=(None, hid_dim))
    score = score_dense(decoded_seq)
    score = dot([score, encoded_seq_input], axes=(2,2))
    attention = Activation('softmax')(score)
    context = dot([attention, encoded_seq_input], axes=(2,1))
    concat = concatenate([context, decoded_seq], axis=2)
    attentional = attention_dense(concat)
    
    latent_image = TimeDistributed(Reshape((HID_IMAGE_SIZE, HID_IMAGE_SIZE, 1)))(attentional)
    latent_image = TimeDistributed(upsample)(latent_image)
    
    decoder_model = Model([decoder_input, encoded_seq_input, decoder_state_input], [decoder_state, latent_image])
    decoder_model.save("decoder_model.h5")
    
    import pickle
    with open('train_history.pickle', 'wb') as f:
        pickle.dump(history.history, f)

def create_latent_images():
    model = load_model("model.h5")
    data_dir = TRAIN_PATH
    shutil.rmtree("result/")
    dir_names = sorted(os.listdir(data_dir))
    generator = data_gen(data_dir, 1)
    for i in range(len(dir_names) - 1):
        (encoder_input, decoder_input), _ = next(generator)
        latent_image = model.predict([encoder_input, decoder_input])
        target_dir = os.path.join("result/", dir_names[i+1])
        os.makedirs(target_dir)
        for j in range(latent_image.shape[1]):
            img = latent_image[0][j]
            save_path = os.path.join(target_dir, "{0:05d}.jpg".format(j + 1))
            save_img(img, save_path)

if __name__ == "__main__":
    train(epochs=10)
    #create_latent_images()