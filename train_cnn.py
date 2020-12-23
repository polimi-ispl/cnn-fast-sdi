import os
import numpy as np
import keras
from data_generator import DataGeneratorPRNU as DataGeneratorPRNU_train
from data_generator import DataGeneratorPRNU as DataGeneratorPRNU_valid
from utility_dataset import load_prnu, load_res, preprocessing_function
from architectures import makeNetwork
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import time
seed = 42


def make_loss_metric():
    from keras import backend as K

    def loss_bin(y_true, y_pred):
        loss_bin = K.mean(keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=True, axis=-1))
        return loss_bin

    def binary_metric(y_true, y_pred):
        binary_metric = keras.metrics.categorical_accuracy(y_true, y_pred)
        return binary_metric

    return loss_bin, binary_metric


# ------------------------------------------------------Train ----------------------------------------------------------

def train(train_data_generator, valid_data_generator, model_path, learning_rate=0.001, num_epochs=500, workers=16,
          verbose=True, base_network='Pcn'):
    """
    Trains the model
    :param train_data_generator: generator of training data
    :param val_data_generator: generator of validation data
    :param model_path: directory where to store model and trained weights
    :param size_block: number of output classes
    :param learning_rate: learning rate
    :param num_epochs: number of epochs to train
    :param workers: number of workers
    :param verbose: boolean value, False to suppress verbose output, True for verbose output
    :return:
    """
    model = makeNetwork(input_shape=(None, None, 2), base_network=base_network, num_classes=2,
                        include_activation=False)

    trainable_count = int(
        np.sum([keras.backend.count_params(p) for p in set(model.trainable_weights)]))  # count the trainable weights

    # path output files
    os.makedirs(model_path, exist_ok=True)  # Make model path
    model_file = os.path.join(model_path, 'model.json')
    weights_file = os.path.join(model_path, 'model_weights.h5')
    log_file = os.path.join(model_path, 'model_log.csv')

    with open(log_file, 'a' if os.path.isfile(log_file) else 'w') as fid:
        fid.write('np:%d\n' % trainable_count)

    # Save Model
    with open(model_file, 'w') as json_file:
        json_file.write(model.to_json())

    # Define optimizer
    optimizer = keras.optimizers.adam(lr=learning_rate)

    # Define loss function and metrics
    my_loss, my_metric = make_loss_metric()
    my_metric.__name__ = 'my_metric'
    model.compile(optimizer=optimizer, loss=my_loss, metrics=[my_metric])

    # Define utilities during training
    callbacks_list = [
        # Save the weights only when the accuracy on validation-set improves
        keras.callbacks.ModelCheckpoint(weights_file, monitor='val_my_metric', verbose=verbose,
                                        save_best_only=True, mode='max'),
        # Save a CSV file with info
        keras.callbacks.CSVLogger(log_file, separator=';', append=True),
        # Stop the training if the accuracy on validation-set does not improve for 30 epochs
        keras.callbacks.EarlyStopping(monitor='val_my_metric', mode='max', patience=30,
                                      verbose=verbose)]

    # Model summary
    model.summary()

    # Train the model
    if verbose:
        print("Start Train")
        print('use_multiprocessing:', (workers > 1))
        print("Number of trainable parameters: {:d}".format(trainable_count))

    start_time = time.time()

    history = model.fit_generator(generator=train_data_generator,
                                  epochs=num_epochs,
                                  steps_per_epoch=len(train_data_generator),
                                  use_multiprocessing=(workers > 1),
                                  workers=workers,
                                  validation_data=valid_data_generator,
                                  validation_steps=len(valid_data_generator),
                                  callbacks=callbacks_list,
                                  verbose=verbose)

    elapsed_time_secs = time.time() - start_time

    history_file_path = os.path.join(model_path, 'history.npy')
    time_file_path = os.path.join(model_path, 'training_time.npy')

    np.save(history_file_path, history.history)
    np.save(time_file_path, elapsed_time_secs)


# ------------------------------------------------------Main -----------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # Training/Validation dataset: devices selected from VISION (20 devices) and DRESDEN (16 devices)
    def_dev_list_train = 'D01;D02;D03;D05;D06;D07;D08;D11;D12;D16;D17;D19;D20;D21;D22;D24;D25;D28;D31;D34;Canon_Ixus55_0;' \
                         'Canon_Ixus70_0;Casio_EX-Z150_0;FujiFilm_FinePixJ50_0;Nikon_CoolPixS710_0;Nikon_D200_0;Nikon_D70_0;' \
                         'Nikon_D70s_0;Olympus_mju_1050SW;Panasonic_DMC-FZ50_0;Pentax_OptioA40_0;Pentax_OptioW60_0;Praktica_DCZ5-9_0;' \
                         'Ricoh_GX100_0;Rollei_RCP-7325XS_0;Samsung_NV15_0'
    def_dev_list_valid = 'D01;D02;D03;D05;D06;D07;D08;D11;D12;D16;D17;D19;D20;D21;D22;D24;D25;D28;D31;D34;Canon_Ixus55_0;' \
                         'Canon_Ixus70_0;Casio_EX-Z150_0;FujiFilm_FinePixJ50_0;Nikon_CoolPixS710_0;Nikon_D200_0;Nikon_D70_0;' \
                         'Nikon_D70s_0;Olympus_mju_1050SW;Panasonic_DMC-FZ50_0;Pentax_OptioA40_0;Pentax_OptioW60_0;Praktica_DCZ5-9_0;' \
                         'Ricoh_GX100_0;Rollei_RCP-7325XS_0;Samsung_NV15_0'

    parser.add_argument('--list_dev_train', type=str, default=def_dev_list_train) # list of training devices
    parser.add_argument('--list_dev_valid', type=str, default=def_dev_list_valid) # list of validation devices
    parser.add_argument('--image_size', type=int, default=720) # minimum size of the image
    parser.add_argument('--crop_size', type=int, default=320) # size of the central patch, to be cropped
    parser.add_argument('--random_cropping', type=int, default=40)  # 40 pixel for random cropping during training
    parser.add_argument('--model_dir', type=str, default='./model')  # directory with CNN weights
    parser.add_argument('--gpu', type=str, default='0') # gpu to be used
    parser.add_argument('--num_epochs_train', type=int, default=500) # number of training epochs
    parser.add_argument('--learning_rate', type=float, default=0.001) # learning rate
    parser.add_argument('--batch_size_train', type=int, default=72) # batch size in training
    parser.add_argument('--batch_size_valid', type=int, default=72) # batch size in validation
    parser.add_argument('--size_block', type=int, default=2) # number of classes
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--base_network', type=str, default='Pcn') # CNN architecture to be used

    config, _ = parser.parse_known_args()
    if config.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu  # set the GPU device

    configSess = tf.ConfigProto()
    # Allowing the GPU memory to grow avoid preallocating all the GPU memory
    configSess.gpu_options.allow_growth = True
    set_session(tf.Session(config=configSess))

    list_dev_train = config.list_dev_train.split(';')
    list_dev_valid = config.list_dev_valid.split(';')

    print("Preparing train data loader")
    #
    train_prnu = [load_prnu(item, config.image_size)[:, :, None] for item in list_dev_train]
    train_data = [load_res(np.load('Noises_lists/train/list_%s.npy' % item).tolist(), config.image_size) for item in
                  list_dev_train]

    # Generator of training data
    train_data_generator = DataGeneratorPRNU_train(config.batch_size_valid, train_prnu, train_data,
                                                   preprocessing_function0=preprocessing_function,
                                                   preprocessing_function1=preprocessing_function, random_crop=True,
                                                   size_crop=config.crop_size, horizontal_flip=True, rot_90=True,
                                                   seed=seed)

    train_data_generator.print_info()

    print("Preparing valid data loader")
    valid_prnu = [load_prnu(item, config.image_size)[:, :, None] for item in list_dev_valid]
    valid_data = [load_res(np.load('Noises_lists/valid/list_%s.npy' % item).tolist(), config.image_size) for item in
                  list_dev_valid]

    # Generator of validation data
    valid_data_generator = DataGeneratorPRNU_valid(config.batch_size_valid, valid_prnu, valid_data,
                                                   preprocessing_function0=preprocessing_function,
                                                   preprocessing_function1=preprocessing_function, random_crop=True,
                                                   size_crop=config.crop_size, horizontal_flip=True, rot_90=True,
                                                   seed=seed+1)

    valid_data_generator.print_info()

    # training
    print('Starting training')
    train(train_data_generator=train_data_generator,
          valid_data_generator=valid_data_generator,
          model_path=os.path.abspath(config.model_dir),
          learning_rate=config.learning_rate,
          num_epochs=config.num_epochs_train,
          workers=config.workers,
          base_network=config.base_network)

    # Release memory
    keras.backend.clear_session()