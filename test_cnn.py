import os
import numpy as np
import keras
from utility_dataset import load_prnu, load_res, preprocessing_function
from tqdm import tqdm
from architectures import makeNetwork as make_network
import tensorflow as tf
import time
from keras.backend.tensorflow_backend import set_session


# --------------------------------------------------- Main -------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # define the list of test devices
    def_dev_list = 'D01;D02;D03;D05;D06;D07;D08;D09;D10;D11;D12;D14;D15;D16;D17;D18;D19;D20;D21;D22;D24;D25;D26;D27;' \
                   'D28;D29;D30;D31;D32;D33;D34;D35;Canon_Ixus55_0;Canon_Ixus70_0;Canon_Ixus70_1;Canon_Ixus70_2;' \
                   'Casio_EX-Z150_0;Casio_EX-Z150_1;Casio_EX-Z150_2;Casio_EX-Z150_3;Casio_EX-Z150_4;' \
                   'FujiFilm_FinePixJ50_0;FujiFilm_FinePixJ50_1;FujiFilm_FinePixJ50_2;Nikon_CoolPixS710_0;' \
                   'Nikon_CoolPixS710_1;Nikon_CoolPixS710_2;Nikon_CoolPixS710_3;Nikon_CoolPixS710_4;Nikon_D200_0;' \
                   'Nikon_D200_1;Nikon_D70_0;Nikon_D70_1;Nikon_D70s_0;Nikon_D70s_1;Olympus_mju_1050SW;' \
                   'Panasonic_DMC-FZ50_0;Panasonic_DMC-FZ50_1;Panasonic_DMC-FZ50_2;Pentax_OptioA40_0;' \
                   'Pentax_OptioA40_1;Pentax_OptioA40_2;Pentax_OptioA40_3;Pentax_OptioW60_0;Praktica_DCZ5-9_0;' \
                   'Praktica_DCZ5-9_1;Praktica_DCZ5-9_2;Praktica_DCZ5-9_3;Praktica_DCZ5-9_4;Ricoh_GX100_0;' \
                   'Ricoh_GX100_1;Ricoh_GX100_2;Ricoh_GX100_3;Ricoh_GX100_4;Rollei_RCP-7325XS_0;Rollei_RCP-7325XS_1;' \
                   'Rollei_RCP-7325XS_2;Samsung_NV15_0;Samsung_NV15_1;Samsung_NV15_2;Sony_DSC-H50_0;Sony_DSC-H50_1;' \
                   'Sony_DSC-T77_0;Sony_DSC-T77_1;Sony_DSC-T77_2;Sony_DSC-T77_3;Sony_DSC-W170_0'
    parser.add_argument('--list_dev_test', type=str, default=def_dev_list) # list of test devices
    parser.add_argument('--crop_size', type=int, default=320) # size of central patch
    parser.add_argument('--model_dir', type=str, default='./model')  # directory with CNN weights
    parser.add_argument('--output_file', type=str, default='./output/results.npz')  # output file with result
    parser.add_argument('--gpu', type=str, default='0') # gpu to be used
    parser.add_argument('--base_network', type=str, default='Pcn') # CNN architecture to be used

    config, _ = parser.parse_known_args()
    if config.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu  # set the GPU device

    configSess = tf.ConfigProto()
    # Allowing the GPU memory to grow avoid preallocating all the GPU memory
    configSess.gpu_options.allow_growth = True
    set_session(tf.Session(config=configSess))

    list_dev = config.list_dev_test.split(';')
    output_file = config.output_file
    crop_size = config.crop_size
    num_dev = len(list_dev)
    base_network = config.base_network

    # define the model used
    model = make_network(input_shape=(None, None, 2), base_network=base_network, num_classes=2,
                         model_path=config.model_dir)

    print("Preparing test data loader")
    # list with the device PRNUs
    list_prnu = [preprocessing_function(load_prnu(item, crop_size)) for item in list_dev]
    list_prnu = np.stack(list_prnu, 0)

    # list with the noise residuals of the images
    list_content = [np.load('Noises_lists/test/list_%s.npy' % item).tolist() for item in list_dev]

    print('Starting Test')
    score_mat = [None for x in range(num_dev)]

    time_list = []

    for indexD in range(num_dev):  # loop on devices
        list_residuals = list_content[indexD]  # list with the residuals of the device
        num_residuals = len(list_residuals)

        # initialize the matrix with the scores
        score_mat[indexD] = [None for x in range(num_residuals)]

        for indexR in tqdm(range(num_residuals)):  # loop on the residuals of the device

            residue = load_res([list_residuals[indexR], ], crop_size)  # load the residue
            score_mat[indexD][indexR] = np.nan * np.ones((num_dev, 1))

            res_processed = preprocessing_function(residue[:, :, 0])
            res_processed = np.tile(res_processed[None, :, :], (num_dev, 1, 1))
            data = np.stack((list_prnu, res_processed), -1)

            start_time = time.time()

            # predict
            score_mat[indexD][indexR][:, 0] = model.predict_on_batch(data)[:, 1]

            elapsed_time = time.time() - start_time
            elapsed_time = elapsed_time / num_dev

            time_list.append(elapsed_time)

    # save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    score_mat_array = np.array([None, score_mat])
    np.savez(output_file, list_dev=list_dev, score_mat=score_mat_array, time_list=time_list)

    # Release memory
    keras.backend.clear_session()
