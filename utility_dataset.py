# utilities for reading device PRNUs and image noise residuals

import h5py
import numpy as np
from tqdm import tqdm

vision_dev_list = 'D01;D02;D03;D05;D06;D07;D08;D09;D10;D11;D12;D14;D15;D16;D17;D18;D19;D20;D21;D22;D24;D25;D26;D27;' \
               'D28;D29;D30;D31;D32;D33;D34;D35'
vision_dev_list = vision_dev_list.split(';')
dresden_dev_list = 'Canon_Ixus55_0;Canon_Ixus70_0;Canon_Ixus70_1;Canon_Ixus70_2;' \
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
dresden_dev_list = dresden_dev_list.split(';')

# PRNU root vision
vision_prnu_dir = 'Vision/prnu_flat_%s.npy'
# PRNU root dresden
dresden_prnu_dir = 'Dresden/prnu_flat_%s.npy'


def preprocessing_function(x):
    return x / (np.sqrt(np.mean(np.square(x))) + np.finfo(float).eps)


def load_prnu(dev_id, size=None):
    # load device PRNU and crop it
    # dev_id = device identifier
    # size = crop size, if None the PRNU is not cropped

    # PRNU of devices were saved as .npy files as they were computed following the implementation at:
    # https://github.com/polimi-ispl/prnu-python

    if np.in1d(dev_id, vision_dev_list):
        try:
            with h5py.File(vision_prnu_dir % dev_id, 'r') as f:
                prnu = np.copy(f['prnu'].value).transpose((1, 0))
        except:
            prnu = np.load(vision_prnu_dir % dev_id)
    else:
        try:
            with h5py.File(dresden_prnu_dir % dev_id, 'r') as f:
                prnu = np.copy(f['prnu'].value).transpose((1, 0))
        except:
            prnu = np.load(dresden_prnu_dir % dev_id)

    prnu = np.copy(prnu, order='C')
    if size is not None:
        r0 = (prnu.shape[0] - size) // 2
        r1 = (prnu.shape[1] - size) // 2
        prnu = prnu[r0:(r0 + size), r1:(r1 + size)]
        assert (prnu.shape == (size, size))

    assert (prnu.dtype == np.float32)
    assert (not (np.isfortran(prnu)))

    return prnu


def load_res(list_noises, size=None, dtype=np.float32):
    # load image noise residual and crop it
    # dev_id = device identifier
    # size = crop size, if None the residue is not cropped

    # noise residuals of every image were saved as .npy files as they were computed following the implementation at:
    # https://github.com/polimi-ispl/prnu-python

    shape_img = None
    crop_frame = [0, 0]
    list_res = list()
    for index, filename in tqdm(enumerate(list_noises)):

        res = np.load(filename)
        res = np.copy(res, order='C')

        if shape_img is None:
            shape_img = (res.shape[0], res.shape[1])
            if size is not None:
                crop_frame[0] = (res.shape[0] - size) // 2
                crop_frame[1] = (res.shape[1] - size) // 2

        # check if the img is rotated by 90 deg
        elif shape_img[0] == res.shape[1] and shape_img[1] == res.shape[0]:
            res = res.T
        else:
            assert (shape_img[0] == res.shape[0])
            assert (shape_img[1] == res.shape[1])

        if size is not None:
            res = res[crop_frame[0]:(crop_frame[0] + size), crop_frame[1]:(crop_frame[1] + size)]
            assert (res.shape[0] == size)
            assert (res.shape[1] == size)

        assert (not (np.isfortran(res)))
        assert (res.dtype == dtype)

        res = np.expand_dims(res, -1)
        list_res.append(res)

    res = np.concatenate(list_res, -1)
    assert (not (np.isfortran(res)))
    assert (res.dtype == dtype)

    return res

