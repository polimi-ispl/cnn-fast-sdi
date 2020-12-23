import numpy as np
from keras.utils import np_utils, Sequence


class DataGeneratorPRNU(Sequence):

    def __init__(self, batch_size, list_data0, list_data1,
                 size_block=2, num_blocks=18, to_categorical=2,
                 preprocessing_function0=None,
                 preprocessing_function1=None,
                 random_crop=True,
                 size_crop=320,
                 horizontal_flip=False, rot_90=False, seed=None):
        """
        :param batch_size: size of batches to create
        :param list_data0: matrices list for data 0 (Nrows,Ncols, 1) (device PRNUs)
        :param list_data1: matrices list for data 1 (Nrows,Ncols,Nsamples) (image noise residuals)
        :param size_block: number of classes
        :param num_blocks: number of devices in list_data0 and in list_data1
        :param to_categorical: if > 1, convert the labels to categorical
        :param preprocessing_function0: method to preprocess data0 (default None)
        :param preprocessing_function1: method to preprocess data1 (default None)
        :param random_crop: True if crop is random and False if the crop is central
        :param size_crop: integer, dimension of the image patch
        :param horizontal_flip: if True, selected data will be randomly horizontally flipped
        :param rot_90: if True, selected data will be randomly rotate
        :param seed: integer

        "list_data0", "list_data1" are lists where each element is associated with a different device
        Every element of "list_data0" is a matrix (Nrows x Ncols x 1) with the device PRNU
        Every element of "list_data1" is a matrix (Nrows x Ncols x Nresidues) with the residues related to the device
        """

        # Set info
        self.size_block = int(size_block)
        self.num_blocks = int(num_blocks)
        self.to_categorical = int(to_categorical)
        self.batch_size = int(batch_size)
        self.num_combine = self.size_block * self.num_blocks
        self.list_data0 = list_data0
        self.list_data1 = list_data1
        self.num_classes = len(self.list_data0)
        self.step_blocks = self.num_classes // self.num_blocks

        assert (len(self.list_data0) == len(self.list_data1))  # same number of classes
        assert (self.size_block >= 2)
        assert (self.size_block <= self.num_classes)
        assert (self.num_blocks >= 1)
        assert (self.num_blocks <= self.num_classes)
        assert ((self.batch_size % self.num_combine) == 0)

        self.data_shape0 = self.list_data0[0].shape[0]
        self.data_shape1 = self.list_data0[0].shape[1]
        self.num_samples = 0
        print(self.num_classes, self.data_shape0, self.data_shape1)

        for index in range(self.num_classes):

            # check dimensions
            assert (len(self.list_data0[index].shape) == 3)
            assert (len(self.list_data1[index].shape) == 3)
            assert (self.list_data0[index].shape[0] == self.data_shape0)  # same num of rows in list_data0
            assert (self.list_data0[index].shape[1] == self.data_shape1)  # same num of cols in list_data0
            assert (self.list_data1[index].shape[0] == self.data_shape0)  # same num of rows in list_data1
            assert (self.list_data1[index].shape[1] == self.data_shape1)  # same num of cols in list_data1
            self.num_samples = max(self.num_samples, self.list_data0[index].shape[2])
            self.num_samples = max(self.num_samples, self.list_data1[index].shape[2])
            print(index, self.list_data0[index].shape[2], self.list_data1[index].shape[2])

        self.batch_size_for_combine = self.batch_size // self.num_combine
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size_for_combine))
        self.preprocessing_function0 = preprocessing_function0
        self.preprocessing_function1 = preprocessing_function1
        self.random_crop = random_crop
        self.size_crop = size_crop
        self.horizontal_flip = horizontal_flip
        self.rot_90 = rot_90
        if seed is None:
            seed = np.random.random_integers(100000)
        self.randomState = np.random.RandomState(seed)

        # Data for iteration
        self._indices0 = [np.arange(0, self.list_data0[index].shape[2]) for index in range(self.num_classes)]
        self._indices1 = [np.arange(0, self.list_data1[index].shape[2]) for index in range(self.num_classes)]
        self._crop_0 = (self.data_shape0 - self.size_crop) // 2
        self._crop_1 = (self.data_shape1 - self.size_crop) // 2

        # Initialize random permutation
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return self.num_batches

    def on_epoch_end(self):
        """
        Update indices after one epoch
        """
        for item in self._indices0:
            self.randomState.shuffle(item)
        for item in self._indices1:
            self.randomState.shuffle(item)

    def print_info(self):
        print(self.batch_size, self.num_batches, self.num_blocks, self.size_block)

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        assert (index < self.num_batches)
        offset = self.batch_size_for_combine * index

        X_batch = []
        Y_batch = []
        indicesC = np.arange(self.num_classes)

        for k in range(self.batch_size_for_combine):
            # set the flags for horizontal flip and rot90
            flag_horizontal_flip = (self.randomState.randint(0, 2) == 1) if self.horizontal_flip else False
            flag_rot_90 = self.randomState.randint(0, 4) if self.rot_90 else 0

            if self.random_crop:
                # Random Crop
                crop_0 = self.randomState.randint(low=0, high=(self.data_shape0 - self.size_crop + 1), size=[])
                crop_1 = self.randomState.randint(low=0, high=(self.data_shape1 - self.size_crop + 1), size=[])
            else:
                # Central Crop
                crop_0 = self._crop_0
                crop_1 = self._crop_1

            self.randomState.shuffle(indicesC)  # shuffle index of classes
            for indexCa in range(self.num_blocks):
                cA = indicesC[indexCa * self.step_blocks]  # class A

                # select a noise residue from class A
                index1A = self._indices1[cA][offset % len(self._indices1[cA])]
                img1A = self.list_data1[cA][crop_0:(crop_0 + self.size_crop), crop_1:(crop_1 + self.size_crop), index1A]

                if flag_horizontal_flip:
                    # horizontal flip
                    img1A = np.flip(img1A, 1)

                if flag_rot_90 > 0:
                    # rot90
                    img1A = np.rot90(img1A, k=flag_rot_90, axes=(0, 1))

                if self.preprocessing_function1 is not None:
                    # preprocessing_function
                    img1A = self.preprocessing_function1(img1A.astype(np.float32))

                for indexCb in range(self.size_block):
                    cB = indicesC[(indexCa * self.step_blocks + indexCb) % self.num_classes]  # class B

                    # select PRNU of class B
                    index0B = self._indices0[cB][offset % len(self._indices0[cB])]
                    img0B = self.list_data0[cB][crop_0:(crop_0 + self.size_crop), crop_1:(crop_1 + self.size_crop),
                            index0B]

                    if flag_horizontal_flip:
                        # horizontal flip
                        img0B = np.flip(img0B, 1)

                    if flag_rot_90 > 0:
                        # rot90
                        img0B = np.rot90(img0B, k=flag_rot_90, axes=(0, 1))

                    if self.preprocessing_function0 is not None:
                        # preprocessing_function
                        img0B = self.preprocessing_function0(img0B.astype(np.float32))

                    X_batch.append(np.stack((img0B, img1A), -1))
                    Y_batch.append(cA == cB)

            offset = offset + 1
            if offset >= self.num_samples: break

        X_batch = np.asarray(X_batch)
        Y_batch = np.asarray(Y_batch)
        if self.to_categorical > 1:
            Y_batch = np_utils.to_categorical(Y_batch, self.to_categorical)

        return X_batch, Y_batch