import os
import keras
from bipooling import PairwiseCorrPooling


def makePairwiseCorrNetwork(input_shape, num_blocks=3):
    model = keras.models.Sequential()
    model.add(
        keras.layers.Conv2D(16, 3, strides=(1, 1), dilation_rate=(1, 1), padding='valid', input_shape=input_shape))
    model.add(keras.layers.LeakyReLU(alpha=0.3))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='valid', data_format=None))

    for _ in range(1, num_blocks):
        model.add(keras.layers.Conv2D(64, 5, strides=(1, 1), dilation_rate=(1, 1), padding='valid'))
        model.add(keras.layers.LeakyReLU(alpha=0.3))
        model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='valid', data_format=None))

    model.add(PairwiseCorrPooling(pooling=False))

    return model


def addBand(x):
    from tensorflow import concat
    y = concat((x, 0.0 * x[:, :, :, :1]), 3)
    return y


def addLastLayers(base_network, num_classes=2, pooling_logit=-1, include_activation=True, input_shape=None):
    model = keras.models.Sequential()
    if input_shape is not None:
        model.add(keras.layers.Lambda(addBand, input_shape=input_shape))
    model.add(base_network)

    model.add(keras.layers.Conv2D(num_classes, 1, strides=(1, 1), padding='valid', name='ConvClass'))
    if pooling_logit < 0:
        model.add(keras.layers.GlobalAveragePooling2D())
    elif pooling_logit == 0:
        model.add(keras.layers.GlobalAveragePooling2D())
        model.add(keras.layers.Reshape((1, 1, -1)))
    elif pooling_logit > 1:
        model.add(keras.layers.AveragePooling2D(pool_size=pooling_logit, padding='valid'))

    if include_activation:
        model.add(keras.layers.Activation('softmax'))
        if pooling_logit >= 0:
            model.add(keras.layers.GlobalAveragePooling2D())

    return model


def makeNetwork(input_shape, base_network='Pcn', num_classes=2, pooling_logit=-1, include_activation=True,
                model_path=None):
    if base_network == 'Inc':
        # Inception ResNet V2
        from keras.applications.inception_resnet_v2 import InceptionResNetV2
        imagenet_input_shape = (input_shape[0], input_shape[1], 3)
        base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=imagenet_input_shape)
    elif base_network == 'EffB4':
        # EfficientNetB4
        import efficientnet.keras as efn
        imagenet_input_shape = (input_shape[0], input_shape[1], 3)
        base_model = efn.EfficientNetB4(include_top=False, weights='imagenet', input_shape=imagenet_input_shape)
    elif base_network == 'EffB2':
        # EfficientNetB2
        import efficientnet.keras as efn
        imagenet_input_shape = (input_shape[0], input_shape[1], 3)
        base_model = efn.EfficientNetB2(include_top=False, weights='imagenet', input_shape=imagenet_input_shape)
    elif base_network == 'EffB0':
        # EfficientNetB0
        import efficientnet.keras as efn
        imagenet_input_shape = (input_shape[0], input_shape[1], 3)
        base_model = efn.EfficientNetB0(include_top=False, weights='imagenet', input_shape=imagenet_input_shape)
    elif base_network == 'Pcn':
        # PairwiseCorrNetwork
        num_blocks = 3
        base_model = makePairwiseCorrNetwork(input_shape=input_shape, num_blocks=num_blocks)
        input_shape = None
    else:
        assert (False)

    model = addLastLayers(base_model, num_classes=num_classes, pooling_logit=pooling_logit,
                          include_activation=include_activation, input_shape=input_shape)

    if model_path is not None:
        final_weights_path = os.path.join(model_path, 'model_weights.h5')
        model.load_weights(final_weights_path)
        print('loaded ', final_weights_path)

    return model