from keras.models import Model
from keras.optimizers import Nadam
from keras.layers import BatchNormalization, Convolution2D, Input, merge
from keras.layers.core import Activation, Layer
from keras.utils.visualize_util import plot

'''
Keras Customizable Residual Unit

This is as simplified implementation of the basic (no bottlenecks) full pre-activation residual unit from He, K., Zhang, X., Ren, S., Sun, J., "Identity Mappings in Deep Residual Networks" (http://arxiv.org/abs/1603.05027v2).
'''

def conv_block(feat_maps_out, prev):
    prev = BatchNormalization(axis=1, mode=2)(prev) # Specifying the axis and mode allows for later merging
    prev = Activation('relu')(prev)
    prev = Convolution2D(feat_maps_out, 3, 3, border_mode='same')(prev) 
    prev = BatchNormalization(axis=1, mode=2)(prev) # Specifying the axis and mode allows for later merging
    prev = Activation('relu')(prev)
    prev = Convolution2D(feat_maps_out, 3, 3, border_mode='same')(prev) 
    return prev


def skip_block(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = Convolution2D(feat_maps_out, 1, 1, border_mode='same')(prev)
    return prev 


def Residual(feat_maps_in, feat_maps_out, prev_layer):
    '''
    A customizable residual unit with convolutional and shortcut blocks

    Args:
      feat_maps_in: number of channels/filters coming in, from input or previous layer
      feat_maps_out: how many output channels/filters this block will produce
      prev_layer: the previous layer
    '''

    skip = skip_block(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block(feat_maps_out, prev_layer)

    print('Residual block mapping '+str(feat_maps_in)+' channels to '+str(feat_maps_out)+' channels built')
    return merge([skip, conv], mode='sum') # the residual connection


if __name__ == "__main__":
    # NOTE: Toy example shows structure
    img_rows = 28  
    img_cols = 28 

    inp = Input((1, img_rows, img_cols))
    cnv1 = Convolution2D(64, 7, 7, subsample=[2,2], activation='relu', border_mode='same')(inp)
    r1 = Residual(64, 128, cnv1)
    # An example residual unit coming after a convolutional layer. NOTE: the above residual takes the 64 output channels
    # from the Convolutional2D layer as the first argument to the Residual function
    r2 = Residual(128, 128, r1)
    r3 = Residual(128, 256, r2)
    out = Convolution2D(1, 1, 1, border_mode='same', activation='sigmoid')(r3)

    model = Model(input=inp, output=out)
    model.compile(optimizer=Nadam(lr=1e-5), loss='mean_squared_error')

    plot(model, to_file='./toy_model.png', show_shapes=True)
