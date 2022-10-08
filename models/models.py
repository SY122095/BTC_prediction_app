import tensorflow as tf
from keras.layers import Input
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects


###---------------------------------------WaveNet---------------------------------------###
def mish(x):
    '''活性化関数の定義'''
    return tf.keras.layers.Lambda(lambda x: x*K.tanh(K.softplus(x)))(x)
get_custom_objects().update({'mish': mish})

def WaveNetResidualConv1D(num_filters, kernel_size, stacked_layer):
    '''WaveNetの構築'''
    def build_residual_block(l_input):
        resid_input = l_input
        for dilation_rate in [2**i for i in range(stacked_layer)]:
            l_sigmoid_conv1d = tf.keras.layers.Conv1D(num_filters, kernel_size, dilation_rate=dilation_rate, padding='same', activation='sigmoid')(l_input)
            l_tanh_conv1d = tf.keras.layers.Conv1D(num_filters, kernel_size, dilation_rate=dilation_rate, padding='same', activation='mish')(l_input)
            l_input = tf.keras.layers.Multiply()([l_sigmoid_conv1d, l_tanh_conv1d])
            l_input = tf.keras.layers.Conv1D(num_filters, 1, padding='same')(l_input)
            resid_input = tf.keras.layers.Add()([resid_input, l_input])
        return resid_input
    return build_residual_block

def wavenet_training(x_train, x_valid, y_train, y_valid, num_filters=16, kernel_size=8, batchsize=128, lr=0.0001):
    '''wavenetによる学習'''
    num_filters_ = num_filters
    kernel_size_ = kernel_size
    stacked_layers_ = [20, 12, 8, 4, 1]
    shape_ = (None, x_train.shape[2])
    l_input = Input(shape=(shape_))
    x = tf.keras.layers.Conv1D(num_filters_, 1, padding='same')(l_input)
    x = WaveNetResidualConv1D(num_filters_, kernel_size_, stacked_layers_[0])(x)
    x = tf.keras.layers.Conv1D(num_filters_*2, 1, padding='same')(l_input)
    x = WaveNetResidualConv1D(num_filters_*2, kernel_size_, stacked_layers_[1])(x)
    x = tf.keras.layers.Conv1D(num_filters_*4, 1, padding='same')(l_input)
    x = WaveNetResidualConv1D(num_filters_*4, kernel_size_, stacked_layers_[2])(x)
    x = tf.keras.layers.Conv1D(num_filters_*8, 1, padding='same')(l_input)
    x = WaveNetResidualConv1D(num_filters_*8, kernel_size_, stacked_layers_[3])(x)
    x = tf.keras.layers.Conv1D(num_filters_*16, 1, padding='same')(l_input)
    x = WaveNetResidualConv1D(num_filters_*16, kernel_size_, stacked_layers_[4])(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    l_output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=[l_input], outputs=[l_output])
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=optimizer)
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(x_train, y_train, validation_data=[x_valid, y_valid], callbacks=[es_callback], epochs=200, batch_size=batchsize)
    return model
###---------------------------------------WaveNet---------------------------------------###



###---------------------------------------Logistic---------------------------------------###
###---------------------------------------Logistic---------------------------------------###



###---------------------------------------Light GBM---------------------------------------###
###---------------------------------------Light GBM---------------------------------------###