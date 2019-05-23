import keras
import keras.backend as K

def calu(x):
    """ Capped linear unit (0-1)"""
    return K.minimum(1.0, K.maximum(0.0, x))

def wmae(ytrue, ypred):
    """ Weighted mean average error """
    err=K.abs(K.mean(K.abs(ytrue-ypred)) - ytrue + .5)
    return err

def get_nn_model():
    """
    Returns neural network model for use with statistical data
    """
    # -- INPUT --
    input1  = keras.layers.Input(shape=(6,), dtype='float', name='input1')
    
    # -- HIDDEN LAYERS --
    # Dense block 1
    x       = keras.layers.Dense(64)(input1)
    x       = keras.layers.BatchNormalization()(x)
    x       = keras.layers.Activation('relu')(x)
    
    # Dense block 1
    x       = keras.layers.Dense(64)(x)
    x       = keras.layers.BatchNormalization()(x)
    x       = keras.layers.Activation('relu')(x)
    
    # Dense block 1
    x       = keras.layers.Dense(64)(x)
    x       = keras.layers.BatchNormalization()(x)
    x       = keras.layers.Activation('relu')(x)
    
    # -- OUTPUT --
    x       = keras.layers.Dense(1)(x)
    out1    = keras.layers.Activation("sigmoid", name='prognosis')(x)
    
    # Model
    model = keras.Model(inputs=input1, outputs=out1)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mean_absolute_error'],)
    return model

def get_timedata_model(input_length):
    """
    Returns convolutional neural network model for use with time series data
    """
    # -- INPUT --
    input1  = keras.layers.Input(shape=(input_length, 1), dtype='float', name='input1')
    
    # -- HIDDEN LAYERS --
    # Convolutional block 1
    x       = keras.layers.Conv1D(64, 16)(input1)
    x       = keras.layers.MaxPool1D(16)(x)
    x       = keras.layers.BatchNormalization()(x)
    x       = keras.layers.Activation("relu")(x)
    
    # Convolutional block 2
    x       = keras.layers.Conv1D(128, 16)(x)
    x       = keras.layers.MaxPool1D(16)(x)
    x       = keras.layers.BatchNormalization()(x)
    x       = keras.layers.Activation("relu")(x)
    
    # Dense block 1
    x       = keras.layers.Flatten()(x)
    x       = keras.layers.Dense(128)(x)
    x       = keras.layers.BatchNormalization()(x)
    x       = keras.layers.Activation('relu')(x)
    
    # Dense block 2
    x       = keras.layers.Dense(64)(x)
    x       = keras.layers.BatchNormalization()(x)
    x       = keras.layers.Activation('relu')(x)
    
    # -- OUTPUT --
    x       = keras.layers.Dense(1)(x)
    out1    = keras.layers.Activation("sigmoid", name='prognosis')(x)
    
    # Model
    model = keras.Model(inputs=input1, outputs=out1)
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],)
    return model

def get_spectrum_model(input_length):
    """
    Returns convolutional neural network model for use with spectrum data
    """
    # -- INPUT --
    input1  = keras.layers.Input(shape=(input_length, 2), dtype='float', name='input1')
    
    # -- HIDDEN LAYERS --
    # Convolutional block 1
    x       = keras.layers.Conv1D(64, 4)(input1)
    x       = keras.layers.MaxPool1D(4)(x)
    x       = keras.layers.BatchNormalization()(x)
    x       = keras.layers.Activation("relu")(x)
    
    # Convolutional block 1
    x       = keras.layers.Conv1D(64, 8)(x)
    x       = keras.layers.MaxPool1D(4)(x)
    x       = keras.layers.BatchNormalization()(x)
    x       = keras.layers.Activation("relu")(x)
    
    # Dense block 1
    x       = keras.layers.Flatten()(x)
    x       = keras.layers.Dense(128)(x)
    x       = keras.layers.BatchNormalization()(x)
    x       = keras.layers.Activation('relu')(x)
    
    # Dense block 2
    x       = keras.layers.Dense(64)(x)
    x       = keras.layers.BatchNormalization()(x)
    x       = keras.layers.Activation('relu')(x)
    
    # -- OUTPUT --
    x       = keras.layers.Dense(1)(x)
    out1    = keras.layers.Activation("sigmoid", name='prognosis')(x)
    
    # Model
    model = keras.Model(inputs=input1, outputs=out1)
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],)
    return model

# Health index models
def basic_hi(dataset, current_file, last_file):
    """ Basic lifetime based HI function """
    hi = current_file/last_file
    return hi

def linear_hi(dataset, current_file, last_file):
    """ Linear HI increasing from 0 to 1 after "degradation_start" set in
        dataset dictionary """
    split = dataset["degradation_start"]
    hi = np.maximum(0, (current_file-split)/(last_file-split))
    return hi

def exp_hi(dataset, current_file, last_file):
    """ Exponential HI designed to work with curve fitted function parameters
        set in dataset description json file """
    time_spacing = eval(dataset.get("time_spacing") or\
                   self.metadata.get("default_time_spacing"))
    
    split = dataset["split"]
    if file > split:
        curve = lambda x, k, M: k*np.power(x, M)
        params = dataset["curve_params"]
        total_failure = curve(time_spacing(last_file-split), **params)
        current_degradation = curve(time_spacing(current_file-split), **params)
        label = current_degradation / total_failure # normalize
        return label
    else:
        return 0.0

if __name__ == "__main__":
    model = get_timedata_model(2560)
    print(model.summary())