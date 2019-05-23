import datagen
import models

if __name__ == '__main__':
    import pandas as pd
    import keras
    import numpy as np
    import json
    import matplotlib.pyplot as plt
    import plotr
    
    # -- SCRIPT HYPERPARAMETERS --
    # General dataset parameters
    subsampling = 1

    # Training dataset parameters
    training_inclusion = (0,1,1)
    # Testing dataset parameters
    testing_inclusion = (0,2,10)
    # Settings for loading and saving models
    models_path = "saved_models"
    model_name = "spectrum_model"
    save_model_after_training =  False
    
    # Training parameters
    batch_size = 64
    epochs = 50
    
    metadata_path = "../data/index.json"
    metadata = None
    with open(metadata_path) as json_file:
        metadata = json.load(json_file)
    
    # -- SCRIPT FUNCTIONALITY --
    # Instantiate training generator
    train_gen = datagen.SpectrumDataSequence(metadata, "train", models.basic_hi,
                                         batch_size=batch_size,
                                         subsampling=subsampling,
                                         shuffle=True,
                                         inclusion=training_inclusion,
                                         noise_scale=.2,
                                         normalize=False,)
    val_gen = datagen.SpectrumDataSequence(metadata, "test", models.basic_hi,
                                        batch_size=batch_size,
                                        subsampling=subsampling,
                                        shuffle=True,
                                        inclusion=testing_inclusion,
                                        noise_scale=.1,
                                        normalize=False,)
    
    sample_length = train_gen.metadata["file_length"]//subsampling//2+1
    
    # Create model and print model summary
    model = models.get_spectrum_model(sample_length)
    print(model.summary())

    path_string = "/".join((models_path,model_name))+".h5"

    csv_logger = keras.callbacks.CSVLogger("logs/spectrumdata_train.csv")
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit_generator(train_gen,
                        epochs=epochs,
                        validation_data=val_gen,
                        callbacks=[early_stop],)
    if save_model_after_training:
        model.save(path_string)
