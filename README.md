<h2>Source code material for</h2>
<h1>Lifetime Based Health Indicator for Bearings using Convolutional Neural Networks</h1>

This is the initial commit. Some of the code had some last minute changes to
improve readability, and has not been tested afterwards. This might result in
a minor error or two. A lot of optimization and code reduction still needs to
be made.

The data files goes in the data folder. Look into `data/index.json` to understand
the current structuring or change it if needed. The datasets should only contain
files with accelerometer data.

The scripts to train the models are `main/timedata_training.py` and `main/spectrum_training.py`.
They will save the trained models to a `main/saved_models/` folder. The training logs
will be saved in a `main/logs/` folder. At the moment, these folder
must be manually created

The scripts to test the models are `main/timedata_prediction.py` and `main/spectrum_prediction.py`.
They will load the trained models from the `main/saved_models/` folder.
They will also save the predictions, figures and statistics to the main/validation folder,
which must be manually created.