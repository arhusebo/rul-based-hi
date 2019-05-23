import numpy as np
import json
import keras
import datagen
import models
import plotr
import matplotlib.pyplot as plt
import sys
from scipy import stats
import pandas as pd

def get_pred_outputs(model, generator, verbose=0):
    print(" -- PREDICTING ON TESTING GENERATOR -- ")
    preds = model.predict_generator(generator, verbose=verbose)
    output = np.reshape(preds, (-1,))
    print(" -- FINISHED PREDICTIONS -- ")
    return output

def get_true_outputs(generator, verbose=0):
    print(" -- COLLECTING TRUE OUTPUTS -- ")
    generator_length = len(generator)
    output = np.empty(generator_length)
    for i, io in enumerate(generator):
        if not i%10 and verbose == 1:
            sys.stdout.write(f"\rCollecting output {i}/{generator_length}")
            sys.stdout.flush()
        inp, out = io
        output[i] = out
    print(" -- FINISHED COLLECTING OUTPUTS -- ")
    return output

metadata_path = "../data/index.json"
metadata_all = None
with open(metadata_path) as json_file:
    metadata_all = json.load(json_file)

subsampling = 1
testing_inclusion = (0, 2, 2)

model_path = "saved_models/timedata_model.h5"
fig_path = "../validation/timedata_prediction"
stat_data_path = "../validation/"

legend = []
preds = []
trues = []

pearson_data = {}
spearman_data = {}
mae_data = {}

for dataset in metadata_all["test_data"]:
    metadata = metadata_all
    metadata["train_data"] = []
    metadata["test_data"] = [dataset]
    
    test_gen = datagen.TimeDataSequence(metadata, "test", models.basic_hi,
                                        batch_size=1,
                                        subsampling=subsampling,
                                        shuffle=False,
                                        inclusion=testing_inclusion,
                                        noise_scale=.1,
                                        normalize=True,)
    
    model = keras.models.load_model(model_path)
    
    pred_outputs = get_pred_outputs(model, test_gen, verbose=1)
    true_outputs = get_true_outputs(test_gen, verbose=1)
    
    trues.append(true_outputs)
    preds.append(pred_outputs)
    
    bearing_name = dataset['path'].split('/')[-2]
    
    pearson_data[bearing_name] = stats.pearsonr(pred_outputs, true_outputs)
    spearman_data[bearing_name] = stats.spearmanr(pred_outputs, true_outputs)
    mae_data[bearing_name] = np.mean(np.abs(pred_outputs-true_outputs))
    
    legend.append(bearing_name)

stat_data = pd.DataFrame({"pearson": pearson_data, "spearman": spearman_data, "mae": mae_data})
stat_data.to_json(f"{stat_data_path}/stats.json")
print(stat_data)

data = pd.DataFrame({"trues":trues, "preds":preds})
data.to_json("validation/timeseries_data.json")

plotr.plot_hi_subplots(trues, preds, names=legend)
plt.savefig(f"{fig_path}/timeseries_hi_all.png")
plt.show()