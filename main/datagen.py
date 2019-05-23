import os
import numpy as np
from scipy import stats
import pandas
import keras

class TimeDataSequence(keras.utils.Sequence):
    """
    Keras Sequence model data batch generator to be used with vibrational data
    for training and testing models batch wise.
    
    metadata    -- Dictionary of metadata used for training (loaded from json)
    purpose     -- Purpose of generator; "train" or "test"
    health_index-- Function that takes following arguments
                    
                    (dataset, current_file, last_file)
                    
                    Should return the desired health indicator as a function of
                    current file
    batch_size  -- Batch size used for training
    subsampling -- Number of subsampled signals to divide each signal into
    inclusion   -- Tuple containing values used for including parts of dataset
                    
                    (first, last, every)
                    
                    Useful for dividing same datasets into different generators
                    for training, validation, testing etc.
    noise_scale -- Scale (stdev) of noise to apply to each sample
    normalize   -- Whether to perform a per-sample normalization
    shuffle     -- Whether the dataset should be shuffled before each epoch
    verbose     -- Whether to output status messages
    """
    def __init__(self, metadata, purpose, health_index,
                 batch_size=10,
                 subsampling=1,
                 inclusion=None,
                 noise_scale=0,
                 normalize=True,
                 shuffle=True,
                 verbose=True,):
        self.metadata = metadata
        self.purpose = purpose
        self.health_index = health_index
        self.batch_size = batch_size
        self.subsampling = subsampling
        self.noise_scale = noise_scale
        self.normalize = normalize
        self.shuffle = shuffle
        self.verbose = verbose
        
        if purpose == "train": self.datasets = metadata["train_data"]
        elif purpose == "test": self.datasets = metadata["test_data"]
        
        self.sample_length = metadata["file_length"] // subsampling
        self.n_files = 0
        self.files = []
        for dataset in self.datasets:
            dp = self.metadata["path"] + dataset["path"] # dataset path
            files = sorted(os.listdir(dp))
            self.n_files += len(files)
            self.files.append(files)

        self.sample_ids = np.arange(self.n_files*subsampling)
            
        # if inclusion parameter is given:
        # - generate inclusion ranges for every interval
        # - extract only IDs from inclusion ranges
        if inclusion:
            include_first, include_last, include_every = inclusion
            inclusion_ranges = []
            for i in range(len(self.sample_ids)//include_every):
                sub_range = range(i*include_every+include_first,
                                  i*include_every+include_last)
                inclusion_ranges.append(sub_range)
            inclusion_array = np.array(inclusion_ranges).flatten()
            self.sample_ids = np.take(self.sample_ids, inclusion_array)
        
        # initial shuffle
        self.on_epoch_end()
        
        if self.verbose:
            print(f"{self.purpose.capitalize()} generator initialized!")
            print(f"{self.n_files} files in {len(self.datasets)} datasets")
    
    def __len__(self):
        return len(self.sample_ids)//self.batch_size
    
    def __getitem__(self, index):
        batch_samples, batch_labels = self.generate_batch(index)
        return batch_samples, batch_labels
    
    def generate_batch(self, index):
        batch_samples = np.zeros((self.batch_size, self.sample_length, 1))
        batch_labels = np.zeros((self.batch_size))
        selection = np.arange(index*self.batch_size, (index+1)*self.batch_size)
        ids = np.take(self.ids_remaining, selection) # pick batch-size amount of samples
        for i, id in enumerate(ids):
            path_id = 0 # should be renamed dataset_id
            next_sample_position = 0
            for j in range(len(self.datasets)):
                # This loop looks at each dataset to check if the current file-ID is
                # inside of current dataset. When correct dataset is found,
                # the dataset-index is stored inside path_id and loop is broken.
                # --> Should be optimized at later date.
                n_dataset_samples = len(self.files[j])*self.subsampling
                next_sample_position += n_dataset_samples
                if id < next_sample_position:
                    path_id = j
                    break
            
            dataset = self.datasets[path_id]    
            file_id = id//self.subsampling
            file_id -= sum(map(len, self.files[:path_id]))
            subsample_id = id%self.subsampling

            # setting labels
            last_file = len(self.files[path_id])
            batch_labels[i] = self.health_index(dataset, file_id, last_file)

            # determine which channel to use
            channels = [dataset["channel"]]
            sep = dataset.get("sep") or self.metadata.get("default_sep")
            fp = self.metadata["path"] + dataset["path"] + self.files[path_id][file_id]
            data = pandas.read_csv(fp, sep=sep, usecols=channels, header=None).values
            index_from = subsample_id*self.sample_length
            index_to = (subsample_id+1)*self.sample_length
            batch_samples[i] = data[index_from:index_to]
            if self.noise_scale:
                stdev = np.std(batch_samples[i])
                noise = np.random.normal(scale=stdev*self.noise_scale,
                                         size=(batch_samples[i].shape))
                batch_samples[i] += noise
            if self.normalize:
                # Per-sample min-max normalization
                norm = np.linalg.norm(batch_samples[i], ord=np.inf)
                batch_samples[i]/=norm
                
        return batch_samples, batch_labels

    def on_epoch_end(self):
        self.ids_remaining = np.copy(self.sample_ids)
        if self.shuffle:
            np.random.shuffle(self.ids_remaining)
    
class SpectrumDataSequence(TimeDataSequence):
    """
    Extension of TimeDataSequence model. Estimates a complex frequency spectrum
    using FFT and outputs a rank three tensor:
    (batch_size, floor(signal_length/2)+1, 2)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        samples, labels = super().generate_batch(index)
        spectra = np.zeros((len(samples), 2, self.metadata["file_length"]//2+1))
        for i, sample in enumerate(samples):
            fft_h = np.fft.rfft(sample, axis=0, norm="ortho").flatten()
            spectra[i][0] = np.abs(fft_h.real)
            spectra[i][1] = np.abs(fft_h.imag)
        spectra = np.reshape(spectra, (spectra.shape[0], spectra.shape[2], spectra.shape[1]))
        return spectra, labels

class StatisticalDataSequence(TimeDataSequence):
    """
    Extension of TimeDataSequence model. Extracts and outputs statistical
    parameters from time series data. Parameterers are organized per sample as
    follows: rms, skewness, kurtosis, activity, mobility, complexity
    """
    @staticmethod
    def rms(y):
        return np.sqrt(np.sum(np.power(y, 2)))/y.size
    @staticmethod
    def hjorth_activity(y):
        return np.var(y)
    @staticmethod
    def hjorth_mobility(y):
        return np.sqrt(np.var(np.gradient(y))/np.var(y))
    @staticmethod
    def hjorth_complexity(y):
        return __class__.hjorth_mobility(np.gradient(y))/__class__.hjorth_mobility(y)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        samples, labels = super().generate_batch(index)
        params = np.zeros((len(samples), 6))
        for i, sample in enumerate(samples):
            sample = sample.flatten()
            params[i][0] = self.rms(sample) # rms
            params[i][1] = stats.skew(sample) # skewness
            params[i][2] = stats.kurtosis(sample) # kurtosis
            params[i][3] = self.hjorth_activity(sample)
            params[i][4] = self.hjorth_mobility(sample)
            params[i][5] = self.hjorth_complexity(sample)
        
        return params, labels

if __name__ == '__main__':
    # for testing
    import json
    import models
    metadata_path = "../data/index.json"
    metadata = None
    with open(metadata_path) as json_file:
        metadata = json.load(json_file)
    
    test_gen = TimeDataSequence(metadata, "test", models.basic_hi,
                                batch_size=2,
                                subsampling=1,
                                inclusion=(0,1,4),
                                noise_scale=0,
                                normalize=True,
                                shuffle=False,)
    
    import matplotlib.pyplot as plt
    samples, labels = test_gen[15]
    plt.plot(samples[0])
    plt.show()
        