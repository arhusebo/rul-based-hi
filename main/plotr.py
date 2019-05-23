import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LinearRegression
from scipy.stats.mstats import theilslopes

error_color = lambda e: 'red' if e > .2 else 'orange' if e > .1 else 'blue'

def windowed_average(values, window):
    """ Padded windowed average of values with chosen window size """
    weights = np.ones(window) / window
    values_pad = np.pad(values, (window//2, window-1-window//2), mode="edge")
    avg_pred = np.convolve(values_pad, weights, mode='valid')
    return avg_pred

def rul(hi, sample_number, interval):
    """ Get RUL from HI, sample number in a dataset and recording interval """
    rul = (1-hi)*sample_number/hi
    return rul * interval

def lstsq_hi(values):
    """ Get a in f(values) = a*values """
    X_ = np.arange(len(values))
    XX = np.vstack((X_, np.ones_like(X_))).T
    p = np.linalg.lstsq(XX[:,:-1], values)[0]
    return p
    
def momentum_pred(values, momentum=.9, damping=.9):
    """ Experimental trend estimation based on dynamic momentum """
    momentum = min(1, max(momentum, 0))
    damping = min(1, max(damping, 0))
    dY = 0
    Y = np.zeros(values.shape)
    Y[0] = values[0]
    for i, value in enumerate(values[1:]):
        diff = value - Y[i]
        dY += (1-momentum) * diff# * max(value, .05)
        Y[i+1] = Y[i] + dY * (1-damping)
        Y[i+1] = min(1, max(Y[i+1], 0))
    return Y

def plot_confusion(ytrue, ypred, ax, average_window=0, momentum=0, damping=0):
    """ Plots a HI trend on an axes object with respect to HI predictions """
    ytrue = np.reshape(ytrue, -1,)
    ypred = np.reshape(ypred, -1,)
    
    error = np.abs(ytrue-ypred)
    X = range(len(ytrue))
    ax.scatter(X, ypred,
               label="Prediction",
               s=2,
               alpha=1,
               #color=[error_color(e) for e in error]
               c=error,
               cmap="cool")
    ax.plot(X, ytrue, label="Optimal")
    if average_window:
        avg_pred = windowed_average(ypred, average_window)
        ax.plot(X, avg_pred, label="Moving Average")
    if momentum:
        mpred = momentum_pred(ypred, momentum, damping)
        ax.plot(X, mpred, label="Momentum")
    ax.set_ylabel("HI")
    ax.set_xlabel("Sample number")
    ax.grid()

def plot_rul(ytrue, ypred, ax, timestep=10, average_window=0, split=0):
    """ Plots a RUL trend on an axes object with respect to HI predictions """
    ytrue = np.reshape(ytrue, -1,)
    ypred = np.reshape(ypred, -1,)
    
    X = np.arange(len(ytrue))

    #ax.scatter(X, rulpred, label="Prediction", s=2)
    ax.plot(X*timestep, rul(ytrue, X, 10), label="Optimal")
    avg_pred = windowed_average(rul(ypred, X, timestep), average_window)
    ax.plot(X*timestep, avg_pred, label="Moving Average")

    params1 = [lstsq_hi(ypred[:x]) if x>0 else 0 for x in X]
    ax.plot(X*timestep, [rul(params1[x]*x, x, timestep) for x in X], label="Linear Estimation")
    
    ax.set_ylim(0)
    ax.set_xlim(0, X[-1]*timestep)
    
    ax.set_ylabel("RUL [s]")
    ax.set_xlabel("Time [s]")
    ax.grid()
    
    truerul = timestep*(len(ytrue)-split)
    splitrul = avg_pred[split]
    splitlinrul = rul(params1[split]*split, split, timestep)
    err = lambda rul: 100*(truerul-rul)/(truerul)
    score = np.vectorize(lambda e: np.exp(-np.log(.5)*(e/5)) if e <= 0 else np.exp(np.log(.5)*(e/20)))
    
    return([splitrul, splitlinrul, score(err(splitrul)), score(err(splitlinrul))])

def plot_hi_subplots(trues, preds, average_window=100, names=None):
    """ Plots a set of HI subplots for multiple datasets """
    size = len(trues)
    fig = plt.figure(figsize=(8.27, 11.69))
    for i in range(size):
        ax = fig.add_subplot(int(np.ceil(size/3)), 3, 1+i)
        plot_confusion(trues[i], preds[i], ax, average_window=100)
        if names: ax.set_title(names[i])
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right")
    
def plot_rul_subplots(trues, preds, average_window=100, names=None, splits=None):
    """ Plots a set of RUL subplots for multiple datasets """
    size = len(trues)
    fig = plt.figure(figsize=(8.27, 11.69))
    for i in range(size):
        ax = fig.add_subplot(int(np.ceil(size/3)), 3, 1+i)
        ruls = plot_rul(trues[i], preds[i], ax, average_window=100, split=splits[i])
        if names: ax.set_title(names[i])
        if splits: ax.axvline(splits[i]*10, color="black", linestyle="--", alpha=.5, label="RUL estimation time")
        print(f"{names[i]}: avg pred rul = {ruls[0]}, lin approx rul = {ruls[1]}")
        print(f"\t avg pred score = {ruls[2]}, lin approx score = {ruls[3]}")
        print("_____________________")
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right")

def plot_absolute_error(ytrue, ypred, ax, average_window=0):
    """ Plots the absolute HI prediction errors on an axes object """
    ytrue = ytrue.flatten()
    ypred = ypred.flatten()
    error = np.abs(ytrue-ypred)

    ax.scatter(ytrue, error,
               label="Prediction error",
               s=2,
               color='red')
    if average_window:
        avg_error = windowed_average(error, average_window)
        ax.plot(ytrue[average_window//2:-average_window//2+1], avg_error, label="Error MA")
    ax.set_xlim(max(ytrue), min(ytrue))
    ax.set_ylabel("Absolute error [hrs]")
    ax.set_xlabel("Remaining lifetime [hrs]")
    ax.grid()

if __name__ == '__main__':
    # for testing
    import numpy as np
    ytrue = np.linspace(0, .8, 800)
    ypred = np.linspace(0, .8, 800) + np.random.normal(scale=.05, size=800) +\
            .1*np.sin(np.linspace(0, 4*np.pi, 800))
    p = lstsq_hi(ypred)
    print(p)
