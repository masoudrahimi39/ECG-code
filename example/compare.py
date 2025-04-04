import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.spatial.distance import euclidean
from scipy.signal import correlate
from scipy.stats import pearsonr

def compute_correlation(sig1, sig2, bias1, bias2, scale1, scale2, lag):
    signal1 = sig1 - bias1
    signal2 = sig2 - bias2
    signal1 = signal1 * scale1
    signal2 = signal2 * scale2
    if lag < 0:
        signal1 = signal1[-lag:]
        signal2 = signal2[:lag]
    else:
        signal1 = signal1[:-lag]
        signal2 = signal2[lag:]
    
    valid = signal2<np.quantile(signal2, 0.99)
    signal1 = signal1[valid]
    signal2 = signal2[valid]
    # signal1[signal1>np.quantile(signal1, 0.95)] = 2
    # signal1[signal1<np.quantile(signal1, 0.05)] = -2
    # signal2[signal2>np.quantile(signal2, 0.95)] = 2
    # signal2[signal2<np.quantile(signal2, 0.05)] = -2
    return pearsonr(signal1, signal2)[0]
    # return np.corrcoef(signal1, signal2)[0,1]

def improve_bias(signal, json, bias_signal, bias_json, scale_signal, scale_json, lag, search_area):
    # Try different bias values around the current bias to improve correlation
    best_correlation = compute_correlation(signal, json, bias_signal, bias_json, scale_signal, scale_json, lag)
    best_bias = bias_signal
    
    # Search in a range around current bias with decreasing step sizes
    for bias_new in np.linspace(bias_signal+search_area, bias_signal-search_area, 10):
        correlation = compute_correlation(signal, json, bias_new, bias_json, scale_signal, scale_json, lag)
        if correlation > best_correlation:
            best_correlation = correlation
            best_bias = bias_new
            
    return best_bias, best_correlation

def improve_scale(signal, json, bias_signal, bias_json, scale_signal, scale_json, lag, search_area):
    # Try different scale values around the current scale to improve correlation
    best_correlation = compute_correlation(signal, json, bias_signal, bias_json, scale_signal, scale_json, lag)
    best_scale = scale_signal
    
    for scale_new in np.linspace(scale_signal+search_area, scale_signal-search_area, 10):
        correlation = compute_correlation(signal, json, bias_signal, bias_json, scale_new, scale_json, lag)
        if correlation > best_correlation:
            best_correlation = correlation
            best_scale = scale_new
            
    return best_scale, best_correlation

def improve_lag(signal, json, bias_signal, bias_json, scale_signal, scale_json):
    lag = np.argmax(correlate(
        (json-bias_json)*scale_json,
        (signal-bias_signal)*scale_signal)) \
            - (len(signal) - 1)
    return lag

def visualize_signal(signal, json, bias_signal, bias_json, scale_signal, scale_json, lag, save_path):
    correlation = compute_correlation(signal, json, bias_signal, bias_json, scale_signal, scale_json, lag)
    
    sig1 = (signal-bias_signal)*scale_signal
    sig2 = (json-bias_json)*scale_json

    if lag < 0:
        sig1 = sig1[-lag:]
        sig2 = sig2[:lag]
    else:
        sig1 = sig1[:-lag]
        sig2 = sig2[lag:]

    plt.plot(sig2, label='real')
    plt.plot(sig1, label='predicted')
    plt.title(f'Correlation: {correlation}')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()


# Read the signal from app.txt
with open('example/ex2/app.txt', 'r') as f:
    signal = [float(line.strip()) for line in f.readlines()]

# Read the JSON file
with open('example/ex2/0_3by1_lead_0.json', 'r') as f:
    json_data = json.load(f)

signal_array = np.array(signal)
json_array = np.array(json_data)

new_indices = np.linspace(0, len(signal_array)-1, len(json_array))
signal_array = np.interp(new_indices, np.arange(len(signal_array)), signal_array)

# find the initial biases, scales, and drifts
bias_signal = np.mean(signal_array)
bias_json = np.mean(json_array)

scale_signal = 1/np.quantile(np.abs(signal_array-bias_signal), 0.9)
scale_json = 1/np.quantile(np.abs(json_array-bias_json), 0.9)

lag = improve_lag(signal_array, json_array, bias_signal, bias_json, scale_signal, scale_json)

correlation = compute_correlation(signal_array, json_array, bias_signal, bias_json, scale_signal, scale_json, lag)

visualize_signal(signal_array, json_array, bias_signal, bias_json, scale_signal, scale_json, lag, f'example/tmp_.png')

print(f"Initial Values")
print(f"Bias: {bias_signal}, Scale: {scale_signal}, Lag: {lag}, Correlation: {correlation}")

for i in range(20):
    search_scale = 10**(-i/10)
    bias_signal, correlation = improve_bias(signal_array, json_array, bias_signal, bias_json, scale_signal, scale_json, lag, search_scale*bias_signal)
    scale_signal, correlation = improve_scale(signal_array, json_array, bias_signal, bias_json, scale_signal, scale_json, lag, search_scale*scale_signal)
    lag = improve_lag(signal_array, json_array, bias_signal, bias_json, scale_signal, scale_json)
    correlation = compute_correlation(signal_array, json_array, bias_signal, bias_json, scale_signal, scale_json, lag)
    print(f"Iteration {i+1}")
    print(f"Bias: {bias_signal}, Scale: {scale_signal}, Lag: {lag}, Correlation: {correlation}")

    visualize_signal(signal_array, json_array, bias_signal, bias_json, scale_signal, scale_json, lag, f'example/tmp_{i}.png')
