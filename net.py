import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
import pywt
from scipy.signal import butter, freqz, filtfilt


window_size = 200
dec_len = 28 #sym14
sampling_frequency = 2000


def WaveletLength(window_size, dwt_level, dec_len): 
    Wl = np.floor((window_size + dec_len - 1) / 2)
    for i in range(0,dwt_level-1):
        Wl = int((Wl + dec_len - 1) / 2)
    return Wl

def Standardize(data):
    if np.std(data) != 0:
            data = (data - np.mean(data))/np.std(data)
    else:
            data = data - np.mean(data)
    return data 

def Normalize(data):
    data = (data - np.min(data))/(np.max(data) - np.min(data))
    return data

def DWT(data, window_size, dwt_level=4):
    wl = WaveletLength(window_size, dwt_level, dec_len)
    output = np.zeros([wl,data.shape[1]*2])
    i = 0
    j = 0
    while i < data.shape[1]:
        output[:,j], output[:,j+1],_,_,_ = pywt.wavedec(data[:,i], pywt.Wavelet('sym14'), mode='sym', level=dwt_level)
        j += 2
        i += 1
    return output


def ExtractFeatures(data, window_size, sampling_frequency, dwt_level=4):
    #dwt_level =0 -> nie ma transformaty falkowej i okna długosci takiej jak na poczatku
    if dwt_level !=0:
        wl = WaveletLength(window_size, dwt_level, dec_len)
    else:
        wl = window_size
        
    features = np.zeros([6,data.shape[1]])
    for i in range(0, data.shape[1]):
        features[0,i] = np.mean(np.abs(data[:,i])) #MAV
        features[1,i] = np.mean(data[:,i]**2) #VAR
        features[2,i] = np.sqrt(features[1,i]) #RMS
        
        power_spectrum = np.abs(np.fft.fft(data[:,i]))** 2
        freqs = np.fft.fftfreq(wl, 1/sampling_frequency)
        mask_negative = freqs >= 0
        freqs = freqs[mask_negative]
        power_spectrum = power_spectrum[mask_negative]
        
        features[3,i] = np.sum(freqs * power_spectrum)/np.sum(power_spectrum) #MNF
        features[4,i] = np.mean(power_spectrum) #MNP
        features[5,i] = np.sum(power_spectrum)/2
    return features

 # Ensure each sample has 200 samples
def pad_to_length(arr, target_length=200):
    if arr.shape[0] < target_length:
        padding = np.zeros((target_length - arr.shape[0], arr.shape[1]))
        return np.vstack((arr, padding))
    else:
        return arr[:target_length, :]  # Trim if longer than 200

#Filtering data 

# Creating  3-rd order butterworth filter inside generator 
low = 10 
high = 500  



class TSGenerator(Sequence):
    def __init__(self, data, batch_size=32, window_size=200, sampling_frequency=2000, order=3, step_size=20):
        self.data = data
        self.sampling_frequency = sampling_frequency
        nyq = 0.5 * self.sampling_frequency
        self.b, self.a =  butter(order, [low/nyq, high/nyq], btype='band')
        self.window_size = window_size
        self.batch_size = batch_size
        self.step_size = step_size
        self.indices = np.arange(0, len(data) - window_size, step_size)
    
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = np.array([pad_to_length(self.data[i:i + self.window_size],target_length=self.window_size) for i in batch_indices])
        
        #Channels 
        input_data = np.array([sample[:, :10] for sample in batch_data])
        labels = np.array([np.mean(sample[:, 10:32], axis=0) for sample in batch_data])
        for q in range(0, labels.shape[0]):
            labels[q] = Standardize(np.array(labels[q]))
            labels[q] = Normalize(labels[q])

        padlen = 3* max(len(self.b),len(self.a)) - 1

        if input_data.shape[1] > padlen:
            tab1 = np.zeros([len(batch_indices), 200, 10])
            tab2 = np.zeros([len(batch_indices),37,20])
            tab3 = np.zeros([len(batch_indices), 6, 20])
            for d in range(0, tab1.shape[0]):
                tab1[d] = filtfilt(self.b, self.a, input_data[d], axis=0)
                tab2[d] = DWT(tab1[d], self.window_size)
                tab3[d] = ExtractFeatures(tab2[d], self.window_size, self.sampling_frequency)
            return tab3, labels
        else: 
            print(f"Data was too trimmed. Len: {len(input_data)}, dim: {input_data[0]}")
            return None, None
    
    def on_epoch_end(self):
        pass

class TSGenerator_no_feature(Sequence):
    def __init__(self, data, batch_size=32, window_size=200, sampling_frequency=2000, order=3, step_size=20):
        self.data = data
        self.sampling_frequency = sampling_frequency
        nyq = 0.5 * self.sampling_frequency
        self.b, self.a =  butter(order, [low/nyq, high/nyq], btype='band')
        self.window_size = window_size
        self.batch_size = batch_size
        self.step_size = step_size
        self.indices = np.arange(0, len(data) - window_size, step_size)
    
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = np.array([pad_to_length(self.data[i:i + self.window_size],target_length=self.window_size) for i in batch_indices])
        
        #Channels 
        input_data = np.array([sample[:, :10] for sample in batch_data])
        labels = np.array([np.mean(sample[:, 10:32], axis=0) for sample in batch_data])
        for q in range(0, labels.shape[0]):
            labels[q] = Standardize(np.array(labels[q]))
            labels[q] = Normalize(labels[q])

        padlen = 3* max(len(self.b),len(self.a)) - 1

        if input_data.shape[1] > padlen:
            tab1 = np.zeros([len(batch_indices), 200, 10])
            tab2 = np.zeros([len(batch_indices),37,20])
            for d in range(0, tab1.shape[0]):
                tab1[d] = filtfilt(self.b, self.a, input_data[d], axis=0)
                tab2[d] = DWT(tab1[d], self.window_size)
            return tab2, labels
        else: 
            print(f"Data was too trimmed. Len: {len(input_data)}, dim: {input_data[0]}")
            return None, None
    
    def on_epoch_end(self):
        pass

class TSGenerator_no_DWT(Sequence):
    def __init__(self, data, batch_size=32, window_size=200, sampling_frequency=2000, order=3, step_size=20):
        self.data = data
        self.sampling_frequency = sampling_frequency
        nyq = 0.5 * self.sampling_frequency
        self.b, self.a =  butter(order, [low/nyq, high/nyq], btype='band')
        self.window_size = window_size
        self.batch_size = batch_size
        self.step_size = step_size
        self.indices = np.arange(0, len(data) - window_size, step_size)
    
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = np.array([pad_to_length(self.data[i:i + self.window_size],target_length=self.window_size) for i in batch_indices])
        
        #Channels 
        input_data = np.array([sample[:, :10] for sample in batch_data])
        labels = np.array([np.mean(sample[:, 10:32], axis=0) for sample in batch_data])
        for q in range(0, labels.shape[0]):
            labels[q] = Standardize(np.array(labels[q]))
            labels[q] = Normalize(labels[q])

        padlen = 3* max(len(self.b),len(self.a)) - 1

        if input_data.shape[1] > padlen:
            tab1 = np.zeros([len(batch_indices), 200, 10])
            tab2 = np.zeros([len(batch_indices),37,20])
            tab3 = np.zeros([len(batch_indices), 6, 20])
            for d in range(0, tab1.shape[0]):
                tab1[d] = filtfilt(self.b, self.a, input_data[d], axis=0)
            return tab1, labels
        else: 
            print(f"Data was too trimmed. Len: {len(input_data)}, dim: {input_data.shape[0]} {input_data.shape[1]} {input_data.shape[2]}")
            return None, None
    
    def on_epoch_end(self):
        pass
class MLP(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.mlp = tf.keras.layers.Dense(units, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.1)
    def call(self, x):
        x = self.mlp(x)
        x = self.dropout(x)
        return x
       
class Regression_block(tf.keras.layers.Layer):
    def __init__(self, output_channels, **kwargs):
        super().__init__(**kwargs)
        self.mlp1 = MLP(128)
        self.mlp2 = MLP(128)
        self.final = tf.keras.layers.Dense(output_channels, activation='relu')
        self.GAP = tf.keras.layers.GlobalAveragePooling1D()
    def call(self, x):
        x = self.GAP(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.final(x)
        return x
    
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, input_channels, **kwargs):
        super().__init__(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=1, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.conv2 = tf.keras.layers.Conv1D(filters=input_channels, kernel_size=1)
        self.add = tf.keras.layers.Add()

    def call(self, x):
        out = self.layernorm(x)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        x = self.add([x, out])
        return x

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, attn_units, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=attn_units, num_heads=4)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        attn_output = self.dropout(attn_output)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class Transformer_block(tf.keras.layers.Layer):
    def __init__(self, attention_units, input_channels, **kwargs):
        super().__init__(**kwargs)
        self.mha = SelfAttention(attention_units)
        self.ff = FeedForward(input_channels)

    def call(self, x):
        x = self.mha(x)
        x = self.ff(x)
        return x









