import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
import pywt

#%%Stałe
window_size = 200
dwt_level = 4
dec_len = 28 #sym14
sampling_frequency = 100
#%% Wczytanie danych
alldata = np.genfromtxt('t1.ASC', delimiter = ';', skip_header=8 , autostrip = True)[:,1:-1]
test = alldata[3000:3200,4:]
testx = test[:,1:]
testy = test[:,0]
print(testx.shape)
print(testy.shape)

#%% Funkcje do preprocessingu


def WaveletLength(window_size, dwt_level, dec_len): 
    Wl = np.floor((window_size + dec_len - 1) / 2)
    for i in range(0,dwt_level-1):
        Wl = int((Wl + dec_len - 1) / 2)
    return Wl

def Standardize(data):
    for i in range(0,data.shape[1]):
        data[:,i] = (data[:,i] - np.mean(data[:,i]))/np.std(data[:,i])
    return data

def DWT(data, window_size, dwt_level):
    wl = WaveletLength(window_size, dwt_level, dec_len)
    output = np.zeros([wl,data.shape[1]*2])
    i = 0
    j = 0
    while i < data.shape[1]:
        output[:,j], output[:,j+1],_,_,_ = pywt.wavedec(data[:,i], pywt.Wavelet('sym14'), mode='sym', level=4)
        j += 2
        i += 1
    return output


def ExtractFeatures(data, window_size, sampling_frequency, dwt_level):
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
        print(power_spectrum)
        mask_negative = freqs >= 0
        freqs = freqs[mask_negative]
        power_spectrum = power_spectrum[mask_negative]
        
        features[3,i] = np.sum(freqs * power_spectrum)/np.sum(power_spectrum) #MNF
        features[4,i] = np.mean(power_spectrum) #MNP
        features[5,i] = np.sum(power_spectrum)/2
    plt.plot(freqs,power_spectrum)
    return features

#TODO
#funkcje do filtracji

#%%
class TSGenerator(Sequence):
    def __init__(self, data, window_size, batch_size, step_size=1):
        self.data = data
        self.window_size = window_size
        self.batch_size = batch_size
        self.step_size = step_size
        self.indices = np.arange(0, len(data) - window_size, step_size)
    
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = np.array([self.data[i:i + self.window_size] for i in batch_indices])
        
        
        #dobrze przyporządkować kanały i wgl 
        input_data = np.array([sample[:, :5] for sample in batch_data])
        labels = np.array([np.mean(sample[:, -1]) for sample in batch_data])
        
        return input_data, labels
    
    def on_epoch_end(self):
        pass
    
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=4, **kwargs)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
    def call(self, x):
       attn_output = self.mha(query=x, value=x)
       attn_output = self.dropout(attn_output)
       x = self.add([x, attn_output])
       x = self.layernorm(x)
       return x
   
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.conv1 = tf.keras.layers.Conv1D(filters=3, kernel_size=1, activation = 'relu')
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.conv2 = tf.keras.layers.Conv1D(filters=5, kernel_size=1)
        self.add = tf.keras.layers.Add()
    def call(self, x):
        out = self.layernorm(x)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        x = self.add([x, out])
        return x
    
class MLP(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__()
        self.mlp = tf.keras.layers.Dense(units, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.1)
    def call(self, x):
        x = self.mlp(x)
        x = self.dropout(x)
        return x
       
class Regression_block(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mlp1 = MLP(128)
        self.mlp2 = MLP(128)
        self.final = tf.keras.layers.Dense(1, activation='relu')
        self.GAP = tf.keras.layers.GlobalAveragePooling1D()
    def call(self, x):
        x = self.GAP(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.final(x)
        return x
    
class Transformer_block(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = SelfAttention(256)
        self.ff = FeedForward()
    def call(self, x):
        x = self.mha(x)
        x = self.ff(x)
        return x

#%%
input_layer = tf.keras.Input(shape=(200,5))
transformer1 = Transformer_block()(input_layer)
transformer2 = Transformer_block()(transformer1)
transformer3 = Transformer_block()(transformer2)
transformer4 = Transformer_block()(transformer3)
output_layer = Regression_block()(transformer4)
#%%

model = tf.keras.Model(input_layer, output_layer)
model.summary()
model.compile(optimizer='rmsprop', loss='mean_squared_error')
#%%
train_generator = TSGenerator(data, window_size = 200, batch_size = 32, step_size=20)
model.fit(train_generator, epochs=10)
#%%
tdata = test[:,:5]
tlabel = np.mean(test[:,-1])

#%%
model.predict(np.expand_dims(tdata, 0))



