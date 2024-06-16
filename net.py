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
    for i in range(0,data.shape[1]):
        data[:,i] = (data[:,i] - np.mean(data[:,i]))/np.std(data[:,i])
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
        batch_data = np.array([self.data[i:i + self.window_size] for i in batch_indices])
        
        #Channels 
        input_data = np.array([sample[:, :10] for sample in batch_data])
        labels = np.array([np.mean(sample[:, 10:32], axis=0) for sample in batch_data])
        labels = Standardize(labels)

        input_data = filtfilt(self.b, self.a, input_data, axis=0)
        tab = np.zeros([len(batch_indices),37,20])
        for d in range(0, tab.shape[0]):
            tab[d] = DWT(input_data[d], self.window_size)

        tab2 = np.zeros([len(batch_indices), 6, 20])
        for q in range(0, tab2.shape[0]):
            tab2[q]= ExtractFeatures(tab[q], self.window_size, self.sampling_frequency)
        return tab, labels
    
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


class Regression_block_embedded(tf.keras.layers.Layer):
    def __init__(self, output_channels, **kwargs):
        super().__init__(**kwargs)
        self.mlp1 = MLP(128)
        self.mlp2 = MLP(128)
        self.final = tf.keras.layers.Dense(output_channels, activation='relu')
        self.GAP = tf.keras.layers.GlobalAveragePooling2D()
    def call(self, x):
        x = self.GAP(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.final(x)
        return x

class Input_Embedding(tf.keras.layers.Layer):
    def __init__(self, seq_len=37, embedding_dim=120, channels=20):
        super(Input_Embedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.channels = channels
        self.conv = tf.keras.layers.Conv1D(embedding_dim * channels, 1, groups=channels)  # Each channel embedded separately

    def call(self, inputs):
        x = self.conv(inputs)
        x = tf.reshape(x, (-1, tf.shape(x)[1], self.channels, self.embedding_dim))
        
        return x
    
class Trig_Encoding(tf.keras.layers.Layer):
    def __init__(self, seq_len = 37, d_model = 120, channels = 20):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_encoding = self.add_weight(
            "pos_encoding",
            shape=[seq_len, channels , d_model],
            initializer="zeros",
            trainable=False
        )

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    
def positional_encoding(seq_len, d_model, channels):
    angle_rads = np.arange(seq_len)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    pos_encoding = tf.tile(tf.expand_dims(pos_encoding, axis=2), [1, 1, channels, 1])
    return tf.cast(pos_encoding, dtype=tf.float32)


class Trainable_Encoding(tf.keras.layers.Layer):
    def __init__(self, seq_len = 37, d_model = 120, channels = 20):
        super(Trainable_Encoding, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_encoding = self.add_weight(
            "pos_encoding",
            shape=[seq_len, channels , d_model],
            initializer=tf.initializers.RandomNormal(),
            trainable=True
        )

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class Embedding_Layer(tf.keras.layers.Layer):
    def __init__(self, input_len, embedding_dim, num_channels, TrainEncoding = False,**kwargs):
        super().__init__(**kwargs)

        self.input_embedding = Input_Embedding(input_len,embedding_dim,num_channels)
        if TrainEncoding:
            self.positional_encoding = Trainable_Encoding(input_len,embedding_dim,num_channels)
        else:
            self.positional_encoding = Trig_Encoding(input_len, embedding_dim,num_channels)
    def call(self, inputs):
        x = self.input_embedding(inputs)
        x = self.positional_encoding(x)
        return x
    
    
class split_input(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.hf_indices = np.arange(0,20,2)
        self.lf_indices = np.arange(1,20,2)
    def call(self,inputs):
        hf_input = tf.gather(inputs, indices=self.hf_indices, axis=-1)
        lf_input = tf.gather(inputs, indices=self.lf_indices, axis=-1)
        return(hf_input, lf_input)


#%%

trainx = np.random.normal(1, 0.5, [100*200, 10])
trainy = np.random.normal(1, 0.5, [100*200, 22])



train = np.concatenate((trainx, trainy), axis=1)
train_gen = TSGenerator(train)
a = train_gen[0]

lfembedding = Embedding_Layer(37,10,10,False)
hfembedding = Embedding_Layer(37,10,10,False)

lfembedding.set_weights(positional_encoding(37,10,10))
hfembedding.set_weights(positional_encoding(37,10,10))

input_layer = tf.keras.Input(shape=(37,20))
hf, lf = split_input()(input_layer)

hfemb = hfembedding(hf)
lfemb = lfembedding(lf)

hft1 = Transformer_block(attention_units=128,input_channels=10)(hfemb)
hft2 = Transformer_block(attention_units=128,input_channels=10)(hft1)

lft1 = Transformer_block(attention_units=128,input_channels=10)(lfemb)
lft2 = Transformer_block(attention_units=128,input_channels=10)(lft1)

concat = tf.keras.layers.Concatenate(axis=-1)([hft2,lft2])
x = tf.keras.layers.Dense(256, activation='relu')(concat)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output_layer = Regression_block_embedded(output_channels= 22)(x)

model = tf.keras.Model(input_layer, output_layer)

model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.summary()
model.fit(train_gen,epochs=1)

#%%
emb = Embedding_Layer(37, 10, 20, False)
emb.set_weights(positional_encoding(37,10,20))

input_layer = tf.keras.Input(shape=(37,20))
embedded = emb(input_layer)
transformer1 = Transformer_block(attention_units=128,input_channels=10)(embedded)
transformer2 = Transformer_block(attention_units=128,input_channels=10)(transformer1)
output_layer = Regression_block_embedded(output_channels= 22)(transformer2)

model = tf.keras.Model(input_layer, output_layer)
model.summary()
model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.fit(train_gen,epochs=1)
