
#1D-CNN+LSTM
model = keras.Sequential([
         tf.keras.layers.InputLayer(input_shape=(winlen,channels)),
        tf.keras.layers.Conv1D(filters=20, kernel_size=8),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(filters=40, kernel_size=6),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(filters=80, kernel_size=4),
        tf.keras.layers.MaxPooling1D(2),

        tf.keras.layers.Dense(100,activation='swish'),

        tf.keras.layers.LSTM(20,return_sequences = True,dropout = 0.5),
        tf.keras.layers.LSTM(30,return_sequences = True,dropout = 0.5),
        tf.keras.layers.LSTM(40,dropout = 0.3),


        tf.keras.layers.Dense(40,activation='swish'),
        tf.keras.layers.Dense(30,activation='swish'),
        tf.keras.layers.Dense(20,activation='swish'),

        tf.keras.layers.Dense(8,activation = 'softmax')
    ])


# 1CNN
model = keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(250,5)),
        tf.keras.layers.Conv1D(filters=20, kernel_size=8,activation='swish'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(filters=40, kernel_size=6,activation='swish'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(filters=60, kernel_size=4,activation='swish'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(filters=80, kernel_size=2,activation='swish'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),


        tf.keras.layers.Dense(240,activation='swish'),
        tf.keras.layers.Dense(120,activation='swish'),
        tf.keras.layers.Dense(60,activation='swish'),
        tf.keras.layers.Dense(30,activation='swish'),
        tf.keras.layers.Dense(15,activation='swish'),

        tf.keras.layers.Dense(,activation = )
    ])


# LSTM
model = keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(250,5)),

        tf.keras.layers.LSTM(32,return_sequences=True,activation='tanh',dropout = 0.5),
        tf.keras.layers.LSTM(64,activation='tanh',dropout = 0.5),


        tf.keras.layers.Dense(60,activation='swish'),
        tf.keras.layers.Dense(30,activation='swish'),
        tf.keras.layers.Dense(15,activation='swish'),

        tf.keras.layers.Dense(,activation =)
    ])




#REGRESS 2D-CNN
model = keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(250,5)),
        tf.keras.layers.Reshape((250,5,1)),
        tf.keras.layers.Conv2D(20,(8,1)),
        tf.keras.layers.MaxPool2D((2,1)),
        tf.keras.layers.Conv2D(40,(4,2)),
        tf.keras.layers.MaxPool2D((2,1)),
        tf.keras.layers.Conv2D(60,(2,2)),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Dropout(0.5),


        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(240,activation='swish'),
        tf.keras.layers.Dense(120,activation='swish'),
        tf.keras.layers.Dense(60,activation='swish'),
        tf.keras.layers.Dense(30,activation='swish'),
        tf.keras.layers.Dense(15,activation='swish'),

        tf.keras.layers.Dense(,activation = )
    ])

model.compile(loss= tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

#2D-CNN+LSTM
model = keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(250,5)),
        tf.keras.layers.Reshape((250,5,1)),
        tf.keras.layers.Conv2D(20,(8,1)),
        tf.keras.layers.MaxPool2D((2,1)),
        tf.keras.layers.Conv2D(40,(4,2)),
        tf.keras.layers.MaxPool2D((2,2)),

        tf.keras.layers.Reshape((59,80)),

        tf.keras.layers.LSTM(20,return_sequences = True,dropout = 0.5),
        tf.keras.layers.LSTM(30,return_sequences = True,dropout = 0.5),
        tf.keras.layers.LSTM(40,dropout = 0.3),

        tf.keras.layers.Dense(40,activation='swish'),
        tf.keras.layers.Dense(30,activation='swish'),
        tf.keras.layers.Dense(20,activation='swish'),

        tf.keras.layers.Dense(,activation = )
    ])

model.compile(loss= tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

