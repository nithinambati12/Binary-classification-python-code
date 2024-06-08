Python 3.11.0 (main, Oct 24 2022, 18:26:48) [MSC v.1933 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> # Heart Attack Analytics Prediction Using Binary Classification
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from sklearn.model_selection import train_test_split
>>> import sklearn
>>> import pandas as pd
>>> import keras
2024-06-08 13:42:15.381194: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-06-08 13:42:16.748946: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
>>> from keras.models import Sequential
>>> from keras.layers import Dense
>>> import tensorflow as tf
>>> from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
>>> from sklearn.preprocessing import MinMaxScaler
>>> df = pd.read_csv("C:/Users/nithi/OneDrive/Documents/data sets/heart.csv")
>>> df.head()
   age  sex  cp  trtbps  chol  fbs  restecg  thalachh  exng  oldpeak  slp  caa  thall  output
0   63    1   3     145   233    1        0       150     0      2.3    0    0      1       1
1   37    1   2     130   250    0        1       187     0      3.5    0    0      2       1
2   41    0   1     130   204    0        0       172     0      1.4    2    0      2       1
3   56    1   1     120   236    0        1       178     0      0.8    2    0      2       1
4   57    0   0     120   354    0        1       163     1      0.6    2    0      2       1
>>> target_column = "output"
>>> numerical_column = df.columns.drop(target_column)
>>> output_rows = df[target_column]
>>> df.drop(target_column,axis=1,inplace=True)
>>> scaler = MinMaxScaler()
>>> scaler.fit(df)
MinMaxScaler()
>>> t_df = scaler.transform(df)
>>> X_train, X_test, y_train, y_test = train_test_split(t_df, output_rows, test_size=0.25, random_state=0)
>>> print('X_train:',np.shape(X_train))
X_train: (227, 13)
>>> print('y_train:',np.shape(y_train))
y_train: (227,)
>>> print('X_test:',np.shape(X_test))
X_test: (76, 13)
>>> print('y_test:',np.shape(y_test))
y_test: (76,)
>>> basic_model = Sequential()
>>> basic_model.add(Dense(units=16, activation='relu', input_shape=(13,)))
C:\Users\nithi\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\layers\core\dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
2024-06-08 13:44:17.397938: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized touse available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
>>> basic_model= sequential()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'sequential' is not defined. Did you mean: 'Sequential'?
>>> basic_model.add(Dense(units=16, activation='relu', input(shape=(13,))))
  File "<stdin>", line 1
    basic_model.add(Dense(units=16, activation='relu', input(shape=(13,))))
                                                                         ^
SyntaxError: positional argument follows keyword argument
>>> basic_model = Sequential()
>>> basic_model.add(Dense(units=16, activation='relu', input_shape=(13,)))
>>> basic_model.add(Dense(1, activation='sigmoid'))
>>> adam = keras.optimizers.Adam(learning_rate=0.001)
>>> basic_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])
>>> basic_model.fit(X_train, y_train, epochs=100)
Epoch 1/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.5579 - loss: 0.6529
Epoch 2/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.6061 - loss: 0.6407
Epoch 3/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6662 - loss: 0.6356
Epoch 4/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7553 - loss: 0.6178
Epoch 5/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.7129 - loss: 0.6356
Epoch 6/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7662 - loss: 0.6067
Epoch 7/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7616 - loss: 0.6025
Epoch 8/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.7957 - loss: 0.5896
Epoch 9/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.7833 - loss: 0.5821
Epoch 10/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7939 - loss: 0.5796
Epoch 11/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7747 - loss: 0.5637
Epoch 12/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.8156 - loss: 0.5605
Epoch 13/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8037 - loss: 0.5607
Epoch 14/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8190 - loss: 0.5501
Epoch 15/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8284 - loss: 0.5410
Epoch 16/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8140 - loss: 0.5452
Epoch 17/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8280 - loss: 0.5373
Epoch 18/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8256 - loss: 0.5322
Epoch 19/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7838 - loss: 0.5387
Epoch 20/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8235 - loss: 0.5171
Epoch 21/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8064 - loss: 0.5164
Epoch 22/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8420 - loss: 0.4992
Epoch 23/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.8539 - loss: 0.4894
Epoch 24/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step - accuracy: 0.8132 - loss: 0.5067
Epoch 25/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.8281 - loss: 0.4803
Epoch 26/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8409 - loss: 0.4824
Epoch 27/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8275 - loss: 0.4865
Epoch 28/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8549 - loss: 0.4737
Epoch 29/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8180 - loss: 0.4823
Epoch 30/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8109 - loss: 0.4834
Epoch 31/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8484 - loss: 0.4559
Epoch 32/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8411 - loss: 0.4627
Epoch 33/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8446 - loss: 0.4329
Epoch 34/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8598 - loss: 0.4360
Epoch 35/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8606 - loss: 0.4125
Epoch 36/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8351 - loss: 0.4386
Epoch 37/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8266 - loss: 0.4300
Epoch 38/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8392 - loss: 0.4319
Epoch 39/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8600 - loss: 0.3974
Epoch 40/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.8697 - loss: 0.3908
Epoch 41/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.8180 - loss: 0.4402
Epoch 42/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.8508 - loss: 0.4163
Epoch 43/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8456 - loss: 0.3969
Epoch 44/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8499 - loss: 0.3917
Epoch 45/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8342 - loss: 0.3947
Epoch 46/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8405 - loss: 0.4155
Epoch 47/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8315 - loss: 0.4250
Epoch 48/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8492 - loss: 0.3872
Epoch 49/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8486 - loss: 0.3809
Epoch 50/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8524 - loss: 0.3834
Epoch 51/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8521 - loss: 0.3953
Epoch 52/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8252 - loss: 0.4044
Epoch 53/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8720 - loss: 0.3598
Epoch 54/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step - accuracy: 0.8473 - loss: 0.3952
Epoch 55/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8674 - loss: 0.3605
Epoch 56/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.8440 - loss: 0.3824
Epoch 57/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step - accuracy: 0.8539 - loss: 0.3550
Epoch 58/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8580 - loss: 0.3811
Epoch 59/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step - accuracy: 0.8415 - loss: 0.3807
Epoch 60/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8550 - loss: 0.3784
Epoch 61/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8364 - loss: 0.3848
Epoch 62/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step - accuracy: 0.8294 - loss: 0.4020
Epoch 63/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8503 - loss: 0.3519
Epoch 64/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8380 - loss: 0.3876
Epoch 65/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 844us/step - accuracy: 0.8418 - loss: 0.3602
Epoch 66/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8569 - loss: 0.3614
Epoch 67/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step - accuracy: 0.8462 - loss: 0.3815
Epoch 68/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8784 - loss: 0.3669
Epoch 69/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8555 - loss: 0.3529
Epoch 70/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step - accuracy: 0.8336 - loss: 0.3706
Epoch 71/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8469 - loss: 0.3715
Epoch 72/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8262 - loss: 0.4114
Epoch 73/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8559 - loss: 0.3534
Epoch 74/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8523 - loss: 0.3563
Epoch 75/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8603 - loss: 0.3569
Epoch 76/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8628 - loss: 0.3682
Epoch 77/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8426 - loss: 0.3894
Epoch 78/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8597 - loss: 0.3666
Epoch 79/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.8408 - loss: 0.3701
Epoch 80/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8477 - loss: 0.3359
Epoch 81/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8881 - loss: 0.3367
Epoch 82/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step - accuracy: 0.8404 - loss: 0.3662
Epoch 83/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8532 - loss: 0.3587
Epoch 84/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8610 - loss: 0.3594
Epoch 85/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8443 - loss: 0.3682
Epoch 86/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8497 - loss: 0.3763
Epoch 87/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8433 - loss: 0.3721
Epoch 88/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8566 - loss: 0.3440
Epoch 89/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8608 - loss: 0.3551
Epoch 90/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8205 - loss: 0.3878
Epoch 91/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step - accuracy: 0.8581 - loss: 0.3476
Epoch 92/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8549 - loss: 0.3501
Epoch 93/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8368 - loss: 0.3660
Epoch 94/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8593 - loss: 0.3540
Epoch 95/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8514 - loss: 0.3479
Epoch 96/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8099 - loss: 0.3983
Epoch 97/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step - accuracy: 0.8838 - loss: 0.3167
Epoch 98/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8618 - loss: 0.3245
Epoch 99/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8415 - loss: 0.3623
Epoch 100/100
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8521 - loss: 0.3305
<keras.src.callbacks.history.History object at 0x000001DCF13F5B50>
>>> loss_and_metrics = basic_model.evaluate(X_test, y_test)
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step - accuracy: 0.8337 - loss: 0.4097
>>> print(loss_and_metrics)
[0.40885743498802185, 0.8157894611358643]
>>> print('Loss = ',loss_and_metrics[0])
Loss =  0.40885743498802185
>>> print('Accuracy = ',loss_and_metrics[1])
Accuracy =  0.8157894611358643
>>> predicted = basic_model.predict(X_test)
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
>>> predicted = tf.squeeze(predicted)
>>> predicted = np.array([1 if x >= 0.5 else 0 for x in predicted])
>>> actual = np.array(y_test)
>>> conf_mat = confusion_matrix(actual, predicted)
>>> displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
>>> displ.plot()
<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay object at 0x000001DCF15B2FD0>
>>> plt.show()
>>>
