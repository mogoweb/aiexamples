from keras.datasets import mnist
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np

np.random.seed(1671)

NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10  # 输出类别数量
OPTIMIZER = SGD()  # SGD优化器
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2  # 训练集中用于验证集的比例

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train是60000个28*28的数据，窄化为60000*784
RESHAPE = 784

X_train = X_train.reshape(60000, RESHAPE)
X_test = X_test.reshape(10000, RESHAPE)
X_train = X_train.astype('float32')
Y_test = X_test.astype('float32')

# 归一化
X_train = X_train / 255
X_test = X_test / 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 将类向量转换为二值类别矩阵
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)
print(Y_train.shape, 'train lables')

# N_HIDDEN个隐藏层
# 10个softmax输出
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPE, )))
model.add(Activation('relu'))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,
                    validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("Test score:", score[0])
print("Test accuracy:", score[1])
