import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

trX = np.linspace(-1, 1, 101)
trY = 3 * trX + np.random.randn(*trX.shape) * 0.33

print(trY)