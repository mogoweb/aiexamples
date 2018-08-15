import matplotlib.pyplot as plt
import keras
from keras.datasets import boston_housing
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import imdb
from keras.datasets import reuters
from termcolor import colored


def show_shapes(x_train, y_train, x_test, y_test, color='green'):
    print(colored('Training shape:', color, attrs=['bold']))
    print('  x_train.shape:', x_train.shape)
    print('  y_train.shape:', y_train.shape)
    print(colored('\nTesting shape:', color, attrs=['bold']))
    print('  x_test.shape:', x_test.shape)
    print('  y_test.shape:', y_test.shape)


def plot_data(my_data, cmap=None):
    plt.axis('off')
    fig = plt.imshow(my_data, cmap=cmap)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    print(fig)


def show_sample(x_train, y_train, idx=0, color='blue'):
    print(colored('x_train sample:', color, attrs=['bold']))
    print(x_train[idx])
    print(colored('\ny_train sample:', color, attrs=['bold']))
    print(y_train[idx])


def show_sample_image(x_train, y_train, idx=0, color='blue', cmap=None):
    print(colored('Label:', color, attrs=['bold']), y_train[idx])
    print(colored('Shape:', color, attrs=['bold']), x_train[idx].shape)
    print()
    plot_data(x_train[idx], cmap=cmap)


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
show_shapes(x_train, y_train, x_test, y_test)
print("\n**********************************\n")
show_sample(x_train, y_train)


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
show_shapes(x_train, y_train, x_test, y_test)
print("\n**********************************\n")
show_sample_image(x_train, y_train)


(x_train, y_train), (x_test, y_test) = cifar100.load_data()
show_shapes(x_train, y_train, x_test, y_test)
print("\n**********************************\n")
show_sample_image(x_train, y_train)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
show_shapes(x_train, y_train, x_test, y_test)
print("\n**********************************\n")
show_sample_image(x_train, y_train, cmap='gray')


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
show_shapes(x_train, y_train, x_test, y_test)
print("\n**********************************\n")
show_sample_image(x_train, y_train, idx=100)


(x_train, y_train), (x_test, y_test) = imdb.load_data()
show_shapes(x_train, y_train, x_test, y_test)
print("\n**********************************\n")
show_sample(x_train, y_train)


(x_train, y_train), (x_test, y_test) = reuters.load_data()
show_shapes(x_train, y_train, x_test, y_test)
print("\n**********************************\n")
show_sample(x_train, y_train)


