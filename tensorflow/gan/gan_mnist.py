import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import imageio

initializer = xavier_initializer()


# 为生成器生成随机噪声
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
Z_dim = 100

# 生成器参数设置
G_W1 = tf.Variable(initializer([100, 128]), name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')
G_W2 = tf.Variable(initializer([128, 784]), name='G_W2')
G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')
theta_G = [G_W1, G_W2, G_b1, G_b2]


# 生成器网络
def generator(z):
  G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
  G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
  G_prob = tf.nn.sigmoid(G_log_prob)

  return G_prob

# 为判别器准备的MNIST图像输入设置
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')

# 判别器参数设置
D_W1 = tf.Variable(initializer(shape=[784, 128]), name='D_W1')
D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')
D_W2 = tf.Variable(initializer(shape=[128, 1]), name='D_W2')
D_b2 = tf.Variable(tf.zeros(shape=[1]), name="D_W2")
theta_D = [D_W1, D_W2, D_b1, D_b2]


# 判别器网络
def discriminator(x):
  D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
  D_logit = tf.matmul(D_h1, D_W2) + D_b2
  D_prob = tf.nn.sigmoid(D_logit)

  return D_prob, D_logit


G_sample = generator(Z)

D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# GAN原始论文中的损失函数
D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

# 仅更新D(X)的参数, var_list=theta_D
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
# 仅更新G(X)的参数, var_list=theta_G
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


def sample_Z(m, n):
  return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
  fig = plt.figure(figsize=(4, 4))
  gs = gridspec.GridSpec(4, 4)
  gs.update(wspace=0.05, hspace=0.05)

  for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

  return fig


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
mb_size = 100

if not os.path.exists('output/'):
    os.makedirs('output/')

i = 0
for it in range(60000):
  if it % 10000 == 0:
    samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

    fig = plot(samples)
    plt.savefig('output/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
    i += 1
    plt.close(fig)

  X_mb, _ = mnist.train.next_batch(mb_size)

  _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
  _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

  if it % 1000 == 0:
    print('Iter: {}'.format(it))
    print('D loss: {:.4}'.format(D_loss_curr))
    print('G_loss: {:.4}'.format(G_loss_curr))
    print()

images = []
for file_name in os.listdir('output'):
    if file_name.endswith('.png'):
        file_path = os.path.join('output', file_name)
        images.append(imageio.imread(file_path))

imageio.mimsave(os.path.join("output", 'samples.gif'), images, fps=1)
