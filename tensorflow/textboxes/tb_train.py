import numpy as np
import matplotlib.pyplot as plt
import keras
import os
import time
import pickle

from tb_model import TB300
from ssd_utils import PriorUtil
from ssd_utils import load_weights
from ssd_data import InputGenerator
from ssd_training import SSDLoss, LearningRateDecay, Logger

from data_icdar2015fst import GTUtility
gt_util_train = GTUtility('data/ICDAR2015_FST/')
gt_util_val = GTUtility('data/ICDAR2015_FST/', test=True)

model = TB300()

prior_util = PriorUtil(model)

initial_epoch = 0

weights_path = '~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
layer_list = [('block1_conv1', 'conv1_1'),
              ('block1_conv2', 'conv1_2'),
              ('block2_conv1', 'conv2_1'),
              ('block2_conv2', 'conv2_2'),
              ('block3_conv1', 'conv3_1'),
              ('block3_conv2', 'conv3_2'),
              ('block3_conv3', 'conv3_3'),
              ('block4_conv1', 'conv4_1'),
              ('block4_conv2', 'conv4_2'),
              ('block4_conv3', 'conv4_3'),
              ('block5_conv1', 'conv5_1'),
              ('block5_conv2', 'conv5_2'),
              ('block5_conv3', 'conv5_3')]
#load_weights(model, weights_path, layer_list)

weights_path = 'ssd300_voc_weights_fixed.hdf5'
weights_path = 'checkpoints/201710132146_tb300_synthtext_horizontal10/weights.004.h5'; initial_epoch = 5
weights_path = 'checkpoints/201710141431_tb300_synthtext_horizontal10/weights.019.h5'; initial_epoch = 20

load_weights(model, weights_path)

freeze = ['conv1_1', 'conv1_2',
          'conv2_1', 'conv2_2',
          'conv3_1', 'conv3_2', 'conv3_3',
          #'conv4_1', 'conv4_2', 'conv4_3',
          #'conv5_1', 'conv5_2', 'conv5_3',
         ]

# TextBoxes paper
# Momentum 0.9, weight decay 5e-4
# lerning rate initially set to 1e−3 and decayed to 1e−4 after 40k iterations
# SynthText for 50k iterations, finetune on ICDAR 2013 for 2k iterations

experiment = 'tb300_synthtext_horizontal10'
#experiment = 'tb300_icdar'

epochs = 100
batch_size = 32

gen_train = InputGenerator(gt_util_train, prior_util, batch_size, model.image_size, augmentation=True,
                           hflip_prob=0.0, vflip_prob=0.0, do_crop=False)
gen_val = InputGenerator(gt_util_val, prior_util, batch_size, model.image_size, augmentation=True,
                         hflip_prob=0.0, vflip_prob=0.0, do_crop=False)

# freeze layers
for l in model.layers:
    l.trainable = not l.name in freeze

checkdir = './checkpoints/' + time.strftime('%Y%m%d%H%M') + '_' + experiment
if not os.path.exists(checkdir):
    os.makedirs(checkdir)

with open(checkdir+'/source.py','wb') as f:
    source = ''.join(['# In[%i]\n%s\n\n' % (i, In[i]) for i in range(len(In))])
    f.write(source.encode())

#optim = keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=0, nesterov=True)
optim = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# weight decay
regularizer = keras.regularizers.l2(5e-4) # None if disabled
for l in model.layers:
    if l.__class__.__name__.startswith('Conv'):
        l.kernel_regularizer = regularizer

loss = SSDLoss(alpha=1.0, neg_pos_ratio=3.0)

model.compile(optimizer=optim, loss=loss.compute, metrics=loss.metrics)

history = model.fit_generator(
        gen_train.generate(), # generator
        steps_per_epoch=gen_train.num_batches, # steps_per_epoch,
        epochs=epochs,
        verbose=1,
        callbacks=[
            keras.callbacks.ModelCheckpoint(checkdir+'/weights.{epoch:03d}.h5', verbose=1, save_weights_only=True),
            Logger(checkdir),
             # learning rate decay usesd with sgd
             #LearningRateDecay(methode='linear', base_lr=1e-3, n_desired=40000, desired=0.1, bias=0.0, minimum=0.1)
        ],
        validation_data=gen_val.generate(),
        validation_steps=gen_val.num_batches,
        class_weight=None,
        #max_queue_size=10,
        workers=1,
        #use_multiprocessing=False,
        initial_epoch=initial_epoch)

_, inputs, images, data = gt_util_val.sample_random_batch(batch_size=16, input_size=model.image_size)

preds = model.predict(inputs, batch_size=1, verbose=1)

for i in range(3):
    res = prior_util.decode(preds[i], confidence_threshold=0.6, keep_top_k=100)
    if len(data[i]) > 0:
        plt.figure(figsize=[10]*2)
        plt.imshow(images[i])
        prior_util.plot_results(res, classes=gt_util.classes, show_labels=True, gt_data=data[i])
        plt.show()