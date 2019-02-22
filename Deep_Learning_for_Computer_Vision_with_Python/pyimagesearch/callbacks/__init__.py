from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class TrainingMonitor(BaseLogger):
  def __init__(self, fig_path, json_path=None, start_at=0):
    super(TrainingMonitor, self).__init__()
    self.fig_path = fig_path
    self.json_path = json_path
    self.start_at = start_at


  def on_train_begin(self, logs={}):
    self.H = {}

    if self.json_path is not None:
      if os.path.exists(self.json_path):
        self.H = json.loads(open(self.json_path).read())

        if self.start_at > 0:
          for k in self.H.keys():
            self.H[k] = self.H[k][:self.start_at]


  def on_train_end(self, epoch, logs={}):
    for (k, v) in logs.items():
      l = self.H.get(k, [])
      l.append(v)
      self.H[k] = l

    if self.json_path is not None:
      f = open(self.json_path, "w")
      f.write(json.dump(self.H))
      f.close()

    if len(self.H["loss"]) > 1:
      N = np.arange(0, len(self.H["loss"]))
      plt.style.use("ggplot")
      plt.figure()
      plt.plot(N, self.H["loss"], label="train_loss")
      plt.plot(N, self.H["val_loss"], label="val_loss")
      plt.plot(N, self.H["acc"], label="train_acc")
      plt.plot(N, self.H["val_acc"], label="val_acc")
      plt.title("Training loss and accuracy [epoch {}]".format(len(self.H["loss"])))
      plt.xlabel("Epoch #")
      plt.ylabel("Loss/Accuracy")
      plt.legend()

      plt.savefig(self.fig_path)
      plt.close()
