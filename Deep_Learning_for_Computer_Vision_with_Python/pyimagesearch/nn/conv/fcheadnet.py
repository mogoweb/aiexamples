from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense


class FCHeadNet:
  @staticmethod
  def build(base_model, classes, D):
    # initialize the head model that will be placed on top of
    # the base, then add a FC layer
    head_model = base_model.output
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(D, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)

    # add a softmax layer
    head_model = Dense(classes, activation="softmax")(head_model)

    return head_model
