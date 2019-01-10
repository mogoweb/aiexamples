import numpy as np
import cv2


labels = ["dog", "cat", "panda"]
np.random.seed(1)

# randomly initialize our weight matrix and bias vector -- in a
# *real* training and classification task, these parameters would
# be *learned* by our model, but for the sake of this example,
# let’s use random values
W = np.random.randn(3, 3072)
b = np.random.randn(3)

# load our example image, resize it, and then flatten it into our
# "feature vector" representation
origin = cv2.imread("dog.4148.jpg")
image = cv2.resize(origin, (32, 32)).flatten()

# compute the output scores by taking the dot product between the
# weight matrix and image pixels, followed by adding in the bias
scores = W.dot(image) + b

# loop over the scores + labels and display them
for (label, score) in zip(labels, scores):
  print("[INFO] {}: {:.2f}".format(label, score))

# draw the label with the highest score on the image as our
# prediction
cv2.putText(origin, "Label: {}".format(labels[np.argmax(scores)]),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow("Image", origin)
cv2.waitKey(0)