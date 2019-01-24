import numpy as np


def rank5_accuracy(preds, labels):
  # initialize the rank-1 and rank-5 accuracies
  rank1 = 0
  rank5 = 0

  # loop over the predictions and ground-truth labels
  for (p, gt) in zip(preds, labels):
    # sort the probabilities by their index in descending
    # order so that the more confident guesses are at the
    # front of the list
    p = np.argsort(p)[::-1]

    if gt in p[:5]:
      rank5 += 1

    if gt == p[0]:
      rank1 +=1

  # compute the final rank-1 and rank-5 accuracies
  rank1 /= float(len(labels))
  rank5 /= float(len(labels))

  return (rank1, rank5)