import argparse
import pickle
import h5py
import sys

sys.path.append("../..")
from pyimagesearch.utils.ranked import rank5_accuracy

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help="path to HDF5 database")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
args = vars(ap.parse_args())

print("[INFO] loading pre-trained model ...")
print(args["model"])
model = pickle.loads(open(args["model"], "rb").read())

db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0] * 0.75)

print("[INFO] predictinng ...")
preds = model.predict_proba(db["features"][i:])
(rank1, rank5) = rank5_accuracy(preds, db["labels"][i:])

print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))

db.close()
