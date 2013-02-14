import sys
from contest_dataset import ContestDataset

from main import Tanh

def usage():
    print """usage: python make_submission.py model.pkl submission.csv
Where model.pkl contains a trained pylearn2.models.mlp.MLP object.
The script will make submission.csv, which you may then upload to the
kaggle site."""


if len(sys.argv) != 3:
    usage()
    print "(You used the wrong # of arguments)"
    quit(-1)

_, model_path, out_path = sys.argv

import os
if os.path.exists(out_path):
    usage()
    print out_path+" already exists, and I don't want to overwrite anything just to be safe."
    quit(-1)

from pylearn2.utils import serial
try:
    model = serial.load(model_path)
except Exception, e:
    usage()
    print model_path + "doesn't seem to be a valid model path, I got this error when trying to load it: "
    print e

dataset = ContestDataset(which_set='public_test',
            base_path = '../data',
            preprocessor = None)

dataset = dataset.get_test_set()

X = model.get_input_space().make_batch_theano()
Y = model.fprop(X, apply_dropout=False)

from theano import tensor as T

y = T.argmax(Y, axis=1)

from theano import function

y = function([X], y)(dataset.X.astype(X.dtype))

out = open(out_path, 'w')
for i in xrange(y.shape[0]):
    out.write('%d\n' % y[i])
out.close()


