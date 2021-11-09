import argparse
import sys

print()
print(' '.join(sys.argv))
print()
parser = argparse.ArgumentParser('Options for model configuration and training')

parser.add_argument('--input_channels', type=int, default=6)
parser.add_argument('--starting_channels', type=int, default=64)
parser.add_argument('--state_channels', type=int, default=64)
parser.add_argument('--model', type=str, default='bidirectional', choices=['unidirectional', 'bidirectional'], help="Choose between unidirectional and bidirectional models")
parser.add_argument('--cell_type', type=str, default='sgm', choices=['lstm', 'gru', 'sgm'], help="Recurrent cell type")
parser.add_argument('--model_name', type=str, default=None, help="Name of the model. By default, of format {model}.{cell_type}")
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--rtx', action='store_true')
parser.add_argument('--dataset', type=str, default='UCSD', choices=['UCSD', 'ICCP'])
parser.add_argument('--ref_idx', type=int, default=1)
args = parser.parse_args()

from sequences import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import tensorflow as tf

try:
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
except:
    pass

if args.rtx:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
from models import *
from losses import *

from utils import *
import cv2
import numpy as np


model_name = (args.model + '.' + args.cell_type) if args.model_name is None else args.model_name
model = models[args.model](
    input_channels=args.input_channels,
    starting_channels=args.starting_channels,
    out_channels=3,
    state_channels=args.state_channels,
    cell=args.cell_type,
    name=model_name,
)
model([tf.zeros((1, 3, 128, 128, args.input_channels)), tf.zeros((1, 128, 128, args.input_channels))])
model.summary()

########################################################
# Configure data loader
val_data_loader = DataLoader(batch_size=1, dataset=args.dataset, split='val')
#############################################################
# Define loss and metrics
losses = [MSE_TM] #, EDGE, GRAD]
metrics = [PSNR_L, PSNR_T]


def val_fn():
    if not os.path.exists('results'):
        os.mkdir('results')

    progbar = tf.keras.utils.Progbar(len(val_data_loader))
    step = 1

    for i in range(len(val_data_loader)):
        loss = 0
        loss_vals = []
        metric_vals = []

        if not os.path.exists('results/' + model_name + f'/val_{args.dataset}/' + str(i)):
            os.makedirs('results/' + model_name + f'/val_{args.dataset}/' + str(i))

        X, Y, exp = val_data_loader[i]

        img_ref = X[:, args.ref_idx, :, :, :]
        pred = model([X, img_ref], training=False)
        for l in losses:
            _loss = tf.reduce_mean(l(Y, pred))
            loss_vals.append((l.__name__.lower(), tf.reduce_mean(_loss)))
            loss += _loss

        for m in metrics:
            _metric = tf.reduce_mean(m(Y, pred))
            metric_vals.append((m.__name__.lower(), tf.reduce_mean(_metric)))

        radiance_writer(
            os.path.join('results', model_name, f'val_{args.dataset}', str(i), 'output.hdr'),
            np.squeeze(pred, axis=0)
        )
        cv2.imwrite(
            os.path.join('results', model_name, f'val_{args.dataset}', str(i), 'output.png'),
            np.clip(255. * tonemap(cv2.cvtColor(np.squeeze(pred, axis=0), cv2.COLOR_RGB2BGR)), 0., 255.).astype(
                np.uint8),
        )

        progbar.update(step, loss_vals + metric_vals)
        step += 1
    print()


print("Loading weights from ", args.weights)
model.load_weights(args.weights)
print("Finished loading")
val_fn()

