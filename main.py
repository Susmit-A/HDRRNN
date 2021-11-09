import argparse
import sys

print()
print(" ".join(sys.argv))
print()
parser = argparse.ArgumentParser('Options for model configuration and training')

parser.add_argument('--num_epochs', type=int, default=100, help="Number of iterations to run training for")
parser.add_argument('--starting_epoch', type=int, default=0, help="Resume training at specific iteration")
parser.add_argument('--weights', type=str, default=None, help="Load pretrained weights")
parser.add_argument('--input_channels', type=int, default=6, help="Number of input channels. Default is 6: 3 for an "
                                                                  "image in the sequence, 3 for the reference image")
parser.add_argument('--dataset', type=str, default='UCSD', choices=['UCSD', 'ICCP'])

parser.add_argument('--starting_channels', type=int, default=64)
parser.add_argument('--state_channels', type=int, default=64)
parser.add_argument('--model', type=str, default='bidirectional', choices=['unidirectional', 'bidirectional'], help="Choose between unidirectional and bidirectional models")
parser.add_argument('--cell_type', type=str, default='sgm', choices=['lstm', 'gru', 'sgm'], help="Recurrent cell type")
parser.add_argument('--model_name', type=str, default=None, help="Name of the model. By default, of format {model}.{cell_type}")
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--start_lr', type=float, default=0.0002)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--steps', type=int, default=5000, help="Number of steps per iteration.")
parser.add_argument('--save_format', type=str, default='tf')
parser.add_argument('--decay-rate', type=float, default=0.5)

# Optimizer-dependent: Ignored if not applicable
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nesterov', type=bool, default=True)
parser.add_argument('--amsgrad', type=bool, default=False)
parser.add_argument('--eps', type=float, default=1e-7)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)

parser.add_argument('--save_val', action='store_true', help="Save validation images at the end of every epoch")
parser.add_argument('--gpu', type=str)
parser.add_argument('--rtx', action='store_true', help="Use mixed-precision training (recommended, but disabled by default)")
parser.add_argument('--workers', type=int, default=4)

args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from sequences import DataLoader

if args.rtx:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

from losses import *
from utils import *
import cv2
import numpy as np
from models import models


###########################################################
# Initialize model
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

if args.weights is not None:
    print()
    print("Loading weights from ", args.weights)
    print()
    model.load_weights(args.weights)


###########################################################
# Set training parameters
start_epoch = args.starting_epoch
start_lr = lr = args.start_lr
epoch = start_epoch
###########################################################
# Configure optimizer


def schd():
    return lr


optimizers = {
    'adam': tf.keras.optimizers.Adam(schd, amsgrad=args.amsgrad, epsilon=args.eps, beta_1=args.beta1,
                                     beta_2=args.beta2),
    'sgd': tf.keras.optimizers.SGD(schd, momentum=args.momentum, nesterov=args.nesterov),
    'nadam': tf.keras.optimizers.Nadam(start_lr, beta_1=args.beta1, beta_2=args.beta2, epsilon=args.eps),
    'adadelta': tf.keras.optimizers.Adadelta(1.0),
    'rmsprop': tf.keras.optimizers.RMSprop(schd, momentum=args.momentum),
}

opt = optimizers[args.optimizer]
if args.rtx:
    opt = mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')


###########################################################
# Configure data loader
data_loader = DataLoader(batch_size=args.batch_size, dataset=args.dataset)
data_seq = tf.keras.utils.OrderedEnqueuer(data_loader, shuffle=True, use_multiprocessing=False)
data_seq.start(workers=args.workers, max_queue_size=10)
data_seq_gen = data_seq.get()

val_data_loader = DataLoader(batch_size=args.batch_size, dataset=args.dataset, split='val')
###########################################################
# Define loss and metrics
losses = [MSE_TM]
metrics = [PSNR_L, PSNR_T]
###########################################################
# Define validation function


def val_fn(epoch):
    if not os.path.exists('results'):
        os.mkdir('results')

    print("\nValidation - epoch ", epoch)
    progbar = tf.keras.utils.Progbar(len(val_data_loader))
    step = 1

    if os.path.exists('results/' + model_name + '/' + str(epoch)):
        os.system('rm -r ' + 'results/' + model_name + '/' + str(epoch))
    os.makedirs('results/' + model_name + '/' + str(epoch))
    model.save_weights(os.path.join('results', model_name + '/' + str(epoch), model.name + '.' + args.save_format))

    for i in range(len(val_data_loader)):
        loss = 0
        loss_vals = []
        metric_vals = []

        os.mkdir('results/' + model_name + '/' + str(epoch) + '/' + str(i))

        X, Y, exp = val_data_loader[i]

        img_ref = X[:, 1, :, :, :]
        pred = model([X, img_ref], training=False)
        for l in losses:
            _loss = tf.reduce_mean(l(Y, pred))
            loss_vals.append((l.__name__.lower(), tf.reduce_mean(_loss)))
            loss += _loss

        for m in metrics:
            _metric = tf.reduce_mean(m(Y, pred))
            metric_vals.append((m.__name__.lower(), tf.reduce_mean(_metric)))

        if args.save_val:
            radiance_writer(
                os.path.join('results', model_name + '/' + str(epoch), str(i), str(i) + '.hdr'),
                np.squeeze(pred, axis=0)
            )
            radiance_writer(
                os.path.join('results', model_name + '/' + str(epoch), str(i), str(i) + '_gt.hdr'),
                np.squeeze(Y, axis=0)
            )
            cv2.imwrite(
                os.path.join('results', model_name + '/' + str(epoch), str(i), str(i) + '.png'),
                np.clip(255. * tonemap(cv2.cvtColor(np.squeeze(pred, axis=0), cv2.COLOR_RGB2BGR)), 0., 255.).astype(
                    np.uint8),
            )
            cv2.imwrite(
                os.path.join('results', model_name + '/' + str(epoch), str(i), str(i) + '_gt.png'),
                np.clip(255. * tonemap(cv2.cvtColor(np.squeeze(Y, axis=0), cv2.COLOR_RGB2BGR)), 0., 255.).astype(
                    np.uint8)
            )

        progbar.update(step, loss_vals + metric_vals)
        step += 1
    print()


###########################################################
# Start training


model.summary(line_length=160)
while epoch < args.num_epochs:
    if (epoch + 1) % 25 == 0:
        lr = lr * args.decay_rate
    print("\nTraining - epoch ", epoch, " | Learning rate = ", schd())
    data_loader.shuffle()
    step = 1
    if args.steps is None:
        steps = int(len(data_loader))
    else:
        steps = args.steps
    progbar = tf.keras.utils.Progbar(steps)
    for i in range(steps):
        loss_vals = []
        metric_vals = []

        X, Y, exp = next(data_seq_gen)

        img_ref = X[:, 1, :, :, :]
        with tf.GradientTape() as tape:
            loss = 0

            pred = model([X, img_ref])

            for l in losses:
                _loss = tf.reduce_mean(l(Y, pred))
                loss_vals.append((l.__name__.lower(), _loss))
                loss += _loss

            if args.rtx:
                scaled_loss = opt.get_scaled_loss(loss)

        for m in metrics:
            _metric = tf.reduce_mean(m(Y, pred))
            metric_vals.append((m.__name__.lower(), _metric))

        if args.rtx:
            scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
            grads = opt.get_unscaled_gradients(scaled_gradients)
        else:
            grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        progbar.update(step, loss_vals + metric_vals)
        step += 1
    val_fn(epoch)
    epoch += 1
