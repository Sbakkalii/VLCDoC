from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import numpy as np
FLAGS = flags.FLAGS


def define_trainer_flags():
    """Defining all the necessary flags."""

    # Data parameters
    flags.DEFINE_string(name='dataset', default='rvlcdip', help='Dataset to use')
    flags.DEFINE_string(name='datapath', default='/path/to/dataset/', help='Path, where datasets are stored')

    # Training parameters
    flags.DEFINE_string(name='backbone', default='Multimodal', help='Backbone architecture')
    flags.DEFINE_integer(name='img_size', default=224, help='The Input Image Size')
    flags.DEFINE_integer(name='epochs', default=100, help='Number of training epochs')
    flags.DEFINE_string(name='num_gpu', default='0', help='The number of GPU to use')
    flags.DEFINE_string(name='optimizer', default='AdamW', help='Use optimizer to train the model"')
    flags.DEFINE_float(name='lr', default=1e-4, help='Learning rate')
    flags.DEFINE_integer(name='lr_decay_step', default=20, help='Number of epochs after which to decay the learning rate')
    flags.DEFINE_integer(name='num_classes', default=16, help='Number of classes')
    flags.DEFINE_integer(name='max_length', default=128, help='max sequence length')
    flags.DEFINE_float(name='lr_decay_rate', default=0.5, help='Weight decay rate')
    flags.DEFINE_integer(name='patience', default=25, help='Patience until early stopping. -1 means no early stopping')
    flags.DEFINE_boolean(name='enable_function', default=True, help='Ena:ble Function?')
    flags.DEFINE_boolean(name='mirrored_strategy', default=True, help='Enable Function?')
    flags.DEFINE_string(name='train_mode', default='keras_fit', help='Use either "keras_fit" or "custom_loop"')
    flags.DEFINE_string(name='mode', default='train', help='Training Mode')
    flags.DEFINE_string(name='split', default='Train_Data', help='Data Split')
    flags.DEFINE_float(name="best_val_loss", default=np.inf, help="Validation loss [np.inf]")
    flags.DEFINE_boolean(name='early_stopping_triggered', default=False, help='Trigger Early Stopping')
    flags.DEFINE_integer(name='batch_size', default=96, help='Batch size for ProtoTransfer')

    # Saving and loading parameters
    flags.DEFINE_boolean(name='save', default=True, help='Whether to save the best model')
    flags.DEFINE_string(name='save_path', default='/path/to/save/checkpoints/', help='Save checkpoints path')
    flags.DEFINE_string(name='load_path', default='/path/to/load/checkpoints/', help='load checkpoints')
    flags.DEFINE_string(name='data_directory', default='/home/data/directory/', help='Optionally load a model')
    flags.DEFINE_boolean(name='load_last', default=False, help='Optionally load from default model path')
    flags.DEFINE_boolean(name='load_best', default=False, help='Optionally load from default best model path')


def flags_dict():
    """Define the flags.
    Returns:
      Command line arguments as Flags.
    """

    kwargs = {'dataset': FLAGS.dataset,
              'datapath': FLAGS.datapath,
              'backbone': FLAGS.backbone,
              'num_classes': FLAGS.num_classes, 
              'split': FLAGS.split,  
              'img_size': FLAGS.img_size,
              'epochs': FLAGS.epochs,
              'max_length': FLAGS.max_length,
              'optimizer': FLAGS.optimizer,
              'num_gpu': FLAGS.num_gpu,
              'lr': FLAGS.lr,
              'lr_decay_step': FLAGS.lr_decay_step,
              'lr_decay_rate': FLAGS.lr_decay_rate,
              'patience': FLAGS.patience,
              'enable_function': FLAGS.enable_function,
              'early_stopping_triggered': FLAGS.early_stopping_triggered,
              'best_val_loss': FLAGS.best_val_loss,
              'batch_size': FLAGS.batch_size,
              'save': FLAGS.save,
              'save_path': FLAGS.save_path,
              'train_mode': FLAGS.train_mode,
              'data_directory': FLAGS.data_directory,
              'mode': FLAGS.mode,
              }
    return kwargs
