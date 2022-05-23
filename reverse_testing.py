import os
import time

import logging
from logging import handlers

from definitions import WEIGHT_DIR
from utils import dataset
# from utils.plot import plot_spectrogram

import numpy as np
import tensorflow as tf
from models.SCNN18 import SCNN18

LOG = logging.getLogger(__name__)


def initLog(debug=False):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M',
        handlers=[logging.StreamHandler(), handlers.RotatingFileHandler('output.log', "w", 1024 * 1024 * 100, 3, "utf-8")]
    )
    LOG.setLevel(logging.DEBUG if debug else logging.INFO)
    tf.get_logger().setLevel('ERROR')


initLog()

dataset_list = [
    'SCNN-Jamendo-train.h5',
    'SCNN-FMA-C-1-fixed-train.h5',
    'SCNN-FMA-C-2-fixed-train.h5',
    'SCNN-KTV-train.h5',
    'SCNN-Taiwanese-CD-train.h5',
    'SCNN-Taiwanese-stream-train.h5',
    'SCNN-Chinese-CD-train.h5',
    'SCNN-Classical-train.h5',
    # ---- Test
    'SCNN-Jamendo-test.h5',
    'SCNN-FMA-C-1-fixed-test.h5',
    'SCNN-FMA-C-2-fixed-test.h5',
    'SCNN-KTV-test.h5',
    'SCNN-Taiwanese-CD-test.h5',
    'SCNN-Taiwanese-stream-test.h5',
    'SCNN-Chinese-CD-test.h5',
    'SCNN-Classical-test.h5',
    # ---- Only have one
    'SCNN-MIR-1k-train.h5',
    'SCNN-Instrumental-non-vocal.h5',
    'SCNN-A-Cappella-vocal.h5',
    'SCNN-test-hard.h5',
]

dataset_train_list = [
    # 'SCNN-Jamendo-train.h5',
    # 'SCNN-FMA-C-1-fixed-train.h5',
    # 'SCNN-FMA-C-2-fixed-train.h5',
    # 'SCNN-KTV-train.h5',
    # 'SCNN-Taiwanese-CD-train.h5',
    # 'SCNN-Taiwanese-stream-train.h5',
    # 'SCNN-Chinese-CD-train.h5',
    # 'SCNN-Classical-train.h5',
    # ---- Test
    # 'SCNN-Jamendo-test.h5',
    # 'SCNN-FMA-C-1-fixed-test.h5',
    # 'SCNN-FMA-C-2-fixed-test.h5',
    # 'SCNN-KTV-test.h5',
    # 'SCNN-Taiwanese-CD-test.h5',
    # 'SCNN-Taiwanese-stream-test.h5',
    # 'SCNN-Chinese-CD-test.h5',
    # 'SCNN-Classical-test.h5',
    # ---- Only have one
    # 'SCNN-MIR-1k-train.h5',
    # 'SCNN-Instrumental-non-vocal.h5',
    # 'SCNN-A-Cappella-vocal.h5',
    # 'SCNN-test-hard.h5',
    'SCNN-RWC.h5',
]

times = 10
# Setting visible of gpus
batch_size = 150
nb_classes = 2
nb_epoch = 160
sample_size = 32000
input_shape = (sample_size, 1)
# Calculat time
start = time.time()
training = True
total_acc = 0
acc_list = []
# Config gpus
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        LOG.error(e)

input_shape = (32000, 1)
classes = 2
lr = 1.0

logging = []
reversed_accs = {key: [] for key in dataset_train_list}
origin_accs = {key: [] for key in dataset_train_list}


def run_test(train_dataset, origin_dataset, times):

    for i in range(times):
        strategy = tf.distribute.MirroredStrategy(devices=[f'/gpu:{_i}' for _i in range(3)])
        with strategy.scope():
            model = SCNN18(input_shape, classes).model()
            model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adadelta(lr), metrics=['accuracy'])
            model.load_weights(os.path.join(WEIGHT_DIR, "FMA-C-1", f"2021-02-13_11_SCNN18_SCNN-FMA-C-1-fixed-train_h5_2GPU-{i}.h5"))

        X = dataset.get_dataset_without_label(train_dataset)
        Y = np.where(model.predict(dataset.load(train_dataset).batch(batch_size)) >= 0.5, 1, 0)
        real_train_ds = dataset.load(train_dataset).batch(batch_size)
        train_ds = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size)
        # val_ds = dataset.load(validation_dataset).batch(batch_size)
        origin_ds = dataset.load(origin_dataset).batch(batch_size)

        strategy = tf.distribute.MirroredStrategy(devices=[f'/gpu:{i}' for i in range(3)])
        with strategy.scope():
            model_retrain = SCNN18(input_shape, classes).model()
            model_retrain.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adadelta(lr), metrics=['accuracy'])

        model_retrain.fit(train_ds, epochs=nb_epoch, validation_data=None)
        _reversed_acc = model_retrain.evaluate(origin_ds)[1]
        _origin_acc = model.evaluate(real_train_ds)[1]
        LOG.info(
            f"{origin_dataset.replace('.h5', '')} retrain with pseudo lable {train_dataset.replace('.h5', '')} {i}: {_reversed_acc:.5f}, {_origin_acc:.5f}"
        )
        reversed_accs[train_dataset].append(_reversed_acc)
        origin_accs[train_dataset].append(_origin_acc)
        model_retrain.save_weights(
            os.path.join(
                WEIGHT_DIR, 'reverse_testing_RWC_FMA-C-1',
                f"SCNN_{origin_dataset.replace('.h5', '')}_with_pseudo_{train_dataset.replace('.h5', '')}_{i}.h5"
            )
        )


for train_dataset in dataset_train_list:
    run_test(train_dataset, 'SCNN-FMA-C-1-fixed-train.h5', times)

for train_dataset in dataset_train_list:
    LOG.info(f"{train_dataset} reversed retrain avg acc: {np.sum(reversed_accs[train_dataset])/len(reversed_accs[train_dataset])}")
    LOG.info(f"{train_dataset} avg acc: {np.sum(origin_accs[train_dataset])/len(origin_accs[train_dataset])}")
