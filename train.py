import os
import time
import json
import argparse
import importlib
import multiprocessing

import logging
from logging import handlers

from datetime import datetime
from definitions import LOG_DIR, WEIGHT_DIR, DATA_DIR
from utils import dataset
from utils.voting import voting
# from utils.plot import plot_spectrogram
from utils.encoder import NumpyEncoder
from utils.explainable_block import explainable_block

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

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


def get_optimizer(optimizer, lr):
    optimizer = optimizer.lower()
    if optimizer == 'adadelta':
        return tf.optimizers.Adadelta() if lr == 0 else tf.optimizers.Adadelta(learning_rate=lr)
    elif optimizer == 'adagrad':
        return tf.optimizers.Adagrad() if lr == 0 else tf.optimizers.Adagrad(learning_rate=lr)
    elif optimizer == 'adam':
        return tf.optimizers.Adam() if lr == 0 else tf.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'adamax':
        return tf.optimizers.Adamax() if lr == 0 else tf.optimizers.Adamax(learning_rate=lr)
    elif optimizer == 'sgd':
        return tf.optimizers.SGD() if lr == 0 else tf.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'rmsprop':
        return tf.optimizers.RMSprop() if lr == 0 else tf.optimizers.RMSprop(learning_rate=lr)
    else:
        raise Exception("Not valid optimizer!")


def run(
    model,
    train_ds_path,
    val_ds_path,
    train_ds_size=None,
    train_ds_indexes=None,  # type list => index of dataset 取第0,1,3,5個資料 => [0, 1, 3, 5]
    additional_ds_path=None,
    additional_ds_size=None,
    additional_ds_indexes=None,  # type list => index of dataset 取第0,1,3,5個資料 => [0, 1, 3, 5]
    test_ds_paths=[],
    classes=2,  # 分類類別
    sample_size=[32000, 1],  # 訓練音訊頻率
    epochs=160,
    batch_size=150,
    times=10,  # 總共跑幾次
    tag=None,
    lr=1.0,  # learning rate
    optimizer='adadelta',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    num_gpus=2,  # number of gpus
    training=True,
    debug=False,
    explainable=False,
    filter_x=45,
    filter_y=120,
    magnification=4,
    seed=None,
    use_saved_inital_weight=False,
    enabled_transfer_learning=False,
    enabled_transfer_learning_weights=[],
    verbose=1
):
    initLog(debug)
    Model = importlib.import_module(f'models.{model}').__getattribute__(model)
    start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in range(num_gpus)])
    if tag is None:
        training = True
        tag = datetime.now().strftime("%Y-%m-%d_%H")
        tag += '_' + model + '_' + train_ds_path.replace('.', '_').replace(' ', '_').replace('/', '_') + f'_{num_gpus}GPU'

    input_shape = tuple(sample_size)
    tags = []
    LOG.info(f'training set: {train_ds_path}')
    LOG.info(f'testing sets: {test_ds_paths}')

    # Run thread training --------------------------------------------------------------------------------------------
    def train(t, q):
        # Config gpus
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                LOG.error(e)

        # Load dataset
        if train_ds_indexes is not None and isinstance(list(train_ds_indexes), list):
            train_ds = dataset.create_new_dataset_from_index(train_ds_path, list(train_ds_indexes))
        else:
            train_ds = dataset.load(train_ds_path)

        dataset_length = [i for i, _ in enumerate(train_ds)][-1] + 1
        val_ds = dataset.load(val_ds_path)
        dataset_length = [i for i, _ in enumerate(val_ds)][-1] + 1
        if t == 0:
            LOG.info(f'{str(val_ds)} size: {dataset_length}')

        # Tensorboard
        tensorboard_callback = TensorBoard(log_dir=f'{LOG_DIR}/{tag}/{t}')

        # Add more training dataset
        if additional_ds_path:
            if additional_ds_indexes is not None and isinstance(list(additional_ds_indexes), list):
                additional_ds = dataset.create_new_dataset_from_index(additional_ds_path, list(additional_ds_indexes))
            else:
                additional_ds = dataset.load(additional_ds_path)

            if additional_ds_size is not None:
                dataset_length = [i for i, _ in enumerate(additional_ds)][-1] + 1
                additional_ds = additional_ds.shuffle(
                    dataset_length, seed=seed, reshuffle_each_iteration=False
                ).take(additional_ds_size if additional_ds_size < dataset_length else dataset_length)

            if enabled_transfer_learning:
                train_ds = additional_ds
            else:
                train_ds = train_ds.concatenate(additional_ds)

            dataset_length = [i for i, _ in enumerate(train_ds)][-1] + 1
            train_ds = train_ds.shuffle(dataset_length, seed=seed, reshuffle_each_iteration=False)

        dataset_length = [i for i, _ in enumerate(train_ds)][-1] + 1

        if train_ds_size is not None:
            train_ds = train_ds.shuffle(
                dataset_length, seed=seed, reshuffle_each_iteration=False
            ).take(train_ds_size if train_ds_size < dataset_length else dataset_length)

        dataset_length = [i for i, _ in enumerate(train_ds)][-1] + 1

        if t == 0:
            LOG.info(f'{str(train_ds)} size: {dataset_length}')

        train_ds = train_ds.batch(batch_size).shuffle(dataset_length, reshuffle_each_iteration=True)
        val_ds = val_ds.batch(batch_size)

        strategy = tf.distribute.MirroredStrategy(devices=[f'/gpu:{i}' for i in range(num_gpus)])
        with strategy.scope():
            _model = Model(input_shape, classes).model()
            _model.compile(loss=loss, optimizer=get_optimizer(optimizer, lr), metrics=metrics)
        if t == 0:
            _model.summary(print_fn=LOG.info)
        if use_saved_inital_weight:
            if os.path.exists(os.path.join(WEIGHT_DIR, f'{model}_inital_weights.h5')):
                LOG.info(f"load {model} inital weight")
                _model.load_weights(os.path.join(WEIGHT_DIR, f'{model}_inital_weights.h5'))
            else:
                LOG.info(f"create {model} inital weight")
                _model.save_weights(os.path.join(WEIGHT_DIR, f'{model}_inital_weights.h5'))

        if enabled_transfer_learning and enabled_transfer_learning_weights:
            _model.load_weights(os.path.join(WEIGHT_DIR, enabled_transfer_learning_weights[t % len(enabled_transfer_learning_weights)] + '.h5'))

        LOG.info(f'Training {tag}-{t} start')
        _model.fit(train_ds, epochs=epochs, verbose=verbose, validation_data=val_ds, callbacks=[tensorboard_callback])
        _model.save_weights(os.path.join(WEIGHT_DIR, tag + f"-{t}" + ".h5"))
        q.put(f'{tag}-{t}')

    # -------------------------------------------------------------------------------------------------

    if training:
        q = multiprocessing.Queue()
        for t in range(times):
            p = multiprocessing.Process(target=train, args=(t, q))
            p.start()
            p.join()
            tags.append(q.get())
    else:  # For test
        tags = [f'{tag}-{i}' for i in range(times)]

    tag = tag.replace('.', '_').replace(' ', '_').replace('/', '_').replace('\\', '_')
    LOG.info(f'Model: {tag}')
    mgr = multiprocessing.Manager()
    # Testing ------------------------------------------------------------------------------------------
    cls_results = {s: mgr.list() for s in test_ds_paths}
    cls_results['ground_truth'] = {}
    total_acc = mgr.list([0 for _ in test_ds_paths])
    acc_list = mgr.list([mgr.list() for _ in test_ds_paths])

    LOG.info('Run test')
    for index, _tag in enumerate(tags):

        def test():
            strategy = tf.distribute.MirroredStrategy(devices=[f'/gpu:{i}' for i in range(num_gpus)])
            with strategy.scope():
                _model = Model(input_shape, classes).model()
                _model.compile(loss=loss, optimizer=get_optimizer(optimizer, lr), metrics=metrics)
            _model.load_weights(os.path.join(WEIGHT_DIR, _tag + '.h5'))

            # Evaluation
            for i, test_ds_path in enumerate(test_ds_paths):
                test_ds = dataset.load(test_ds_path).batch(batch_size)
                score, acc = _model.evaluate(test_ds, verbose=0)
                result = _model.predict(test_ds)
                cls_results[test_ds_path].append(np.where(result >= 0.5, 1, 0))
                acc_list[i].append(acc)
                total_acc[i] += acc
                LOG.debug(f'no.{index + 1}, score={score}, acc={acc}')
                del test_ds
            del _model

        p = multiprocessing.Process(target=test)
        p.start()
        p.join()

    # Explainable block test
    if explainable:
        try:
            Model_ex = importlib.import_module(f'models.{model}_EX').__getattribute__(f'{model}_EX')
            LOG.info(f'Run {model}_EX explainable block test')

            # Magnification = 4
            # filter_x = 45  # 63
            # filter_y = 120  # 1024
            max_x_position_bias = 63 - filter_x
            max_y_position_bias = 1024 - filter_y
            cls_results_ex = {}
            total_acc_ex = mgr.list()
            acc_list_ex = mgr.list()
            LOG.info(f'filter_x:{filter_x}, filter_y: {filter_y}, Magnification:{magnification}')
            # Iterate only one model
            for index, _tag in enumerate(tags):
                bias_count = 0
                for y_position_bias in range(0, max_y_position_bias, filter_y):
                    for x_position_bias in range(0, max_x_position_bias, 5):
                        bias_key = f'x{x_position_bias}~{x_position_bias+filter_x}_y{y_position_bias}~{y_position_bias+filter_y}={magnification}'
                        LOG.debug(f'no.{index+1}, {bias_key}:')
                        if index == 0:
                            cls_results_ex[bias_key] = {s: mgr.list() for s in test_ds_paths}
                            total_acc_ex.append(mgr.list([0 for _ in test_ds_paths]))
                            acc_list_ex.append(mgr.list([mgr.list() for _ in test_ds_paths]))

                        def test():
                            # Create models
                            strategy = tf.distribute.MirroredStrategy(devices=[f'/gpu:{i}' for i in range(num_gpus)])
                            with strategy.scope():
                                _model_ex, _model_revise_spectrogram, _model_origin_spectrogram = Model_ex(
                                    input_shape, classes, x_position_bias, y_position_bias, filter_x, filter_y, magnification
                                ).model()
                                _model_ex.compile(loss=loss, optimizer=get_optimizer(optimizer, lr), metrics=metrics)
                            _model_ex.load_weights(os.path.join(WEIGHT_DIR, _tag + '.h5'))

                            # Evaluate all test datasets
                            for i, test_ds_path in enumerate(test_ds_paths):
                                test_ds = dataset.load(test_ds_path).batch(batch_size)
                                score, acc = _model_ex.evaluate(test_ds, verbose=0)
                                result = _model_ex.predict(test_ds)

                                # Append result
                                cls_results_ex[bias_key][test_ds_path].append(np.where(result >= 0.5, 1, 0))
                                acc_list_ex[bias_count][i].append(acc)
                                total_acc_ex[bias_count][i] += acc
                                LOG.debug(f'no.{index + 1}, score={score}, acc={acc}')
                                del test_ds
                            del _model_ex, _model_revise_spectrogram, _model_origin_spectrogram

                        p = multiprocessing.Process(target=test)
                        p.start()
                        p.join()
                        bias_count += 1

            bias_count = 0
            # Iterate all block
            for y_position_bias in range(0, max_y_position_bias, filter_y):
                for x_position_bias in range(0, max_x_position_bias, 5):
                    bias_key = f'x{x_position_bias}~{x_position_bias+filter_x}_y{y_position_bias}~{y_position_bias+filter_y}={magnification}'
                    LOG.debug(bias_key)
                    for i, test_ds_path in enumerate(test_ds_paths):
                        LOG.debug(f"Dataset {test_ds_path}")
                        reverses = []
                        # Iterate one model
                        for index in range(len(tags)):
                            reversed = explainable_block(cls_results_ex[bias_key][test_ds_path][index], cls_results[test_ds_path][index])
                            reverses.append(reversed)
                            LOG.debug(f"第{index+1}次正確率：{acc_list_ex[bias_count][i][index]*100:.4f}, 反轉率: {reversed*100:.4f}%")
                        average_acc = total_acc_ex[bias_count][i] / len(acc_list_ex[bias_count][i])
                        average_reverse = np.sum(reverses) / len(reverses)
                        LOG.info(
                            f"x{x_position_bias},y{y_position_bias},m{magnification} avg_acc: {average_acc*100:.6f}%, 反轉率: {average_reverse*100:.4f}%"
                        )
            json.dump(
                cls_results_ex,
                open(os.path.join(DATA_DIR, 'json', f'{tag}_{filter_x}_{filter_y}_{magnification}_cls_result_ex.json'), "w"),
                cls=NumpyEncoder
            )
        except Exception as err:
            LOG.error(err)
            LOG.info(f'Cannot run {model}_EX explainable block test')

    for i, test_ds_path in enumerate(test_ds_paths):
        LOG.info(f"Dataset {test_ds_path}")
        for index in range(times):
            LOG.info(f"第{index+1}次正確率：{acc_list[i][index]:.4f}")
        average_acc = total_acc[i] / len(acc_list[i])
        LOG.info(f"Average_acc: {average_acc*100:.6f}%")

        ground_truth = np.array(dataset.get_ground_truth(test_ds_path))
        cls_results['ground_truth'][test_ds_path] = ground_truth
        voting_acc, _, _ = voting(cls_results[test_ds_path], ground_truth, f'{tag}_{test_ds_path}')
        LOG.info(f"Voting_acc: {voting_acc*100:.6f}%")

    json.dump(cls_results, open(os.path.join(DATA_DIR, 'json', f'{tag}_cls_result.json'), "w"), cls=NumpyEncoder)
    end = time.time()
    elapsed = end - start
    LOG.info(f"Time taken: {elapsed:.3f} seconds.")


_examples = '''examples:
  # Train SCNN 18Layers using the keras:
  python %(prog)s \\
        --model SCNN18 \\
        --train_ds_path SCNN-Jamendo-train.h5 \\
        --val_ds_path SCNN-Jamendo-test.h5 \\
        --test_ds_paths SCNN-test-hard.h5 FMA-C-1-fixed-SCNN-Test.h5 SCNN-Jamendo-test.h5 \\
        --additional_ds_path SCNN-MIR-1k-train.h5 \\
        --classes 2 \\
        --sample_size 32000 1 \\
        --epochs 160 \\
        --batch_size 150 \\
        --loss categorical_crossentropy \\
        --optimizer adadelta \\
        --metrics accuracy \\
        --lr 1.0 \\
        --times 10 \\
        --training
'''


def main():
    parser = argparse.ArgumentParser(description="Train SCNN 18Layers", epilog=_examples, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', required=True, help="SCNN18,SCNN36,AutoEncoderRemoveVocal")
    parser.add_argument('--train_ds_path', required=True, help='Training dataset path')
    parser.add_argument('--train_ds_size', help='Cut training dataset', type=int)
    parser.add_argument('--val_ds_path', required=True, help='validation dataset path')
    parser.add_argument('--test_ds_paths', help='Testing dataset paths(default: %(default)s)', nargs='+', default=[])
    parser.add_argument('--additional_ds_path', help='Additional dataset path(default: %(default)s)', default=None)
    parser.add_argument('--additional_ds_size', help='Additional dataset size(default: %(default)s)', type=int)
    parser.add_argument('--classes', help='Output class number(default: %(default)s)', default=2, type=int)
    parser.add_argument('--sample_size', help='Audio sample size(default: %(default)s)', nargs='+', type=int, default=[32000, 1])
    parser.add_argument('--epochs', help="epochs (default: %(default)s)", default=160, type=int)
    parser.add_argument('--batch_size', help="batch_size (default: %(default)s)", default=150, type=int)
    parser.add_argument('--loss', help="loss(default: %(default)s)", default='categorical_crossentropy', type=str)
    parser.add_argument('--optimizer', help="optimizer(default: %(default)s)", default='adadelta', type=str)
    parser.add_argument('--metrics', help="metrics(default: %(default)s)", nargs='+', default=['accuracy'])
    parser.add_argument('--lr', help="learning rate(default: %(default)s for optimizer default value)", default=0.0, type=float)
    parser.add_argument('--times', help="Loop times(default: %(default)s)", default=10, type=int)
    parser.add_argument('--training', help="Is training?(default: %(default)s)", default=False, action='store_true')
    parser.add_argument('--explainable', help="Run explainable?(default: %(default)s)", default=False, action='store_true')
    parser.add_argument('--filter_x', help="Explainable filter_x(default: %(default)s)", default=45, type=int)
    parser.add_argument('--filter_y', help="Explainable filter_y(default: %(default)s)", default=120, type=int)
    parser.add_argument('--magnification', help="Explainable magnification(default: %(default)s)", default=4, type=int)
    parser.add_argument('--tag', help="weights tag?(default: %(default)s)", default=None, type=str)
    parser.add_argument('--num_gpus', help="Number of gpus(default: %(default)s)", default=2, type=int)
    parser.add_argument('--debug', help="Is debuging?(default: %(default)s)", default=False, action='store_true')
    parser.add_argument('--seed', help="Random seed (default: %(default)s)", type=int)
    parser.add_argument('--verbose', help="Verbose (default: %(default)s)", default=1, type=int)

    args = parser.parse_args()

    run(**vars(args))


if __name__ == "__main__":
    main()
