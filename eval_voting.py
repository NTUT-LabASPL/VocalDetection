import os
import time
import argparse
import importlib
import multiprocessing
import logging

import numpy as np
from definitions import WEIGHT_DIR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import dataset  # noqa: E402
from utils.voting import voting  # noqa: E402

from train import initLog, get_optimizer, run as run_training  # noqa: E402

LOG = logging.getLogger(__name__)


def run(
    model: str,
    voting_tag: str,
    voting_times: int,
    train_ds_path: str,
    val_ds_path: str,
    test_ds_paths: list,
    test_add_retrain_sizes: list,
    test_retrain_times: int = 1,
    test_retrain_has_random: bool = True,
    classes=2,  # 分類類別
    sample_size=[32000, 1],  # 訓練音訊頻率
    epochs=160,
    batch_size=150,
    lr=1.0,  # learning rate
    optimizer='adadelta',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    num_gpus=2,  # number of gpus
    debug: bool = False,
    explainable=False,
    filter_x=45,
    filter_y=120,
    magnification=4,
    seed=None,
    use_saved_inital_weight=False,
    enabled_transfer_learning=False,
    verbose=1,
    skip_origin=False
):
    initLog(debug)

    import tensorflow as tf

    Model = importlib.import_module(f'models.{model}').__getattribute__(model)
    start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in range(num_gpus)])
    input_shape = tuple(sample_size)

    voting_tags = [f'{voting_tag}-{i}' for i in range(voting_times)]

    tag = voting_tag.replace('.', '_').replace(' ', '_').replace('/', '_').replace('\\', '_')
    LOG.info(f'Model: {tag}')
    mgr = multiprocessing.Manager()
    # Testing ------------------------------------------------------------------------------------------
    cls_results = {s: mgr.list() for s in test_ds_paths}
    cls_results['ground_truth'] = {}
    total_acc = mgr.list([0 for _ in test_ds_paths])
    acc_list = mgr.list([mgr.list() for _ in test_ds_paths])

    LOG.info('Run test')
    for index, _tag in enumerate(voting_tags):

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

    if not skip_origin:
        training_kwargs = {
            'test_ds_paths': [ds.replace('train', 'test') for ds in test_ds_paths],
            'times': test_retrain_times,
            'tag': train_ds_path,
            'classes': classes,
            'sample_size': sample_size,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'optimizer': optimizer,
            'loss': loss,
            'metrics': metrics,
            'num_gpus': num_gpus,
            'training': True,
            'seed': seed,
            'use_saved_inital_weight': use_saved_inital_weight,
            'verbose': verbose
        }

        p = multiprocessing.Process(target=run_training, args=(model, train_ds_path, val_ds_path), kwargs=training_kwargs)
        p.start()
        p.join()

    for i, test_ds_path in enumerate(test_ds_paths):
        LOG.info(f"Dataset {test_ds_path}")
        for index in range(voting_times):
            LOG.info(f"第{index+1}次正確率：{acc_list[i][index]:.4f}")
        average_acc = total_acc[i] / len(acc_list[i])
        LOG.info(f"Average_acc: {average_acc*100:.6f}%")

        # Dataset must have train and test set

        real_test_ds_path = test_ds_path.replace('train', 'test')
        ground_truth = np.array(dataset.get_ground_truth(test_ds_path))
        cls_results['ground_truth'][test_ds_path] = ground_truth
        voting_acc, _, voting_rate_list = voting(cls_results[test_ds_path], ground_truth, f'{tag}_{test_ds_path}')
        LOG.info(f"Voting_acc: {voting_acc*100:.6f}%")
        voting_rate_list = np.array(sum(voting_rate_list, [])[::-1])
        LOG.info(f"Voting_rate_list_size: {len(voting_rate_list)}")
        LOG.info(f"Voting_unconfident_top_10: {voting_rate_list[:10]}")

        for length in test_add_retrain_sizes:
            if length > voting_rate_list.shape[0]:
                break
            if test_retrain_has_random:
                # base line training
                training_kwargs = {
                    'test_ds_paths': [real_test_ds_path],
                    'train_ds_size': length,
                    'times': test_retrain_times,
                    'tag': f'{test_ds_path}_RNG_{length}',
                    'classes': classes,
                    'sample_size': sample_size,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'lr': lr,
                    'optimizer': optimizer,
                    'loss': loss,
                    'metrics': metrics,
                    'num_gpus': num_gpus,
                    'training': True,
                    'seed': seed,
                    'use_saved_inital_weight': use_saved_inital_weight,
                    'verbose': verbose
                }

                p = multiprocessing.Process(target=run_training, args=(model, test_ds_path, real_test_ds_path), kwargs=training_kwargs)
                p.start()
                p.join()

                # train_ds + base line training
                training_kwargs = {
                    'additional_ds_path': test_ds_path,
                    'additional_ds_size': length,
                    'test_ds_paths': [real_test_ds_path],
                    'times': test_retrain_times,
                    'tag': f'{train_ds_path}+{test_ds_path}_RNG_{length}',
                    'classes': classes,
                    'sample_size': sample_size,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'lr': lr,
                    'optimizer': optimizer,
                    'loss': loss,
                    'metrics': metrics,
                    'num_gpus': num_gpus,
                    'training': True,
                    'seed': seed,
                    'use_saved_inital_weight': use_saved_inital_weight,
                    'enabled_transfer_learning': enabled_transfer_learning,
                    'enabled_transfer_learning_weights': voting_tags,
                    'verbose': verbose
                }
                p = multiprocessing.Process(target=run_training, args=(model, train_ds_path, val_ds_path), kwargs=training_kwargs)
                p.start()
                p.join()

            # train_ds + uncertain base line training
            training_kwargs = {
                'additional_ds_path': test_ds_path,
                'additional_ds_indexes': voting_rate_list[:length],
                'test_ds_paths': [real_test_ds_path],
                'times': test_retrain_times,
                'tag': f'{train_ds_path}+{test_ds_path}_UNC_{length}',
                'classes': classes,
                'sample_size': sample_size,
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'optimizer': optimizer,
                'loss': loss,
                'metrics': metrics,
                'num_gpus': num_gpus,
                'training': True,
                'seed': seed,
                'use_saved_inital_weight': use_saved_inital_weight,
                'enabled_transfer_learning': enabled_transfer_learning,
                'enabled_transfer_learning_weights': voting_tags,
                'verbose': verbose
            }
            p = multiprocessing.Process(target=run_training, args=(model, train_ds_path, val_ds_path), kwargs=training_kwargs)
            p.start()
            p.join()

    end = time.time()
    elapsed = end - start
    LOG.info(f"Time taken: {elapsed:.3f} seconds.")


_examples = '''examples:
  # Train SCNN 18Layers using the keras:
  python %(prog)s \\
        --model SCNN18 \\
        --voting_tag 2021-01-23/20210123-12_SCNN18_SCNN-Jamendo-train_h5 \\
        --voting_times 21 \\
        --train_ds_path SCNN-Jamendo-train.h5 \\
        --val_ds_path SCNN-Jamendo-test.h5 \\
        --test_ds_paths SCNN-Taiwanese-stream-train.h5 SCNN-Classical-test.h5 \\
        --test_add_retrain_sizes 100 200 300 400 500 600 700 800 900 1000 \\
        --test_retrain_times 1 \\
        --test_retrain_has_random \\
        --classes 2 \\
        --sample_size 32000 1 \\
        --epochs 160 \\
        --batch_size 150 \\
        --loss categorical_crossentropy \\
        --optimizer adadelta \\
        --metrics accuracy \\
        --lr 1.0 \\
        --seed 0
'''


def main():
    parser = argparse.ArgumentParser(description="Train SCNN 18Layers", epilog=_examples, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', required=True, help="SCNN18,SCNN36,AutoEncoderRemoveVocal")
    parser.add_argument('--voting_tag', required=True, help="Trained Model tag")
    parser.add_argument('--voting_times', required=True, help="How many trained models?(default: %(default)s)", default=21, type=int)
    parser.add_argument('--train_ds_path', required=True, help='Training dataset path')
    parser.add_argument('--val_ds_path', required=True, help='validation dataset path')
    parser.add_argument(
        '--test_ds_paths',
        help='Testing dataset paths; Required pair dataset include train and test ; Use train in here; (default: %(default)s)',
        nargs='+',
        default=['train.h5']
    )
    parser.add_argument('--test_add_retrain_sizes', help='Add some test_set to train_set(default: %(default)s)', type=int, nargs='+', default=[100])
    parser.add_argument('--test_retrain_times', required=True, help="How many times do you train?(default: %(default)s)", default=1, type=int)
    parser.add_argument('--test_retrain_has_random', help="Also trained in random?(default: %(default)s)", default=False, action='store_true')
    parser.add_argument('--classes', help='Output class number(default: %(default)s)', default=2, type=int)
    parser.add_argument('--sample_size', help='Audio sample size(default: %(default)s)', nargs='+', type=int, default=[32000, 1])
    parser.add_argument('--epochs', help="epochs (default: %(default)s)", default=160, type=int)
    parser.add_argument('--batch_size', help="batch_size (default: %(default)s)", default=150, type=int)
    parser.add_argument('--loss', help="loss(default: %(default)s)", default='categorical_crossentropy', type=str)
    parser.add_argument('--optimizer', help="optimizer(default: %(default)s)", default='adadelta', type=str)
    parser.add_argument('--metrics', help="metrics(default: %(default)s)", nargs='+', default=['accuracy'])
    parser.add_argument('--lr', help="learning rate(default: %(default)s for optimizer default value)", default=0.0, type=float)
    parser.add_argument('--explainable', help="Run explainable?(default: %(default)s)", default=False, action='store_true')
    parser.add_argument('--filter_x', help="Explainable filter_x(default: %(default)s)", default=45, type=int)
    parser.add_argument('--filter_y', help="Explainable filter_y(default: %(default)s)", default=120, type=int)
    parser.add_argument('--magnification', help="Explainable magnification(default: %(default)s)", default=4, type=int)
    parser.add_argument('--num_gpus', help="Number of gpus(default: %(default)s)", default=2, type=int)
    parser.add_argument('--debug', help="Is debuging?(default: %(default)s)", default=False, action='store_true')
    parser.add_argument('--seed', help="Random seed (default: %(default)s)", type=int)
    parser.add_argument('--verbose', help="Verbose (default: %(default)s)", default=1, type=int)
    parser.add_argument('--use_saved_inital_weight', help="use saved inital weight(default: %(default)s)", default=False, action='store_true')
    parser.add_argument('--skip_origin', help="skip training origin weight(default: %(default)s)", default=False, action='store_true')
    parser.add_argument('--enabled_transfer_learning', help="enabled transfer learnning(default: %(default)s)", default=False, action='store_true')
    args = parser.parse_args()

    run(**vars(args))


if __name__ == "__main__":
    main()
