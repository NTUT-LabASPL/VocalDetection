import os
import numpy as np
from utils.plot import plot_voting
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from definitions import DIAGRAM_DIR


def voting(cls_results, ground_truth, title="Default"):
    """
    To calculate voting accuracy.

        cls_results == np.where(cls.predict(test_ds) >= 0.5, 1, 0)
        ground_truth == [ y.numpy() for _, y in dataset.load(test_ds_path)]
        title(str)
    """
    ground_truth = np.array(ground_truth)
    voting_0_count = np.zeros(ground_truth.shape[0])
    voting_1_count = np.zeros(ground_truth.shape[0])

    for v in np.array(cls_results):
        for i, value in enumerate(v):
            if value[0] == 1:
                voting_1_count[i] += 1
            else:
                voting_0_count[i] += 1

    voting_result_dict = {}
    voting_result_list = [[] for _ in range(int(len(cls_results) / 2) + 1)]
    voting_result = np.zeros(ground_truth.shape[0])

    for i in range(ground_truth.shape[0]):
        x, y = voting_0_count[i], voting_1_count[i]
        if x > y:
            temp = y
            y = x
            x = temp
        else:
            voting_result[i] = 1
        label = f'{int(x)}::{int(y)}'
        if voting_result_dict.get(label, None) is None:
            voting_result_dict[label] = 0
        voting_result_dict[label] += 1
        voting_result_list[int(min(x, y))].append(i)

    plot_voting(voting_result_dict, ground_truth.shape[0], title)

    return np.sum(voting_result == ground_truth[:, 0]) / ground_truth.shape[0], voting_result_dict, voting_result_list


def voting_LR_score(x, y_1, y_2, y_3, y_4, title="Default"):
    x = np.array(x).reshape(-1, 1)
    y_1 = np.array(y_1).reshape(-1, 1)
    y_2 = np.array(y_2).reshape(-1, 1)
    y_3 = np.array(y_3).reshape(-1, 1)
    y_4 = np.array(y_4).reshape(-1, 1)

    lr1 = LinearRegression().fit(x, y_1)
    lr2 = LinearRegression().fit(x, y_2)
    lr3 = LinearRegression().fit(x, y_3)
    lr4 = LinearRegression().fit(x, y_4)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18, 8))
    axs[0][0].axis([0.0, 1., 0.0, 1.])
    axs[0][1].axis([0.0, 1., 0.0, 1.])
    axs[1][0].axis([0.0, 1., 0.0, 1.])
    axs[1][1].axis([0.0, 1., 0.0, 1.])
    scores = [lr1.score(x, y_1) * 100, lr2.score(x, y_2) * 100, lr3.score(x, y_3) * 100, lr4.score(x, y_4) * 100]

    axs[0][0].set_title(f'score: {scores[0]:.3f}')
    axs[0][1].set_title(f'score: {scores[1]:.3f}')
    axs[1][0].set_title(f'score: {scores[2]:.3f}')
    axs[1][1].set_title(f'score: {scores[3]:.3f}')

    axs[0][0].scatter(x, y_1, label='y_1')
    axs[0][1].scatter(x, y_2, label='y_2')
    axs[1][0].scatter(x, y_3, label='y_3')
    axs[1][1].scatter(x, y_4, label='y_4')
    axs[0][0].legend(loc='upper left', bbox_to_anchor=(1, 1.05))
    axs[0][1].legend(loc='upper left', bbox_to_anchor=(1, 1.05))
    axs[1][0].legend(loc='upper left', bbox_to_anchor=(1, 1.05))
    axs[1][1].legend(loc='upper left', bbox_to_anchor=(1, 1.05))
    axs[0][0].plot(x, lr1.predict(x), color='blue', linewidth=1)
    axs[0][1].plot(x, lr2.predict(x), color='blue', linewidth=1)
    axs[1][0].plot(x, lr3.predict(x), color='blue', linewidth=1)
    axs[1][1].plot(x, lr4.predict(x), color='blue', linewidth=1)
    fig.savefig(os.path.join(DIAGRAM_DIR, title + '.png'), bbox_inches='tight', facecolor='w')
    plt.close('all')

    return scores
