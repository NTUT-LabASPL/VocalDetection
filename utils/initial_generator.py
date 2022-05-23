from time import sleep

import numpy as np
import matplotlib.pyplot as plt


class InitialGenerator:
    @classmethod
    # deprecated
    def initial_sin(cls, nums, duration, first_coeff=4, end_coeff=150, exp_mode=True):
        initial_stride = end_coeff / first_coeff
        amp_list = np.empty(shape=(duration, 0))
        exp_coeff = initial_stride ** (1 / nums)
        for i in range(0, nums):
            x = int(first_coeff * (exp_coeff ** i))
            times = np.arange(0.0, duration, 1 / (duration - i))
            if exp_mode:
                times = np.arange(0.0, duration, 1 / x)
            amplitudes = np.sin(times * 2.0 * np.pi)
            if len(amplitudes) > duration:
                amplitudes = amplitudes[:duration]
            amplitudes = np.reshape(amplitudes, [-1, 1])
            amp_list = np.append(amp_list, amplitudes, axis=1)
            print(i)
        return amp_list

    @classmethod
    # deprecated
    def initial_cos(cls, nums, duration, first_coeff=4, end_coeff=150, exp_mode=True):
        initial_stride = end_coeff / first_coeff
        amp_list = np.empty(shape=(duration, 0))
        exp_coeff = initial_stride ** (1 / nums)
        for i in range(0, nums):
            x = int(first_coeff * (exp_coeff ** i))
            times = np.arange(0.0, duration, 1 / (duration - i))
            if exp_mode:
                times = np.arange(0.0, duration, 1 / x)
            amplitudes = np.cos(times * 2.0 * np.pi)
            if len(amplitudes) > duration:
                amplitudes = amplitudes[:duration]
            amplitudes = np.reshape(amplitudes, [-1, 1])
            amp_list = np.append(amp_list, amplitudes, axis=1)
        return amp_list

    @classmethod
    def generate_wave(cls, duration, cycle, sin_wave=True, initial_angle=0, window='hann'):
        frequency = cycle / duration
        hanning = np.hanning(duration)
        hamming = np.hamming(duration)
        duration = np.arange(duration)
        # print(2 * np.pi * duration * frequency)
        # duration = np.add(duration, 1.57)
        if sin_wave:
            amp = (np.sin(2 * np.pi * duration * frequency + initial_angle * np.pi / 180.)).astype(np.float32)
            if window == 'hann':
                amp = np.multiply(hanning, amp)
            elif window == 'hamming':
                amp = np.multiply(hamming, amp)
            return amp
        else:
            amp = (np.cos(2 * np.pi * duration * frequency)).astype(np.float32)
            if window == 'hann':
                amp = np.multiply(hanning, amp)
            elif window == 'hamming':
                amp = np.multiply(hamming, amp)
            return amp

    @classmethod
    def initial_staggered_wave(cls, cycle_nums, duration):
        total_num = cycle_nums + cycle_nums
        result = np.empty(shape=(duration, 0))
        for i in range(0, total_num):
            samples = cls.generate_wave(duration, int(i / 2), sin_wave=True)
            if i % 2 == 1:
                samples = cls.generate_wave(duration, int(i / 2), sin_wave=False)
            samples = np.reshape(samples, [-1, 1])
            # print(samples, result.shape, total_num)
            result = np.append(result, samples, axis=1)
        return result

    @classmethod
    def initial_wave(cls, cycle_nums, duration, initial_cycle=0, sin_wave=True, exp=False, inverse=False, exp_coeff=1,
                     initial_angle=0,
                     window='hann'):
        result = np.empty(shape=(duration, 0))
        if inverse:
            current_cycle = cycle_nums
            while current_cycle > 1:
                samples = cls.generate_wave(duration, current_cycle, sin_wave=sin_wave, initial_angle=initial_angle,
                                            window=window)
                samples = np.reshape(samples, [-1, 1])
                result = np.append(result, samples, axis=1)
                current_cycle *= 0.98
                print(current_cycle)
            return result
        if exp:
            current_cycle = 1
            pre_cycle = -1
            cycles = []
            index = 1
            while current_cycle < cycle_nums:
                # cycle = current_cycle
                # print(cycle)
                cycle = int(current_cycle)
                if cycle == pre_cycle:
                    current_cycle *= exp_coeff
                    continue
                # print("%d: %d" % (index, cycle))
                index += 1
                cycles.append(cycle)
                samples = cls.generate_wave(duration, cycle, sin_wave=sin_wave, initial_angle=initial_angle,
                                            window=window)
                samples = np.reshape(samples, [-1, 1])
                result = np.append(result, samples, axis=1)
                current_cycle *= exp_coeff
                pre_cycle = cycle
            return result
        else:
            for i in range(initial_cycle, cycle_nums, exp_coeff):
                samples = cls.generate_wave(duration, i, sin_wave=sin_wave, initial_angle=initial_angle, window=window)
                if i >= 512:
                    samples = np.multiply(2, samples)
                samples = np.reshape(samples, [-1, 1])
                # print(samples, result.shape, total_num)
                result = np.append(result, samples, axis=1)
            return result

            # plt.plot(np.hamming(200))
            # plt.show()

            # InitialGenerator.initial_wave(cycle_nums=1024, duration=2048, sin_wave=True, exp=True,
            #                               window='hann', exp_coeff=1.00523)
            # plt.plot(InitialGenerator.generate_wave(200, 5, initial_angle=0, window='hamming'))
            # plt.show()
