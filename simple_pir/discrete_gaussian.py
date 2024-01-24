import numpy as np


class Sampler:
    """
    To generate the discrete Gaussian distribution with zero mean and standard error of 6.4 as in the original paper.
    """
    cdf_table = [
        0.5, 0.987867, 0.952345, 0.895957, 0.822578, 0.736994, 0.644389, 0.549831, 0.457833, 0.372034,
        0.295023, 0.22831, 0.172422, 0.127074, 0.0913938, 0.0641467, 0.0439369, 0.0293685, 0.0191572,
        0.0121949, 0.00757568, 0.00459264, 0.00271706, 0.00156868, 0.000883826, 0.000485955, 0.000260749,
        0.000136536, 6.97696e-05, 3.47923e-05, 1.69316e-05, 8.041e-06, 3.72665e-06, 1.68549e-06,
        7.43923e-07, 3.20426e-07, 1.34687e-07, 5.52484e-08, 2.21163e-08, 8.63973e-09,
        3.29371e-09, 1.22537e-09, 4.44886e-10, 1.57625e-10, 5.45004e-11, 1.83896e-11,
        6.05535e-12, 1.94583e-12, 6.10194e-13, 1.86736e-13, 5.57679e-14, 1.62532e-14,
        4.62263e-15, 1.28303e-15, 3.47522e-16, 9.18597e-17, 2.36954e-17, 5.96487e-18,
        1.46533e-18, 3.5129e-19, 8.21851e-20, 1.87637e-20, 4.18062e-21, 9.08991e-22, 1.92875e-22,
        3.99383e-23, 8.07049e-24, 1.5915e-24, 3.06275e-25, 5.75194e-26, 1.05418e-26, 1.88542e-27,
        3.29081e-28, 5.60522e-29, 9.31708e-30, 1.51135e-30, 2.39247e-31, 3.69594e-32,
        5.57187e-33, 8.19735e-34, 1.17691e-34, 1.64896e-35, 2.25463e-36, 3.00841e-37,
        3.91737e-38, 4.97795e-39, 6.1731e-40, 7.47055e-41, 8.82266e-42, 1.01682e-42, 1.14363e-43,
        1.25523e-44, 1.34449e-45, 1.40537e-46, 1.43357e-47, 1.42708e-48, 1.38634e-49,
        1.31429e-50, 1.21593e-51, 1.0978e-52, 9.67246e-54, 8.31661e-55, 6.97835e-56, 5.71421e-57,
        4.56622e-58, 3.56086e-59, 2.70987e-60, 2.01252e-61, 1.45858e-62, 1.03161e-63,
        7.12032e-65, 4.79601e-66, 3.15252e-67, 2.02224e-68, 1.26591e-69, 7.73344e-71, 4.6104e-72,
        2.68226e-73, 1.52287e-74, 8.4376e-76, 4.56219e-77, 2.40727e-78, 1.23958e-79, 6.22901e-81,
        3.05465e-82, 1.46185e-83, 6.82713e-85, 3.11152e-86, 1.3839e-87,
    ]
    def __init__(self):
        self.np_generator = np.random.default_rng()

    def sample_dgaussian(self):
        while True:
            x = self.np_generator.integers(0, len(self.cdf_table))
            y = self.np_generator.random(dtype=np.float64)
            if y < self.cdf_table[x]:
                break
        if self.np_generator.random() > 0.5:
            x = - x
        return x


class NpSampler:
    """
    This is a numpy implementation, which is faster, by rounding random normal floats into integers
    Caution: the produced distribution could be slightly different to the original implementation
    """
    def __init__(self):
        self.np_generator = np.random.default_rng()

    def sample_dgaussian(self, shape=1):
        normal_floats = self.np_generator.standard_normal(shape, dtype=np.float64)
        normal_ints = np.round(normal_floats * 6.4).astype(np.int32)
        return normal_ints


default_sampler = NpSampler()


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt


    def test_original_sampler():
        print("Test original sampler...")
        sampler = Sampler()

        start_time = time.time()
        samples = [sampler.sample_dgaussian() for _ in range(1000_000)]
        end_time = time.time()
        print(f"Time consumed for 1M samples: {end_time-start_time:2.4f}s. "
              f"Mean {np.mean(samples):.4f}  Std. {np.std(samples):.4f}")

        plt.hist(samples, bins=101, range=(-50.5, 50.5))
        plt.show()


    def test_np_sampler():
        print("Test NPSampler...")
        sampler = NpSampler()

        start_time = time.time()
        samples = sampler.sample_dgaussian(1000_000)
        end_time = time.time()
        print(f"Time consumed for 1M samples: {end_time - start_time:.4f}s. "
              f"Mean {np.mean(samples):.4f}  Std. {np.std(samples):.4f}")

        plt.hist(samples, bins=101, range=(-50.5, 50.5))
        plt.show()

    test_original_sampler()
    test_np_sampler()
