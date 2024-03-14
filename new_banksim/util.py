import numpy as np


class Util:
    id = 0

    @staticmethod
    def get_random_uniform(max_size):
        return np.random.uniform(0, max_size)

    @staticmethod
    def get_random_log_normal(mean, standard_deviation):
        return np.random.lognormal(mean, standard_deviation)

    @staticmethod
    def get_random_default_probability(a, b):
        return np.random.beta(a, b)

    @classmethod
    def get_unique_id(cls):
        cls.id += 1
        return cls.id
