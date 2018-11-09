from sklearn.model_selection import BaseCrossValidator
from CGRtools.preparer import CGRpreparer
from collections import defaultdict
from random import shuffle as r_shuffle
import numpy as np
from sklearn.utils.validation import indexable


class TransformationOut(BaseCrossValidator):
    def __init__(self, n_splits=5, n_repeats=5, shuffle=False):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.n_repeats = n_repeats

    def split(self, X, y=None, groups=None):
        cgr = CGRpreparer()
        X, y, groups = indexable(X, y, groups)
        cgrs = [cgr.condense(r) for r in X]

        structure_condition = defaultdict(set)
        condition_structure = defaultdict(set)

        for structure, condition in zip(cgrs, groups):
            structure_condition[structure].add(condition)
            condition_structure[condition].add(structure)

        train_data = defaultdict(list)
        test_data = []

        for n, (structure, condition) in enumerate(zip(cgrs, groups)):
            train_data[structure].append(n)
            if len(condition_structure[condition]) > 1:
                test_data.append(n)

        if self.n_splits > len(train_data):
            raise ValueError("Cannot have number of splits n_splits=%d greater"
                             " than the number of transformations: %d."
                             % (self.n_splits, len(train_data)))

        structures_weight = {x: len(y) for x, y in train_data.items()}
        train_indexes = list(train_data.keys())

        fold_mean_size = len(cgrs) // self.n_splits

        for idx in range(self.n_repeats):
            if self.shuffle:
                r_shuffle(train_indexes)
            fold_len = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            n = 0
            fold = {}

            for i in train_indexes:
                if fold_len[n] + structures_weight[i] <= fold_mean_size:
                    fold.setdefault(n, []).extend(train_data[i])
                    fold_len[n] += structures_weight[i]
                elif n == self.n_splits - 1:
                    break
                else:
                    n += 1
                    fold.setdefault(n, []).extend(train_data[i])
                    fold_len[n] += structures_weight[i]

            for i in fold:
                train_index = []
                test = []
                for j in range(self.n_splits):
                    if not j == i:
                        train.extend(fold[j])
                    else:
                        for c in fold[j]:
                            if groups is not None:
                                if c in test_data:
                                    test.append(c)
                            else:
                                test.append(c)
                yield train_index, test_index