import numpy as np
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed
class ParallelSplit():
    def __init__(self, n_jobs, method='chunk', axis=0):
        if method not in ['chunk', 'step']:
            raise ValueError('method has to be chunk or step')
        self.n_jobs = n_jobs
        self.method = method
        self.axis = axis

    def _indexer(self, n_jobs, arr_length, fold_num, method):
        '''
        return the splited index for mulptithread computing.
        n_jobs: number of splits
        arr_length: the length of the array to be partition
        fold_num: the number 
        '''
        if method=='chunk':
            chunk_size = arr_length // n_jobs
            start_index = fold_num * chunk_size
            end_index = (fold_num + 1) * chunk_size
            if arr_length - end_index < n_jobs:
                return np.arange(start_index, arr_length)
            return np.arange(start_index, end_index)
        elif method == 'step':
            return np.arange(fold_num, arr_length, n_jobs)

    def split(self, X):
        return (self._indexer(self.n_jobs,
                              arr_length=X.shape[self.axis],
                              fold_num=i,
                              method=self.method) for i in range(self.n_jobs))

def _apply(df, func, axis, **kwds):
    return df.apply(func, axis=axis, **kwds)

def parallel_apply(df, func, axis=1, n_jobs=1, verbose=0, **kwds):
    split_axis = 1 - axis
    output = Parallel(n_jobs=n_jobs,
                      verbose=verbose)(
        delayed(_apply)(df.iloc[index, :] if axis==1 else df.iloc[:, index],
                        func,
                        axis,
                        **kwds) for index in ParallelSplit(n_jobs,
                                                         axis=split_axis).split(df))
    return pd.concat(output, axis=split_axis)