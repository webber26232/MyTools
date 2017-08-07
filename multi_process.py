import numpy as np
class ParallelSplit():
    def __init__(self,n_jobs,method='chunk'):
        self.n_jobs = n_jobs
        self.method = method
   
    def _indexer(self,n_jobs,arr_length,fold_num,method='chunk'):
        '''
        return the splited index for mulptithread computing.
        n_jobs: number of splits
        arr_length: the length of the array to be partition
        fold_num: the number 
        '''
        if method=='chunk':
            chunk_size = int(arr_length / n_jobs)
            start_index = fold_num * chunk_size
            end_index = (fold_num + 1) * chunk_size
            if arr_length - end_index < chunk_size:
                return np.arange(start_index, arr_length)
            return np.arange(start_index, end_index)
        elif method == 'step':
            return np.arange(fold_num, arr_length, n_jobs)
    
    def split(self,X):
        return (self._indexer(self.n_jobs,X.shape[0],i,method=self.method) for i in range(self.n_jobs))