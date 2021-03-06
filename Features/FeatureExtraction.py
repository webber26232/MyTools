﻿from sklearn.base import BaseEstimator, TransformerMixin
from collections import Iterable
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

class HCCTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, grouping_column, label_to_use=None, threshold='mean',
                 coeficient=1, alpha=0.01,
                 low_observations=1 ,fill_value=-1, inplace=True):

        if isinstance(grouping_column, Iterable) \
        and not isinstance(grouping_column, str):
            self.grouping_column = [x for x in grouping_column]
        else:
            self.grouping_column = grouping_column

        if low_observations is not None and not (isinstance(fill_value, int)) \
        and (not isinstance(fill_value,float)):
            raise ValueError('low_observations must be None or a digit number')

        if not (isinstance(fill_value, int)) \
        and (not isinstance(fill_value, float)) \
        and (fill_value != 'global'):
            raise ValueError('fill_value must be "global" or a digit number')

        if not isinstance(label_to_use, Iterable) \
        or isinstance(label_to_use,str):
            label_to_use = [label_to_use]

        self.label_to_use = label_to_use
        self.threshold = threshold
        self.coeficient = coeficient
        self.alpha = alpha
        self.low_observations = low_observations
        self.fill_value = fill_value
        self.inplace = inplace
        
        self.pre_fix = _prefix_genertor(self.grouping_column)


    def _reset(self):

        if hasattr(self, 'mapping_'):
            self.mapping_ = None
            self.global_ratio = None

    def fit(self, X, y):
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        if X.shape[0] != y.size:
            raise ValueError('X and y must have same size')
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X has to be a pandas DataFrame')
            
        self._reset()
        
        if self.label_to_use is not None:
            extra_labels = set(self.label_to_use) - set(y)
            if len(extra_labels)>0:
                print('Runtime Warning: Output contains labels '
                      'out of y: {0}'.format(list(extra_labels)))
        else:
            self.label_to_use = list(set(y))
            extra_labels = set([])
            

        group_count = _get_posterior(X, self.grouping_column, y, extra_labels)
        record_count = group_count.sum(axis=1)
        self.global_ratio = group_count.sum(axis=0) / y.size
        for additional_label in extra_labels:
            self.global_ratio[additional_label] = 0        
        
        if isinstance(self.threshold,str):
            if self.threshold == 'median':
                threshold_value = record_count.median()
            elif self.threshold == 'mean':
                threshold_value = y.size/group_count.index.size
        elif isinstance(self.threshold,int):
            threshold_value = self.threshold
        elif isinstance(self.threshold,float):
            threshold_value = y.size * self.threshold
        lambda_value =  1.0 / (1.0
                               + np.exp((threshold_value - record_count)
                               * self.coeficient))    

        for label in self.label_to_use:
            ratio = group_count[label] / record_count
            weight = (ratio * lambda_value +
                      (1 - lambda_value) * self.global_ratio[label])
            randoms = (1 +
                       (np.random.uniform(size=group_count.shape[0]) - 0.5)
                       * self.alpha)
            group_count[self.pre_fix + str(label)] = weight * randoms
        
        self.mapping_ = group_count[[self.pre_fix + str(label) for label in self.label_to_use]]
        if self.low_observations is not None and self.fill_value != 'global':
            if isinstance(self.low_observations,float):
                observation_count = record_count.quantile(self.low_observations)
            else:
                observation_count = self.low_observations
            self.mapping_.where(record_count>observation_count,
                                self.fill_value,
                                inplace=True)
        return self

    def transform(self, X):
        
        _X = pd.merge(X, self.mapping_, how='left',
                     left_on=self.grouping_column,
                     right_index=True)
        
        if isinstance(self.fill_value, int) \
        or isinstance(self.fill_value, float):
            for label in self.label_to_use:
                _X[self.pre_fix+str(label)].fillna(self.fill_value,
                                                   inplace=True)
        else:
            for label in self.label_to_use:
                _X[self.pre_fix + str(label)].fillna(self.global_ratio[label],
                                                    inplace=True)

        if self.inplace:
            if isinstance(self.grouping_column, Iterable) \
            and not isinstance(self.grouping_column, str):
                for column in self.grouping_column:
                    column_name = ''
                    if isinstance(column,str) \
                    or isinstance(column,int) \
                    or isinstance(column,float):
                        column_name = column
                    elif isinstance(column, pd.Series):
                        column_name = column.name
                    else:
                        continue
                    if column_name in _X.columns:
                        _X.drop(column_name,axis=1,inplace=True)
            elif isinstance(self.grouping_column,str) \
            or isinstance(self.grouping_column,int) \
            or isinstance(self.grouping_column,float):
                if self.grouping_column in _X.columns:
                    _X.drop(self.grouping_column,axis=1,inplace=True)
        return _X


def _prefix_genertor(grouping_column):
    pre_fix = ''
    if isinstance(grouping_column,Iterable) and not isinstance(grouping_column,str):
        for i in range(len(grouping_column)):
            if isinstance(grouping_column[i],str) \
            or isinstance(grouping_column[i],int) \
            or isinstance(grouping_column[i],float):
                pre_fix += (str(grouping_column[i]) + '_')
            elif isinstance(grouping_column[i],pd.Series):
                pre_fix += (grouping_column[i].name + '_')
            else:
                pre_fix += (str(i) + '_')
    else:
        if isinstance(grouping_column,str) \
        or isinstance(grouping_column,int) \
        or isinstance(grouping_column,float):
            pre_fix += (str(grouping_column) + '_')
        elif isinstance(grouping_column,pd.Series):
            pre_fix += (grouping_column.name + '_')
        else:
            pre_fix += (str(0) + '_')
    return pre_fix

def _get_posterior(X, grouping_column, y, extra_labels=[]):
    if isinstance(grouping_column, Iterable) \
    and not isinstance(grouping_column,str):
        groups = X.groupby([y] + grouping_column).size()
    else:
        groups = X.groupby([y] + [grouping_column]).size()
    group_count = groups.unstack(level=0, fill_value=0)
    for additional_label in extra_labels:
        group_count[additional_label] = 0
    return group_count


def CV_basian_encoding(grouping_column, label_to_use,
                       train, y, test=None,
                       fill_value=None, cv=None, inplace=False):
    if not isinstance(y,np.ndarray):
        y = np.array(y)
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=888)
    elif isinstance(cv,int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=888)
    pre_fix = _prefix_genertor(grouping_column)
    if not isinstance(label_to_use,Iterable) or isinstance(label_to_use,str):
        label_to_use = [label_to_use]
        
    train_code = np.zeros((train.shape[0], len(label_to_use)))
    prefixed_labels = [pre_fix + str(label) for label in label_to_use]
    for train_index, test_index in cv.split(train, y):
        group_count = _get_posterior(train.iloc[train_index],
                             grouping_column,
                             y[train_index])
        group_count[prefixed_labels] = group_count[label_to_use] \
            .divide(group_count.sum(axis=1), axis=0)
        train_code[test_index] = pd.merge(train.iloc[test_index],
                                       group_count[prefixed_labels],
                                       how='left', left_on=grouping_column,
                                       right_index=True)[prefixed_labels].values

    train_code = pd.DataFrame(train_code,
                              columns=prefixed_labels,
                              index=train.index)
    if fill_value is not None:
        train_code.fillna(fill_value, inplace=True)

    test_code = None
    if test is not None:
        test_code = np.zeros((test.shape[0], len(label_to_use)))
        group_count = _get_posterior(train, grouping_column, y)
        group_count[prefixed_labels] = group_count[label_to_use] \
            .divide(group_count.sum(axis=1), axis=0)
        test_code[:] = test.merge(group_count[prefixed_labels], how='left',
                          left_on=grouping_column,
                          right_index=True)[prefixed_labels].values
        test_code = pd.DataFrame(test_code,
                                 columns=prefixed_labels,
                                 index=test.index)
        if fill_value is not None:
            test_code.fillna(fill_value, inplace=True)
    return train_code, test_code

def _get_grouped(X,grouping_column, method='min'):
    if isinstance(grouping_column, Iterable) \
    and not isinstance(grouping_column, str):
        groups = X.groupby(grouping_column)
    else:
        groups = X.groupby([grouping_column])
    return groups.min()


class CountEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value=None):
        self.fill_value = fill_value
    
    def fit(self,y):
        count = y.value_counts()
        self.count_code = pd.Series(range(count.size), index=count.index)
        return self
    def transform(self, y):
        code = y.map(self.count_code)
        nulls = code.isnull()
        if nulls.any():
            if isinstance(self.fill_value, int):
                return code.fillna(self.fill_value)
            elif isinstance(self.fill_value, str) and self.fill_value == 'auto':
                na_index = code[nulls].index
                count = y.loc[na_index].value_counts()
                na_count_code = -pd.Series(range(1, count.size+1),
                                           index=count.index)
                return count.replace(na_count_code)
        else:
            return code
        
def add_feature(df, cat, met, method='rank', asc_flg=True):
    if method == 'rank':
        return df.groupby(cat)[met].rank(method='average',
                         pct=True,
                         ascending=asc_flg)
    elif method == 'diff':
        return df[met] - df[cat].map(df.groupby(cat)[met].median())
    elif method == 'median':
        return df[cat].map(df.groupby(cat)[met].median())
    elif method == 'std':
        return df[cat].map(df.groupby(cat)[met].std()).fillna(0)