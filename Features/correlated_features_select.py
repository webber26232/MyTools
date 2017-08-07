import numpy as np
import pandas as pd
def _test_grouped_features(base_X,y,feature_group,score_function,best_score=None,max_features=4,method='iter'):
    if not isinstance(feature_group,pd.DataFrame):
        raise TypeError('feature_group must be a DataFrame')
    if not isinstance(method,str):
        raise TypeError('method must be "iter" or "rec"')
    if method != 'iter' and method != 'rec':
        raise ValueError('method must be "iter" or "rec"')
    first_round_scores = np.array([]) 
    for feature in feature_group:
        train_X = pd.concat([base_X,feature_group[feature]],axis=1)
        print(method,'working on',feature)
        first_round_scores = np.append(first_round_scores,score_function(train_X,y))
    
    print(first_round_scores)
    feature_to_keep = (feature_group.T[first_round_scores < best_score]).T #CDF
    print(feature_to_keep.columns)
    if feature_to_keep.shape[1] == 0:
        return feature_to_keep, best_score, max_features
    elif feature_to_keep.shape[1] == 1:
        return feature_to_keep, first_round_scores.min(), max_features
    else:
        scores_to_keep = first_round_scores[first_round_scores < best_score]
        max_features -= 1
        if max_features == 0:
            min_index = scores_to_keep.argmin()       
            return feature_to_keep.iloc[:,min_index:min_index+1], scores_to_keep.min(), max_features
        if method == 'rec':
            second_round_scores = np.array([])
            second_round_features = []
            rounds = []
            for i in range(feature_to_keep.shape[1]-1):
                base_feature = feature_to_keep.iloc[:,i:i+1]
                print('\nBased on',base_feature.columns)
                features, score, rec_round = _test_grouped_features(pd.concat([base_X,base_feature],axis=1),
                                                           feature_to_keep.iloc[:,i+1:],
                                                           best_score=scores_to_keep[i],
                                                           max_features=max_features,
                                                           method=method)
                if score < scores_to_keep[i]:
                    second_round_scores = np.append(second_round_scores,score)
                    second_round_features.append(pd.concat([base_feature,features],axis=1))
                    rounds.append(rec_round)
            if len(second_round_features) == 0 or second_round_scores.min() >= scores_to_keep.min():
                min_index = scores_to_keep.argmin()
                return feature_to_keep.iloc[:,min_index:min_index+1],scores_to_keep.min(), max_features
            else:
                return second_round_features[second_round_scores.argmin()], second_round_scores.min(), rounds[second_round_scores.argmin()]
        elif method == 'iter':
            min_score = scores_to_keep.min()
            min_index = scores_to_keep.argmin()
            base_feature = feature_to_keep.iloc[:,min_index:min_index+1]
            print('\nBased on',base_feature.columns)
            features, score, iter_round = _test_grouped_features(pd.concat([base_X,base_feature],axis=1),
                                                       feature_to_keep.drop(feature_to_keep.columns[min_index],axis=1),
                                                       best_score=scores_to_keep[i],
                                                       max_features=max_features,
                                                       method=method)
            if score < min_score:
                return pd.concat([base_feature,features],axis=1),score, iter_round
            else:
                return base_feature, min_score, iter_round
            
def test_grouped_features(base_X,y,feature_group,score_function,best_score=None,max_features=-1,rec_round=2):
    if max_features < 0:
        max_features = feature_group.shape[1]    
    if rec_round > 0 and feature_group.shape[1]<=20:
        features, score, rounds = _test_grouped_features(base_X,
                                                         y,
                                                         feature_group,
                                                         score_function=score_function,
                                                         best_score=best_score,
                                                         max_features=rec_round,
                                                         method='rec')
        if score < best_score and rounds == 0 and max_features - rec_round > 0:
            iter_features, score, _= _test_grouped_features(pd.concat([base_X,features],axis=1),
                                                                      y,
                                                                      feature_group.drop(features.columns,axis=1),
                                                                      score_function=score_function,
                                                                      best_score=score,
                                                                      max_features=max_features-rec_round,
                                                                      method='iter')
            features = pd.concat([features,iter_features],axis=1)
    else:
        features, score, _ =  _test_grouped_features(base_X,
                                                     y,
                                                     feature_group,
                                                     score_function=score_function,
                                                     best_score=best_score,
                                                     max_features=max_features,
                                                     method='iter')
    return features, score