import pandas as pd
import requests
from ..multiprocess import indexer
from geopy import distance
from sklearn.externals.joblib import Parallel, delayed


def ambience_collect(address_cod_frame,key_column,ifm_dict,keys,retry_times,n_jobs):
    lines = []
    try:
        tmp = address_cod_frame.drop_duplicates(subset=key_column)[['latitude','longitude',key_column]]
        table = Parallel(n_jobs=n_jobs)(delayed(task_distributor)(tmp.iloc[indexer(n_jobs,tmp.shape[0],i)],ifm_dict,keys,retry_times) for i in range(n_jobs))
        for s_list in table:
            lines.extend(s_list)
    finally:
        if len(lines)>0:
            ambs = pd.concat(lines,axis=1).T.drop(['latitude','longitude'],axis=1)
            ambs.to_csv('ambience.csv',index=False)
            address_cod_frame.merge(ambs,how='left',left_on=key_column,right_on=key_column).to_csv('cood_with_ambs.csv',index=False)

def task_distributor(address_cod_frame,ifm_dict,keys,retry_times,start_index,step):
    url = 'https://maps.googleapis.com/maps/api/place/radarsearch/json?location={0},{1}&type={2}&radius={3}&key='
    l=[]
    keyI = 1
    for i in range(address_cod_frame.shape[0]):
        s = address_cod_frame.iloc[i].copy()
        for ifm in ifm_dict:
            radius = ifm_dict[ifm][0]
            result = _get_ambs(url.format(s['latitude'],s['longitude'],ifm,radius),keys,keyI,retry_times)
            
            while result[0] == 'keyIncrement':
                result = result[1]
                keyI += 1
            if result[0] == 'ok':
                df = result[1]
                s[ifm + '_' + str(radius)] = df.shape[0]
                if len(ifm_dict[ifm])>1:
                    radius_small = ifm_dict[ifm][1]
                    s[ifm + '_' + str(radius_small)] = df[(df.apply(lambda x:distance.vincenty(x,s[['latitude','longitude']]).meters,axis=1) <= radius_small)].shape[0]
            elif result[0] == 'zero':
                s[ifm + '_' + str(radius)] = 0
                if len(ifm_dict[ifm])>1:
                    s[ifm + '_' + str(ifm_dict[ifm][1])] = 0
            elif result[0] == 'urlError':
                print(ifm,radius)
            elif result[0] == 'no more keys':
                print(result[0])
                return l
            else:
                print(result[0])
        print(s)
        l.append(s)
    return l
    

    
    
def _get_ambs(url,keys,keyI,retry_times=5):
    if retry_times<1:
        return 'urlError',url
    if keyI>=len(keys):
        return 'no more keys', None
    try:
        #print(url+keys[keyI])
        results = requests.get(url+keys[keyI]).json()
        #results = {'status':0}
    except:
        return _get_ambs(url,keys,keyI,retry_times-1)

    if results['status'] == 'OK':
        return 'ok', pd.DataFrame([x['geometry']['location'] for x in results['results']])
    elif results['status'] == 'OVER_QUERY_LIMIT':
        return 'keyIncrement',_get_ambs(url,keys,keyI+1,retry_times)
    elif results['status'] == 'ZERO_RESULTS':
        return 'zero',None
    else:
        return 'keyError,parameterError',None

def mistake_fix(frame,key_column,keys,ifm_dict,retry_times):
    url = 'https://maps.googleapis.com/maps/api/place/radarsearch/json?location={0},{1}&type={2}&radius={3}&key='
    keyI = 1
    column_list = [ifm+'_'+str(x) for ifm in ifm_dict for x in ifm_dict[ifm]]
    null_columns = frame[column_list].columns[frame[column_list].isnull().any()]
    print(null_columns)
    for column in null_columns:
        element = column.split('_')
        type, radius = '_'.join(element[:-1]), element[-1]
        for key, table in frame[frame[column].isnull()].groupby(key_column):
            index = table.index
            lat = table['latitude'].mean()
            lng = table['longitude'].mean()
            print(url.format(lat,lng,type,radius))
            result = _get_ambs(url.format(lat,lng,type,radius),keys,keyI,retry_times)
            while result[0] == 'keyIncrement':
                result = result[1]
                keyI += 1
            if result[0] == 'ok':
                frame.loc[index,column] = result[1].shape[0]
            elif result[0] == 'zero':
                frame.loc[index,column] = 0
    return frame

        


if __name__=='__main__':
    n_jobs=30
    #df = pd.read_csv('cood.csv',encoding='gbk')
    df = pd.read_csv('cood_with_ambs.csv',encoding='gbk')
    keys=[]
    ifm_dict = {'convenience_store':[500],
    'home_goods_store':[500],
    'department_store':[2000],
    'bar':[500],'cafe':[500],'restaurant':[300],
    'train_station':[3000],'bus_station':[1000,300],
    'subway_station':[2000,500],
    'laundry':[1000],'bank':[1000],'pharmacy':[1000],'church':[1000],'school':[500]}
    #ifm_dict = {'grocery_or_suppermarket':[300]}
    retry_times = 5
    key_column = 'formatted_adrs'
    mistake_fix(df,key_column,keys,ifm_dict,retry_times).to_csv('cood_with_ambs_patch.csv',index=False)
    #ambience_collect(df,key_column,ifm_dict,keys,retry_times,n_jobs)

'''
ifm_dict = {'convenience_store':[500],
'grocery_or_suppermarket':[500],
'home_goods_store':[500],
'department_store':[2000],
'bar':[500],'cafe':[500],'restaurant':[300],
'train_station':[3000],'bus_station':[1000,300],
'subway_station':[2000,500],
'laundry':[1000],'bank':[1000],'pharmacy':[1000],'church':[1000],'school':[500]}'''
