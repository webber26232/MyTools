import pandas as pd
import requests
from ..multiprocess import indexer
from sklearn.externals.joblib import Parallel, delayed


def format_adrs(column,city,keys,retry_times=5,n_jobs=1,save_file=None):
    try:
        if save_file is not None:
            addressTable = pd.read_csv('fmtedAdrs.csv',encoding='gbk')
            existedAdrs = set(addressTable['original_adrs'].fillna(''))
    except:
        addressTable = pd.DataFrame()
        existedAdrs = set()
    newAdrs = list(set(column.str.replace('\(.*?\)','').str.replace('&',' and ').str.strip().str.lower().str.replace(' +',' ')) - existedAdrs)
    print(newAdrs)
    if len(newAdrs) > 0:
        lines=[]
        try:
            table = Parallel(n_jobs=n_jobs)(delayed(_formating_distribute)(city,newAdrs.iloc[indexer(n_jobs,newAdrs.shape[0],i)],keys,retry_times) for i in range(n_jobs))
            for s_list in table:
                lines.extend(s_list)
        finally:
            if len(lines)>0:
                addressTable = pd.concat([addressTable,pd.concat(lines,axis=1).T])
                addressTable.to_csv('fmtedAdrs.csv',index=False)
                return column.map(addressTable.reset_index('original_adrs')['formatted_adrs'])
    

def _formating_distribute(city,newAdrs,keys,retry_times):
    url = 'https://maps.googleapis.com/maps/api/geocode/json?address={0}&key='
    keyI=0
    l=[]
    try:
        for i in range(len(newAdrs)):
            adrs = newAdrs[i]
            print(adrs)
            if city.lower() not in adrs.lower():
                city_adrs = adrs +' ' + city
            else:
                city_adrs = adrs
            result = _get_adrs_information(url.format(city_adrs.replace(' ','+')),keys,keyI,retry_times)
            while result[0] == 'keyIncrement':
                result = result[1]
                keyI += 1
            if result[0] == 'ok':
                result[1]['original_adrs'] = adrs
                l.append(result[1])
                print('success')
            elif result[0] == 'no more keys':
                print('no more keys')
                break
            else:
                print(result[0],adrs)
    finally:
        return l

def _get_adrs_information(url,keys,keyI,retry_times):
    if retry_times<1:
        return 'urlError',url
    if keyI >= len(keys):
        return 'no more keys', None
    try:
        jsondict = requests.get(url+keys[keyI]).json()
    except:
        return _get_adrs_information(url,keys,keyI,retry_times-1)

    if jsondict['status'] == 'OK':
        results = jsondict['results'][0]
        s = pd.Series([results['formatted_address']],index=['formatted_adrs'])
        for element in results['address_components']:
            s[element['types'][0]]=element['short_name']
        s['lat'] = results['geometry']['location']['lat']
        s['lng'] = results['geometry']['location']['lng']
        return 'ok', s
    elif jsondict['status'] == 'OVER_QUERY_LIMIT':
        return 'keyIncrement',_get_adrs_information(url,keys,keyI+1,retry_times)
    elif jsondict['status'] == 'ZERO_RESULTS':
        return 'zero', None
    elif jsondict['status'] == 'UNKNOWN_ERROR':
        return _get_adrs_information(url,keys,keyI,retry_times-1)
    else:
        return 'keyError,parameterError',None

if __name__=='__main__':
    train = pd.read_json('train.json')
    test = pd.read_json('test.json')
    df = pd.concat([train,test],axis=0)
    keys=[]
    n_jobs = 20
    retry_times = 5
    format_adrs(df.street_address,'New York',keys,retry_times,n_jobs)
