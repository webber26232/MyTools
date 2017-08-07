import pandas as pd
import requests
from sklearn.externals.joblib import Parallel, delayed


def task_distribute(fmtedAdrs,keys,start_index,step):
    url = 'https://maps.googleapis.com/maps/api/geocode/json?address={0}&key='
    keyI=0
    l=[]
    for i in range(start_index,len(fmtedAdrs),step):
        adrs = fmtedAdrs[i]
        result = _get_coordinate(url.format(adrs.replace(' ','+')),keys,keyI)
        while result[0] == 'keyIncrement':
            result = result[1]
            keyI += 1
        if result[0] == 'ok':
            result[1]['Address'] = adrs
            l.append(result[1])
            print(result[1])
        else:
            l.append(pd.Series([adrs],index=['Address']))
            print(result[0])
    return l

def _get_coordinate(url,keys,keyI,attempt_time=1):
    if attempt_time>5:
        return 'urlError',url
    if keyI >= len(keys):
        return 'no more keys', None
    try:
        jsondict = requests.get(url+keys[keyI]).json()
    except:
        return _get_coordinate(url,keys,keyI,attempt_time+1)
    
    if jsondict['status'] == 'OK':
        results = jsondict['results'][0]
        s = pd.Series([])
        s['lat'] = results['geometry']['location']['lat']
        s['lng'] = results['geometry']['location']['lng']
        return 'ok',s
    elif jsondict['status'] == 'OVER_QUERY_LIMIT':
        return 'keyIncrement',_get_coordinate(url,keys,keyI+1,attempt_time)
    elif jsondict['status'] == 'ZERO_RESULTS':
        return 'zero', None
    elif jsondict['status'] == 'UNKNOWN_ERROR':
        return _get_coordinate(url,keys,keyI,attempt_time+1)
    else:
        return 'keyError,parameterError',None
    
if __name__ == '__main__':
    adrsTable = pd.read_csv('fmtedAddress.csv')
    keys=['AIzaSyC1ugNifJdJgT38RRsa2cRfNpjifzQXKCU','AIzaSyBsXZFvuHxXKfedMpX3ajutUeNeE7QBIek','AIzaSyA02G2uESQ7KxAh7p2PLWKZsuDZtoXGq4Y','AIzaSyDGCcRGIN1BVOjZgKJSp3Fsd3OWhxXmFNc']
    n_jobs = 10
    l = []
    
    try:
        table = Parallel(n_jobs=n_jobs)(delayed(task_distribute)(adrsTable.Address[adrsTable.lat.isnull()].dropna().unique(),keys,start_index,n_jobs) for start_index in range(n_jobs))
        for s_list in table:
            l.extend(s_list)      
    finally:
        if len(l)>0:
            pd.concat(l,axis=1).T.to_csv('coordinate.csv',index=False)