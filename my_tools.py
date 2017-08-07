

def confusionMatrix(actual,predict,recall=True,precision=True,text=False):	
	dict = {}
	dict['accuarcy'] = (actual==predict).value_counts()[True]/float(predict.size)
	confustion = pd.crosstab(actual,predict,margins=True,rownames='Acutual',colnames='Predict')
	for label in confustion.index:
		if label not in confustion.columns:
			confustion[label] = 0
	dict['confustion'] = confustion[confustion.index.tolist()]
	if precision:
		dict['precision'] = confustion.iloc[:-1].divide(confustion.iloc[-1],axis=1).fillna(0)
	if recall:
		dict['recall'] = confustion.iloc[:,:-1].divide(confustion.iloc[:,-1],axis=0).fillna(0)
	if text:
		return {'confusion':dict['confustion'],'precision':dict['precision'].applymap(lambda x: "{0:.2f}%".format(x*100)),'recall':dict['recall'].applymap(lambda x: "{0:.2f}%".format(x*100)),'accuracy':"{0:.2f}%".format(dict['accuarcy']*100)}
	else:
		return dict
		
def encode(s):
	u = s.unique()
	d = {}
	d['c2d'] = dict(zip(u,range(len(u))))
	d['d2c'] = dict(zip(range(len(u)),u))
	return d
	
def is_holiday(s,format=None):
    s = pd.to_datetime(s,format)
    from pandas.tseries.holiday import USFederalHolidayCalendar
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=s.min(), end=s.max())
    return s.isin(holidays)*1
	
def is_business_day(s,format=None):
    s = pd.to_datetime(s,format)
    return s.dt.dayofweek.replace({2:1,3:1,4:1,5:1,6:0}) & (is_holiday(s)==0) * 1