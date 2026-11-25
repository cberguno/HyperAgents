# Trusted preprocessor. Transform a past-only 28-series package; no dates or names.
import json, math

def transform(raw):
 p=json.loads(raw)
 if not isinstance(p,dict) or set(p)!={'schema','columns','series'} or p['schema']!='weekly-series-v1' or p['columns']!=['weekly_return','mean_feature']: raise ValueError('bad source schema')
 series=p['series']
 if not isinstance(series,list) or len(series)!=28 or not all(isinstance(s,list) and s for s in series) or len({len(s) for s in series})!=1: raise ValueError('bad shape')
 output=[]
 for history in series:
  values=[]; prior=[]
  for row in history:
   if not isinstance(row,list) or len(row)!=2 or any(x is not None and (type(x) not in (float,int) or not math.isfinite(x)) for x in row): raise ValueError('bad value')
   r,f=row; prior.append(r)
   trailing=prior[-4:]; valid=[v for v in trailing if v is not None]
   avg=sum(valid)/len(valid) if len(valid)==len(trailing) else None
   spread=(sum((v-avg)**2 for v in valid)/len(valid))**0.5 if avg is not None else None
   values.append([r,f,avg,spread])
  output.append(values)
 return {'schema':'anonymous-series-v2','columns':['value_1','value_2','value_3','value_4'],'series':output}
