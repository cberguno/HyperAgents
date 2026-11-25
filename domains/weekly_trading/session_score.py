# Trusted host only: never expose this module's inputs to candidate code.
import datetime as dt
import math
import random
import duckdb
import exchange_calendars as xc

NAMES='AEHR BAND BJRI CBRL COHU CSIQ FCEL FIVN GT HUBG ICHR INOD JBLU NKTR NTLA PLUG RARE REPL RUN SEDG SRPT STNE TNDM TRIP URGN WEN XRAY ZYME'.split()
FIRST=dt.date(2019,1,2)
LAST=dt.date(2026,11,23)

def score_week(path, week_start, selected, seed='1', cost_bps=5):
    week=dt.date.fromisoformat(week_start)
    if week.weekday()!=0 or week < FIRST or week+dt.timedelta(days=6)>LAST: raise ValueError('week must be a complete Monday-to-Sunday window within the data range')
    if not isinstance(selected,list) or any(type(i) is not int or i<0 or i>=len(NAMES) for i in selected) or len(set(selected))!=len(selected): raise ValueError('invalid selection')
    if type(cost_bps) not in (int,float) or not math.isfinite(cost_bps) or not 0<=cost_bps<5000: raise ValueError('invalid cost')
    cal=xc.get_calendar('XNAS',start=str(FIRST-dt.timedelta(days=7)),end=str(LAST+dt.timedelta(days=7)))
    dates=[s.date() for s in cal.sessions_in_range(str(week),str(week+dt.timedelta(days=6)))]
    if not dates: raise ValueError('empty trading week')
    weeks=sorted({s.date()-dt.timedelta(days=s.date().weekday()) for s in cal.sessions_in_range(str(FIRST),str(week))})
    index=weeks.index(week)
    if index<1: raise ValueError('no prior-week history')
    order=NAMES[:]; random.Random(seed+':'+str(index)).shuffle(order)
    rows=duckdb.connect().execute('select d,symbol,open,close,first_minute,last_minute,bar_count from read_parquet(?) where d >= ? and d <= ?', [str(path),week,week+dt.timedelta(days=6)]).fetchall()
    data={}
    for d,s,o,c,first,last,count in rows:
        if s not in NAMES or d not in dates or (d,s) in data: raise ValueError('unexpected or duplicate session row')
        session=cal.date_to_session(str(d)); opening=cal.session_open(session).to_pydatetime(); closing=cal.session_close(session).to_pydatetime()
        if not o or not c or o<=0 or c<=0 or o>=9223372036854775807 or c>=9223372036854775807 or not first or not last or not count or count<1 or not opening<=first<=last<closing: raise ValueError('invalid proxy or session timestamp')
        data[d,s]=c/o-1
    if any((d,s) not in data for d in dates for s in NAMES): raise ValueError('missing name-session, cannot score week')
    strategy=benchmark=1.0; cost=cost_bps/10000
    for d in dates:
        x=[data[d,s] for s in order]; b=sum(x)/len(x)
        benchmark*=1+b-cost*(2+b)
        if selected:
            r=sum(x[i] for i in selected)/len(selected); strategy*=1+r-cost*(2+r)
    return {'week_start':str(week),'sessions':[str(d) for d in dates],'universe_size':len(order),'strategy_return':strategy-1,'benchmark_return':benchmark-1,'excess_return':strategy-benchmark,'price_type':'first and last available in-session minute bars; not auction fills','cost_bps_per_side':cost_bps}
