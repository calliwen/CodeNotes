# coding:utf-8

import time
import datetime
from datetime import datetime

# 以 now_time 为基准， 计算前 hours 小时的 时间戳
def beforeHour2Date( hours, now_time ):
    hours = int(hours)
    timestamp = now_time - hours*60*60
    return timestamp

def beforeDays_timestamp(days, now_time):
    return beforeHour2Date( int(days)*24, now_time )
def beforeWeeks_timestamp(week_num, now_time):
    return beforeDays_timestamp( int(week_num)*7, now_time )

# 将 timestamp 为基准， 转换为 year week 的形式
def timestamp_to_yearWeek( timestamp ):
    form = "%Y-%m-%d"
    date = time.strftime( form, time.localtime( timestamp ) )
    date_time_object = datetime.strptime( date, form )
    year = date_time_object.isocalendar()[0]
    week = date_time_object.isocalendar()[1]
    return str("{}年-第{}周").format( year, week )

# 获取以 timestamp 为基准， 前 num_latesWeek 个 自然周的 year-week 形式的 list 表；（不包括timestamp所在的自然周）
def get_latest_yearWeek( timestamp, num_latesWeek=12 ):
    latest_yearWeek = [  ]
    for num in range( 1, num_latesWeek+1, 1 ):
        bf_timestamp = beforeWeeks_timestamp( num, timestamp )
        yearWeek = timestamp_to_yearWeek( bf_timestamp )
        latest_yearWeek.append( yearWeek )
    return sorted( latest_yearWeek, reverse=False )


now_time = time.time()
now_date = time.strftime( "%Y-%m-%d", time.localtime( now_time ) ) 
now_yearWeek = timestamp_to_yearWeek( now_time )
print( now_yearWeek )

res = get_latest_yearWeek( now_time, num_latesWeek=12 )
for item in res:
    print( item )

