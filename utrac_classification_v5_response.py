# 問題單貼標籤程式
import pyodbc
import pandas as pd
from datetime import datetime, date, timedelta
# print(sklearn.__version__)
from SQLextract import sql_connection
from sentence_seg_v2 import *
# from model_predict_v2 import *
from model_predict_v3_response import *
# from update_google_v2_response import *
# from apscheduler.schedulers.blocking import BlockingScheduler
# import logging

# logging.basicConfig()
# logging.getLogger('apscheduler').setLevel(logging.DEBUG)


def utrac_classification(utrac_no,sentence, value):
    # 包含資料前處理(斷詞)、預測
    data = {'utrac_no':[utrac_no], 
            'sentence':[sentence],
            'value':[value]}
    df = pd.DataFrame(data, columns=['utrac_no', 'sentence', 'value'])
    # print(df)
    data_cut = cut_sentence(df, 'sentence')
    # print(data_cut)
    predict_result = rule_base_predict(data_cut)
    print(predict_result)
    predict_label = predict_result.iloc[0]['分類結果']
    return predict_label


if __name__ == '__main__':
    utrac_no = input('enter utrac no:')
    sentence = input('enter sentence: ')
    value = input('enter value:')
    utrac_classification(utrac_no, sentence, value)

# scheduler = BlockingScheduler()
# scheduler.add_job(utrac_classification, 'cron', hour=14, minute=12)

# scheduler.start()





