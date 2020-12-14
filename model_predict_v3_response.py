# print(sklearn.__version__)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from gensim.models import Word2Vec
# import xgboost as xgb
import pickle
import pandas as pd
import numpy as np
import requests
import jieba
import codecs  
import re
import requests
import monpa
# monpa.load_userdict('train_monpa.txt')
import time
w2v = Word2Vec.load("word2vec_utrac_QA.model")
from tensorflow import keras


# w2v = Word2Vec.load('word2vec_faq_qa_utrac_v2.model')
def split_word(lst, index):
    word_lst=str(lst[index]).split(" ")
    return word_lst 

def w2v_sent(sent_cut):
    global w2v
    vec_lst=[]
    for i in sent_cut:
        if i in w2v.wv.vocab:
            vec_lst.append(i)
    if len(vec_lst)>=1:
    # print(len(w2v[vec_lst]))
        text_vector_sent = np.mean(w2v[vec_lst], axis=0)
    else:
        text_vector_sent=np.array(0)
        text_vector_sent.resize(600)
    return text_vector_sent

def get_two_hight(input_lst, cate_lst):
  #提供前兩個標簽
  # first_lst=[]
  # first_lst_score=[]
  # second_lst=[]
  # second_lst_score=[]
  # print(input_lst)
  # print(cate_lst)
  flat_list = [item for sublist in input_lst for item in sublist]
  # print(flat_list)
  # for i in input_lst:
  #   i=i.tolist()
  #   a1=i.index(sorted(i)[-1])
  #   a2=i.index(sorted(i)[-2])
  #   first_itme =cate_lst[a1]
  #   sec_itme=cate_lst[a2]
    # first_lst.append(i)
  # print(flat_list)
  # print(cate_lst)
  #   first_lst_score.append(sorted(i)[-1])
  #   second_lst.append(sec_itme)
  #   second_lst_score.append(sorted(i)[-2])
  #提供標籤清單
  df_result = pd.DataFrame({'標籤':cate_lst, '分數': flat_list}, columns=['標籤', '分數'])
  # return first_lst,second_lst
  return df_result

# rele-base
# 提供 50.新建置機關問題, 502.課務相關問題, 51.介接, 52.差旅費的標記
def rule_base_predict(data):
    data['pred_class']=0
    data_lst = data[['abs_detail_response','value', 'pred_class']].to_numpy().tolist()
    start = time.time()
    print(data_lst)
    test_lst = rule_base_1st(data_lst)
    end = time.time()
    print('完成時間: ', end-start)
    a =pd.DataFrame(test_lst, columns=['a', 'b', '分類結果'])
    # print(a.head())
    a = a[['分類結果']]
    data = data.reset_index(drop=True)
    result_pre = pd.concat([a, data], axis=1)
    return result_pre

def rule_base_1st(input_lst_1):
      # 1st classification
    value =['個人權限設定']
    #新建置
    pattern_501=['匯入']
    pattern_502=['課務']

    #其他類別
    pattern_51=['公文系統','公文介接','介接']
    pattern_52=['差旅費']

    #未休假加班費
    pattern_42=['未休假','不休假', '休假補助','修補費','休補費' ]


    #改成整句
    for i in input_lst_1:
        i[0] = i[0].split(" ")
        # print(i)
        if value[0] in i:
            i[2]=408
        elif any(s in i[0] for s in pattern_501):
          i[2]=50
        elif any(s in i[0] for s in pattern_502):
          i[2]=502
        elif any(s in i[0] for s in pattern_51):
          i[2]=51
        elif any(s in i[0] for s in pattern_52):
          i[2]=52
        elif any(s in i[0] for s in pattern_42):
          i[2]=42
        else:
          start = time.time()
          i[2] = model_predict(i)
          print(i)
          end = time.time()
          print('需求時間: ', end-start)

        print('預測結果:', i)
    return input_lst_1

# model_predict_sub-class
def model_predict(input_lst):
    global w2v
    # w2v = Word2Vec.load("word2vec_utrac_QA.model")
    model_1st ="keras_1st_original_response_v3"
    # model_1st = 'xgb_1st_v2.pkl'
    model_22="keras_original_response_22_v3"
    model_41 ="keras_original_response_41_v3"
    model_16="keras_original_response_16_v3"
    model_17='keras_original_response_17_v3'
    model_1 = "keras_original_response_1_v3"
    model_405="keras_original_response_405_v3"
    model_406="keras_original_response_406_v3"
    model_407="keras_original_response_407_v3"

    # 三個模型_加權
    # model_41_22_16=
    # model_41_22= 'SVC_41_22.pickle'
    # loaded_model_22 = pickle.load(open(model_22, 'rb'))
    # print('load ok')
    loaded_model_1st = keras.models.load_model(model_1st)
    # loaded_model_1st = pickle.load(open(model_1st, 'rb'))
    loaded_model_1 = keras.models.load_model(model_1)
    loaded_model_16 = keras.models.load_model(model_16)
    loaded_model_17 = keras.models.load_model(model_17)
    loaded_model_22 = keras.models.load_model(model_22)
    loaded_model_405 = keras.models.load_model(model_405)
    loaded_model_406 = keras.models.load_model(model_405)
    loaded_model_407 = keras.models.load_model(model_407)
    loaded_model_41 = keras.models.load_model(model_41)
    # loaded_model_41 = pickle.load(open(model_41, 'rb'))
    # loaded_model_16 = pickle.load(open(model_16, 'rb'))
    
    input_lst[0]=' '.join([str(elem) for elem in input_lst[0]]) 
    input_lst[0]= re.sub(r'[^\w\s]','',input_lst[0])
    print(input_lst)
    input_lst=split_word(input_lst, 0)
    input_lst_value=split_word(input_lst, 1)
    # print(input_lst)
    sent_train_x=w2v_sent(input_lst)
    # print(sent_train_x)
    sent_train_x_value=w2v_sent(input_lst_value)
    # print(sent_train_x_value)
    sent_x =np.concatenate((np.array(sent_train_x), np.array(sent_train_x_value)))
    # print(sent_train_x)
    # print(np.array([sent_x]).shape)
    # print(loaded_model_1st.summary())
    # for i in input_lst:
    #     i=w2v_sent(i)
    #     sent_train_x.append(i)
    #加權
    predict_1st = loaded_model_1st.predict(np.array([sent_x]))
    # print(predict_1st)
    # cate_1st = loaded_model_16_41_22.classes_
    cate_1st = ['0', '1', '16','17', '22','405','406','407', '41']
    df_1st = get_two_hight(predict_1st, cate_1st)
    df_1st.set_index('標籤', inplace=True)
    df_1st = df_1st.sort_values(by=['分數'],  ascending=False)
    # print(df_1st)
    # print('1st_pred:', df_1st.index.tolist())
    index_lst = df_1st.index.tolist()
    index_score = df_1st['分數'].tolist()
    # print(index_score)
    if index_lst[0]=='0' and index_score[0]>0.9:
      # print(index_lst[1])
      predict_label='0'
    
    else:

      # print(df_1st)

      # print('1st 標籤')
      # print(df_1st)
      dict_1st = df_1st.to_dict()['分數']

      cate_lst_1=['1_0','1_1','1_2']
      cate_lst_16=['16_0','16_1','16_2','16_4','16_5','16_6','16_9']
      cate_lst_17=['17_0','17_1','17_3','17_4','17_6']
      cate_lst_22=['22_0','22_1','22_10','22_12','22_13','22_2','22_3']
      cate_lst_405=['405_0','405_1','405_10','405_2','405_3','405_4','405_5', '405_6', '405_7']
      cate_lst_406=['406_0','406_1','406_2','406_3','406_4','406_5','406_6','406_7','406_8']
      cate_lst_407=['407_0','407_1']
      cate_lst_41=['41_0','41_10','41_11','41_2','41_4','41_6','41_7','41_8','41_9']
      
      predict_1 = loaded_model_1.predict_proba(np.array([sent_x]))
      df_1 = get_two_hight(predict_1, cate_lst_1)
      df_1['加權分數'] = df_1['分數']*dict_1st['1']
      
      predict_22 = loaded_model_22.predict_proba(np.array([sent_x]))
      df_22 = get_two_hight(predict_22, cate_lst_22)
      df_22['加權分數'] = df_22['分數']*dict_1st['22']
      
      predict_16= loaded_model_16.predict_proba(np.array([sent_x]))
      df_16 = get_two_hight(predict_16, cate_lst_16)
      df_16['加權分數'] = df_16['分數']*dict_1st['16']

      predict_17 = loaded_model_17.predict_proba(np.array([sent_x]))
      df_17 = get_two_hight(predict_17, cate_lst_17)
      df_17['加權分數'] = df_17['分數']*dict_1st['17']
      
      predict_405 = loaded_model_405.predict_proba(np.array([sent_x]))
      df_405 = get_two_hight(predict_405, cate_lst_405)
      df_405['加權分數'] = df_405['分數']*dict_1st['405']
      
      predict_406= loaded_model_406.predict_proba(np.array([sent_x]))
      df_406 = get_two_hight(predict_406, cate_lst_406)
      df_406['加權分數'] = df_406['分數']*dict_1st['406']

      predict_407 = loaded_model_407.predict_proba(np.array([sent_x]))
      df_407 = get_two_hight(predict_407, cate_lst_407)
      df_407['加權分數'] = df_407['分數']*dict_1st['407']
      
      predict_41= loaded_model_41.predict_proba(np.array([sent_x]))
      df_41 = get_two_hight(predict_41, cate_lst_41)
      df_41['加權分數'] = df_41['分數']*dict_1st['41']

      df_result = pd.concat([df_41, df_16, df_1, df_17, df_22, df_405, df_406, df_407])
      # df_result = pd.concat([df_result, df_22])
      df_result = df_result.sort_values(by=['加權分數'],  ascending=False)
      df_result = df_result.drop_duplicates(subset='標籤')
      predict_label = df_result.iloc[0,0]
      print(df_result)
      # data['rule_3label']=0
      # data_lst = data[['問題','abs_detail_response', 'rule_3label']]
      # test_lst =data_lst.to_numpy().tolist()
      # test_lst = rule_base_1st(test_lst)
      # a =pd.DataFrame(test_lst, columns=['a', 'b', '第一層分類'])
      # a = a[['第一層分類']]
      # data = data.reset_index(drop=True)
      # result_pre = pd.concat([a, data], axis=1)
      # pred_result = result_pre['第一層分類'].tolist()
      # if 41 in pred_result:
      #   df_x = result_pre[['單號','abs_detail_response']].to_numpy().tolist()
      #   for i in df_x:
      #       i[1] = re.sub(r'[^\w\s]','',i[1])
        
      #   lst_train_x=split_word(df_x)
      #   sent_train_x=[]
      #   for i in lst_train_x:
      #       i=w2v_sent(i)
      #       sent_train_x.append(i)
      #   predict_1st = loaded_model_41.predict_proba(sent_train_x)
      #   cate_lst_41=loaded_model_41.classes_
      #   [result_1st, result_2nd] = get_two_hight(predict_1st, cate_lst_41)
      # if 22 in pred_result or 30 in pred_result:
      #   df_x = result_pre[['單號','abs_detail_response']].to_numpy().tolist()
      #   for i in df_x:
      #       i[1] = re.sub(r'[^\w\s]','',i[1])
        
      #   lst_train_x=split_word(df_x)
      #   sent_train_x=[]
      #   for i in lst_train_x:
      #       i=w2v_sent(i)
      #       sent_train_x.append(i)
      #   predict_1st = loaded_model_22.predict_proba(sent_train_x)  
      #   cate_lst_22=loaded_model_22.classes_
      #   [result_1st, result_2nd] = get_two_hight(predict_1st, cate_lst_22)  
      
      # if 16 in pred_result:
      #   df_x = result_pre[['單號','abs_detail_response']].to_numpy().tolist()
      #   for i in df_x:
      #       i[1] = re.sub(r'[^\w\s]','',i[1])
        
      #   lst_train_x=split_word(df_x)
      #   sent_train_x=[]
      #   for i in lst_train_x:
      #       i=w2v_sent(i)
      #       sent_train_x.append(i)
      #   predict_1st = loaded_model_16.predict_proba(sent_train_x)
      #   cate_lst_16=loaded_model_16.classes_
      #   [result_1st, result_2nd] = get_two_hight(predict_1st, cate_lst_16)   
      # else:
      #   [result_1st, result_2nd]=[0,0]
      # predict_1st= [result_1st, result_2nd]
      # print(predict_1st)
      # data['標籤']=str(predict_1st)
      # data['標籤']=data['標籤'].astype(str)
      # data['date'] = pd.DatetimeIndex(data['時間戳記']).date
    print(predict_label)
    return predict_label

#使用rule_base做為權重+all
def model_predict_v2(data):
    global w2v
    # w2v = Word2Vec.load("word2vec_utrac_QA.model")
    model_22="xgb_22_v5.pkl"
    model_41 ="xgb_41_v5.pkl"
    model_16="xgb_16_v5.pkl"
    # model_41_22_16 ='svc_16_41_22_v2.pickle'
    # 三個模型_加權
    # model_41_22_16=
    # model_41_22= 'SVC_41_22.pickle'
    loaded_model_22 = pickle.load(open(model_22, 'rb'))
    # print('load ok')
    # loaded_model_16_41_22 = pickle.load(open(model_41_22_16, 'rb'))
    loaded_model_41 = pickle.load(open(model_41, 'rb'))
    loaded_model_16 = pickle.load(open(model_16, 'rb'))
    data['rule_3label']=0
    data_lst = data[['問題','abs_detail_response', 'rule_3label']]
    test_lst =data_lst.to_numpy().tolist()
    test_lst = rule_base_1st(test_lst)
    a =pd.DataFrame(test_lst, columns=['a', 'b', '第一層分類'])
    a = a[['第一層分類']]
    data = data.reset_index(drop=True)
    result_pre = pd.concat([a, data], axis=1)
    result_pre['第一層分類'] = result_pre['第一層分類'].astype(str)
    pred_result = result_pre['第一層分類'].tolist()
    print(pred_result)
    if '41' in pred_result:
      df_x = result_pre[['單號','abs_detail_response']].to_numpy().tolist()
      for i in df_x:
          i[1] = re.sub(r'[^\w\s]','',i[1])
      
      lst_train_x=split_word(df_x)
      sent_train_x=[]
      for i in lst_train_x:
          i=w2v_sent(i)
          sent_train_x.append(i)
      predict_41 = loaded_model_41.predict_proba(sent_train_x)
      cate_lst_41=loaded_model_41.classes_
      df_41 = get_two_hight(predict_41, cate_lst_41)
      df_41['加權分數'] = df_41['分數']*0.98
      predict_22 = loaded_model_22.predict_proba(sent_train_x)
      cate_lst_22=loaded_model_22.classes_
      df_22 = get_two_hight(predict_22, cate_lst_22)
      df_22['加權分數'] = df_22['分數']*0.01
      predict_16 = loaded_model_16.predict_proba(sent_train_x)
      cate_lst_16=loaded_model_16.classes_
      df_16 = get_two_hight(predict_16, cate_lst_16)
      df_16['加權分數'] = df_16['分數']*0.01
    if '22' in pred_result:
      df_x = result_pre[['單號','abs_detail_response']].to_numpy().tolist()
      for i in df_x:
          i[1] = re.sub(r'[^\w\s]','',i[1])
      
      lst_train_x=split_word(df_x)
      sent_train_x=[]
      for i in lst_train_x:
          i=w2v_sent(i)
          sent_train_x.append(i)
      predict_41 = loaded_model_41.predict_proba(sent_train_x)
      cate_lst_41=loaded_model_41.classes_
      df_41 = get_two_hight(predict_41, cate_lst_41)
      df_41['加權分數'] = df_41['分數']*0.01
      predict_22 = loaded_model_22.predict_proba(sent_train_x)
      cate_lst_22=loaded_model_22.classes_
      df_22 = get_two_hight(predict_22, cate_lst_22)
      df_22['加權分數'] = df_22['分數']*0.98
      predict_16 = loaded_model_16.predict_proba(sent_train_x)
      cate_lst_16=loaded_model_16.classes_
      df_16 = get_two_hight(predict_16, cate_lst_16)
      df_16['加權分數'] = df_16['分數']*0.01
    
    if '16' in pred_result:
      df_x = result_pre[['單號','abs_detail_response']].to_numpy().tolist()
      for i in df_x:
          i[1] = re.sub(r'[^\w\s]','',i[1])
      
      lst_train_x=split_word(df_x)
      sent_train_x=[]
      for i in lst_train_x:
          i=w2v_sent(i)
          sent_train_x.append(i)
      predict_41 = loaded_model_41.predict_proba(sent_train_x)
      cate_lst_41=loaded_model_41.classes_
      df_41 = get_two_hight(predict_41, cate_lst_41)
      df_41['加權分數'] = df_41['分數']*0.01
      print('16_41', df_41)
      predict_22 = loaded_model_22.predict_proba(sent_train_x)
      cate_lst_22=loaded_model_22.classes_
      df_22 = get_two_hight(predict_22, cate_lst_22)
      df_22['加權分數'] = df_22['分數']*0.01
      print(df_22)
      predict_16 = loaded_model_16.predict_proba(sent_train_x)
      cate_lst_16=loaded_model_16.classes_
      df_16 = get_two_hight(predict_16, cate_lst_16)
      df_16['加權分數'] = df_16['分數']*0.98
      print(df_16)
    df_result = pd.concat([df_41, df_16])
    df_result = pd.concat([df_result, df_22])
    df_result = df_result.sort_values(by=['加權分數'],  ascending=False)
    df_result = df_result.drop_duplicates(subset='標籤')
    # data['rule_3label']=0
    # data_lst = data[['問題','abs_detail_response', 'rule_3label']]
    # test_lst =data_lst.to_numpy().tolist()
    # test_lst = rule_base_1st(test_lst)
    # a =pd.DataFrame(test_lst, columns=['a', 'b', '第一層分類'])
    # a = a[['第一層分類']]
    # data = data.reset_index(drop=True)
    # result_pre = pd.concat([a, data], axis=1)
    # pred_result = result_pre['第一層分類'].tolist()
    # if 41 in pred_result:
    #   df_x = result_pre[['單號','abs_detail_response']].to_numpy().tolist()
    #   for i in df_x:
    #       i[1] = re.sub(r'[^\w\s]','',i[1])
      
    #   lst_train_x=split_word(df_x)
    #   sent_train_x=[]
    #   for i in lst_train_x:
    #       i=w2v_sent(i)
    #       sent_train_x.append(i)
    #   predict_1st = loaded_model_41.predict_proba(sent_train_x)
    #   cate_lst_41=loaded_model_41.classes_
    #   [result_1st, result_2nd] = get_two_hight(predict_1st, cate_lst_41)
    # if 22 in pred_result or 30 in pred_result:
    #   df_x = result_pre[['單號','abs_detail_response']].to_numpy().tolist()
    #   for i in df_x:
    #       i[1] = re.sub(r'[^\w\s]','',i[1])
      
    #   lst_train_x=split_word(df_x)
    #   sent_train_x=[]
    #   for i in lst_train_x:
    #       i=w2v_sent(i)
    #       sent_train_x.append(i)
    #   predict_1st = loaded_model_22.predict_proba(sent_train_x)  
    #   cate_lst_22=loaded_model_22.classes_
    #   [result_1st, result_2nd] = get_two_hight(predict_1st, cate_lst_22)  
    
    # if 16 in pred_result:
    #   df_x = result_pre[['單號','abs_detail_response']].to_numpy().tolist()
    #   for i in df_x:
    #       i[1] = re.sub(r'[^\w\s]','',i[1])
      
    #   lst_train_x=split_word(df_x)
    #   sent_train_x=[]
    #   for i in lst_train_x:
    #       i=w2v_sent(i)
    #       sent_train_x.append(i)
    #   predict_1st = loaded_model_16.predict_proba(sent_train_x)
    #   cate_lst_16=loaded_model_16.classes_
    #   [result_1st, result_2nd] = get_two_hight(predict_1st, cate_lst_16)   
    # else:
    #   [result_1st, result_2nd]=[0,0]
    # predict_1st= [result_1st, result_2nd]
    # print(predict_1st)
    # data['標籤']=str(predict_1st)
    # data['標籤']=data['標籤'].astype(str)
    # data['date'] = pd.DatetimeIndex(data['時間戳記']).date
    return df_result

# def rule_base_predict(data):
#     data['date'] = pd.DatetimeIndex(data['時間戳記']).date
#     data['rule_3label']=0
#     data_lst = data[['問題','abs_detail_response', 'rule_3label']]
#     test_lst =data_lst.to_numpy().tolist()
#     test_lst = rule_base_1st(test_lst)
#     a =pd.DataFrame(test_lst, columns=['a', 'b', '第一層分類'])
#     a = a[['第一層分類']]
#     data = data.reset_index(drop=True)
#     result_pre = pd.concat([a, data], axis=1)
#     result_pre['第一層分類'] = result_pre['第一層分類'].astype(str)
#     result_pre['rele_3label_2']=0
#     test_2nd = result_pre[['第一層分類','abs_detail_response','rele_3label_2', 'cut_abs_detail']]
#     test_2nd_lst = test_2nd.to_numpy().tolist()
#     test_2nd_lst=rule_base_2nd(test_2nd_lst)
#     b =pd.DataFrame(test_2nd_lst, columns=['a', 'b', '第二層分類','d'])
#     b = b[['第二層分類']]
#     result_pre = result_pre.reset_index(drop=True)
#     result_pre = pd.concat([b, result_pre], axis=1)
#     result_pre['第一層分類'] = result_pre['第一層分類'] .str.replace('\_\d+\_\d+', '')
#     result_pre['第一層分類'] = result_pre['第一層分類'] .str.replace('\_\d+', '')
#     return result_pre

# def rule_base_1st(input_lst_1):
#       # 1st classification
#     pattern_1_6_1 =['已請', '已請領', '已使用', '已領']
#     pattern_1_1_1=['補休', '加班補休', '補修'] 
#     pattern_1=['補修', '補休', '加班補休', '補休期限']
#     pattern_1_0_1=['大批', '屆期', '管理者', '行事曆', '國定假日', '排班','差假']
#     pattern_1_0_2=['申請', '加班費'] #all
#     pattern_1_0_3=['請領', '方式'] #all
#     pattern_1_1=['調整', '修改', '更新', '調整為', '修正', '修改為', '修改成', '改', '改成', '變更']
#     pattern_1_400=['一般加班', '專案加班']
#     pattern_0_2=['加班費請領', '請領加班費', '加班費', '加班費計算', '請領加班費'] 
#     pattern_0_2_11=['加班', '費'] #all
#     pattern_0_2_12=['加班', '費用'] #all
#     pattern_0_2_1=['已請領', '請領','申請', '請領時數']
#     pattern_0_2_2 = ['上限']
#     pattern_0_2_3 = ['時數']
#     pattern_0_2_4 = ['退回','退件']
#     pattern_23_1 = ['列印', '個人列印']
#     pattern_23_3=['單位', '小時'] #all
#     pattern_23_2=['顯示', '沒有', '錯誤', '無法', '未能', '不符']
#     pattern_23_5=['計算', '方式'] #all
#     pattern_23_8=['功能', '承辦人'] #all
#     pattern_23_4=['有', '誤'] #all
#     pattern_23_7=['沒','看到'] #all
#     pattern_23_6=['有誤', '疑義'] 
#     pattern_23_9=['3分之1', '3分之2', '三分之一', '三分之二']
#     pattern_23_10=['不要', '出現', '補印'] #all
#     pattern_23_12=['關閉', '新增','有', '增加'] #any
#     pattern_23_13=['權限', '功能'] #all
#     pattern_23_11=['屆期', '權限']
#     pattern_23_15=['預算科目']
#     pattern_23_14=['核發', '紀錄'] #all
#     pattern_23_16=['重送']
#     pattern_23_17=['彙整']
#     pattern_22_1 = ['總表','清冊', '列印總表', '加班費印領清冊', '報表', '明細表', '產制']
#     pattern_22_2 =['加班', '統計', '資料檔'] #all
#     pattern_1_2=['加班時數','時數','加班資料', '加班', '專案加班']
#     pattern_1_3 =['加班資料維護']
#     pattern_1_4=['修改', '正確']
#     # pattern_1_3=['加班資料維護', '加班資料']
#     pattern_2_1=['重新計算','重算', '計算', '核算', '給予', '核給', '核發', '計算出', '統計', '不可異動', '計入']
#     pattern_2_2=['加班時數','時數', '加班', '專案加班', '加班資料'] 
#     pattern_2_3 =['無', '有效', '加班時數']  #將無有效加班時數包含在加班時數計算
#     pattern_2_6 =['錯誤', '有誤', '誤', '正確']
#     pattern_2_5 =['無效', '加班時數']
#     pattern_2_7 = ['加班計算']
#     pattern_2_8 = ['有效', '加班時數']
#     pattern_2_9=['刷', '卡', '設定'] #all
#     pattern_2_4 =['未顯示', '匯入','顯示', '進入', '出現'] #將加班時數未顯示歸列在此
#     pattern_4_1=['機關差勤規定', '參數']
#     pattern_4_2_1 =['更改','修改']
#     pattern_4_2_2 =['刷上','下班卡']
#     pattern_4_1_1=['中午','午休','傍晚']
#     pattern_4_1_2=['上班', '免刷卡','加班']
#     pattern_4_1_3=['跨夜', '加班']
#     pattern_4_1_4 =['加班','刷']
#     pattern_4_1_5 =['中午','扣除']
#     pattern_5_1 =['撤銷']
#     pattern_5_2 =['重送', '補登', '重新', '修改']
#     pattern_5_3 =['首頁']
#     # pattern_6_1=['新增', '修正', '建立', '補', '更新']
#     pattern_6_2=['刷卡資料', '下班卡', '上班卡', '卡別', '加班進', '卡', '加班出']
#     pattern_6_3=['下班時間', '無法', '核算']
#     pattern_6_4=['上班', '時數', '核給'] #all
#     pattern_6_5=['出勤']
#     # pattern_6_6=['值班', '輪班'] 
#     pattern_8_1=['版更', '抽換', '程式', '工程師', '升級', '版本更新']
#     pattern_8_2=['版本', '更新']
#     pattern_8_3=['公版','那週']
#     pattern_9_1=['免刷卡', '無刷卡']
#     pattern_10_1=['無申請', '無法', '無', '重新']
#     pattern_10_2=['錯誤訊息', '錯']
#     pattern_10_4=['申請加班']
#     pattern_10_3=['申報', '申請', '輸入', '新增', '送', '申請中']
#     pattern_10_5=['一般加班', '加班', '專案加班', '一班加班']
#     pattern_10_6=['加班單', '送', '不出']
#     pattern_10_7=['錯誤']
#     pattern_12_1=['未滿' , '不足', '少', '差']
#     pattern_12_2=['小時', '一小時', '分鐘']
#     pattern_13_1=['時數', '加班資料'] #
#     pattern_13_2=['上限'] #
#     pattern_14_2=['不可異動', '異動', '無效', '加班資料', '加班', '公假補休', '未核算']
#     pattern_14_1=['刪除']
#     pattern_17_1=['流程']
#     pattern_33_1 =['批核', '差假']
#     pattern_21_1=['完畢', '加班申請', '無法', '不能', '錯誤訊息', '簽核']
#     pattern_21_2=['發生', '未知', '錯誤'] #all
#     pattern_21_3=['無法', '申請', '差假']#all
#     pattern_0_1=['自行', '測試']
#     pattern_13_3=['未核算']
#     pattern_0_3=['加班', '性質']
#     pattern_0_3_1=['加班費性質', '加班性質']
#     pattern_0_3_2=['加班', '屬性']
#     pattern_0_3_3=['加班', '休息日']
#     pattern_0_3_4=['加班', '國定假日']
#     pattern_20_1=['上班別', '彈性', '加班時數', '正常']
#     pattern_20_2=['彈性']
#     pattern_20_3=['擴大', '彈班'] #all

#     pattern_30_1=['餘數', '合併', '加班餘數', '併', '含分', '餘數計算']
#     pattern_30_2=['分計', '併計', '分鐘數']
#     pattern_30_6=['加班','顯示', '分']
#     pattern_30_7=['加班','出現', '分']
#     pattern_30_8=['加班','產出', '分']
#     pattern_30_3=['流程模組發生錯誤']
#     pattern_30_4=['差', '旅費', '差費', '公出差', '差旅費', '出差', '差旅費單', '公出', '車資']
#     pattern_30_5=['出', '差']

#     pattern_31_1 =['申報', '申請']
#     pattern_31_2 =['加班']
#     pattern_32_1 =['代理人', '代理']
#     pattern_34_1=['請假', '喪假', '公假補休', '休假', '假', '病假', '休', '寒休', '勤惰']
#     pattern_34_2 =['計','統計','時數', '計算', '更正', '判定', '修正']
#     pattern_48_2=['線上刷卡','線上', '在家','手機']
#     pattern_48_3=['刷卡', '簽到', '打卡']
#     pattern_48_4=['線上', '刷', '卡']  
#     # pattern_problem_1=['加班']
#     pattern_problem_2=['加班補休','補休', '補修', '加班費', '班費', '時數', '加班時數']
#     pattern_problem_3=['核算', '申請', '未核算']

#     pattern_solution_1=['調整','小時', '計算']
#     pattern_solution_2=['版更', '抽換', '程式', '工程師', '升級', '版本更新', 'Bug', 'bug']
#     pattern_solution_3=['版本', '更新']
#     #22_1
#     pattern_22_1_1 = ['無', '無法', '未', '未能'] 
#     pattern_22_1_2=['有效', '核算', '採算',  '核計'] 
#     pattern_22_1_3 =['未核算', '未顯示']

#     #405_3 跨科室代理
#     pattern_405_3_1=['跨']
#     pattern_405_3_2=['單位', '科室', '處室']
#     for i in input_lst_1:
#         i[1] = i[1].split(" ")
#         if any(s in i[1] for s in pattern_405_3_1) and any(s in i[1] for s in pattern_405_3_2):
#           i[2]='405_3'
#         elif any(s in i[1] for s in pattern_22_1_1) and any(s in i[1] for s in pattern_22_1_2) and any(s in i[1] for s in pattern_10_5):
#           i[2]=22
#         elif any(s in i[1] for s in pattern_22_1_3) and any(s in i[1] for s in pattern_10_5):
#           i[2]=22
#         elif all(s in i[1] for s in pattern_2_3) and not (any(s in i[1] for s in pattern_0_2_1) and any(s in i[1] for s in pattern_0_2_2)) and not any(s in i[1] for s in pattern_6_5) and not all(s in i[1] for s in pattern_2_9):
#           i[2]=22
#         elif all(s in i[1] for s in pattern_2_5) and not (any(s in i[1] for s in pattern_0_2_1) and any(s in i[1] for s in pattern_0_2_2)) and not any(s in i[1] for s in pattern_6_5) and not all(s in i[1] for s in pattern_2_9):
#           i[2]=22
#         elif any(s in i[1] for s in pattern_30_1):
#           i[2]=30
#         elif any(s in i[1] for s in pattern_30_2):
#           i[2]=30
#         elif all(s in i[1] for s in pattern_30_6):
#           i[2]=30
#         elif all(s in i[1] for s in pattern_30_7):
#           i[2]=30
#         elif all(s in i[1] for s in pattern_30_8):
#           i[2]=30
#         elif any(s in i[1] for s in pattern_1) and not any(s in i[1] for s in pattern_1_0_1) and not all(s in i[1] for s in pattern_1_0_3) and not any(s in i[1] for s in pattern_1_6_1):
#           i[2]=1
#         elif any(s in i[1] for s in pattern_17_1):
#           i[2]=17
#         elif (any(s in i[1] for s in pattern_48_2) and any(s in i[1] for s in pattern_48_3)) or all(s in i[1] for s in pattern_48_4):
#           i[2]=16 #48
#         elif all(s in i[1] for s in pattern_20_1):
#           i[2]=16 #20
#         elif all(s in i[1] for s in pattern_20_2) and not any(s in i[1] for s in pattern_2_2):
#           i[2]=16 #20
#         elif all(s in i[1] for s in pattern_20_3):
#           i[2]=16 #20
#         elif any(s in i[1] for s in pattern_6_2) or all(s in i[1] for s in pattern_6_4) or all(s in i[1] for s in pattern_6_3) and not any(s in i[1] for s in pattern_1_2) and not any(s in i[1] for s in pattern_1_3) and not any(s in i[1] for s in pattern_5_1) and not any(s in i[1] for s in pattern_22_1):
#           i[2]=16
#         elif any(s in i[1] for s in pattern_32_1):
#           i[2]='405_1' #401
#         elif any(s in i[1] for s in pattern_33_1):
#           i[2]=405 #21
#         elif any(s in i[1] for s in pattern_33_1) and any(s in i[1] for s in pattern_21_1):
#           i[2]=405 #21
#         elif any(s in i[1] for s in pattern_33_1) and all(s in i[1] for s in pattern_21_2):
#           i[2]=405 #21
#         elif any(s in i[1] for s in pattern_30_3):
#           i[2]=405 #21
#         elif all(s in i[1] for s in pattern_21_3):
#           i[2]=405 #21
#         elif any(s in i[1] for s in pattern_4_1):
#           if any(s in i[1] for s in pattern_4_1_1) or any(s in i[1] for s in pattern_4_1_2) or any(s in i[1] for s in pattern_4_1_3) or any(s in i[1] for s in pattern_4_1_4):
#             i[2]=16 
#         elif any(s in i[1] for s in pattern_4_2_1) and all(s in i[1] for s in pattern_4_2_2):
#           i[2]=16 
#         elif all(s in i[1] for s in pattern_4_1_5):
#           i[2]=16 
#         elif any(s in i[1] for s in pattern_4_1):
#           i[2]=16 
#         elif any(s in i[1] for s in pattern_30_4) and not (any(s in i[1] for s in pattern_2_1) and any(s in i[1] for s in pattern_2_2)) and not any(s in i[1] for s in pattern_23_15):
#           i[2]='405_4' #232
#         elif all(s in i[1] for s in pattern_30_5) and not (any(s in i[1] for s in pattern_2_1) and any(s in i[1] for s in pattern_2_2)) and not any(s in i[1] for s in pattern_23_15):
#           i[2]='405_4' #232
#         elif any(s in i[1] for s in pattern_0_2):
#           i[2]=41
#         elif any(s in i[1] for s in pattern_23_1) and not any(s in i[1] for s in pattern_23_11) and not all(s in i[1] for s in pattern_23_14) and not any(s in i[1] for s in pattern_0_2_4) and not any(s in i[1] for s in pattern_23_16) and not any(s in i[1] for s in pattern_23_17):
#           i[2]=41
#         elif all(s in i[1] for s in pattern_23_3) and any(s in i[1] for s in pattern_23_2) and not any(s in i[1] for s in pattern_23_11) and not all(s in i[1] for s in pattern_23_14) and not any(s in i[1] for s in pattern_0_2_4) and not any(s in i[1] for s in pattern_23_17):
#           i[2]=41
#         elif any(s in i[1] for s in pattern_1_1_1) and any(s in i[1] for s in pattern_1_6_1):
#           i[2]=1  
#         elif (any(s in i[1] for s in pattern_0_2) or all(s in i[1] for s in pattern_0_2_11) or all(s in i[1] for s in pattern_0_2_12)) and any(s in i[1] for s in pattern_23_2) and not any(s in i[1] for s in pattern_9_1) and not any(s in i[1] for s in pattern_23_11) and not all(s in i[1] for s in pattern_23_14) and not any(s in i[1] for s in pattern_0_2_4) and not any(s in i[1] for s in pattern_22_1) and not any(s in i[1] for s in pattern_23_17):
#           i[2]=41
#         elif (any(s in i[1] for s in pattern_0_2) or all(s in i[1] for s in pattern_0_2_11) or all(s in i[1] for s in pattern_0_2_12)) and all(s in i[1] for s in pattern_23_7) and not any(s in i[1] for s in pattern_9_1) and not any(s in i[1] for s in pattern_23_11) and not all(s in i[1] for s in pattern_23_14) and not any(s in i[1] for s in pattern_0_2_4) and not any(s in i[1] for s in pattern_23_17):
#           i[2]=41
#         elif (any(s in i[1] for s in pattern_0_2) or all(s in i[1] for s in pattern_0_2_11) or all(s in i[1] for s in pattern_0_2_12)) and (all(s in i[1] for s in pattern_23_4) or any(s in i[1] for s in pattern_23_6)) and not any(s in i[1] for s in pattern_9_1) and not any(s in i[1] for s in pattern_23_11) and not all(s in i[1] for s in pattern_23_14) and not any(s in i[1] for s in pattern_0_2_4) and not any(s in i[1] for s in pattern_23_17):
#           i[2]=41
#         elif (any(s in i[1] for s in pattern_0_2) or all(s in i[1] for s in pattern_0_2_11) or all(s in i[1] for s in pattern_0_2_12)) and any(s in i[1] for s in pattern_23_9) and not any(s in i[1] for s in pattern_9_1) and not any(s in i[1] for s in pattern_23_11) and not all(s in i[1] for s in pattern_23_14) and not any(s in i[1] for s in pattern_0_2_4) and not any(s in i[1] for s in pattern_23_17):
#           i[2]=41
#         elif (any(s in i[1] for s in pattern_0_2) or all(s in i[1] for s in pattern_0_2_11) or all(s in i[1] for s in pattern_0_2_12)) and all(s in i[1] for s in pattern_23_5):
#           i[2]=41 #2334
#         elif (any(s in i[1] for s in pattern_0_2) or all(s in i[1] for s in pattern_0_2_11) or all(s in i[1] for s in pattern_0_2_12)) and all(s in i[1] for s in pattern_23_8):
#           i[2]=41 #2334
#         elif (any(s in i[1] for s in pattern_0_2) or all(s in i[1] for s in pattern_0_2_11) or all(s in i[1] for s in pattern_0_2_12)) and all(s in i[1] for s in pattern_23_10):
#           i[2]=41 #2334
#         elif (any(s in i[1] for s in pattern_0_2) or all(s in i[1] for s in pattern_0_2_11) or all(s in i[1] for s in pattern_0_2_12)) and (any(s in i[1] for s in pattern_23_12) and all(s in i[1] for s in pattern_23_13)):
#           i[2]=41 #2334
#         elif (any(s in i[1] for s in pattern_0_2) or all(s in i[1] for s in pattern_0_2_11) or all(s in i[1] for s in pattern_0_2_12)) and any(s in i[1] for s in pattern_0_2_1) and any(s in i[1] for s in pattern_0_2_4):
#           i[2]=41 #2334
#         elif (any(s in i[1] for s in pattern_0_2) or all(s in i[1] for s in pattern_0_2_11) or all(s in i[1] for s in pattern_0_2_12)) and any(s in i[1] for s in pattern_0_2_4):
#           i[2]=41 #2334
#         elif any(s in i[1] for s in pattern_22_1) or all(s in i[1] for s in pattern_22_2) and not any(s in i[1] for s in pattern_9_1):
#           i[2]=41 #5
#         elif any(s in i[1] for s in pattern_5_1) and any(s in i[1] for s in pattern_5_2):
#           i[2]='405_2' #25
#         elif any(s in i[1] for s in pattern_5_1) and any(s in i[1] for s in pattern_5_3):
#           i[2]='405_2' #25
#         elif any(s in i[1] for s in pattern_5_1):
#           i[2]='405_2' #25
#         elif any(s in i[1] for s in pattern_34_1) and any(s in i[1] for s in pattern_34_2):
#           i[2]=405 #403
#         elif any(s in i[1] for s in pattern_2_2) and any(s in i[1] for s in pattern_13_2):
#         # elif any(s in i[1] for s in pattern_13_1) and any(s in i[1] for s in pattern_13_2) and not any(s in i[1] for s in pattern_1_4):
#           i[2]='22_2' #13
#         elif any(s in i[1] for s in pattern_14_1) and any(s in i[1] for s in pattern_14_2) and not any(s in i[1] for s in pattern_10_3):
#           i[2]='22_3_0' #14
#         elif any(s in i[1] for s in pattern_10_1) and any(s in i[1] for s in pattern_10_3) and any(s in i[1] for s in pattern_10_5) and not any(s in i[1] for s in pattern_4_1) and not any(s in i[1] for s in pattern_6_2) and not any(s in i[1] for s in pattern_2_1) :
#           i[2]='405_5' #10
#         elif any(s in i[1] for s in pattern_10_1) and all(s in i[1] for s in pattern_10_4) and not any(s in i[1] for s in pattern_4_1) and not any(s in i[1] for s in pattern_6_2) and not any(s in i[1] for s in pattern_2_1) :
#           i[2]='405_5' #10
#         elif any(s in i[1] for s in pattern_10_2) and all(s in i[1] for s in pattern_10_4) and not any(s in i[1] for s in pattern_4_1) and not any(s in i[1] for s in pattern_6_2) and not any(s in i[1] for s in pattern_2_1) :
#           i[2]='405_5' #10
#         # elif any(s in i[1] for s in pattern_10_2) and any(s in i[1] for s in pattern_10_5) and not any(s in i[1] for s in pattern_4_1) and not any(s in i[1] for s in pattern_6_2) :
#         #   i[2]=10
#         elif all(s in i[1] for s in pattern_10_6):
#           i[2]='405_5' #10
#         # elif any(s in i[1] for s in pattern_10_3) and any(s in i[1] for s in pattern_10_5) and any(s in i[1] for s in pattern_10_7) and not any(s in i[1] for s in pattern_4_1) and not any(s in i[1] for s in pattern_6_2) :
#         #   i[2]=10
#         elif all(s in i[1] for s in pattern_0_3) or any(s in i[1] for s in pattern_0_3_1) or all(s in i[1] for s in pattern_0_3_4) or all(s in i[1] for s in pattern_0_3_2) or all(s in i[1] for s in pattern_0_3_3):
#           i[2]="41_6" #6

#         elif (any(s in i[1] for s in pattern_1_1) and all(s in i[1] for s in pattern_1_400)):
#           i[2]=400
#         elif any(s in i[1] for s in pattern_9_1) and any(s in i[1] for s in pattern_2_2) and not any(s in i[1] for s in pattern_5_1) and not (any(s in i[1] for s in pattern_2_1) and any(s in i[1] for s in pattern_2_2)) and not (any(s in i[1] for s in pattern_6_2) or all(s in i[1] for s in pattern_2_9)):
#           i[2]=22
#         elif any(s in i[1] for s in pattern_13_3) and not all(s in i[1] for s in pattern_2_9):
#           i[2]=22
#         elif any(s in i[1] for s in pattern_1_3) and not (any(s in i[1] for s in pattern_0_2_1) and any(s in i[1] for s in pattern_0_2_2)) and not any(s in i[1] for s in pattern_6_5) and not all(s in i[1] for s in pattern_2_9):
#           i[2]=22
#         elif len(i[0])<30 and any(s in i[1] for s in pattern_2_1) and not (any(s in i[1] for s in pattern_0_2_1) and any(s in i[1] for s in pattern_0_2_2)) and not any(s in i[1] for s in pattern_6_5) and not all(s in i[1] for s in pattern_2_9):
#           i[2]=22
#         elif len(i[0])>=30 and any(s in i[1] for s in pattern_2_1) and any(s in i[1] for s in pattern_2_2) and not (any(s in i[1] for s in pattern_0_2_1) and any(s in i[1] for s in pattern_0_2_2)) and not any(s in i[1] for s in pattern_6_5) and not all(s in i[1] for s in pattern_2_9):
#           i[2]=22
#         elif all(s in i[1] for s in pattern_2_3) and not (any(s in i[1] for s in pattern_0_2_1) and any(s in i[1] for s in pattern_0_2_2)) and not any(s in i[1] for s in pattern_6_5) and not all(s in i[1] for s in pattern_2_9):
#           i[2]=22
#         elif all(s in i[1] for s in pattern_2_5) and not (any(s in i[1] for s in pattern_0_2_1) and any(s in i[1] for s in pattern_0_2_2)) and not any(s in i[1] for s in pattern_6_5) and not all(s in i[1] for s in pattern_2_9):
#           i[2]=22
#         elif all(s in i[1] for s in pattern_2_7) and not (any(s in i[1] for s in pattern_0_2_1) and any(s in i[1] for s in pattern_0_2_2)) and not any(s in i[1] for s in pattern_6_5) and not all(s in i[1] for s in pattern_2_9):
#           i[2]=22
#         elif all(s in i[1] for s in pattern_2_8) and not (any(s in i[1] for s in pattern_0_2_1) and any(s in i[1] for s in pattern_0_2_2)) and not any(s in i[1] for s in pattern_6_5) and not all(s in i[1] for s in pattern_2_9):
#           i[2]=22
#         elif any(s in i[1] for s in pattern_1_2) and any(s in i[1] for s in pattern_2_6) and not (any(s in i[1] for s in pattern_0_2_1) and any(s in i[1] for s in pattern_0_2_2)) and not any(s in i[1] for s in pattern_6_5) and not all(s in i[1] for s in pattern_2_9):
#           i[2]=22
#         elif any(s in i[1] for s in pattern_2_2) and any(s in i[1] for s in pattern_2_4) and not (any(s in i[1] for s in pattern_0_2_1) and any(s in i[1] for s in pattern_0_2_2)) and not any(s in i[1] for s in pattern_6_5) and not all(s in i[1] for s in pattern_2_9):
#           i[2]=22
#         elif (any(s in i[1] for s in pattern_1_1) and any(s in i[1] for s in pattern_1_2)) or any(s in i[1] for s in pattern_1_3) and not (any(s in i[1] for s in pattern_2_1) or any(s in i[1] for s in pattern_6_5)) and not (any(s in i[1] for s in pattern_0_2_1) and any(s in i[1] for s in pattern_0_2_2)) and not all(s in i[1] for s in pattern_2_9):
#           i[2]=22
#         elif (any(s in i[1] for s in pattern_0_2) or all(s in i[1] for s in pattern_0_2_11) or all(s in i[1] for s in pattern_0_2_12)) and any(s in i[1] for s in pattern_0_2_1) and any(s in i[1] for s in pattern_0_2_2):
#           i[2]=41 #2331

#         elif any(s in i[1] for s in pattern_34_1):
#           i[2]=405
#         else:
#           i[2]=0
#     return input_lst_1

def rule_base_2nd(input_lst_2):
    #問題keyword調整
    #加班餘數
    pattern_30_1=['餘數', '合併', '加班餘數', '併', '含分', '餘數計算']
    pattern_30_2=['分計', '併計', '分鐘數']
    pattern_30_6=['加班', '顯示', '分']
    pattern_30_7=['加班','出現', '分']
    pattern_30_8=['加班', '產出', '分']
    pattern_30_9 =['月底', '結算','手動', '餘數表','加班費']
    pattern_30_10=['費用', '作業']
    pattern_30_11=['按鈕', '手動', '結算']

    #6 bug
    pattern_s_6_1 =['版更', '抽換', '工程師', '升級', '版本更新'] #any
    pattern_s_6_2=['版本', '更新'] #all
    pattern_s_6_3=['公版','那週'] #all
    pattern_s_6_4=['Bug', 'bug']
    pattern_s_6_7=['程式'] #all
    pattern_s_6_5=['問題','修正'] #any
    pattern_s_6_6=['研議'] # and not
    # 1.和加班資料維護有關的bug(加班餘數)
    pattern_problem_1_2=['加班補休','補休', '補修', '加班費', '班費', '時數', '加班時數', '加班資料維護']
    pattern_problem_1_3=['核算', '申請', '未核算', '更改', '調整', '新增']
    pattern_problem_1_4=['誤差']
    pattern_problem_1_5=['後台', '後臺']

    pattern_solution_1_1=['調整','小時', '計算', '改為']
    pattern_solution_1_2=['版更', '抽換', '程式', '工程師', '升級', '版本更新', 'Bug', 'bug', 'BUG']
    pattern_solution_1_3=['版本', '更新']
    pattern_solution_1_4=['加班時數','加班資料', '加班資料維護']
    pattern_solution_1_5=['異動']
    pattern_solution_1_9=['分鐘', '分鐘數']
    pattern_solution_1_6=['修正','修改','改為']
    pattern_solution_1_7=['加班資料維護']
    pattern_solution_1_8=['一般加班','專案加班', '加班時數', '加班申請']
    #2. 未被告知餘數結算方式變更(加班餘數)
    pattern_problem_2_1=['沒有'] #all
    pattern_problem_2_2=['自動', '未能','無法'] 
    pattern_problem_2_3=['累積', '結算','併計', '結轉', '累計', '合計', '按鈕'] #pattern_problem_1_2 with and pattern_2_2 with
    pattern_problem_2_4=['加', '總'] #all
    pattern_problem_2_5=['異常'] #with 2_4
    pattern_solution_2_6=['自動', '手動', '執行']
    
    #3. 和結算後，專案加班餘數變一般加班有關(加班餘數)
    pattern_solution_3_1=['上限', '超過']
    pattern_problem_3_1=['專案加班', '未', '計入']
    #4. 刪除加班時數而修改餘數 此問題應由人事自行處理，加入對內宣導信件(加班餘數)
    pattern_problem_4_1=['加班','結算'] #all
    pattern_problem_4_2=['刪除', '修正', '移除', '休改為']
    pattern_problem_4_3 =['餘數']

    #5. 未滿一小時(人事操作問題)(加班餘數)
    pattern_solution_5_1=['滿', '未滿']
    pattern_solution_5_2=['一小時', '1小時']

    #6. 結算早於核算(人事操作問題)(加班餘數)
    pattern_solution_6_1=['修改', '核算', '加總', '計算到', '簽核'] 
    pattern_solution_6_2=['加班資料', '加班時數'] 
    pattern_solution_6_3=['加班', '資料'] #all
    pattern_solution_6_5=['加班', '時數'] #all
    pattern_solution_6_4=['太早', '合併'] #all
    pattern_problem_6_1=['手動','未核算'] #all
    pattern_problem_6_2=['未', '計算', '當月'] #all
    pattern_problem_6_3=['未', '併計'] #all
    pattern_problem_6_4=['沒有', '併計'] #all

    # 7. 差勤組別/臨時組別(加班餘數)
    pattern_solution_7_1 = ['差勤', '臨時'] 
    pattern_solution_7_2=['DTS', 'dts', 'Dts']
    pattern_solution_7_3=['組別']

    #9. 結算計算的標示不夠明確(建議可調整為標示一段時間後到XX地方查看之類的)(加班餘數)
    pattern_solution_9_1=['無', '累積', '餘數'] #all
    pattern_solution_9_3=['無', '餘數', '合併'] #all
    pattern_solution_9_2=['背景', '程式'] #all
    pattern_solution_9_4=['背景', '系統'] #all
    pattern_problem_9_3=['尚未', '完成', '餘數', '結算'] #all
    pattern_problem_9_4=['沒有', '餘數','未', '併計'] #all

    # 其他(加班餘數)
    pattern_solution_0=['跨夜', '勞基法人員', '勞基法']
    #加班費
    #勞基法人員加班費性質錯誤、無加班費性質
    pattern_0_3=['加班', '性質']
    pattern_0_3_1=['加班費性質', '加班性質']
    pattern_0_3_2=['加班', '加班費', '加班費計算', '費率']
    pattern_0_3_3=['國定假日', '休息日','屬性', '平日']
    # pattern_0_3_4=[ ]
    # 請領超過上限
    pattern_3_1_2=['上限', '請領上限']
    #一般假日無法申請超過24小時(因為費用組別沒有修改到) （同上）

    #人事由加班資料維護修改同仁加班費請領時數，造成同仁無法正常列印報表。 #4_3
    pattern_r41_1_1 =['調整', '修改', '更新', '調整為', '修正', '修改為', '修改成', '改', '改成', '變更']
    pattern_r41_1_2=['人事', '後台']
    pattern_r41_1_3=['加班費性質']
    pattern_r41_1_4=['無','錯誤']

    #人事異動過加班費請領資料，無法正常退回加班費
    pattern_r4_8_1 =['退回']
    #機關要求啟用分批請領功能
    pattern_r4_7_1 =['分批', '分批列印']
    pattern_r4_7_2=['分', '批'] 

    #41_4無效操作
    pattern_41_4_1=['無效', '操作'] #all
    pattern_41_4_2=['無效操作']
    pattern_41_4_3=['組別', '職務類別'] #all

    #41_9 同一日加班無法送出兩次請領
    pattern_41_9_1=['當日', '同日', '當天']
    pattern_41_9_2=['請領', '退回']
    pattern_41_9_3=['一起']
    #加班時數
    # 無有效加班時數(加班時數不可異動)
    #22
    # 加班時數核算錯誤
    #22_1
    pattern_22_1 = ['無', '無法', '未', '未能'] 
    pattern_22_2=['有效', '核算', '採算',  '核計'] 
    pattern_22_3 =['未核算', '未顯示']

    # 加班補休
    # 要求手動選取加班補休
    pattern_1_5_1 =['選用', '選擇'] 
    pattern_r1_1_3_2=['拉長','拉大'] #any
    pattern_r1_1_3_3=['時數使用紀錄']
    pattern_1_5_3 =['手動', '自行', '先']

    # 未休假加班費
    pattern_02_2_1=['上限', '清除'] #and
    pattern_02_3_1_1=['+']
    pattern_02_3_1_2=['國旅','卡'] #and
    pattern_02_3_1_3=['出國']
    pattern_02_3_1_4=['法規'] #for cut_detail
    pattern_02_2_2=['勤惰統計', '重新'] #and
    pattern_02_2_3=['未', '批核'] # and
    pattern_02_2_4_1=['離職', '離退']
    pattern_02_2_4_2=['版更', '抽換', '更新', '程式', '工程師', '升級', '版本更新', 'Bug', 'bug', 'BUG']
    pattern_02_2_5=['版更','抽換', '程式', '工程師', '升級', '版本更新', 'Bug', 'bug', 'BUG','版本']
    pattern_02_5_1=['手冊']
    pattern_02_5_2=['退回']
    pattern_02_5_3=['轉移']
    pattern_02_1_2=['費用', '作業'] #and for cut_detail
    pattern_02_1_3=['未', '休假', '加班費'] #and for cut_detail
    pattern_02_1_4=['更新', '已休', '天數'] #and for cut_detail
    pattern_02_1_1=['費用作業', '未休假加班費', '休假結算調查', '年終休假結算'] #or for cut_detail
    pattern_02_4_1=['錯誤', '錯誤訊息'] 
    pattern_02_4_3=['組別']
    pattern_02_4_2=['組別', '資料庫', '參數'] 
    pattern_02_5_4=['列印', '清冊']
    #16. 差勤相關
    #16_1. 忘刷卡次數修改、上限設定
    pattern_16_1_1=['忘打', '忘刷', '忘'] 
    pattern_16_1_2=['上限', '次數']
    #16_2 差勤資料列印
    pattern_16_2_1=['列印']
    #16_3 有上下班卡，但顯示刷卡不一致(上下班卡未自動轉出勤)(輪班人員)
    pattern_16_3_1=['一致', '異常'] 
    pattern_16_3_2=['輪班', '班表']
    #16_4 IP無法使用
    pattern_16_4_1=['ip','位置'] #all
    #16_5 刷卡資料未匯入
    pattern_16_5_1=['匯入','轉入']

    # test_p = test[['similarity cluster_solution', 'topic', 'cut_all','rule_3label', 'cut_detail']] #for 未休假加班費(包含兩個類聚算法)
    for i in input_lst_2:
      i[1] = i[1].split(' ')
      i[3] = i[3].split(' ')
      # print(i)

    # 30
      if i[0]=='405_3':
        i[2]='405_3'
        i[0]='405'
      elif i[0]=='405_2':
        i[2]='405_2'
        i[0]='405'
      elif i[0]=='405_1':
        i[2]='405_1'
        i[0]='405' 
      elif i[0]=='405_4':
        i[2]='405_4'
        i[0]='405' 
      elif i[0]=='405_5':
        i[2]='405_5'
        i[0]='405' 
      elif i[0]=='405_6':
        i[2]='405_6'
        i[0]='405'   
      elif i[0]=='30':
        if any(s in i[1] for s in pattern_solution_7_1) and any(s in i[1] for s in pattern_solution_7_3):
          i[2]='7'
        elif any(s in i[1] for s in pattern_solution_7_2):
          i[2]='7'
        elif any(s in i[1] for s in pattern_solution_7_3):
          i[2]='7'
        elif all(s in i[1] for s in pattern_problem_6_4) and any(s in i[1] for s in pattern_solution_6_1) and (any(s in i[1] for s in pattern_solution_6_2) or all(s in i[1] for s in pattern_solution_6_3) or all(s in i[1] for s in pattern_solution_6_4)):
          i[2]='6'
        elif all(s in i[1] for s in pattern_problem_6_1) and any(s in i[1] for s in pattern_solution_6_1) and (any(s in i[1] for s in pattern_solution_6_2) or all(s in i[1] for s in pattern_solution_6_3) or all(s in i[1] for s in pattern_solution_6_4)) and not any(s in i[1] for s in pattern_30_10) and not any(s in i[1] for s in pattern_30_11):
          i[2]='6'
        elif all(s in i[1] for s in pattern_problem_6_2) and any(s in i[1] for s in pattern_solution_6_1) and (any(s in i[1] for s in pattern_solution_6_2) or all(s in i[1] for s in pattern_solution_6_3) or all(s in i[1] for s in pattern_solution_6_4)) and not any(s in i[1] for s in pattern_30_10) and not any(s in i[1] for s in pattern_30_11):
          i[2]='6'
        elif all(s in i[1] for s in pattern_problem_6_3) and any(s in i[1] for s in pattern_solution_6_1) and (any(s in i[1] for s in pattern_solution_6_2) or all(s in i[1] for s in pattern_solution_6_3) or all(s in i[1] for s in pattern_solution_6_4)) and not any(s in i[1] for s in pattern_30_10) and not any(s in i[1] for s in pattern_30_11):
          i[2]='6'
        elif all(s in i[1] for s in pattern_solution_6_4) and not any(s in i[1] for s in pattern_30_10) and not any(s in i[1] for s in pattern_30_11):
          i[2]='6'
        elif any(s in i[1] for s in pattern_solution_0):
          i[2]='0'   
        elif all(s in i[1] for s in pattern_problem_4_1) and any(s in i[1] for s in pattern_problem_4_2):
          i[2]='4'
        elif any(s in i[1] for s in pattern_problem_4_3) and any(s in i[1] for s in pattern_problem_4_2):
          i[2]='4'
        elif any(s in i[1] for s in pattern_solution_3_1) and any(s in i[1] for s in pattern_solution_1_8):
          i[2]='3'  
        elif any(s in i[1] for s in pattern_solution_3_1) and any(s in i[1] for s in pattern_solution_1_8) and any(s in i[1] for s in pattern_problem_2_3):
          i[2]='3' #例外，使用solution去篩選problem
        elif all(s in i[1] for s in pattern_problem_3_1):
          i[2]='3'
        elif any(s in i[1] for s in pattern_solution_5_1) and any(s in i[1] for s in pattern_solution_5_2) and not any(s in i[1] for s in pattern_solution_2_6):
          i[2]='5'
        elif all(s in i[1] for s in pattern_solution_9_1) or all(s in i[1] for s in pattern_solution_9_2) and not any(s in i[1] for s in pattern_30_10) and not any(s in i[1] for s in pattern_30_11):
          i[2]='9'
        elif all(s in i[1] for s in pattern_solution_9_3) or all(s in i[1] for s in pattern_solution_9_4) and not any(s in i[1] for s in pattern_30_10) and not any(s in i[1] for s in pattern_30_11):
          i[2]='9'
        elif all(s in i[1] for s in pattern_problem_9_3) or all(s in i[1] for s in pattern_problem_9_3) and not any(s in i[1] for s in pattern_30_10) and not any(s in i[1] for s in pattern_30_11):
          i[2]='9'
        elif all(s in i[1] for s in pattern_problem_9_4) and not any(s in i[1] for s in pattern_30_10) and not any(s in i[1] for s in pattern_30_11):
          i[2]='9'
        elif any(s in i[1] for s in pattern_problem_2_1) and any(s in i[1] for s in pattern_problem_2_3):
          i[2]='2'  
        elif any(s in i[1] for s in pattern_problem_2_2) and any(s in i[1] for s in pattern_problem_2_3):
          i[2]='2'  
        elif all(s in i[1] for s in pattern_problem_2_4) and any(s in i[1] for s in pattern_problem_2_5):
          i[2]='2' 
        elif any(s in i[1] for s in pattern_30_1) and all(s in i[1] for s in pattern_30_10):
          i[2]='2'
        elif any(s in i[1] for s in pattern_30_1) and any(s in i[1] for s in pattern_30_11):
          i[2]='2' 
        elif any(s in i[1] for s in pattern_problem_1_4) and any(s in i[1] for s in pattern_solution_1_2): 
          i[2]='1'
        elif any(s in i[1] for s in pattern_problem_1_3) and any(s in i[1] for s in pattern_problem_1_5):
          i[2]='1'
        # elif any(s in i[1] for s in pattern_solution_1_4) and any(s in i[1] for s in pattern_solution_1_5):
        #   i[2]='1'
        elif any(s in i[1] for s in pattern_solution_1_4) and any(s in i[1] for s in pattern_problem_1_2):
          i[2]='1'
        elif any(s in i[1] for s in pattern_solution_1_4) and all(s in i[1] for s in pattern_solution_1_3):
          i[2]='1'
        elif any(s in i[1] for s in pattern_solution_1_4) and all(s in i[1] for s in pattern_solution_1_9):
          i[2]='1'
        elif any(s in i[1] for s in pattern_solution_1_6) and any(s in i[1] for s in pattern_solution_1_7):
          i[2]='1'
        elif any(s in i[1] for s in pattern_solution_1_1) and (any(s in i[1] for s in pattern_solution_1_8) or any(s in i[1] for s in pattern_solution_1_4)):
          i[2]='1'  
        # elif any(s in i[0] for s in pattern_30_9):
        #   i[2]='8'  
        else:
          i[2]='0' 
      # elif i[10]=='1_1' or i[10]=='1_2' or i[10]=='1_3' or i[10]=='1_4' or i[10]=='1_5' or i[10]=='1_6':
      elif i[0]=='1':
        # print(i)
        if (any(s in i[1] for s in pattern_1_5_1) or any(s in i[3] for s in pattern_1_5_3)):
          i[2]='1_1'
        elif any(s in i[1] for s in pattern_r1_1_3_2) or any(s in i[3] for s in pattern_r1_1_3_3):
          i[2]='1_1'
        elif any(s in i[1] for s in pattern_1_5_3):
          i[2]='1_1'
        else:
          i[2]='0'
      # elif i[10]==41:
      elif i[0]=='41':
        # print(i)
        if all(s in i[1] for s in pattern_0_3) or any(s in i[1] for s in pattern_0_3_1) or (any(s in i[1] for s in pattern_0_3_2) and any(s in i[1] for s in pattern_0_3_3)) and not (any(s in i[1] for s in pattern_r41_1_3) and any(s in i[1] for s in pattern_r41_1_4)):
          i[2]='41_6'
        elif any(s in i[1] for s in pattern_r4_7_1) or all(s in i[1] for s in pattern_r4_7_2):
          i[2] ='41_7'
        elif any(s in i[1] for s in pattern_3_1_2):
          i[2]='41_2'
        elif (all(s in i[1] for s in pattern_41_4_1) or any(s in i[1] for s in pattern_41_4_2)) or all(s in i[1] for s in pattern_41_4_3):
          i[2]='41_4'
        elif (any(s in i[1] for s in pattern_41_9_1) and any(s in i[1] for s in pattern_41_9_2)) or (any(s in i[1] for s in pattern_41_9_2) and any(s in i[1] for s in pattern_41_9_3)):
          i[2]='41_9'
        elif any(s in i[3] for s in pattern_r4_8_1):
          i[2]='41_8'
        elif any(s in i[1] for s in pattern_r41_1_1) and any(s in i[1] for s in pattern_r41_1_2) and (any(s in i[1] for s in pattern_r41_1_3) and any(s in i[1] for s in pattern_r41_1_4)):
          i[2]='41_3'
        else:
          i[2]='0'
      elif i[0] =='22':
        if any(s in i[1] for s in pattern_22_1) and any(s in i[1] for s in pattern_22_2):
          i[2]='22_1'
        elif any(s in i[1] for s in pattern_22_3):
          i[2]='22_1'
        else:
          i[2]='22_0'
      elif i[0]=='16':
        if any(s in i[1] for s in pattern_16_1_1) and any(s in i[1] for s in pattern_16_1_2):
          i[2]='16_1'
        elif any(s in i[1] for s in pattern_16_2_1):
          i[2]='16_2'
        elif any(s in i[1] for s in pattern_16_3_1) and any(s in i[1] for s in pattern_16_3_2):
          i[2]='16_3'
        elif all(s in i[1] for s in pattern_16_4_1):
          i[2]='16_4'
        elif any(s in i[1] for s in pattern_16_5_1):
          i[2]='16_5'
        else:
          i[2]=0
    return input_lst_2



def convert_vertical(data):
    total_no = data['狀態'].count()
    open_no = data[data['狀態']==1]['單號'].count()
    closed_no=data[data['狀態']==99]['單號'].count()
    counts = data['第一層分類'].value_counts().to_dict()
    # count_data['日期'] = str(data.loc[1, 'date'])
    # count_data['total'] = total_no
    # count_data['open']=open_no
    # count_data['closed']=closed_no
    count_data_class = pd.DataFrame([counts], columns=counts.keys()).to_numpy().tolist()
    flat_list = [item for sublist in count_data_class for item in sublist]
    # print(count_data_class)
    count_data_lst=[str(data.loc[1, 'date']), total_no, open_no, closed_no]
    count_data_lst +=flat_list
    return count_data_lst




    



    



