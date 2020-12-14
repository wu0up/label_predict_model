
# print(sklearn.__version__)
import pandas as pd
import jieba
import codecs  
import re
import monpa
monpa.load_userdict('train_monpa.txt')
# monpa.load_userdict('/content/train_monpa.txt')
import time

def read_data(data, column):
    data[column] = data[column].replace('\r|\n','', regex=True).astype(str)
    # data['回覆1'] = data['回覆1'].replace('\r|\n','', regex=True).astype(str)
    content_lst =data[column].tolist()
    # solution_lst =data['回覆1'].tolist()
    return data, content_lst

def remove_duplicate(x):
    return list(dict.fromkeys(x))

def cut_sentence(data, column):
    data, data_content_1= read_data(data, column)
    data_content_1_k =[]
    # data_solution_1_k =[]
    for row in data_content_1:
        row_1 = monpa.pseg(row)
        data_content_1_k.append(row_1)
    # for row in data_solution_1:
    #     row_1 = monpa.pseg(row)
    #     data_solution_1_k.append(row_1)

    stop_list =['Caa', 'Cab', 'Cba', 'Cbb', 'COLONCATEGORY', 'COMMACATEGORY', 
            'DASHCATEGORY', 'DE', 'ETCCATEGORY', 'EXCLAMATIONCATEGORY',
           'PARENTHESISCATEGORY', 'PAUSECATEGORY', 'PERIODCATEGORY', 
            'QUESTIONCATEGORY', 'SEMICOLONCATEGORY', 'T', 'PER', 'NC', 'Nc', 'LOC','ORG', 
           'P', 'Neu']
  # 去除標點符號
    newList_content_1=[]
    # newList_solution_1=[]
    for line in data_content_1_k:
        row=[]
        for i in line:
            if i[1] not in stop_list and i[0] !=' ':
                row.append(i[0])
        newList_content_1.append(row)
    # for line in data_solution_1_k:
    #     row=[]
    #     for i in line:
    #         if i[1] not in stop_list and i[0] !=' ':
    #             row.append(i[0])
    #     newList_solution_1.append(row)

    # stopwords = [line.strip() for line in codecs.open('/content/ChineseStopWords.txt', 'r', 'utf-8').readlines()] 
    stopwords = [line.strip() for line in codecs.open('ChineseStopWords.txt', 'r', 'utf-8').readlines()] 
  #去除stopword
    text_content_1=[]
    # text_solution_1=[]
    for line in newList_content_1:
        row=[]
        for i in line:
            if i not in stopwords and i!='  ':
                row.append(i)
        text_content_1.append(row)
    # for line in newList_solution_1:
    #     row=[]
    #     for i in line:
    #         if i not in stopwords and i!='  ':
    #             row.append(i)
    #     text_solution_1.append(row)
  #去除重複字
    text_content_1_d=[]
    # text_solution_1_d=[]
    for i in text_content_1:
        text_content_1_d.append(remove_duplicate(i))
    # for i in text_solution_1:
    #     text_solution_1_d.append(remove_duplicate(i))
  #將list變成string
    text_str_1=[]
    for line in text_content_1_d:
        single_string=" ".join(line)
        text_str_1.append(single_string)
    text_str_2=[]
    # for line in text_solution_1_d:
    #     single_string=" ".join(line)
    #     text_str_2.append(single_string)
    # ner_str=[]
    df_content = pd.DataFrame({'cut_abs_detail': text_str_1})
    df_content['abs_detail_response'] = df_content['cut_abs_detail'].astype(str)
    data_1= data.reset_index(drop=True)
    result = pd.concat([data_1, df_content], axis=1)
    return result