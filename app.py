#get-接收問題單的摘要、細節
#post-html file
# 多線程問題_from flask_script import Server
from flask import Flask, jsonify, request, redirect,url_for, render_template
import json
# import webbrowser
# import bs4
from utrac_classification_v5_response import *
from SQLextract import save_label_to_sql

app = Flask(__name__)

@app.route('/predict_model', methods=['POST'])
def predict_model():

    # 如果沒有資料，要怎麼處理
    # 如果傳入的資料格式錯誤，要怎麼處理
    data_dict=request.get_json(force=True)
    utrac_summary = [data_dict['summary']]
    utrac_detail =  [data_dict['detail']]
    utrac_response = [data_dict['comment']] #回覆
    utrac_value = [data_dict['value']]
    utrac_id = [data_dict['utrac_id']] # utrac單號
    utrac_recorder=[data_dict['logged_by']] #填單人員
    utrac_sentence = [data_dict['summary']+' '+data_dict['detail']+' '+data_dict['comment']]
    recorder_lst=[3967, 3202, 1031]
    predict_label = utrac_classification(utrac_id[0], utrac_sentence, utrac_value[0])
    
    #秀蘭-3967, 偉豪-3202, 信銘-1031
    #如果填單人員是資深客服，導到新頁面，再將資料存入sql server

    if utrac_recorder[0] in recorder_lst:
        print(utrac_id)
        save_label_to_sql(predict_label, utrac_id[0], 0)
        return jsonify(label=predict_label, utrac_id=utrac_id[0], redirect=1)

    else:
        print(utrac_id)
        save_label_to_sql(predict_label, utrac_id[0], 0)
        return jsonify(label=predict_label, utrac_id=utrac_id[0], redirect=0)


if __name__ == '__main__':
    app.run(debug=False)