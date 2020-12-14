import pyodbc
import pandas as pd 
import time
from datetime import datetime

def sql_connection(start_date, end_date):
    start_date = str(datetime.strptime(start_date, '%Y-%m-%d'))
    end_date = str(datetime.strptime(end_date, '%Y-%m-%d'))
    server = '103.234.81.137'
    database = 'jtrac'
    username = 'vivian.ou'
    password = 'Origo777'
    #和SQL資料庫連接，輸出Utrac問題單
    cnxn = pyodbc.connect('DRIVER={SQL Server}; SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
    cursor = cnxn.cursor()
    #SQL query->找到某段日期的回覆資料
    sql_query = """SELECT items.id,
        items.sequence_num, 
        items.time_stamp,
        spaces.name,
        spaces.description,
        history.comment, 
        history.time_stamp, 
        history.summary, 
        history.detail, 
        history.status,
        users.name,
        history.cus_int_06,
        history.cus_int_03
        FROM   history 
        LEFT JOIN items ON items.id=history.item_id
        LEFT JOIN spaces ON spaces.id =items.space_id
        LEFT JOIN users ON users.id=history.logged_by
        WHERE  (items.time_stamp > CONVERT(DATETIME, ?, 102)) AND (items.time_stamp <= CONVERT(DATETIME, ?, 102));
        """
    params=(start_date, end_date)
    #將資料存成DataFrame
    result = cursor.execute(sql_query, params)
    lst_result = list(result.fetchall())
    df_columns = ['items_id','items_sequence_num', '時間戳記','spaces_name', '機關', 
                 '回覆', 'history_time_stamp', '摘要', '細節', '狀態','紀錄者', '功能類別', '問題單分類']
    df = pd.DataFrame((tuple(t) for t in lst_result), columns=df_columns) 
    df = df.reset_index(drop=True)
    df['items_sequence_num'] = df['items_sequence_num'].astype('float').astype(int).astype(str)
    df['單號'] = df['spaces_name'].str.cat(df['items_sequence_num'],sep="-")
    df['temp']=df['單號'].str.replace('\-\d+', ' ')
    df['temp'] =df['temp'].str.replace(' ', '') 
    df['主機關代碼'] = df['temp'].str.replace('\_\d+\_\w+', '')
    df['主機關代碼'] = df['temp'].str.replace('\_\w+', '')
    df = df.reset_index(drop=True)
    df=df[['單號','主機關代碼','機關','時間戳記', '摘要','細節','回覆', '狀態','紀錄者','功能類別','spaces_name', '問題單分類']]
    #將所有summary和detail全部填滿(因只有最一開始的問題有此兩個值
    df['摘要']=df.sort_values(['單號','摘要'])['摘要'].ffill()   
    df['細節']=df.sort_values(['單號','細節'])['細節'].ffill()
    #針對回覆進行處理
    df = df.replace(r'\n','', regex=True) 
    df['回覆'] =df['回覆'].str.replace('\n', ' ')
    df['回覆'] =df['回覆'].str.replace('\t', ' ')
    df['回覆1'] =df['回覆'].str.replace(r'^.*原因', '')
    mask =df['回覆1'].str.len()==df['回覆'].str.len()
    df.loc[mask, '回覆1'] = df.loc[mask, '回覆1'].str.replace(r'^.+?(?=[處理])', '')
    #去除非針對系統問題的問題單
    df = df[df['紀錄者']!='自動填單']
    #區分已結案的問題單
    df_closed=df[df['狀態']==99.0]
    #區分未結案的問題單
    df_open=df[df['狀態']==1.0]
    #新增功能類別欄位
    sql_query_function = """SELECT value, text, column_type
    FROM   cus_int_options 
    """
    result_2 = cursor.execute(sql_query_function)
    lst_result_2 = list(result_2.fetchall())
    df2_columns=['value', '功能類別', '欄位']
    # df1 = pd.DataFrame((tuple(t) for t in lst_result_2), columns=df2_columns) 
    # df2 = df2.reset_index(drop=True)
    # df2 = df2[df2['欄位']=='cus_int_06']
    # df2 = df2.reset_index(drop=True)
    # df2 = df2.drop_duplicates('value')
    df1 = pd.DataFrame((tuple(t) for t in lst_result_2), columns=df2_columns) 
    df1 = df1.reset_index(drop=True)
    df2 = df1[df1['欄位'].isin(['cus_int_06'])]
    df3 = df1[df1['欄位'].isin(['cus_int_03'])]
    df2 = df2.reset_index(drop=True)
    df2 = df2.drop_duplicates('value')
    df3 = df3.reset_index(drop=True)
    df3 = df3.drop_duplicates('value')
    df3.columns = ['value', '問題單分類', '欄位']
    # df = df.reset_index(drop=True)
    df_open['功能類別'] = df_open['功能類別'].map(df2.set_index('value')['功能類別'])
    df_closed['功能類別'] = df_closed['功能類別'].map(df2.set_index('value')['功能類別'])
    df_open['問題單分類'] = df_open['問題單分類'].map(df3.set_index('value')['問題單分類'])
    df_closed['問題單分類'] = df_closed['問題單分類'].map(df3.set_index('value')['問題單分類'])
    #去除非針對系統問題的問題單
    drop_facility=['000_AutoTest', '000_ENG_M', '000_UG', '000_UGAls', '000_UGAtt', '000_UGCar', 
    '000_UGCas', '000_UGCsw', '000_UGExp', '000_UGIsm', '000_UGMon', '000_UGOes', '000_UGProposal', 
    '000_UGRas', '000_UGRFP', 'WebITR', 'jTracDlp']
    assignee_lst =['黃羿禎']
    df_closed = df_closed[~df_closed['spaces_name'].isin(drop_facility)]
    df_closed = df_closed[~df_closed['紀錄者'].isin(assignee_lst)] 
    df_open = df_open[~df_open['spaces_name'].isin(drop_facility)]
    df_open = df_open[~df_open['紀錄者'].isin(assignee_lst)]
    closed_lst= df_closed['單號'].tolist()
    df_open = df_open[~df_open['單號'].isin(closed_lst)]
    df_closed = df_closed.drop_duplicates(subset='單號')
    df_open = df_open.drop_duplicates(subset='單號')   
    df_all = pd.concat([df_open, df_closed], axis=0)
    df_all=df_all.drop_duplicates(subset='單號')
    return df_all

def save_label_to_sql(label, id, correct_check):
    #因為此張表之後會存在各個機關端，因此先不在utrac開此表，從json取
    #不過，目前在local端測試，可以先使用sql server的資料
    server = 'VIVIAN-HP\SQLEXPRESS' 
    database = 'QAlabel' 
    # username = 'username' 
    # password = 'yourpassword' 
    
    cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database)
    cursor = cnxn.cursor()
    sql_query="""
                INSERT INTO QAlabel.dbo.jtrac_label
                VALUES
                (?,?,(SELECT Service_label from QAlabel.dbo.jtrac_pred_label_code WHERE jtrac_pred_label_code.Predict_label=?),?);
                """
    param=(int(id), label, label, correct_check)
    cursor.execute(sql_query, param)
    cnxn.commit()
    cursor.close()
    cnxn.close()
    print('in save_label_to_sql')

def extrac_QA_label(label, para):
    server = 'VIVIAN-HP\SQLEXPRESS' 
    database = 'QAlabel' 
    # username = 'username' 
    # password = 'yourpassword' 
    
    cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database)
    cursor = cnxn.cursor()
    sql_query="""
                SELECT para 
                from label_parameter 
                WHERE labels=? AND para_type=? 
                """
    param=(label, para)
    cursor.execute(sql_query, param)
    row = [item[0] for item in cursor.fetchall()]
    cursor.close()
    cnxn.close()
    return row

def update_sql(utrac_id, correct_check):
    server = 'VIVIAN-HP\SQLEXPRESS' 
    database = 'QAlabel' 
    # username = 'username' 
    # password = 'yourpassword' 
    
    cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database)
    cursor = cnxn.cursor()
    sql_query="""
                Update QAlabel.dbo.jtrac_label
                SET correct_check=?
                WHERE id=?;
                """
    param=(correct_check, utrac_id)
    cursor.execute(sql_query, param)
    cnxn.commit()
    cursor.close()
    cnxn.close()
    print('in update_sql')


if __name__ == '__main__':
    start_date = input('輸入起始日期:')
    end_date=input('輸入結束日期:')
    sqlquery = sql_connection(start_date, end_date)
    sqlquery.info()
