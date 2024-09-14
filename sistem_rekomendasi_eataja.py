from fastapi import FastAPI
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import json
import requests
pd.options.mode.chained_assignment = None


app = FastAPI()


@app.get('/rekomendasi/{aidi_resto}/{dipesans}')
def show(aidi_resto, dipesans):

    # Builing Request URL
    #id = "0d93f781-7f1a-4c42-a056-683fd8866793"
    url = "https://api.eataja.com/api/get-order-mitra/" + aidi_resto
    # print(url)

    # Send Request Method to EatAja server
    req = requests.get(url).json()

    # Convert JSON Response to Dataframe using pandas
    dta_ = []
    for i in req['data']:
        dta_.append(i)
    df_ = pd.DataFrame(dta_)
    # print(df_.head(53))

    # Experiment
    #print(df_[['waktu_order', 'total_price']])
    #print(df_[['nama_pemesan', 'menu_order']])

    # Data Preprocessing Step1
    namaColumn = 'menu'
    j = 1
    dta_order = []
    # nama_column=[]
    for i in df_.menu_order:
        dta_order.append(i)
        #namaColumn = "menu" + str(j)

    df_order = pd.DataFrame(dta_order)
    dff2 = pd.DataFrame()

    readyTuse = []
    for i in range(len(df_order)):
        ll = []
        for k in df_order.columns:
            if df_order[k][i] != None:
                a = df_order[k][i]['menu_id']
                ##print(a, k, i)
            # else :
            #     a = 'None'
            #     #print(a)
                ll.append(a)
        dff = pd.DataFrame(ll)
        dff = dff.transpose()
        readyTuse.append(ll)
        dff2 = pd.concat([dff2, dff], ignore_index=True)

    dff2['order_id'] = df_['order_id']

    dff3 = pd.DataFrame()
    final = []
    readyTuseV2 = []
    for i in dff2['order_id'].unique():
        kk = []
        a = dff2[dff2['order_id'] == i]
        a.drop(['order_id'], axis=1, inplace=True)
        for j in a.columns:
            for k in a[j]:
                if k not in kk:

                    kk.append(k)
        final.append(k)
        dff = pd.DataFrame([kk])
        dff.dropna(axis=1, inplace=True)
        dff.insert(0, 'order_id', i)
        # kk.append(i)
        cleanedList = [x for x in kk if str(x) != 'nan']
        readyTuseV2.append(cleanedList)
        dff3 = pd.concat([dff3, dff], ignore_index=True)
    # print(dff3)
    # print(readyTuseV2)

    # DataPreprocessing Step2
    te = TransactionEncoder()
    te_ary = te.fit(readyTuseV2).transform(readyTuseV2)

    df = pd.DataFrame(te_ary, columns=te.columns_)
    # print(df)

    # Creat Models for Recommendation System Using Apriori Algorithm
    frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
    # print(frequent_itemsets)

    # POPULAR MENU
    popular_menu = pd.DataFrame()
    popmenu = pd.DataFrame()
    popular_menu = frequent_itemsets.nlargest(3, 'support')
    popmenu = popmenu.append(popular_menu, ignore_index=True)
    # popmenu

    res = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=0.7)

    res1 = res[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    # print(res1)

    res2 = res1[res1['confidence'] >= 1]
    # print(res2)

    # Testing and Get Specific Item Recommendation
    #
    #
    # Input
    buyed_item = 'ast', 'tes2'
    ordernya = dipesans.split("||")
    order_set = frozenset(ordernya)
    # print(order_set)

    # Search
    res3 = res2[res2["antecedents"] == order_set]
    if not res3.empty:
        # Create If Here, When Recomendation is not ready
        # print(res3)

        recomend_item = res3["consequents"]
        # print(recomend_item)
        # Results
        result = res3.to_json(orient="records")

        # Results in JSON response version
        parsed = json.loads(result)
        #print("this parsed", parsed)

        response = json.dumps(parsed, indent=0)
        #print("Respon JSON", json.dumps(parsed, indent=0))

        #dataframeResult = pd.DataFrame(res3)
        readyResposne = res3.reset_index(drop=True)

        # STEP1
        dta_recom = []
        for i in parsed:
            dta_recom.append(i)

        df_recom = pd.DataFrame(dta_recom)
        # df_recom.head(53)

        # SPET2
        idMenu_recom = df_recom["consequents"]
        # idMenu_recom

        df_idMenu_recom = pd.DataFrame(idMenu_recom)
        # df_idMenu_recom.head(53)

        # STEP3
        dfaja = pd.DataFrame()

        for index, rows in df_idMenu_recom.iterrows():
            rows = rows["consequents"]
            for i in rows:
                urlDetailMenuRecomended = "http://api.eataja.com/api/mitra/menu/" + i
                df_recomend = pd.read_json(urlDetailMenuRecomended)

                df_recomend = df_recomend.transpose()
                dfaja = dfaja.append(df_recomend, ignore_index=True)
        # dfaja.head(25)

        # STEP4
        dfaja = dfaja.drop_duplicates(subset=['id'])
        # dfaja.head(25)

        # STEP5
        result_menuRecomended = dfaja.to_json(orient="records")
        parsedDetailRecomendedItem = json.loads(result_menuRecomended)

        # parsedDetailRecomendedItem

        # FINISH
        return parsedDetailRecomendedItem

    elif not popmenu.empty:
        result = popmenu.to_json(orient="records")
        parsed = json.loads(result)

        dta_recom = []
        for i in parsed:
            dta_recom.append(i)

        df_recom = pd.DataFrame(dta_recom)

        idMenu_recom = df_recom["itemsets"]

        df_idMenu_recom = pd.DataFrame(idMenu_recom)

        dfaja = pd.DataFrame()
        counter = 0
        for index, rows in df_idMenu_recom.iterrows():
            rows = rows["itemsets"]
            for i in rows:
                # time.sleep(1)
                urlDetailMenuRecomended = "http://api.eataja.com/api/mitra/menu/" + i
                df_recomend = pd.read_json(urlDetailMenuRecomended)

                df_recomend = df_recomend.transpose()
                dfaja = dfaja.append(df_recomend, ignore_index=True)
    #         if counter == 50:
    #             break
    #         print(counter)
    #         counter +=1

        # STEP4
        dfaja = dfaja.drop_duplicates(subset=['id'])

        # STEP5
        result_menuRecomended = dfaja.to_json(orient="records")
        parsedDetailRecomendedItem = json.loads(result_menuRecomended)

        return parsedDetailRecomendedItem

    else:
        return {"message": "Belum Ada Rekomendasi"}
