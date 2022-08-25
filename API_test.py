# import json
# from collections import OrderedDict
# from flask import make_response



product_info_list = [[["빼빼로 오리지널", "1500", "15", "1_B_1_1", "/url/빼빼로 오리지널.png", "500"]]]

sql_info_list = [["빼빼로 조안나 어디있어","위치", "빼빼로", "Select Product, Location from common_product2 where'%빼빼로%' and where '%치즈 빼빼로%'",  "빼빼로, 1_A_1_2"]
    ,["빼빼로 조안나 어디있어","위치", "조안나 바닐라", "Select Product, Location from common_product2 where'%빼빼로%' and where '%조안나 딸기%'",  "조안나, 1_A_1_2"]]

page_id = ["1","2"]
semantic_id = "위치"
page_sen = ["빼빼로는 1층 B구역 1선반 1진열장에 있다","조안나 바닐라 1층 A구역 2선반 1진열장에 있다"]
tts_ans = ["빼빼로 오리지널은 1층 B구역 1선반 1진열장에 있다","조안나 바닐라 1층 A구역 2선반 1진열장에 있다"]


qaa=[
["질문1",  ['답변1', '답변2', '답변3']],
["질문2",  ['답변1', '답변2', '답변3']]
]

strength={"str_per1", "str_per1"}
str_per1= {"per1": "성실성", "qa":[qaa]}


qbb=[
["질문1", ['답변1', '답변2', '답변3']],
["질문2",  ['답변1', '답변2', '답변3']]
]
str_per2= {"per2": "개방성", "qa":[qbb]}


print("str_per1", str_per1)
print("str_per2", str_per2)
# print("qa", qa[0])
# print("qa", qa[0][1])
# print("qa", qa[0][1][1])
#print('aa', qa["ans"])


qa={
"per1":"개인", "ans": ['ans1', 'ans2', 'ans3'],
"per2": "목표", "ans": ['ans4', 'ans5', 'ans3']
}

print('aa', qa["per1"])

# object_info_list = []
# for k in range(len(product_info_list)):
#     group_data = OrderedDict()
#     product = OrderedDict()
#     object_info = OrderedDict()
#     sql_process = OrderedDict()
#     product_temp =[]
#     for i in range(len(product_info_list[k])): ## product info
#         product = OrderedDict()
#         product["product_name"] = product_info_list[k][i][0]
#         product["product_price"] = product_info_list[k][i][1]
#         product["product_stock"] = product_info_list[k][i][2]
#         product["product_loc"] = product_info_list[k][i][3]
#         product["product_image"] = product_info_list[k][i][4]
#         product["product_sales"] = product_info_list[k][i][5]
#         product_temp.append(product)
#     ### SQL info
#     sql_process["sql_sen"] = sql_info_list[k][0]
#     sql_process["sql_semantic"] = sql_info_list[k][1]
#     sql_process["sql_name"] = sql_info_list[k][2]
#     sql_process["sql_gen"] = sql_info_list[k][3]
#     sql_process["sql_result"] = sql_info_list[k][4]
#     # Integration
#     group_data["page_id"] = page_id[k]
#     group_data["semantic_id"] = semantic_id
#     group_data["page_sen"] = page_sen[k]
#     group_data["tts_ans"] = tts_ans[k]
#     group_data["product"] = product_temp
#     group_data["sql_process"] = sql_process
#     object_info_list.append(group_data)
# object_info["object_info"] = object_info_list
#
# json_dumps = json.dumps(object_info, ensure_ascii=False, indent='\t')
# #print(json.dumps(object_info, ensure_ascii=False, indent="\t") )
# with open('test.json', 'w', encoding='utf-8') as make_file: ## write
#     json.dump(object_info, make_file, ensure_ascii=False, indent='\t')
#
# with open('test.json', 'r', encoding='utf-8') as f: ## read
#     json_data = json.load(f)
#
# print(json.dumps(object_info, indent="\t", ensure_ascii=False) )
#
# print('page', json_data['object_info'][0]['page_id'])
# print('loc', json_data['object_info'][0]['product'][0]["product_loc"])
# print('sen', json_data['object_info'][0]['sql_process']["sql_sen"])
#
