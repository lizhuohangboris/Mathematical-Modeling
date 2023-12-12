#导入爬虫所需要的库
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import time
import json
 
#生成时间戳
def getTime():
    return int(round(time.time() * 1000))
 
#爬虫代码，传递url、headers、键值对参数。最终爬取的数据以json的形式展示
url='https://data.stats.gov.cn/easyquery.htm?cn=B01'
headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}#浏览器代理
key={}#参数键值对
key['m']='QueryData'
key['dbcode']='hgyd'
key['rowcode']='zb'
key['colcode']='sj'
key['wds']='[]'
key['dfwds']='[{"wdcode":"sj","valuecode":"LAST13"}]'
key['k1']=str(getTime())
r=requests.get(url,headers=headers,params=key,verify=False)
js=json.loads(r.text)
 
 [{"dbcode":"hgjd","id":"A01","isParent":true,"name":"å›½æ°‘ç»æµŽæ ¸ç®—","pid":"","wdcode":"zb"},{"dbcode":"hgjd","id":"A02","isParent":true,"name":"å†œä¸š","pid":"","wdcode":"zb"},{"dbcode":"hgjd","id":"A03","isParent":true,"name":"å·¥ä¸š","pid":"","wdcode":"zb"},{"dbcode":"hgjd","id":"A04","isParent":true,"name":"å»ºç­‘ä¸š","pid":"","wdcode":"zb"},{"dbcode":"hgjd","id":"A05","isParent":true,"name":"äººæ°‘ç”Ÿæ´»","pid":"","wdcode":"zb"},{"dbcode":"hgjd","id":"A06","isParent":true,"name":"ä»·æ ¼æŒ‡æ•°","pid":"","wdcode":"zb"},{"dbcode":"hgjd","id":"A07","isParent":true,"name":"å›½å†…è´¸æ˜“","pid":"","wdcode":"zb"},{"dbcode":"hgjd","id":"A08","isParent":true,"name":"æ–‡åŒ–","pid":"","wdcode":"zb"}]
#数据预处理
#strdata就是键值对的值，同时整个字典类型数据存在于列表里面，那事情就好办啦——遍历列表通过键获取值
length = len(js['returndata']['datanodes'])
 
def getList(l):
    List = []
    for i in range(length):
        List.append(eval(js['returndata']['datanodes'][i]['data']['strdata']))
    return List
lst = getList(length)
 
#将列表转换成9*13的DataFrame
array = np.array(lst).reshape(9,13)#转换成9*13的格式
df = pd.DataFrame(array)
 
#将dataframe进行行列重命名
df.columns = ['2023年5月','2023年4月','2023年3月','2023年2月','2023年1月','2022年12月','2022年11月','2022年10月','2022年9月','2022年8月','2022年7月','2022年6月','2022年5月']
df.index = ['居民消费价格指数(上年同月=100)',
'食品烟酒类居民消费价格指数(上年同月=100)',
'衣着类居民消费价格指数(上年同月=100)',
'居住类居民消费价格指数(上年同月=100)',
'生活用品及服务类居民消费价格指数(上年同月=100)',
'交通和通信类居民消费价格指数(上年同月=100)',
'教育文化和娱乐类居民消费价格指数(上年同月=100)',
'医疗保健类居民消费价格指数(上年同月=100)',
'其他用品和服务类居民消费价格指数(上年同月=100)']
 
print(df)
 