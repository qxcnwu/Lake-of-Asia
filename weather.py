import re
import time
import requests
import numpy as np


def url_maker(M,Y,D):
    return 'https://en.tutiempo.net/climate/'+str(M)+'-'+str(Y)+'/ws-'+str(D)+'.html'

def collection(url):
    headers = {
        'Host': 'en.tutiempo.net',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
        'Referer': 'https://en.tutiempo.net/climate/1937/ws-384570.html',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9'
    }
    r = requests.get(url, headers=headers)
    doc=r.text
    return doc

def found_data(text):
    text_list=[]
    for i in range(8):
        try:
            str2 = re.search(clist[i], text)
            print(str2)
            try:
                text_list.append(float(str2.group(0)[18:-2]))
            except:
                text_list.append(float(str2.group(0)[12:-2]))
        except:
            text_list.append(0)
    return text_list

def data_avg(data):
    c=data.sum(axis=0)/12
    st=c.tolist()
    return st

def read_txt(data,num1):
    path = 'C:/file/data/' + str(num1) + '.iof'
    f = open(path, 'w')
    for k in range(12):
        f.write(str(data[k][0])+','+str(data[k][1])+','+str(data[k][2])+','+str(data[k][3])+','+str(data[k][4])+','+
                str(data[k][5])+','+str(data[k][6])+','+str(data[k][7])+'\n')
    f.close()

clist=[r'class="tc2">\S*?/',r'class="tc3">\S*?/',r'class="tc4">\S*?/',
       r'class="tc5">\S*?/',r'class="tc6">\S*?/',r'class="tc7">\S*?/',
       r'class="tc8">\S*?/',r'class="tc9">\S*?/']
m_list=['01','02','03','04','05','06','07','08','09','10','11','12']

data_y=np.empty((10000,8))
num=0
fd=open('C:/file/2.iof','w')

for i in range(1990,2019):
    data=np.empty((12,8))
    for j in range(12):
        # print('\r[{0}%]'.format(num/19/12*100),end='')
        list= found_data(collection(url_maker(m_list[j],i,351080)))
        data[j]=np.array(list)
        num+=1
        time.sleep(1)
    read_txt(data,i)
    x=data_avg(data)
    fd.write(str(x)+'\n')
fd.close()

