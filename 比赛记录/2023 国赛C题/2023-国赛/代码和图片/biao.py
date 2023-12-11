import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
warnings.filterwarnings("ignore") #忽略警告信息
plt.rcParams['font.sans-serif'] = ['SimHei']
myfont = matplotlib.font_manager.FontProperties(fname=r"simhei.ttf")



检验正态性
In [150]:
#绘制蔬菜各品类时序图（销售月份曲线-规律）
plt.figure(figsize=(20,8),dpi=300)
plt.subplots_adjust(wspace =0.3, hspace =0.3)
n=-1
for i in df_1_data1.columns: 
 s=df_1_data1[i]
 
 n+=1
 plt.subplot(2,4,n+1)
 sns.lineplot(data=s) 
 plt.title(i,fontproperties=myfont)
 plt.grid()
# plt.legend()
plt.savefig('蔬菜各品类销售-月份.png') # 保存图片
plt.show()
# plt.show()