import random
import sys
mac = "02:42:ac:11:00:03"
print(int(mac.replace(":", ""), 16))#转换为10进制
random.seed(2485377892355)
SECRET_KEY = str(random.random())
#根据程序中修改
print(SECRET_KEY)