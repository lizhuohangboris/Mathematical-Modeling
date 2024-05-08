import re

en2 = "时村前花墨写懒炉寒炉寒酒美炉寒温时疑恍酒美"
en = en2[::-1]
en3 = "花墨村前花墨温时村前看醉疑恍村前看醉写懒花墨酒美花墨温时村前看醉酒美花墨炉寒花墨温时看醉看醉看醉温时村前看醉白月看醉花墨看醉炉寒花墨温时花墨炉寒看醉酒美花墨温时看醉写懒看醉疑恍村前温时温"
en1 = en3[::-1]

table = "冻笔新诗懒写寒炉美酒时温醉看墨花月白恍疑雪满前村"
enc = en + en1
print(enc)
index = 0
idx_list = []
while index <= len(enc):
    temp = enc[index:index + 2]
    idx = table.find(temp) // 2
    idx_list.append(idx)
    index += 2

char_list = []
index = 0
while index < len(idx_list):
    if idx_list[index] == 11:
        char_list.append(chr(61 + idx_list[index + 1]))
        index += 2
    else:
        char_list.append(chr(idx_list[index] + ord('0')))
        index += 1

result = ''.join(char_list[:-1])
ascii_str = bytes.fromhex(result).decode('utf-8')

print(ascii_str)