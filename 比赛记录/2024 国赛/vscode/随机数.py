import random

# 生成指定范围内的一个随机整数
def random_integer(start, end):
    return random.randint(start, end)

# 生成指定范围内的一个随机浮点数
def random_float(start, end):
    return random.uniform(start, end)

# 生成指定长度的随机整数列表
def random_integer_list(length, start, end):
    return [random.randint(start, end) for _ in range(length)]

# 生成指定长度的随机浮点数列表
def random_float_list(length, start, end):
    return [random.uniform(start, end) for _ in range(length)]

# 主函数，演示生成随机数
if __name__ == "__main__":
    print("随机整数 (1到100):", random_integer(1, 100))
    print("随机浮点数 (0到1):", random_float(0, 1))
    print("随机整数列表:", random_integer_list(5, 10, 50))
    print("随机浮点数列表:", random_float_list(5, 0, 10))
