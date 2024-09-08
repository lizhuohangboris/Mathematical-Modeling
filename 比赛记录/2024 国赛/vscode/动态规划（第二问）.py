import numpy as np

# 参数：从表格中提取的情况1的数据
P1_defective = 0.10  # 零配件1的次品率
P2_defective = 0.10  # 零配件2的次品率
P1_detect_cost = 2   # 零配件1检测成本
P2_detect_cost = 3   # 零配件2检测成本
P1_price = 4         # 零配件1购买单价
P2_price = 18        # 零配件2购买单价
assembly_cost = 6     # 装配成本
product_detect_cost = 3  # 成品检测成本
market_price = 56     # 市场售价
exchange_loss = 6     # 调换损失（固定的信誉损失与物流费用）
disassembly_cost = 5  # 拆解费用
production_volume = 100  # 生产数量（假设值）

# 更新次品率的函数
def update_defective_rate(defective_rate, detect_effectiveness):
    return defective_rate * (1 - detect_effectiveness)

# 计算每件新产品的生产成本（零配件1 + 零配件2 + 装配成本）
def calculate_new_product_cost():
    return P1_price + P2_price + assembly_cost

# 计算拆解的收益
def calculate_disassembly_saving(p1_rate, p2_rate):
    # 拆解后零配件的再利用率（假设检测后有90%的合格率）
    p1_rate_after_disassembly = update_defective_rate(p1_rate, 0.9)
    p2_rate_after_disassembly = update_defective_rate(p2_rate, 0.9)
    
    # 重新装配的成本减少（节省了重新购买零配件的成本）
    saving = (P1_price * (1 - p1_rate_after_disassembly) + P2_price * (1 - p2_rate_after_disassembly)) * production_volume
    return saving

# 动态规划的一个阶段处理函数，计算利润并加入拆解环节
def dp_stage(p1_rate, p2_rate, detect_p1, detect_p2, detect_product, disassemble):
    cost = 0
    if detect_p1:
        p1_rate = update_defective_rate(p1_rate, 0.9)  # 假设检测有效率为90%
        cost += P1_detect_cost
    if detect_p2:
        p2_rate = update_defective_rate(p2_rate, 0.9)
        cost += P2_detect_cost
    
    # 装配后的成品次品率
    product_defective_rate = p1_rate + p2_rate - p1_rate * p2_rate
    cost += assembly_cost
    
    # 成品检测
    if detect_product:
        product_defective_rate = update_defective_rate(product_defective_rate, 0.9)
        cost += product_detect_cost
    
    # 考虑调换损失，若产品不合格则进行替换
    if product_defective_rate > 0:
        new_product_cost = calculate_new_product_cost()
        total_exchange_loss = product_defective_rate * (new_product_cost + exchange_loss) * production_volume
        cost += total_exchange_loss
    
    # 销售收入
    sales_revenue = (1 - product_defective_rate) * market_price * production_volume
    
    # 如果选择拆解，则计算拆解后的收益和成本
    if disassemble:
        cost += disassembly_cost * production_volume  # 拆解成本
        disassembly_saving = calculate_disassembly_saving(p1_rate, p2_rate)
        cost -= disassembly_saving  # 拆解后节省的成本
    
    # 利润 = 销售收入 - 总成本
    profit = sales_revenue - cost
    
    return profit, product_defective_rate

# 情况1的动态规划求解，加入拆解决策
profit, final_defective_rate = dp_stage(P1_defective, P2_defective, detect_p1=True, detect_p2=True, detect_product=True, disassemble=True)
print(f"总利润: {profit}, 最终成品次品率: {final_defective_rate}")
