import itertools
import matplotlib.pyplot as plt

# Define parameters (based on the provided table)
component_defect_rates = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]  # Defect rates for components
component_purchase_costs = [2, 8, 12, 2, 8, 12, 8, 12]  # Purchase costs for components
component_detection_costs = [1, 1, 2, 1, 1, 2, 1, 2]  # Detection costs for components
semi_product_defect_rates = [0.10, 0.10, 0.10]  # Defect rates for semi-finished products
semi_product_assembly_costs = [8, 8, 8]  # Assembly costs for semi-finished products
semi_product_detection_costs = [4, 4, 4]  # Detection costs for semi-finished products
semi_product_disassembly_costs = [6, 6, 6]  # Disassembly costs for semi-finished products
product_defect_rate = 0.10  # Defect rate for finished product
product_assembly_cost = 8  # Assembly cost for finished product
product_detection_cost = 10  # Detection cost for finished product
product_market_value = 200  # Market value for finished product
product_disassembly_cost = 40  # Disassembly cost for finished product

# Define all possible decision combinations for components, semi-products, product, and disassembly
decisions = list(itertools.product([0, 1], repeat=17))  # 17 variables for decisions

# Function to calculate the probability of a component being defective
def calculate_component_defect_rate(component_defect_rate, detection_flag):
    if detection_flag == 1:
        return 0  # If detected, we assume it is non-defective
    else:
        return component_defect_rate  # If not detected, it follows the regular defect rate

# Function to calculate the probability of a semi-product being defective
def calculate_semi_product_defect_rate(comp_defect_rates, semi_defect_rate, detection_flag):
    prob_all_good = 1.0
    for rate in comp_defect_rates:
        prob_all_good *= (1 - rate)
    
    final_semi_defect_rate = 1 - prob_all_good + prob_all_good * semi_defect_rate
    
    if detection_flag == 1:
        return 0 if final_semi_defect_rate == 0 else 1  # If detected, it must be non-defective
    else:
        return final_semi_defect_rate

# Function to calculate the probability of a finished product being defective
def calculate_product_defect_rate(semi_defect_rates, product_defect_rate, detection_flag):
    prob_all_good = 1.0
    for rate in semi_defect_rates:
        prob_all_good *= (1 - rate)
    
    final_product_defect_rate = 1 - prob_all_good + prob_all_good * product_defect_rate
    
    if detection_flag == 1:
        return 0 if final_product_defect_rate == 0 else 1  # If detected, it must be non-defective
    else:
        return final_product_defect_rate

# Function to calculate the salvage value of a semi-product based on component defect rates and purchase costs
def calculate_semi_product_salvage_value(comp_defect_rates, comp_purchase_costs):
    salvage_value = 0
    for rate, cost in zip(comp_defect_rates, comp_purchase_costs):
        salvage_value += (1 - rate) * cost  # Only non-defective components contribute to the salvage value
    return salvage_value

# Function to calculate the expected profit for a given set of decisions
def calculate_expected_profit(decision):
    comp_detection = decision[:8]  # Detection decisions for components
    semi_detection = decision[8:11]  # Detection decisions for semi-products
    semi_disassembly = decision[11:14]  # Disassembly decisions for semi-products
    prod_detection = decision[14]  # Detection decision for product
    prod_disassembly = decision[15]  # Disassembly decision for defective product

    total_component_cost = 0
    total_component_detection_cost = 0
    comp_defect_rates = []  # Track the defect rates of each component after considering detection
    for i in range(8):
        total_component_cost += component_purchase_costs[i]
        if comp_detection[i]:  # If we choose to detect the component
            total_component_detection_cost += component_detection_costs[i]
        comp_defect_rate = calculate_component_defect_rate(component_defect_rates[i], comp_detection[i])
        comp_defect_rates.append(comp_defect_rate)

    semi_defect_rates = []
    for i in range(3):
        if i < 2:
            current_comp_defect_rates = comp_defect_rates[i * 3:(i + 1) * 3]
        else:
            current_comp_defect_rates = comp_defect_rates[6:8]
        
        semi_prob_defective = calculate_semi_product_defect_rate(current_comp_defect_rates, semi_product_defect_rates[i], semi_detection[i])
        semi_defect_rates.append(semi_prob_defective)

    total_semi_product_cost = 0
    total_semi_product_detection_cost = 0
    total_semi_disassembly_cost = 0
    semi_disassembly_value = 0
    for i in range(3):
        if semi_defect_rates[i] < 1:
            total_semi_product_cost += semi_product_assembly_costs[i]
        if semi_detection[i]:
            total_semi_product_detection_cost += semi_product_detection_costs[i]
        if semi_detection[i] == 1 and semi_defect_rates[i] > 0 and semi_disassembly[i] == 1:
            total_semi_disassembly_cost += semi_product_disassembly_costs[i]
            if i < 2:
                current_comp_defect_rates = comp_defect_rates[i * 3:(i + 1) * 3]
            else:
                current_comp_defect_rates = comp_defect_rates[6:8]
            semi_disassembly_value += calculate_semi_product_salvage_value(current_comp_defect_rates, component_purchase_costs[i * 3:(i + 1) * 3])

    product_prob_defective = calculate_product_defect_rate(semi_defect_rates, product_defect_rate, prod_detection)

    total_product_cost = 0
    if product_prob_defective < 1:
        total_product_cost += product_assembly_cost
    if prod_detection:
        total_product_cost += product_detection_cost
    total_product_disassembly_cost = 0
    product_disassembly_value = 0
    if prod_detection == 1 and product_prob_defective > 0 and prod_disassembly == 1:
        total_product_disassembly_cost += product_disassembly_cost
        for i in range(3):
            if i < 2:
                current_comp_defect_rates = comp_defect_rates[i * 3:(i + 1) * 3]
            else:
                current_comp_defect_rates = comp_defect_rates[6:8]
            product_disassembly_value += calculate_semi_product_salvage_value(current_comp_defect_rates, component_purchase_costs[i * 3:(i + 1) * 3])

    total_cost = (total_component_cost + total_component_detection_cost +
                  total_semi_product_cost + total_semi_product_detection_cost + total_semi_disassembly_cost +
                  total_product_cost + total_product_disassembly_cost)

    total_disassembly_value = semi_disassembly_value + product_disassembly_value

    expected_profit = (1 - product_prob_defective) * product_market_value + total_disassembly_value - total_cost

    return expected_profit

# Sensitivity analysis and plotting
def sensitivity_analysis():
    results = []
    defect_rates_to_test = [0.05, 0.10, 0.15]  # Test a range of defect rates

    for defect_rate in defect_rates_to_test:
        global component_defect_rates
        component_defect_rates = [defect_rate] * 8  # Update defect rates

        best_decision = None
        max_profit = float('-inf')

        for decision in decisions:
            expected_profit = calculate_expected_profit(decision)
            if expected_profit > max_profit:
                max_profit = expected_profit
                best_decision = decision

        results.append((defect_rate, max_profit))

    return results

def plot_sensitivity_results(results):
    rates = [result[0] for result in results]
    profits = [result[1] for result in results]
    
    plt.plot(rates, profits, marker='o')
    plt.xlabel('Defect Rate')
    plt.ylabel('Max Expected Profit')
    plt.title('Sensitivity Analysis of Defect Rate on Profit')
    plt.grid(True)
    plt.show()

# Perform sensitivity analysis and plot results
results = sensitivity_analysis()
plot_sensitivity_results(results)
