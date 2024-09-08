import itertools

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
    """
    Calculates the probability of a component being defective.
    If it's detected, assume non-defective if it passes the test, otherwise it is rejected.
    """
    if detection_flag == 1:
        return 0  # If detected, we assume it is non-defective (only those passing detection continue)
    else:
        return component_defect_rate  # If not detected, it follows the regular defect rate

# Function to calculate the probability of a semi-product being defective
def calculate_semi_product_defect_rate(comp_defect_rates, semi_defect_rate, detection_flag):
    """
    Calculates the probability of a semi-product being defective.
    Compounds the defect rates of all components and adds the semi-product's own defect rate.
    If the semi-product is detected, it can only proceed if it's non-defective.
    """
    prob_all_good = 1.0
    for rate in comp_defect_rates:
        prob_all_good *= (1 - rate)
    
    # Defective if any component is defective, or the assembly of the semi-product fails
    final_semi_defect_rate = 1 - prob_all_good + prob_all_good * semi_defect_rate
    
    if detection_flag == 1:
        return 0 if final_semi_defect_rate == 0 else 1  # If detected, it must be non-defective
    else:
        return final_semi_defect_rate

# Function to calculate the probability of a finished product being defective
def calculate_product_defect_rate(semi_defect_rates, product_defect_rate, detection_flag):
    """
    Calculates the probability of the finished product being defective.
    Compounds the defect rates of all semi-products and adds the finished product's own defect rate.
    If the product is detected, it can only proceed if it's non-defective.
    """
    prob_all_good = 1.0
    for rate in semi_defect_rates:
        prob_all_good *= (1 - rate)
    
    # Defective if either any semi-product is defective, or the assembly of the product fails
    final_product_defect_rate = 1 - prob_all_good + prob_all_good * product_defect_rate
    
    if detection_flag == 1:
        return 0 if final_product_defect_rate == 0 else 1  # If detected, it must be non-defective
    else:
        return final_product_defect_rate

# Function to calculate the salvage value of a semi-product based on component defect rates and purchase costs
def calculate_semi_product_salvage_value(comp_defect_rates, comp_purchase_costs):
    """
    Calculates the salvage value of a semi-product based on the defect rates of its components.
    Only components that are non-defective can contribute to the salvage value, and the value is based on their purchase costs.
    """
    salvage_value = 0
    for rate, cost in zip(comp_defect_rates, comp_purchase_costs):
        salvage_value += (1 - rate) * cost  # Only non-defective components contribute to the salvage value
    return salvage_value

# Function to calculate the expected profit for a given set of decisions
def calculate_expected_profit(decision):
    # Unpack the decision variables
    comp_detection = decision[:8]  # Detection decisions for components
    semi_detection = decision[8:11]  # Detection decisions for semi-products
    semi_disassembly = decision[11:14]  # Disassembly decisions for semi-products
    prod_detection = decision[14]  # Detection decision for product
    prod_disassembly = decision[15]  # Disassembly decision for defective product
    
    # Calculate the expected cost for components
    total_component_cost = 0
    total_component_detection_cost = 0
    comp_defect_rates = []  # Track the defect rates of each component after considering detection
    for i in range(8):
        total_component_cost += component_purchase_costs[i]
        if comp_detection[i]:  # If we choose to detect the component
            total_component_detection_cost += component_detection_costs[i]
        comp_defect_rate = calculate_component_defect_rate(component_defect_rates[i], comp_detection[i])
        comp_defect_rates.append(comp_defect_rate)
    
    # Calculate the probability of each semi-product being defective
    semi_defect_rates = []
    for i in range(3):
        if i < 2:  # First two semi-products are made of 3 components each
            current_comp_defect_rates = comp_defect_rates[i * 3:(i + 1) * 3]
            current_comp_purchase_costs = component_purchase_costs[i * 3:(i + 1) * 3]
        else:  # The third semi-product is made of only 2 components
            current_comp_defect_rates = comp_defect_rates[6:8]
            current_comp_purchase_costs = component_purchase_costs[6:8]
        
        # Semi-product defect probability is based on its components and its own defect rate
        semi_prob_defective = calculate_semi_product_defect_rate(current_comp_defect_rates, semi_product_defect_rates[i], semi_detection[i])
        semi_defect_rates.append(semi_prob_defective)

    # Calculate the expected cost for semi-products
    total_semi_product_cost = 0
    total_semi_product_detection_cost = 0
    total_semi_disassembly_cost = 0
    semi_disassembly_value = 0
    for i in range(3):
        if semi_defect_rates[i] < 1:  # If the semi-product is not guaranteed defective
            total_semi_product_cost += semi_product_assembly_costs[i]
        if semi_detection[i]:  # If we choose to detect the semi-product
            total_semi_product_detection_cost += semi_product_detection_costs[i]
        # Only allow disassembly if the semi-product was detected and found defective
        if semi_detection[i] == 1 and semi_defect_rates[i] > 0 and semi_disassembly[i] == 1:  # If semi-product is detected, defective, and chosen for disassembly
            total_semi_disassembly_cost += semi_product_disassembly_costs[i]
            # Calculate salvage value from disassembled components based on purchase costs
            current_comp_defect_rates = comp_defect_rates[i * 3:(i + 1) * 3] if i < 2 else comp_defect_rates[6:8]
            current_comp_purchase_costs = component_purchase_costs[i * 3:(i + 1) * 3] if i < 2 else component_purchase_costs[6:8]
            semi_disassembly_value += calculate_semi_product_salvage_value(current_comp_defect_rates, current_comp_purchase_costs)

    # Calculate the probability of the finished product being defective
    product_prob_defective = calculate_product_defect_rate(semi_defect_rates, product_defect_rate, prod_detection)

    # Calculate the expected cost for the product
    total_product_cost = 0
    if product_prob_defective < 1:  # If the product is not guaranteed defective
        total_product_cost += product_assembly_cost
    if prod_detection:  # If we choose to detect the product
        total_product_cost += product_detection_cost
    total_product_disassembly_cost = 0
    product_disassembly_value = 0
    # Only allow disassembly if the product was detected and found defective
    if prod_detection == 1 and product_prob_defective > 0 and prod_disassembly == 1:  # If product is detected, defective, and chosen for disassembly
        total_product_disassembly_cost += product_disassembly_cost
        # Calculate salvage value of product based on the salvage value of its semi-products
        for i in range(3):
            current_comp_defect_rates = comp_defect_rates[i * 3:(i + 1) * 3] if i < 2 else comp_defect_rates[6:8]
            current_comp_purchase_costs = component_purchase_costs[i * 3:(i + 1) * 3] if i < 2 else component_purchase_costs[6:8]
            product_disassembly_value += calculate_semi_product_salvage_value(current_comp_defect_rates, current_comp_purchase_costs)
    
    # Calculate the final expected profit
    total_cost = (total_component_cost + total_component_detection_cost +
                  total_semi_product_cost + total_semi_product_detection_cost + total_semi_disassembly_cost +
                  total_product_cost + total_product_disassembly_cost)
    
    # Calculate total disassembly value (semi-products + product)
    total_disassembly_value = semi_disassembly_value + product_disassembly_value
    
    # Calculate expected profit considering defect probabilities
    expected_profit = (1 - product_prob_defective) * product_market_value + total_disassembly_value - total_cost
    
    return expected_profit

# Find the optimal decision by maximizing expected profit
best_decision = None
max_profit = float('-inf')

for decision in decisions:
    expected_profit = calculate_expected_profit(decision)
    if expected_profit > max_profit:
        max_profit = expected_profit
        best_decision = decision

print(f"Best decision: {best_decision}")
print(f"Max expected profit: {max_profit}")
