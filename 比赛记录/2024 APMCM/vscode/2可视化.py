import matplotlib.pyplot as plt
import pandas as pd

# 数据准备
data = {
    "Country": ["USA", "USA", "France", "France", "Germany", "Germany"],
    "Pet Type": ["Cat", "Dog", "Cat", "Dog", "Cat", "Dog"],
    2023: [7380, 8010, 1660, 990, 1570, 1050],
    2022: [7380, 8970, 1490, 760, 1520, 1060],
    2021: [9420, 8970, 1510, 750, 1670, 1030],
    2020: [6500, 8500, 1490, 775, 1570, 1070],
    2019: [9420, 8970, 1300, 740, 1470, 1010],
}

df = pd.DataFrame(data)

# 按国家和宠物种类可视化
years = [2019, 2020, 2021, 2022, 2023]
fig, ax = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

for idx, country in enumerate(["USA", "France", "Germany"]):
    # 提取国家数据
    country_data = df[df["Country"] == country]
    for pet in ["Cat", "Dog"]:
        pet_data = country_data[country_data["Pet Type"] == pet]
        ax[idx].plot(years, pet_data.iloc[0, 2:], label=f"{country} {pet}", marker="o")

    ax[idx].set_title(f"Pet Population Trends in {country} (2019-2023)")
    ax[idx].set_ylabel("Number of Pets (10,000s)")
    ax[idx].grid(True)
    ax[idx].legend()

ax[1].set_xlabel("Year")
plt.tight_layout()

plt.show()
