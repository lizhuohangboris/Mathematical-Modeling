from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import pandas as pd

url_pre = "https://www.wimbledon.com/en_GB/scores/results/day"
url_list = [url_pre + str(i) + ".html" for i in range(1, 22)]

# 创建一个空的DataFrame来存储数据
df = pd.DataFrame()

# 创建一个 WebDriver 实例，并使用 with 语句确保在使用结束后关闭浏览器
with webdriver.Edge() as driver:
    for url in url_list[-3:]:
        driver.get(url)

        # 等待按钮可点击
        button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(.,"Slamtracker Recap")]'))
        )

        # 点击按钮
        ActionChains(driver).click(button).perform()

        # 等待页面加载完成
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'stats-row'))
        )

        # 获取页面源代码
        recap_html = driver.page_source
        soup = BeautifulSoup(recap_html, 'html.parser')

        # 获取球员名字
        head_row = soup.find('div', class_='stats-header-row')
        player1_name_element = head_row.find('div', class_='stats-header-cell t1').find('span', class_='long').find('div', class_='jsx-parser')
        player2_name_element = head_row.find('div', class_='stats-header-cell t2').find('span', class_='long').find('div', class_='jsx-parser')

        player1_name = player1_name_element.text.strip() if player1_name_element else None
        player2_name = player2_name_element.text.strip() if player2_name_element else None

        # 获取统计数据
        stats_rows = soup.find_all('div', class_='stats-row')
        data = {'Player 1 Name': player1_name, 'Player 2 Name': player2_name}

        for row in stats_rows:
            stat_label_element = row.find('div', class_='stats-label')
            if stat_label_element:
                stat_label = stat_label_element.find('span').text.strip()

                player1_stat_element = row.find('div', class_='stats-data t1 leading')
                player2_stat_element = row.find('div', class_='stats-data t2')

                if player1_stat_element and player2_stat_element:
                    player1_stat = player1_stat_element.text.strip()
                    player2_stat = player2_stat_element.text.strip()

                    data[f'Player 1 {stat_label}'] = player1_stat
                    data[f'Player 2 {stat_label}'] = player2_stat

        # 将数据添加到 DataFrame
        df = df.append(data, ignore_index=True)

# 将 DataFrame 保存为 CSV 文件
df.to_csv('C:/Users/92579/Desktop/tennis_stats.csv', index=False)
