# 导入
import os
from DrissionPage import Chromium,ChromiumOptions
import json
import time
import pandas as pd

datalist = []

from datetime import datetime


def write_json_to_file(data, filename='data.json'):
    try:
        # 打开文件并写入JSON数据
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"JSON数据已成功写入到文件: {filename}")
    except Exception as e:
        print(f"写入文件时发生错误: {e}")


def timestamp_to_datetime(timestamp):
# 将时间戳转换为可读的日期格式
    dt = datetime.fromtimestamp(timestamp)
    formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_date


def getdata2():
    for ii in datalist:
        co = ChromiumOptions()
        # co.headless()
        browser = Chromium(co).latest_tab
        browser.listen.start('aweme/v1/web/user/profile/other/?device_platform=webapp&aid')
        browser.get(ii['homeurl'])
        page_source = browser.html
        # 检查验证码/人机验证
        if "验证码" in page_source or "人机验证" in page_source or "滑块" in page_source:
            print("检测到验证码或人机验证，已保存当前数据，请手动处理后重启程序。")
            write_json_to_file(datalist)
            return
        # 检查用户不存在
        if "无法查看" in page_source or "用户不存在" in page_source:
            print(f"检测到用户不存在：{ii['homeurl']}，已跳过。")
            continue  # 跳过本次，处理下一个用户
        res = browser.listen.wait()
        jsondata = res.response.body
        ii["username"] = jsondata['user']['nickname']
        ii["follower_count"] = jsondata['user']['max_follower_count']
        ii["school_name"] = jsondata['user']['school_name']
        ii["ip_location"] = jsondata['user'].get('ip_location', "")


def getdata(count):
    co = ChromiumOptions()
    # co.headless()
    browser = Chromium(co).latest_tab
    browser.listen.start('https://www.douyin.com/aweme/v1/web/tab/feed/?device_platform=webapp&aid=')
    browser.get("https://www.douyin.com/?recommend=1")
    time.sleep(1)
    try:
        browser.ele('x://*[@id="douyin-right-container"]/div[2]/div[1]/div/div/div[2]').click()
        time.sleep(1)
        browser.ele('x://*[@id="douyin-right-container"]/div[2]/div[1]/div/div/div[2]').click()
        time.sleep(1)
        browser.ele('x://*[@id="douyin-right-container"]/div[2]/div[1]/div/div/div[2]').click()
        i = 0
        for packet in browser.listen.steps():
            # 检查是否出现验证码或人机验证
            page_source = browser.html
            if "验证码" in page_source or "人机验证" in page_source or "滑块" in page_source:
                print("检测到验证码或人机验证，已保存当前数据到，请手动处理后重启程序。")
                write_json_to_file(datalist)
                return
            for index, aweme in enumerate(packet.response.body['aweme_list'], 1):
                if 'item_title' in aweme:
                    #print(aweme)
                    ti = ""
                    if len(aweme['video']['big_thumbs']) > 0:
                        ti = aweme['video']['big_thumbs'][0]['duration']
                    data = {
                        "homeurl": "https://www.douyin.com/user/" + aweme.get('author', "").get('sec_uid',""),
                        "url": "https://www.douyin.com/video/" + aweme.get('aweme_id', ""),
                        "caption": aweme.get('caption', ""),
                        "item_title": aweme.get('item_title', ""),
                        "create_time": timestamp_to_datetime(aweme.get('create_time', "")),
                        "time": ti,
                        "collect_count": aweme.get('statistics', {}).get('collect_count', 0),
                        "comment_count": aweme.get('statistics', {}).get('comment_count', 0),
                        "digg_count": aweme.get('statistics', {}).get('digg_count', 0),
                        "share_count": aweme.get('statistics', {}).get('share_count', 0)
                    }
                    #print(data)
                    datalist.append(data)
                    i += 1
                    print("已成功获取到数据：", i)
            browser.ele('x://*[@id="douyin-right-container"]/div[2]/div[1]/div/div/div[2]').click()
            time.sleep(1)
            browser.ele('x://*[@id="douyin-right-container"]/div[2]/div[1]/div/div/div[2]').click()
            time.sleep(1)
            browser.ele('x://*[@id="douyin-right-container"]/div[2]/div[1]/div/div/div[2]').click()
            time.sleep(1)
            browser.ele('x://*[@id="douyin-right-container"]/div[2]/div[1]/div/div/div[2]').click()
            time.sleep(1)
            browser.ele('x://*[@id="douyin-right-container"]/div[2]/div[1]/div/div/div[2]').click()
            time.sleep(1)
            browser.ele('x://*[@id="douyin-right-container"]/div[2]/div[1]/div/div/div[2]').click()
            time.sleep(1)
            browser.ele('x://*[@id="douyin-right-container"]/div[2]/div[1]/div/div/div[2]').click()
            time.sleep(1)
            if i >= count:
                break
    except Exception as e:
        print(f"发生异常：{e}，已保存当前数据到 data11.json")
        write_json_to_file(datalist)
        # 可选：raise e 让主程序退出
        return

def json_array_to_excel(json_filename, excel_filename):
    try:
        # 打开并读取JSON文件
        with open(json_filename, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 检查数据是否为数组
        if isinstance(data, list):
            # 将JSON数组转换为DataFrame
            df = pd.DataFrame(data)

            # 将DataFrame写入Excel文件
            df.to_excel(excel_filename, index=False, engine='openpyxl')
            print(f"成功将JSON数组写入到Excel文件: {excel_filename}")
        else:
            print("JSON文件中的数据不是一个数组。")

    except FileNotFoundError:
        print(f"文件未找到: {json_filename}")
    except json.JSONDecodeError:
        print(f"JSON解析错误: {json_filename}")
    except Exception as e:
        print(f"发生错误: {e}")


def process_aweme_list(data):
    for index, aweme in enumerate(data['aweme_list'], 1):
        if 'item_title' in aweme:
            #print(aweme)
            ti = ""
            if len(aweme['video']['big_thumbs'])>0:
                ti = aweme['video']['big_thumbs'][0]['duration']
            data = {
                "homeurl": "https://www.douyin.com/user/" + aweme.get('authentication_token', ""),
                "url": "https://www.douyin.com/video/" + aweme.get('aweme_id', ""),
                "caption": aweme.get('caption', ""),
                "item_title": aweme.get('item_title', ""),
                "create_time": timestamp_to_datetime(aweme.get('create_time', "")),
                "time": ti,
                "collect_count": aweme.get('statistics', {}).get('collect_count', 0),
                "comment_count": aweme.get('statistics', {}).get('comment_count', 0),
                "digg_count": aweme.get('statistics', {}).get('digg_count', 0),
                "share_count": aweme.get('statistics', {}).get('share_count', 0)
            }
            #print(data)
            #print("--------------------------")
            # print(f"Aweme {index} item_title: {aweme['item_title']}")


def append_json_to_file(new_data, filename='alldata.json'):
    """将新数据追加到json文件（数组）末尾，不重复"""
    try:
        # 读取原有数据
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                old_data = json.load(file)
                if not isinstance(old_data, list):
                    old_data = []
        except (FileNotFoundError, json.JSONDecodeError):
            old_data = []

        # 去重：以url为唯一标识
        old_urls = {item['url'] for item in old_data if 'url' in item}
        new_unique = [item for item in new_data if item.get('url') not in old_urls]

        all_data = old_data + new_unique

        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(all_data, file, ensure_ascii=False, indent=4)
        print(f"已追加{len(new_unique)}条数据到: {filename}")
    except Exception as e:
        print(f"追加数据时发生错误: {e}")


def process_data(input_file, output_file):
    # 读取原始JSON数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []

    # 如果output_file已存在，先读取旧数据
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                old_data = json.load(f)
                if isinstance(old_data, list):
                    processed_data.extend(old_data)
            except Exception:
                pass  # 文件为空或格式不对时忽略

    for item in data:
        # 1. 删除homeurl和url字段
        item.pop('homeurl', None)
        item.pop('url', None)
        # 2. 处理caption和item_title
        caption = item.get('caption', '')
        item_title = item.get('item_title', '')
        # 如果caption为空，删除该条数据
        if not caption.strip():
            continue 
        # 处理item_title
        if not item_title.strip():
            if '#' in caption:
                prefix = caption.split('#', 1)[0].strip()
                item['item_title'] = prefix if prefix else caption
            else:
                item['item_title'] = caption
        # 处理caption（删除第一个#前的内容）
        if '#' in caption:
            new_caption = '#' + caption.split('#', 1)[1]
            item['caption'] = new_caption
        # 3. 填充ip_location和school_name为"未知"
        item['ip_location'] = item.get('ip_location', '未知') if item.get('ip_location') else '未知'
        item['school_name'] = item.get('school_name', '未知') if item.get('school_name') else '未知' 
        # 4. 检查其他字段是否为空（跳过已处理字段）
        required_fields = [
            'caption', 'item_title', 'create_time', 'time',
            'collect_count', 'comment_count', 'digg_count',
            'share_count', 'username', 'follower_count',
            'school_name', 'ip_location'
        ]
        
        has_empty = False
        for field in required_fields:
            value = item.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                has_empty = True
                break
            if field == 'time' and value == "":
                has_empty = True
                break
        
        if has_empty:
            continue
        
        processed_data.append(item)
    
    # 5. 保存处理后的数据到processed.json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    getdata(10)
    getdata2()
    write_json_to_file(datalist)  # 仍然保存本次采集结果
    append_json_to_file(datalist, 'alldata.json')  # 累积保存所有数据
    json_array_to_excel('alldata.json', 'alldata.xlsx')  # 导出所有数据到Excel
    process_data('data.json', 'processed.json')  # 处理数据并保存到processed.json