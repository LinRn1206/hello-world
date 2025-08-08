import json
import hashlib
import re

def process_sensitive_data(data):
    """执行数据合规处理"""
    processed_data = []
    
    for item in data:
        new_item = item.copy()
        
        # 1. 用户名哈希处理（MD5）
        if "username" in new_item:
            username = new_item["username"]
            new_item["username"] = hashlib.md5(username.encode("utf-8")).hexdigest()
        
        # 2. IP属地脱敏（保留前缀）
        if "ip_location" in new_item:
            ip_loc = new_item["ip_location"]
            if "IP属地：" in ip_loc:
                new_item["ip_location"] = "IP属地：**"  # 示例：广东 -> **
            else:
                new_item["ip_location"] = "N/A"
        
        # 3. 删除主页链接字段
        if "homeurl" in new_item:
            del new_item["homeurl"]
        
        # 4. 视频URL部分哈希处理（前20字符+...）
        if "url" in new_item:
            url = new_item["url"]
            if len(url) > 20:
                visible_part = url[:20]
                hidden_part = "*" * (len(url) - 20)
                new_item["url"] = f"{visible_part}{hidden_part}"
            else:
                new_item["url"] = "*" * len(url)
        
        processed_data.append(new_item)
    
    return processed_data

# 读取原始数据
with open("data9.json", "r", encoding="utf-8") as f:
    original_data = json.load(f)

# 执行合规处理
processed_data = process_sensitive_data(original_data)

# 保存处理后的数据
with open("datasensiyive.json", "w", encoding="utf-8") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)

print("数据处理完成，结果已保存到 datasensiyive.json")