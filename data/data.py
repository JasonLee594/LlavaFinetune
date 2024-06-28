import json
import uuid
import os

# 定义每个文件夹对应的对话列表
conversations_list_canjiao = [


    {   "from": "human",
        "value":"<image>\nCan you name some usual defects on this product?"
    },
    {   "from":"gpt",
        "value":"Please relate to the doc in /media/ZihaoLi/xxx.doc.Usual defects on this product can be 'Glue Mark','black point' and 'scratch'."
    }

]

conversations_list_good = [

    {   "from": "human",
        "value":"<image>\nCan you name some usual defects on this product?"
    },
    {   "from":"gpt",
        "value":"Usual defects on this product can be 'Glue Mark','black point' and 'scratch'and so on.Please relate to the doc in /media/ZihaoLi/xxx.doc."
    }

]

conversations_list_yisedian = [
    {   "from": "human",
        "value":"<image>\nCan you name some usual defects on this product?"
    },
    {   "from":"gpt",
        "value":"Usual defects on this product can be 'Glue Mark','black point' and 'scratch'and so on.Please relate to the doc in /media/ZihaoLi/xxx.doc."
    }

]

conversations_list_black = [
    {   "from": "human",
        "value":"<image>\nCan you name some usual defects on this product?"
    },
    {   "from":"gpt",
        "value":"Usual defects on this product can be 'Black point','scratches' ,'broken','hair' and so on."
    }

]

conversations_list_posun = [
    {   "from": "human",
        "value":"<image>\nCan you name some usual defects on this product?"
    },
    {   "from":"gpt",
        "value":"Usual defects on this product can be 'Black point','scratches' ,'broken','hair' and so on."
    }

]

# 创建字典，映射文件夹到对话列表
folder_to_conversations = {
    "black": conversations_list_black,
    "canjiao": conversations_list_canjiao,
    "good": conversations_list_good,
    "posun": conversations_list_posun,
    "yisedian":conversations_list_yisedian
}

# 创建数据集对象列表
dataset_list = []
for folder, conversations in folder_to_conversations.items():
    # 获取文件夹下的所有图片文件
    folder_path = os.path.join(".", folder)
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
    for image_file in image_files:
        for i in range(0, len(conversations), 2):  # 每两个对话对生成一个数据集对象
            dataset_item = {
                "id": str(uuid.uuid4()),
                "image": os.path.join(folder, image_file),  # 使用文件夹和图片文件名
                "conversations": conversations[i:i+2]  # 取两个对话对
            }
            dataset_list.append(dataset_item)

# 将数据集对象列表转换为JSON字符串
json_data = json.dumps(dataset_list, indent=4,ensure_ascii=False)

# 输出JSON字符串
print(json_data)

# 写入JSON文件
with open("dataset.json", "w", encoding="utf-8") as f:
    f.write(json_data)
