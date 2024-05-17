import os
import pickle

def get_y(folder_path, n):
    # 列出文件夹中的所有文件
    files = os.listdir(folder_path)

    # 筛选出 pkl 文件并按名称排序
    pkl_files = sorted([f for f in files if f.endswith('.pkl')])

    # 取前 n 个文件
    selected_files = pkl_files[:n]
    all_text = []
    # 遍历每个文件并打印 x 元素
    for file_name in selected_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            if len(data["y"])==3:
                text = []
                for item in data["y"]:

                    text.append(item)
                print(text)
                all_text.append(text)
    return all_text
# 设置文件夹路径和前 n 个文件数目
folder_path = "downstream"
n = 100

# 打印每个文件的 x 元素
a= get_y(folder_path, n)