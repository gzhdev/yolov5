import numpy as np
import pandas as pd
import os
from shutil import copy
from shutil import move
import time
import random
from PIL import Image


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box.iloc[:, 0] + box.iloc[:, 2]) / 2.0 - 1
    y = (box.iloc[:, 1] + box.iloc[:, 3]) / 2.0 - 1
    w = box.iloc[:, 2] - box.iloc[:, 0]
    h = box.iloc[:, 3] - box.iloc[:, 1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def gen_darknet(dirpath):
    for i in dn['文件名'].unique():
        log = dn[dn['文件名'] == i]
        # print(i)
        # print(log)
        with open(os.path.join(dirpath, log.iloc[0, 1]), 'w', encoding='utf-8') as f:
            for i in log.iloc[:, 2:].values:
                s = ''
                for n in i:
                    s += str(n)
                    s += ' '
                f.write(s)
                f.write('\n')


# 指定工作目录
projectPath = "/content/yolov5"
dataPath = "/content/rawdata"
os.chdir(dataPath)
print(f"指定工作目录为{dataPath}")
# 创建数据文件夹
targetDir = ["dn", "dataset/images/train", "dataset/labels/train", "dataset/images/val",
             "dataset/labels/val"]
for d in targetDir:
    # d = path+d
    os.makedirs(d)
    print(f"创建目标文件夹 {d}")
# 读取图片信息
train_data = pd.read_csv("附件2/图片虫子位置详情表.csv", encoding='gbk')
train_data = train_data.dropna()
train_data = train_data.reset_index(drop=True)

names = {'二化螟': 0, '二点委夜蛾': 1, '棉铃虫': 2, '褐飞虱属': 3, '白背飞虱': 4, '八点灰灯蛾': 5,
         '蝼蛄': 6, '蟋蟀': 7, '甜菜夜蛾': 8, '黄足猎蝽': 9, '稻纵卷叶螟': 10, '甜菜白带野螟': 11,
         '黄毒蛾': 12, '石蛾': 13, '大黑鳃金龟': 14, '粘虫': 15, '稻螟蛉': 16, '甘蓝夜蛾': 17,
         '地老虎': 18, '大螟': 19, '瓜绢野螟': 20, '线委夜蛾': 21, '水螟蛾': 22, '紫条尺蛾': 23,
         '歧角螟': 24, '草地螟': 25, '豆野螟': 26, '干纹冬夜蛾': 27}
train_data["虫子名称"] = train_data.loc[:, "虫子名称"].replace(names)

loc = train_data.loc[:, ('文件名', '左上角x坐标', '左上角y坐标', '右下角x坐标', '右下角y坐标', '虫子名称')]
loc['配置文件'] = loc['文件名'].apply(lambda x: x.replace(".jpg", ".txt"))
loc = loc[['文件名', '左上角x坐标', '左上角y坐标', '右下角x坐标', '右下角y坐标', '虫子名称', '配置文件']]
size = [5472, 3648]
bbox = loc.iloc[:, 1:5]

x, y, w, h = convert(size, bbox)
dn = loc.loc[:, ["文件名", "配置文件", "虫子名称"]]
dn['x'] = x
dn['y'] = y
dn['w'] = w
dn['h'] = h
dn = dn.round(6)
dn['虫子名称'] = dn['虫子名称'].astype(str)
gen_darknet("dn")
print(f"生成DarkNet数据集完成，共{len(os.listdir('dn'))}个文件。")

# 划分数据集
allData = pd.DataFrame(os.listdir("dn"), columns=['txt'])
allData['img'] = allData.loc[:, 'txt'].apply(lambda x: x.replace("txt", "jpg"))
radio = 0.8
p = int(allData.shape[0] * radio + 1)
train = allData.iloc[0:p, :]
val = allData.iloc[p:, :]

print(f"数据集划分完成，训练集包含{train.shape[0]}张，测试集包含{val.shape[0]}张。")
print("开始复制测试集")
T1 = time.time()
for t, j in train.values:
    from_path_t = os.path.join("dn", t)
    from_path_j = os.path.join("附件1", j)
    to_path_t = "dataset/labels/train"
    to_path_j = "dataset/images/train"
    copy(from_path_t, to_path_t)
    copy(from_path_j, to_path_j)
T2 = time.time()
# print(f'程序运行时间:%s毫秒' % ((T2 - T1)*1000))
print(f"测试集复制完成，共耗时{(T2 - T1) * 1000}毫秒，复制了{len(os.listdir('dataset/labels/train'))}张。")

print("开始复制测试集")
T3 = time.time()
for x, j in val.values:
    from_path_t = os.path.join("dn", x)
    from_path_j = os.path.join("附件1", j)
    to_path_t = "dataset/labels/val"
    to_path_j = "dataset/images/val"
    copy(from_path_t, to_path_t)
    copy(from_path_j, to_path_j)
T4 = time.time()
print(f"测试集复制完成，共耗时{(T4 - T3) * 1000}毫秒，复制了{len(os.listdir('dataset/labels/val'))}张。")

move("dataset", projectPath)
print("复制到项目根目录")

os.chdir(projectPath)
print(f"指定工作目录为{projectPath}")

kinds = ['二化螟', '二点委夜蛾', '棉铃虫', '褐飞虱属', '白背飞虱', '八点灰灯蛾', '蝼蛄', '蟋蟀', '甜菜夜蛾', '黄足猎蝽', '稻纵卷叶螟',
         '甜菜白带野螟', '黄毒蛾', '石蛾', '大黑鳃金龟', '粘虫', '稻螟蛉', '甘蓝夜蛾', '地老 虎', '大螟', '瓜绢野螟', '线委夜蛾',
         '水螟蛾', '紫条尺蛾', '歧角螟', '草地螟', '豆野螟', '干纹冬夜蛾']

with open("my_dataset.yaml", 'w', encoding="utf-8") as f:
    f.write("path: dataset\n")
    f.write("train: \n")
    f.write("  - images/train\n")
    f.write("  - labels/train\n")
    f.write("val: \n")
    f.write("  - images/val\n")
    f.write("  - labels/val\n")
    f.write("nc: 28 \n")
    f.write("names: ")
    f.write(str(kinds))

print(f"编写数据集配置文件，位置为 {os.path.join(projectPath,'my_dataset.yaml')}")
