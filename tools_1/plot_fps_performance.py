import matplotlib.pyplot as plt

# 1. 准备数据
data = {
    "SuperOcc-Series": {
        "x": [30.3, 28.2, 18.8, 12.7],
        "y": [42.5, 43.0, 43.6, 44.0],
        "labels": ["SuperOcc-T", "SuperOcc-S", "SuperOcc-M", "SuperOcc-L"],
        "color": "#5B9BD5", "marker": "s", "linestyle": "-" # PPT 主题蓝 (Accent 1)
    },
    "ALOcc-Series": {
        "x": [30.5, 8.2, 6.0],
        "y": [39.3, 43.0, 43.7],
        "labels": ["ALOcc-2D-mini", "ALOcc-2D", "ALOcc-3D"],
        "color": "#70AD47", "marker": "h", "linestyle": "-"
    },
    "OPUS-Series": {
        "x": [25.1, 24.5, 17.8, 8.6],
        "y": [38.4, 39.1, 40.3, 41.2],
        "labels": ["OPUS-T", "OPUS-S", "OPUS-M", "OPUS-L"],
        "color": "#FFC000", "marker": "X", "linestyle": "-" # PPT 橙色 (Accent 2)
    },
    "P—FlashOcc-Series": {
        "x": [41.9, 33.8, 29.8, 26.8],
        "y": [34.8, 35.2, 36.8, 38.5],
        "labels": ["P-FlashOcc-T(1f)", "P-FlashOcc(1f)", "P-FlashOcc(2f)", "P-FlashOcc(8f)"],
        "color": "#ED7D31", "marker": "p", "linestyle": "-" # PPT 灰色 (Accent 3)
    },
    "SparseOcc-Series": {
        "x": [20.9, 15.9],
        "y": [34.0, 35.1],
        "labels": ["SparseOcc(8f)", "SparseOcc(16f)"],
        "color": "#7030A0", "marker": "*", "linestyle": "-" # PPT 金黄 (Accent 4)
    },
    "BEVDetOcc-Series": {
        "x": [9.0, 5.6],
        "y": [29.6, 32.6],
        "labels": ["BEVDetOcc(2f)", "BEVDetOcc(8f)"],
        "color": "#FF33CC", "marker": "D", "linestyle": "-" # PPT 浅蓝 (Accent 5)
    },
    "Others": [
        {"name": "BEVFormer", "x": 4.4, "y": 32.4, "color": "#A5A5A5", "marker": "o"}, # PPT 绿色 (Accent 6)
        {"name": "FB-Occ", "x": 10.3, "y": 39.0, "color": "#264478", "marker": "^"},  # PPT 深蓝
    ]
}

plt.figure(figsize=(14, 10))
ax = plt.gca()

# 设置全局文字颜色为黑色
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

FONT_SIZE_MODELS = 22

# 2. 依次绘制各个系列
linewidth = 2.0
series = data["SuperOcc-Series"]
plt.plot(series["x"], series["y"], color=series["color"], marker=series["marker"],
         markersize=12, linewidth=linewidth, alpha=1.0, label="SuperOcc-Series")
xytexts = [(0, -25), (0, 15), (0, -25), (0, 15)]
for i, txt in enumerate(series["labels"]):
    plt.annotate(txt, (series["x"][i], series["y"][i]), xytext=xytexts[i],
                 textcoords='offset points', fontweight='semibold', ha='center',
                 color='black', fontsize=FONT_SIZE_MODELS)

series = data["ALOcc-Series"]
plt.plot(series["x"], series["y"], color=series["color"], marker=series["marker"],
         markersize=12, linewidth=linewidth, alpha=1.0, label="SuperOcc-Series")
xytexts = [(0, 25), (-50, -25), (-15, 15)]
for i, txt in enumerate(series["labels"]):
    plt.annotate(txt, (series["x"][i], series["y"][i]), xytext=xytexts[i],
                 textcoords='offset points', fontweight='medium', ha='center',
                 color='black', fontsize=FONT_SIZE_MODELS)

series = data["OPUS-Series"]
plt.plot(series["x"], series["y"], color=series["color"], marker=series["marker"],
         markersize=12, linewidth=linewidth, alpha=0.9, label="OPUS-Series")
xytexts = [(-5, -30), (-45, -20), (-5, -30), (0, -30)]
for i, txt in enumerate(series["labels"]):
    plt.annotate(txt, (series["x"][i], series["y"][i]), xytext=xytexts[i],
                 textcoords='offset points', fontweight='medium', ha='center', color='black',
                 fontsize=FONT_SIZE_MODELS)

series = data["P—FlashOcc-Series"]
plt.plot(series["x"], series["y"], color=series["color"], marker=series["marker"],
         markersize=12, linewidth=linewidth, alpha=0.9, label="P—FlashOcc-Series")
xytexts = [(-25, -30), (80, 10), (90, -5), (90, -5)]
for i, txt in enumerate(series["labels"]):
    plt.annotate(txt, (series["x"][i], series["y"][i]), xytext=xytexts[i],
                 textcoords='offset points', fontweight='medium', ha='center', color='black',
                 fontsize=FONT_SIZE_MODELS)

series = data["SparseOcc-Series"]
plt.plot(series["x"], series["y"], color=series["color"], marker=series["marker"],
         markersize=12, linewidth=linewidth, alpha=0.9, label="SparseOcc-Series")
xytexts = [(0, -30), (0, 15)]
for i, txt in enumerate(series["labels"]):
    plt.annotate(txt, (series["x"][i], series["y"][i]), xytext=xytexts[i],
                 textcoords='offset points', fontweight='medium', ha='center', color='black',
                 fontsize=FONT_SIZE_MODELS)

series = data["BEVDetOcc-Series"]
plt.plot(series["x"], series["y"], color=series["color"], marker=series["marker"],
         markersize=12, linewidth=linewidth, alpha=0.9, label="BEVDetOcc-Series")
xytexts = [(0, -30), (0, 15)]
for i, txt in enumerate(series["labels"]):
    plt.annotate(txt, (series["x"][i], series["y"][i]), xytext=xytexts[i],
                 textcoords='offset points', fontweight='medium', ha='center', color='black',
                 fontsize=FONT_SIZE_MODELS)

# 3. 绘制独立模型
for model in data["Others"]:
    tx = 0
    ty = -0.5
    if model["name"] == 'BEVFormer':
        tx = 0.8
        ty = 0.5
    if model["name"] == 'FB-Occ':
        tx = 0
        ty = 0.5
    plt.scatter(model["x"], model["y"], color=model["color"], marker=model["marker"], s=120)
    plt.text(model["x"]-tx, model["y"]-ty, model["name"], fontsize=FONT_SIZE_MODELS, ha='center', va='top', color='black')

# 4. 图表细节美化
plt.xlabel('RTX 4090 FP32 Throughput (FPS)', fontsize=22, labelpad=10, color='black')
plt.ylabel('RayIoU (%)', fontsize=22, labelpad=10, color='black')

plt.xlim(0, 45)
plt.ylim(28, 46)

plt.xticks(range(0, 46, 5), fontsize=20, fontweight='medium')
plt.yticks(range(28, 47, 2), fontsize=20, fontweight='medium')

plt.grid(True, which='both', linestyle='-', linewidth=2.0, alpha=0.4)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_linewidth(2.0)
    # spine.set_edgecolor('black')
    spine.set_alpha(0.4)

# plt.title("RayIoU vs. FPS on Occ3D Benchmark",
#           fontsize=24,
#           fontweight='bold',
#           pad=20) # pad 增加标题与图表的间距

# plt.legend(loc='lower right', fontsize=10, labelcolor='black')
plt.tight_layout()
plt.savefig('rayiou.pdf')
print("!!")
plt.show()















# import matplotlib.pyplot as plt
#
# # 1. 准备数据
# data = {
#     "SuperOcc-Series": {
#         # "x": [25.0, 24.5, 15.1, 7.4],
#         # "y": [36.0, 36.8, 37.6, 38.4],
#         # "labels": ["SuperOcc-T", "SuperOcc-S", "SuperOcc-M", "SuperOcc-L"],
#         # "color": "#5B9BD5", "marker": "s", "linestyle": "-" # PPT 主题蓝 (Accent 1)
#         "x": [30.5, 25.7],
#         "y": [36.5, 37.5],
#         "labels": ["SuperOcc-T", "SuperOcc-S"],
#         "color": "#5B9BD5", "marker": "s", "linestyle": "-"  # PPT 主题蓝 (Accent 1)
#     },
#     "OPUS-Series": {
#         "x": [25.1, 24.5, 17.8, 8.6],
#         "y": [33.2, 34.2, 35.6, 36.2],
#         "labels": ["OPUS-T", "OPUS-S", "OPUS-M", "OPUS-L"],
#         "color": "#70AD47", "marker": "X", "linestyle": "-" # PPT 橙色 (Accent 2)
#     },
#     "P—FlashOcc-Series": {
#         "x": [41.9, 33.8, 29.8, 26.8],
#         "y": [29.1, 29.4, 30.3, 31.6],
#         "labels": ["P-FlashOcc-T(1f)", "P-FlashOcc(1f)", "P-FlashOcc(2f)", "P-FlashOcc(8f)"],
#         "color": "#ED7D31", "marker": "p", "linestyle": "-" # PPT 灰色 (Accent 3)
#     },
#     "SparseOcc-Series": {
#         "x": [20.9, 15.9],
#         "y": [34.0, 35.1],
#         "labels": ["SparseOcc(8f)", "SparseOcc(16f)"],
#         "color": "#FFC000", "marker": "*", "linestyle": "-" # PPT 金黄 (Accent 4)
#     },
#     "BEVDetOcc-Series": {
#         "x": [9.0, 5.6],
#         "y": [36.1, 39.3],
#         "labels": ["BEVDetOcc(2f)", "BEVDetOcc(8f)"],
#         "color": "#FF33CC", "marker": "D", "linestyle": "-" # PPT 浅蓝 (Accent 5)
#     },
#     "Others": [
#         {"name": "BEVFormer", "x": 4.5, "y": 39.3, "color": "#A5A5A5", "marker": "o"}, # PPT 绿色 (Accent 6)
#         {"name": "FB-Occ", "x": 10.3, "y": 30.3, "color": "#264478", "marker": "^"},  # PPT 深蓝
#     ]
# }
#
# plt.figure(figsize=(14, 10))
# ax = plt.gca()
#
# # 设置全局文字颜色为黑色
# plt.rcParams['text.color'] = 'black'
# plt.rcParams['axes.labelcolor'] = 'black'
# plt.rcParams['xtick.color'] = 'black'
# plt.rcParams['ytick.color'] = 'black'
#
# FONT_SIZE_MODELS = 16
#
# # 2. 依次绘制各个系列
# series = data["SuperOcc-Series"]
# plt.plot(series["x"], series["y"], color=series["color"], marker=series["marker"],
#          markersize=12, linewidth=1.5, alpha=1.0, label="SuperOcc-Series")
# xytexts = [(0, -25), (5, 15), (15, 15), (50, 10)]
# for i, txt in enumerate(series["labels"]):
#     plt.annotate(txt, (series["x"][i], series["y"][i]), xytext=xytexts[i],
#                  textcoords='offset points', fontweight=550, ha='center',
#                  color='black', fontsize=FONT_SIZE_MODELS)
#
# series = data["OPUS-Series"]
# plt.plot(series["x"], series["y"], color=series["color"], marker=series["marker"],
#          markersize=12, linewidth=1.5, alpha=0.9, label="OPUS-Series")
# xytexts = [(0, -25), (10, 15), (0, 15), (35, 10)]
# for i, txt in enumerate(series["labels"]):
#     plt.annotate(txt, (series["x"][i], series["y"][i]), xytext=xytexts[i],
#                  textcoords='offset points', ha='center', color='black',
#                  fontsize=FONT_SIZE_MODELS)
#
# series = data["P—FlashOcc-Series"]
# plt.plot(series["x"], series["y"], color=series["color"], marker=series["marker"],
#          markersize=12, linewidth=1.5, alpha=0.9, label="P—FlashOcc-Series")
# xytexts = [(-20, -30), (-10, -25), (80, -5), (80, -5)]
# for i, txt in enumerate(series["labels"]):
#     plt.annotate(txt, (series["x"][i], series["y"][i]), xytext=xytexts[i],
#                  textcoords='offset points', ha='center', color='black',
#                  fontsize=FONT_SIZE_MODELS)
#
# series = data["SparseOcc-Series"]
# plt.plot(series["x"], series["y"], color=series["color"], marker=series["marker"],
#          markersize=12, linewidth=1.5, alpha=0.9, label="SparseOcc-Series")
# xytexts = [(0, -30), (-50, -20)]
# for i, txt in enumerate(series["labels"]):
#     plt.annotate(txt, (series["x"][i], series["y"][i]), xytext=xytexts[i],
#                  textcoords='offset points', ha='center', color='black',
#                  fontsize=FONT_SIZE_MODELS)
#
# series = data["BEVDetOcc-Series"]
# plt.plot(series["x"], series["y"], color=series["color"], marker=series["marker"],
#          markersize=12, linewidth=1.5, alpha=0.9, label="BEVDetOcc-Series")
# xytexts = [(0, -30), (0, 15)]
# for i, txt in enumerate(series["labels"]):
#     plt.annotate(txt, (series["x"][i], series["y"][i]), xytext=xytexts[i],
#                  textcoords='offset points', ha='center', color='black',
#                  fontsize=FONT_SIZE_MODELS)
#
# # 3. 绘制独立模型
# for model in data["Others"]:
#     tx = 0
#     ty = -0.5
#     if model["name"] == 'BEVFormer':
#         tx = 1.3
#         ty = 0.4
#     if model["name"] == 'FB-Occ':
#         tx = 0
#         ty = 0.4
#     plt.scatter(model["x"], model["y"], color=model["color"], marker=model["marker"], s=120)
#     plt.text(model["x"]-tx, model["y"]-ty, model["name"], fontsize=FONT_SIZE_MODELS, ha='center', va='top', color='black')
#
# # 4. 图表细节美化
# plt.xlabel('RTX 4090 FP32 Throughput (FPS)', fontsize=14, labelpad=10, color='black')
# plt.ylabel('Occ3D (mIoU)', fontsize=14, labelpad=10, color='black')
#
# plt.xlim(0, 45)
# plt.ylim(28, 40)
#
# plt.xticks(range(0, 46, 5))
# plt.yticks(range(28, 41, 2))
#
# plt.grid(True, which='both', linestyle='-', linewidth=0.8, alpha=0.8)
# ax.set_axisbelow(True)
#
# for spine in ax.spines.values():
#     spine.set_linewidth(1.5)
#     spine.set_edgecolor('black')
#
# # plt.legend(loc='lower right', fontsize=10, labelcolor='black')
# plt.tight_layout()
# plt.savefig('mious-fps.pdf')
# print("!!")
# plt.show()



















import matplotlib.pyplot as plt

# 1. 准备数据
data = {
    "SuperOcc-Series": {
        "x": [30.3, 28.2, 18.8, 12.7],
        "y": [22.48, 23.15, 23.95, 24.55],
        "labels": ["SuperOcc-T", "SuperOcc-S", "SuperOcc-M", "SuperOcc-L"],
        "color": "#5B9BD5", "marker": "s", "linestyle": "-"
    },
    "Others": [
        # 换成了 PPT 经典的粉紫色和紫色，并调整了重复的 marker
        {"name": "GaussianWorld", "x": 4.4, "y": 22.13, "color": "#70AD47", "marker": "o"},  # 绿色 - 圆形
        {"name": "QuadricFormer", "x": 4.0, "y": 20.12, "color": "#ED7D31", "marker": "*"},  # 橙色 - 星号
        {"name": "GaussianFormer-2", "x": 2.8, "y": 20.02, "color": "#FFC000", "marker": "D"},  # 金黄 - 菱形
        {"name": "GaussianFormer", "x": 2.7, "y": 19.10, "color": "#7030A0", "marker": "^"},  # 紫色 - 上三角 (替换了深蓝)
        {"name": "SurroundOcc", "x": 3.3, "y": 20.30, "color": "#FF33CC", "marker": "p"},  # 粉色 - 五边形 (替换了浅蓝)
        {"name": "TPVFormer*", "x": 2.9, "y": 17.10, "color": "#002060", "marker": "X"},  # 深蓝 - 叉号 (保持区分度)
        {"name": "BEVFormer", "x": 3.3, "y": 16.75, "color": "#A5A5A5", "marker": "v"},  # 灰色 - 下三角 (从五边形改为下三角)
    ]
}

plt.figure(figsize=(14, 10))
ax = plt.gca()

# 设置全局文字颜色
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

FONT_SIZE_MODELS = 22

# 2. 绘制系列模型
series = data["SuperOcc-Series"]
plt.plot(series["x"], series["y"], color=series["color"], marker=series["marker"],
         markersize=12, linewidth=1.5, alpha=1.0, label="SuperOcc-Series")
# 这里的偏移保持你之前的设置
xytexts = [(0, -25), (5, 18), (15, 18), (50, 18)]
for i, txt in enumerate(series["labels"]):
    plt.annotate(txt, (series["x"][i], series["y"][i]), xytext=xytexts[i],
                 textcoords='offset points', fontweight='semibold', ha='center',
                 color='black', fontsize=FONT_SIZE_MODELS)

# 3. 绘制独立模型
for model in data["Others"]:
    tx = 0
    ty = 0.3
    # 针对文字重叠的特定偏移逻辑
    if model["name"] == 'TPVFormer*':
        tx = 0
        ty = -0.6
    if model["name"] == 'SurroundOcc':
        tx = 0
        ty = -0.5
    if model["name"] == 'QuadricFormer':
        tx = -3.5
        ty = -0.1
    if model["name"] == 'GaussianFormer':
        tx = -1.2
        ty = 0.3
    if model["name"] == 'GaussianFormer-2':
        tx = -1.2
        ty = 0.3
    plt.scatter(model["x"], model["y"], color=model["color"], marker=model["marker"], s=150)  # 稍微加大了点的大小
    plt.text(model["x"] - tx, model["y"] - ty, model["name"],
             fontsize=FONT_SIZE_MODELS, ha='center', va='top', color='black', fontweight='medium')

# 4. 细节美化
plt.xlabel('RTX 4090 FP32 Throughput (FPS)', fontsize=22, labelpad=10, color='black')
plt.ylabel('mIoU (%)', fontsize=22, labelpad=10, color='black')

plt.xlim(0, 35)
plt.ylim(16, 26)
plt.xticks(range(0, 36, 5), fontsize=20, fontweight='medium')
plt.yticks(range(16, 27, 2), fontsize=20, fontweight='medium')

plt.grid(True, which='both', linestyle='-', linewidth=2.0, alpha=0.4)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_linewidth(2.0)
    spine.set_alpha(0.4)

# plt.title("mIoU vs. FPS on SurroundOcc Benchmark",
#           fontsize=24,
#           fontweight='bold',
#           pad=20) # pad 增加标题与图表的间距

plt.tight_layout()
plt.savefig('nuscenes-occ.pdf')
plt.show()