import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

content_list = []
length_list = []
length_count = 0
# sp_list = []

with open("/nvme/yuanzhenzhao/reward_score_setting/text_match/xhs_top.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

    for i in range(len(data)):
        content = data[i]["content"]
        leng = len(content)
        if 300 > leng > 180:
            length_count =length_count + 1
            print(content)
        content_list.append(content)
        length_list.append(leng)

print(len(length_list))
print(length_count)
print(np.mean(length_list))
# print(sp_list)

# fig = plt.figure(figsize=(12, 8))
# ax2 = fig.add_subplot(1, 1, 1)
# sns.kdeplot(length_list, ax=ax2, color='teal')
# ax2.set_title('Correlation Values Distribution', fontsize=16)
# ax2.set_xlabel('Correlation Strength')
# ax2.set_ylabel('Density')  # 核密度图的 y 轴通常表示密度
# ax2.axvline(np.mean(length_list), color='darkorange', linestyle='--', label=f'Mean: {np.mean(length_list):.2f}')
# ax2.legend()

# plt.savefig('/nvme/yuanzhenzhao/reward_score_setting/text_match/xhs_top_length', dpi=300, bbox_inches='tight')
