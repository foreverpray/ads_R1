from FlagEmbedding import BGEM3FlagModel
import h5py
import numpy as np

model = BGEM3FlagModel('/nvme/yuanzhenzhao/embedding_model/bge_m3',  
                       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

with h5py.File('/nvme/yuanzhenzhao/reward_score_setting/xiaohongshu_vectors.h5', 'r') as f:
    loaded_vectors = f['vectors'][:]
    print(loaded_vectors.shape) 

# sentences_style = ["小红书文案撰写策略\n- 使用吸引人的标题\n- 描述商品具体的成果和效果，强调标题中的关键词，使其更具吸引力\n- 使用惊叹号、省略号等标点符号增强表达力，营造紧迫感和惊喜感\n- 使用emoji表情符号，来增加文字的活力 \n- 采用具有挑战性和悬念的表述，引发读者好奇心 \n- 蹭热点，蹭高热度话题，使用爆款词，高热度名词 \n- 尽量使用缩略词、习语、过渡短语、感叹词、修饰语和常用语，避免重复短语和不自然的句子结构\n- 需要吸引用户的眼球，从而增加营销的成功率 \n- 以口语化的表达方式，来拉近与读者的距离，但不要直接使用第一人称“我”"]
# sentences_style = ["抖音文案特点：\n短小精悍：由于时长限制，文案需要简洁明了，迅速引起观众的兴趣；\n节奏紧凑：要尽可能地精炼和浓缩内容，避免冗长和复杂的句子；\n语言生动有力：为了吸引观众的注意力，通常使用生动有力、富有感染力的语言，包括网络热梗或流行语等；\n情感共鸣：文案能够引发观众的情感共鸣，使观众产生强烈的认同感和情感联系；\n创意独特：无论是情节设计、角色设定还是语言表达，都需要有新颖独特的元素；\n视觉优先：抖音短视频文案更注重视觉效果和表现力。文案需要与视频内容相配合，通过视觉元素来传达信息和情感；\n引导互动：抖音短视频文案有时会包含引导观众互动的元素，如提问、挑战、话题标签等。这些元素可以激发观众的参与热情，提高视频的互动性和分享率；"]
# sentences_style = ["微博文案特点：\n短小精悍：由于微博的字数限制，微博文案通常短小精悍，言简意赅；\n主题明确：要保持与品牌形象一致的风格和调性，强调产品或服务的独特卖点，提供有价值的信息；\n即时性强：微博文案需要紧跟时事热点和社会话题，及时传递最新的信息和动态；\n个性化表达：微博文案的写作风格和表达方式各具特色，个性化十足；\n互动性强：可以在文案中鼓励读者发表自己的观点或分享自己的经验，以增加互动性"]

sentences_list = []


embeddings_style = model.encode(sentences_style, 
                            batch_size=64, 
                            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']

similarity = loaded_vectors @ embeddings_style.T

print(similarity)

print(similarity.mean())

print(similarity.shape)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# correlation_vector = np.random.randn(1424) * 0.3  # 模拟相关度数据（含正负值）

correlation_vector = similarity

# 创建可视化画布
fig = plt.figure(figsize=(18, 15))
grid = plt.GridSpec(2, 1, hspace=0.4, wspace=0.3)

# 1. 分布直方图
ax1 = fig.add_subplot(grid[0, 0])
sns.histplot(correlation_vector, bins=50, kde=True, ax=ax1, color='teal')
ax1.set_title('Correlation Values Distribution', fontsize=16)
ax1.set_xlabel('Correlation Strength')
ax1.set_ylabel('Frequency')
ax1.axvline(np.mean(correlation_vector), color='darkorange', linestyle='--', label='Mean')
ax1.legend()

# 2. 核密度概率图
ax2 = fig.add_subplot(grid[1, 0])
sns.kdeplot(correlation_vector, ax=ax2, color='teal')
ax2.set_title('Correlation Values Distribution', fontsize=16)
ax2.set_xlabel('Correlation Strength')
ax2.set_ylabel('Density')  # 核密度图的 y 轴通常表示密度
ax2.axvline(np.mean(correlation_vector), color='darkorange', linestyle='--', label='Mean')
ax2.legend()

# plt.tight_layout()
# plt.show()
plt.savefig('/nvme/yuanzhenzhao/reward_score_setting/style_match/xhs_wb_distribution.png', dpi=300, bbox_inches='tight')