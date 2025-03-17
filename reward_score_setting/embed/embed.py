from FlagEmbedding import BGEM3FlagModel
import h5py
import numpy as np

model = BGEM3FlagModel('/nvme/yuanzhenzhao/embedding_model/bge_m3',  
                       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
import json

sentences_list = []
with open("/nvme/yuanzhenzhao/reward_score_setting/sft_v3_marketing_gpt4.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

    for i in range(len(data)):
        if data[i]["task"] == "advertorial_douyin":
            sentences_list.append(data[i]["data"][1])
    
    print(sentences_list)

    
# sentences_style = ["小红书文案撰写策略\n- 使用吸引人的标题\n- 描述商品具体的成果和效果，强调标题中的关键词，使其更具吸引力\n- 使用惊叹号、省略号等标点符号增强表达力，营造紧迫感和惊喜感\n- 使用emoji表情符号，来增加文字的活力 \n- 采用具有挑战性和悬念的表述，引发读者好奇心 \n- 蹭热点，蹭高热度话题，使用爆款词，高热度名词 \n- 尽量使用缩略词、习语、过渡短语、感叹词、修饰语和常用语，避免重复短语和不自然的句子结构\n- 需要吸引用户的眼球，从而增加营销的成功率 \n- 以口语化的表达方式，来拉近与读者的距离，但不要直接使用第一人称“我”"]


embeddings_list = model.encode(sentences_list, 
                            batch_size=64, 
                            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']

# vectors = np.random.rand(1000, 1024)  # 1000个1024维向量

# 保存到HDF5文件
with h5py.File('/nvme/yuanzhenzhao/reward_score_setting/douyin_vectors.h5', 'w') as f:
    f.create_dataset('vectors', data=embeddings_list, compression='gzip')

# 加载数据
with h5py.File('/nvme/yuanzhenzhao/reward_score_setting/douyin_vectors.h5', 'r') as f:
    loaded_vectors = f['vectors'][:]
    print(loaded_vectors.shape) 