import sys
sys.path.append('/home/cottonlove/baseline_code/stylebook')
# print(sys.path)
from utils import read_audio, load_hubert, extract_hubert_vq,apply_fading
import math

DEVICE = "cuda:1"
VOCAB_SIZE = 200 #100
FADE_LEN = 800

SR = 16000
SEG_LEN = 80000 #SR=16000 -> 5초 뽑는거

import torch



# def load_hubert(vocab_size=100, device='cpu'):
#     from textless.data.speech_encoder import SpeechEncoder

#     hubert_model \
#         = SpeechEncoder.by_name(dense_model_name="hubert-base-ls960",
#                                 quantizer_model_name="kmeans",
#                                 vocab_size=vocab_size,
#                                 deduplicate=False,
#                                 need_f0=False).to(device)
    
#     return hubert_model

# def extract_hubert_vq(sig, hubert_model=None, vocab_size=100, device='cpu'):
#     if hubert_model is None:
#         hubert_model = load_hubert(vocab_size=vocab_size, device=device)

#     sig = sig.to(device)
#     encoded = hubert_model(sig)

#     centroids \
#         = hubert_model._modules["quantizer_model"].kmeans_model.cluster_centers_

#     feature = torch.Tensor(centroids[encoded["units"].cpu()]).to(device)

#     return feature


# ## TODO

# Load HuBERT model
print("= Loading HuBERT...")
hubert_model = load_hubert(VOCAB_SIZE, DEVICE)

phone_aligment_text = "/home/cottonlove/LibriTTSCorpusLabel/lab/phone/dev-clean/84/121123/84_121123_000008_000000.lab"

time_phone_list = []
with open(phone_aligment_text, "r") as file:
    for line in file:
        values = line.split()
        time_phone_list.append(values)
# print(time_phone_list) # [['0.0', '0.13', 'V'], ['0.13', '0.19', 'IH1']...]

audio_file_path = "/home/cottonlove/baseline_code/DB/LibriTTS/dev-clean/84/121123/84_121123_000008_000000.wav"

sig = read_audio(audio_file_path, SR).to(DEVICE)
hubert_embeddings = []
phone_labels = []

for i in range(len(time_phone_list)-1):
    start_index = float(time_phone_list[i][0])
    end_index = float(time_phone_list[i][1])
    if len(time_phone_list[i]) == 2:
        phone_label = None
    else:
        phone_label = time_phone_list[i][2]
    seg = sig[math.ceil(start_index*SR):math.floor(end_index*SR)]
    #seg = apply_fading(seg, 160)
    huBERT_outputs = hubert_model(seg.to(DEVICE))
    # print(huBERT_outputs) # dictionary
    for j in range(huBERT_outputs['dense'].shape[0]):
        hubert_embeddings.append(huBERT_outputs['dense'][j][:].unsqueeze(0))
        phone_labels.append(phone_label)
    #print(phone_label, huBERT_outputs['dense'].shape)

concatenated_embeddings = torch.cat(hubert_embeddings, dim=0)
print(concatenated_embeddings.shape)

# print(hubert_embeddings[0].shape)
# import torch
# t = torch.ones(3, 768)
# print(t.shape)
# hubert_embeddings = []
# phone_labels = []
# phone_label = 1
# print(t.shape[0])
# for j in range(t.shape[0]):
#     hubert_embeddings.append(t[j][:].unsqueeze(0))
#     phone_labels.append(phone_label)
# print(hubert_embeddings[0].shape)
# print(hubert_embeddings[1].shape)
# print(hubert_embeddings[2].shape)
# print(phone_labels)
# concatenated_embeddings = torch.cat(hubert_embeddings, dim=0)
# print(concatenated_embeddings.shape)

## TODO- tSNE

from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
import seaborn as sns 
import colorcet as cc

# 2차원으로 차원 축소
n_components = 2

# t-sne 모델 생성
model = TSNE(n_components=n_components, perplexity= 10, n_iter = 2000)
phone_labels_set = list(set(phone_labels))

# 학습한 결과 2차원 공간 값 출력
X_embedded = model.fit_transform(concatenated_embeddings)
# print(model.fit_transform(data.data).shape)#

palette = sns.color_palette(cc.glasbey, len(phone_labels_set))
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=phone_labels, palette=palette)

# 범례를 원하는 위치에 그립니다.
plt.legend(title = "phone label",bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)  # 범례를 오른쪽 상단에 배치합니다.

plt.savefig('figures/plot_hubert_phonelabel.png', bbox_inches='tight')

# number = 3.7

# # 올림
# ceil_result = math.ceil(number)
# print("올림 결과:", ceil_result)  # 출력: 4
# print(type(ceil_result))
# # 내림
# floor_result = math.floor(number)
# print("내림 결과:", floor_result)  # 출력: 3



## TODO - hubert에 넣어줌

# filename = ""

# sig = read_audio(filename, SR).to(DEVICE)

# n_seg = len(sig) // SEG_LEN  # Discard the last frame
# # print("n_seg: ", n_seg)

# for n in range(n_seg):
# # Audio
# seg = sig[n * SEG_LEN:(n + 1) * SEG_LEN] #5초뽑음
# seg = apply_fading(seg, FADE_LEN) #fading해줌

# # HuBERT VQ
# hubert_vq = extract_hubert_vq(seg, hubert_model)
# print("hubert_vq.shape: ", hubert_vq.shape) #([249, 768])
