import random
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from get_text import print_x_elements
from get_y import get_y

# 生成随机 Python 代码
# def generate_random_python_code():
#     code_lengths = [10, 20, 15, 20, 3]  # 定义不同长度的代码
#     code = ""
#     code_length = random.choice(code_lengths)  # 随机选择一个长度
#     for _ in range(code_length):
#         code += random.choice(["import numpy as np\n", "def func():\n", "for i in range(10):\n", "print('Hello, world!')\n"])
#     return code
#
# # 生成长度为3的独热向量
# def one_hot_vector(length):
#     index = random.randint(0, length - 1)
#     return [1 if i == index else 0 for i in range(length)]

# 划分训练集和测试集
num_i = 10
labels = get_y("downstream", num_i)
texts = print_x_elements("downstream", num_i)
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 初始化 Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("codebert_base")

# 模型定义
class CodeClassifier(nn.Module):
    def __init__(self):
        super(CodeClassifier, self).__init__()
        self.codebert = RobertaModel.from_pretrained("codebert_base")
        self.fc = nn.Linear(self.codebert.config.hidden_size, 3)  # 输出维度为3

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        last_token_output = last_hidden_state[:, -1, :]  # 获取最后一个 token 的输出
        output = self.fc(last_token_output)
        return output

# 实例化模型
model = CodeClassifier()

# 定义优化器和损失函数
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

# 训练模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

train_inputs = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
train_labels = torch.tensor(train_labels, dtype=torch.float32)
train_inputs = {key: val.to(device) for key, val in train_inputs.items()}
train_labels = train_labels.to(device)

epochs = 3

# 设置批次大小
batch_size = 1
predictions_list = []
labels_list = []

# 训练模型
model.train()
for epoch in range(epochs):
    # 对训练数据进行分批处理
    for i in range(0, len(train_texts), batch_size):
        optimizer.zero_grad()
        batch_texts = train_texts[i:i + batch_size]
        batch_labels = train_labels[i:i + batch_size]
        batch_inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        batch_labels = torch.tensor(batch_labels.clone().detach(), dtype=torch.float32).to(device)
        batch_inputs = {key: val.to(device) for key, val in batch_inputs.items()}

        outputs = model(**batch_inputs)
        # print(outputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# 评估模型
model.eval()
with torch.no_grad():
    # 对测试数据进行分批处理
    for i in range(0, len(test_texts), batch_size):
        batch_texts = test_texts[i:i + batch_size]
        batch_labels = test_labels[i:i + batch_size]
        batch_inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
        batch_inputs = {key: val.to(device) for key, val in batch_inputs.items()}

        outputs = model(**batch_inputs)
        predictions = torch.round(torch.sigmoid(outputs))
        predictions_list.append(predictions)
        labels_list.append(batch_labels)

# 拼接所有批次的预测值和标签
predictions = torch.cat(predictions_list)
labels = torch.cat(labels_list)
print(predictions)
print(labels)
point = torch.zeros(4,3).to(device)
F1s = torch.zeros(3)
for predictions, labels in zip(predictions, labels):
    point[0] += ((predictions == 1) & (labels == 1)).to(torch.int)  # TP
    point[1] += ((predictions == 0) & (labels == 1)).to(torch.int)  # FN
    point[2] += ((predictions == 1) & (labels == 0)).to(torch.int)  # FP
    point[3] += ((predictions == 0) & (labels == 0)).to(torch.int)  # TN

print(point[0], point[1], point[2], point[3])
for i in range(F1s.shape[0]):
    precision = point[0][i] / (point[0][i] + point[2][i]).clamp(min=1e-6)
    recall = point[0][i] / (point[0][i] + point[1][i]).clamp(min=1e-6)
    F1s[i] = 2 * (precision * recall) / (precision + recall + 1e-6)
precision = point[0][:].sum() / (point[0][:].sum() + point[2][:].sum()).clamp(min=1e-6)
recall = point[0][:].sum()  / (point[0][:].sum() + point[1][:].sum()).clamp(min=1e-6)
micro_F1 =2 * (precision * recall) / (precision + recall + 1e-6)
print(f"F1 score: {F1s},Micro-F1:{micro_F1},Macro-F1:{F1s.mean()},acc:{precision}")
# # 计算混淆矩阵
# conf_matrix = confusion_matrix(labels.cpu().numpy().argmax(axis=1), predictions.cpu().numpy().argmax(axis=1))
#
# # 计算 F1 分数
# f1 = f1_score(labels.cpu().numpy().argmax(axis=1), predictions.cpu().numpy().argmax(axis=1), average='macro')
#
# # 计算准确率
# accuracy = accuracy_score(labels.cpu().numpy().argmax(axis=1), predictions.cpu().numpy().argmax(axis=1))
#
# # 计算召回率
# recall = recall_score(labels.cpu().numpy().argmax(axis=1), predictions.cpu().numpy().argmax(axis=1), average='macro')
#
# print("Confusion Matrix:")
# print(conf_matrix)
# print("F1 Score:", f1)
# print("Accuracy:", accuracy)
# print("Recall:", recall)