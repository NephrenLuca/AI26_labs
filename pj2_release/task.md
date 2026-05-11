## **2 任务一：HMM实现命名实体识别（20分）**

### 2.1 任务内容

手写HMM模型，完成NER任务。

### 2.2 具体要求

- **实现限制**：手写HMM模型，**不能使用机器学习框架**。
- **运行要求**：代码应能在中文、英文两个数据集上完整运行，并生成符合评测脚本格式的预测结果。

------

## **3 任务二：CRF实现命名实体识别（20分）**

### 3.1 任务内容

使用CRF模型完成NER任务。

### 3.2 具体要求

- **框架限制**：可以使用机器学习框架，但**必须理解CRF完成NER的原理**，面试时会提问。
- **结果输出**：在中文、英文两个数据集上完成训练与验证，生成与样例一致的预测文件。

### 3.3 参考资料

- 《Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data》
- 《Discriminative Training Methods for Hidden Markov Models: Theory and Experiments with Perceptron Algorithms》
- 其他自行寻找的资料

------

## **4 任务三：Transformer+CRF实现命名实体识别（20分）**

### 4.1 任务内容

使用Transformer + CRF模型完成NER任务。

### 4.2 具体要求

- **Transformer部分**：可以使用PyTorch等深度学习框架。
- **CRF部分**：**必须手写完成**。
- **预训练模型**：有条件的同学可以使用BERT等预训练模型。

### 4.3 参考资料

- Transformer原文《Attention Is All You Need》
- 其他自行寻找的资料

------

## **5 数据、评测与面试说明**

### 5.1 数据格式

- 提供中文与英文两个数据集，每个包含：
  - `train.txt`
  - `validation.txt`
  - 标签说明文件（如`tag.txt`）
- 数据格式：
  - 每行包含一个token及其实体标签，中间以空格分隔
  - 句子之间以空行分隔
- 标签说明：
  - 中文数据集：33种tag，包含8类实体
  - 英文数据集：9种tag，包含4类实体

### 5.2 评测方式

- 以`check.py`输出的**micro avg F1-score**为准

- 安装依赖：

  text

  ```
  pip install scikit-learn -i https://pypi.mirrors.ustc.edu.cn/simple
  ```

  

- 预测结果文件格式：与`example_data/example_my_result.txt`一致，即每个非空行包含原token与预测标签，保持与测试文件相同的行数和空行位置。