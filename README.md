# PDF-edu-smart
PDF学习/教学神器是一款基于RAG技术的本地AI助手，参考Datawhale经典RAG实践（FAISS + bge-small-zh-v1.5），创新性地调用DeepSeek API代替本地LLM，实现了PDF上传→智能问答→学生通俗讲解+知识结构图→老师完整教案的一站式功能，帮助学生快速精读论文、老师高效备课、研究者高效梳理知识，同时100%本地向量存储保障隐私安全，让学习与教学更智能、更轻松。
## 核心功能

- 📄 上传 PDF → 自动分块 + 本地 FAISS 向量化存储
- 🧑‍🎓 **学生模式**：生成通俗易懂的学习讲解（概念+示例+易错点）
- 👩‍🏫 **教师模式**：自动生成完整教学设计（目标+重难点+过程+板书）
- 💬 **通用问答**：基于上传文档的 RAG 精准问答
- 🗺️ **知识结构图**：自动生成层级关系图（Graphviz，支持下载 SVG 矢量图）

## 快速开始

### 1. 环境要求

- Python 3.10 ~ 3.12
- 推荐有 GPU（embedding 速度会快很多）

### 2. 安装依赖

```bash
git clone https://github.com/你的用户名/PDF-edu-smart.git

cd PDF-edu-smart
pip install -r requirements.txt
```

### 3. 配置 DeepSeek API Key

更新.env 文件，输入API密匙和选择的模型

### 4. 启动程序

```bash
streamlit run main.py
```
