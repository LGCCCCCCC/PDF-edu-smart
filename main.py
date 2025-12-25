# main.py
import streamlit as st
import os
from datetime import datetime
import torch

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import graphviz

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# ==================== 页面基本配置 ====================
st.set_page_config(
    page_title="PDF学习/教学神器 - 本地版",
    page_icon="📚",
    layout="wide"
)

# ==================== 初始化核心组件（使用缓存） ====================
@st.cache_resource
def get_embeddings():
    """使用本地中文 embedding 模型（推荐 bge-small-zh-v1.5）"""
    try:
        return HuggingFaceEmbeddings(
            model_name="./bge-small-zh-v1___5", 
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        st.error(f"Embedding 模型加载失败：{str(e)}")
        st.stop()

@st.cache_resource
def get_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    # 关键修复：检查 API Key 是否有效（避免中文占位符）
    if not api_key or api_key.strip() == "" or "你的API密钥" in api_key or "API" in api_key and len(api_key) < 10:
        st.error(
            "🚫 **DeepSeek API Key 未正确配置！**\n\n"
            "请按以下步骤设置：\n"
            "1. 前往 https://platform.deepseek.com/api_keys 获取你的 API Key\n"
            "2. 在终端或系统环境变量中设置：\n"
            "   ```bash\n"
            "   export DEEPSEEK_API_KEY='sk-your-real-key-here'\n"
            "   ```\n"
            "3. 或者在项目根目录创建 `.env` 文件，写入：\n"
            "   ```\n"
            "   DEEPSEEK_API_KEY=sk-your-real-key-here\n"
            "   ```\n\n"
            "注意：API Key 必须是纯英文+数字，不能包含中文！"
        )
        st.stop()
    
    return ChatDeepSeek(
        model="deepseek-chat",
        api_key=api_key,
        temperature=0.1
    )

embeddings = get_embeddings()
llm = get_llm()

# ==================== 提示词定义（常量字符串，模仿 trip_planner_agent.py） ====================
GENERAL_QA_PROMPT = """你是一个专业的文档问答助手。请严格基于以下上下文回答问题，
如果信息不足或无法回答，请直接说“知识库中暂无相关信息”。

上下文：
{context}

问题：
{question}

回答："""

STUDENT_EXPLAIN_PROMPT = """你是一个耐心的学习讲解助手。请基于以下知识点，为学生生成通俗易懂、结构清晰的学习讲解材料。
包括：核心概念解释、关键点总结、示例说明、常见误区等。

知识点内容：
{knowledge_points}

请生成详细的学习讲解："""

TEACHER_GUIDE_PROMPT = """你是一个专业的教学设计专家。请基于以下知识点，为老师生成完整的教学设计方案。
包括：教学目标、教学重难点、教学过程（导入、新课讲授、巩固练习、总结作业）、板书设计等。

知识点内容：
{knowledge_points}

请生成详细的教学设计："""

SUMMARY_PROMPT_GENERATOR = """请根据以下知识点内容，生成一个用于提取内容结构和知识层次的总结提示词。
该提示词应引导LLM输出清晰的层级式总结（例如：一级标题、二级标题、关键知识点）。

知识点内容：
{knowledge_points}

请严格遵守以下输出规范：
• 不要在总结中使用任何花括号 {{ }} 进行编号、强调或任何其他用途
• 推荐使用 - 或 * 作为列表符号，或使用 1. 2. 3. 的纯数字编号
• 不要模仿或抄袭下面的任何格式要求中的花括号内容
• 输出纯文本总结，不要包含任何代码块标记

请直接输出一个完整的提示词（无需额外说明）："""

STRUCTURE_GRAPH_PROMPT = """你是一个知识结构图专家。请基于以下内容总结，生成Graphviz DOT格式的知识结构图代码。
要求：
- 使用digraph
- 节点使用椭圆形，填充浅蓝色
- 箭头表示层级关系（从上级指向下级）
- 只包含主要知识点和层级关系，避免过多细节
- 输出纯DOT代码，无需额外文字
- **只输出** DOT 代码本身，**不要包含**任何 markdown 标记（如 ```dot
- 第一行必须直接是：digraph G {{
- 最后一行必须是单独的 }}
- 节点名称如果包含中文或空格，必须用双引号 "包裹"
- 使用 digraph，不要用 graph
- 只包含主要知识点和层级关系

内容总结：
{summary}

直接开始输出 DOT 代码，不要有任何前导或后缀文字："""

# ==================== 会话状态管理 ====================
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'all_documents' not in st.session_state:
    st.session_state.all_documents = []

# ==================== 侧边栏 - 上传区 ====================
with st.sidebar:
    st.success("PDF学习/教学神器")
    st.markdown("### 上传 PDF 文件")
    
    uploaded_file = st.file_uploader(
        "拖拽或选择 PDF 文件",
        type=["pdf"],
        accept_multiple_files=False,
        help="文件会自动保存到本地并加入知识库"
    )

    if uploaded_file is not None:
        os.makedirs("uploaded_pdfs", exist_ok=True)
        save_path = os.path.join("uploaded_pdfs", uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        with open("upload_records.txt", "a", encoding="utf-8") as log:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log.write(f"{timestamp} | {uploaded_file.name} | {uploaded_file.size/1024/1024:.2f}MB\n")

        with st.spinner(f"正在处理 {uploaded_file.name}..."):
            try:
                loader = PyPDFLoader(save_path)
                docs = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=150,
                    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
                )
                chunks = text_splitter.split_documents(docs)

                if st.session_state.vectorstore is None:
                    st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
                else:
                    st.session_state.vectorstore.add_documents(chunks)

                st.session_state.all_documents.extend(chunks)

                st.success(f"添加成功！新增 {len(chunks)} 个文本块")

            except Exception as e:
                st.error(f"文件处理失败：{str(e)}")

    st.markdown("---")
    st.caption("最近上传记录")
    if os.path.exists("upload_records.txt"):
        with open("upload_records.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-5:]:
                st.caption(line.strip())

# ==================== 主界面 ====================
role = st.sidebar.selectbox("选择使用角色", ["学生", "老师", "通用问答"], index=0)
st.title(f"{role}模式 - PDF智能学习助手")

if st.session_state.all_documents:
    if st.button("查看当前知识库概览（前5个片段）"):
        st.markdown("### 当前知识库部分内容预览")
        for i, doc in enumerate(st.session_state.all_documents[:5], 1):
            st.markdown(f"**片段 {i}**  · 来源：{doc.metadata.get('source', '未知')}")
            st.text(doc.page_content[:300] + "...")
            st.markdown("---")
else:
    st.info("请在侧边栏上传 PDF 文件以建立知识库")

# ==================== 功能区 ====================
if role == "通用问答":
    st.subheader("向知识库提问")
    question = st.text_area("请输入你的问题：", height=120, placeholder="例如：什么是Transformer的核心思想？")

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        ask_btn = st.button("提问", type="primary", use_container_width=True)

    if ask_btn and question.strip():
        if st.session_state.vectorstore is None:
            st.warning("知识库为空，请先上传 PDF 文件")
        else:
            with st.spinner("检索中..."):
                retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

                prompt = ChatPromptTemplate.from_template(GENERAL_QA_PROMPT)

                chain = (
                    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                     "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                with st.spinner("正在生成回答..."):
                    answer = chain.invoke(question)

                st.markdown("### 回答：")
                st.markdown(answer)

elif role in ["学生", "老师"]:
    st.subheader(f"{role}专用功能区")
    col1, col2 = st.columns(2)

    with col1:
        if st.button(f"功能1：生成{'学习讲解' if role=='学生' else '教学设计'}"):
            if not st.session_state.all_documents:
                st.warning("请先上传 PDF 文件")
            else:
                context = "\n\n".join(d.page_content for d in st.session_state.all_documents[:15])
                prompt_template = STUDENT_EXPLAIN_PROMPT if role == "学生" else TEACHER_GUIDE_PROMPT
                formatted_prompt = prompt_template.format(knowledge_points=context[:8000])

                with st.spinner("正在生成..."):
                    # 使用 HumanMessage 方式调用（更稳定）
                    response = llm.invoke([HumanMessage(content=formatted_prompt)])
                    st.markdown("### 生成结果")
                    st.markdown(response.content)

    with col2:
        if st.button("功能2：生成知识结构图"):
            if not st.session_state.all_documents:
                st.warning("请先上传 PDF 文件")
            else:
                context = "\n\n".join(d.page_content for d in st.session_state.all_documents[:10])
                
                with st.spinner("步骤1/3 生成总结提示..."):
                    gen_p = SUMMARY_PROMPT_GENERATOR.format(knowledge_points=context[:6000])
                    summary_prompt_resp = llm.invoke([HumanMessage(content=gen_p)])
                    summary_prompt = summary_prompt_resp.content

                with st.spinner("步骤2/3 生成内容结构总结..."):
                    summary_resp = llm.invoke([HumanMessage(content=summary_prompt)])
                    summary = summary_resp.content

                with st.spinner("步骤3/3 生成图结构..."):
                    graph_p = STRUCTURE_GRAPH_PROMPT.format_map({"summary": summary})
                    dot_resp = llm.invoke([HumanMessage(content=graph_p)])
                    dot_code = dot_resp.content.strip()

                st.text_area("Graphviz DOT 代码（可复制）", dot_code, height=140)
                st.markdown("### 知识结构图渲染结果")
                try:
                    graph = graphviz.Source(dot_code)
                    
                    # 1. 仍然生成 PNG 用于页面预览（可选，如果你还想保留）
                    img_png = graph.pipe(format='png')
                    st.image(img_png, caption="自动生成的知识结构图（PNG预览）", width='stretch')
                    
                    # 2. 生成 SVG 版本用于下载
                    svg_data = graph.pipe(format='svg').decode('utf-8')   # 转成字符串
                    
                    # 3. 添加下载按钮
                    st.download_button(
                        label="⬇️ 下载矢量 SVG 文件（推荐，任意放大不模糊）",
                        data=svg_data,
                        file_name="知识结构图.svg",
                        mime="image/svg+xml",
                        help="SVG 是矢量格式，适合打印、PPT插入、进一步编辑"
                    )

                except Exception as e:
                    st.error(f"渲染失败：{e}")
                    st.info("备用方案：把下面的 DOT 代码复制到 https://dreampuf.github.io/GraphvizOnline/ 查看")