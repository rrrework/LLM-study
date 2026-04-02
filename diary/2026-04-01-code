import os
import time
from typing import List, Dict
import chromadb
from openai import OpenAI
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ====================== 配置 ======================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Chroma 持久化数据库
chroma_client = chromadb.PersistentClient(path="./chroma_db_papers")
collection = chroma_client.get_or_create_collection(
    name="research_papers_pdf",
    metadata={"hnsw:space": "cosine"}
)

# ====================== 1. PDF 读取 + 自动 chunk 切分 ======================
def extract_and_chunk_pdf(pdf_path: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[Dict]:
    """读取 PDF 并智能切分成 chunks"""
    doc = fitz.open(pdf_path)
    text = ""
    metadata_list = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text")  # 提取纯文本
        text += page_text + "\n\n"
        
        # 记录每页基本元数据
        metadata_list.append({
            "page": page_num + 1,
            "title": os.path.basename(pdf_path).replace(".pdf", ""),
            "source": pdf_path
        })

    doc.close()

    # 使用 RecursiveCharacterTextSplitter 进行智能切分（推荐用于 RAG）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,          # 每个 chunk 大概 800 字符（约 200-300 token）
        chunk_overlap=chunk_overlap,    # 重叠部分，保持上下文连贯
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],  # 优先按段落/句子切分
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_text(text)

    # 为每个 chunk 添加元数据
    chunk_docs = []
    for i, chunk in enumerate(chunks):
        chunk_docs.append({
            "content": chunk,
            "metadata": {
                "title": metadata_list[0]["title"],   # 简化处理，可按页更精细
                "source": pdf_path,
                "chunk_id": i,
                # "page": ... 可根据需要更精确映射页码
            }
        })

    print(f"从 {pdf_path} 提取并切分成 {len(chunks)} 个 chunks")
    return chunk_docs


# ====================== 2. 添加到 Chroma ======================
def add_pdfs_to_chroma(pdf_paths: List[str]):
    """批量添加多个 PDF 到向量数据库"""
    if collection.count() > 0:
        print(f"数据库中已有 {collection.count()} 条记录。如需重新添加，请删除 ./chroma_db_papers 文件夹")
        return

    all_chunks = []
    for pdf_path in pdf_paths:
        chunks = extract_and_chunk_pdf(pdf_path)
        all_chunks.extend(chunks)

    if not all_chunks:
        return

    documents = [c["content"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]

    # 计算 embedding 并添加
    embeddings = [get_embedding(doc) for doc in documents]

    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"成功添加 {len(all_chunks)} 个 chunks 到 Chroma 数据库！")


def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# ====================== 3. 检索 + 生成回答 ======================
def ask_with_rag(user_question: str, top_k: int = 4, temperature: float = 0.2):
    query_embedding = get_embedding(user_question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    # 构建上下文（带标题和来源）
    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(f"【{meta.get('title', '论文')} - Chunk {meta.get('chunk_id')}】\n{doc}")

    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": "你是一个专业的科研论文助手。请严格基于提供的论文片段回答问题，引用具体来源。如果信息不足，请明确说明。"
        },
        {
            "role": "user",
            "content": f"以下是检索到的相关论文片段：\n\n{context}\n\n用户问题：{user_question}"
        }
    ]

    print("🤖 AI 正在基于 PDF 论文回答（流式输出）：\n")
    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=2000,
            stream=True
        )
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        print("\n")
        return full_response
    except Exception as e:
        print(f"错误: {e}")
        return None


# ====================== 使用示例 ======================
if __name__ == "__main__":
    # 把你的 PDF 文件路径放这里（支持多个）
    pdf_files = [
        "papers/attention_is_all_you_need.pdf",   # 替换成你的实际 PDF 路径
        "papers/deepseek_r1.pdf",
        # 添加更多 PDF...
    ]

    # 第一次运行时添加 PDF（会自动读取 + 切分 + 存入数据库）
    add_pdfs_to_chroma(pdf_files)

    # 测试提问
    question = "Transformer 模型相比 RNN 的主要优势是什么？请引用相关论文内容。"
    ask_with_rag(question)

    # 多轮提问（取消注释即可使用）
    # while True:
    #     q = input("\n请输入问题（输入 exit 退出）：")
    #     if q.lower() == "exit":
    #         break
    #     ask_with_rag(q)
