# sample_data

冒烟测试用的小规模样例数据(完整语料因体积与许可未随仓库分发,全量评测产物保留在 `2Milvus_RAG_Qa/RAG评测/`)。

## 使用方法

```bash
# 1. 技术文档:复制到知识库扫描目录后重建(或首次启动自动导入)
mkdir -p 2Milvus_RAG_Qa/data/ai_data
cp sample_data/tech/*.md 2Milvus_RAG_Qa/data/ai_data/
python MAIN.py rebuild

# 2. 招聘 JD:解析入库 MySQL
python -c "from 2Milvus_RAG_Qa.core.jd_structured import build_and_load; build_and_load('sample_data/jobs_sample.md')"

# 3. 冒烟问答
python MAIN.py query "Python 的装饰器是什么"
python MAIN.py query "北京 java 15k 岗位有哪些"
```
