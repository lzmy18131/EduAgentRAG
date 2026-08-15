"""EdeRAG 智能问答系统 — FastAPI Web 接口

启动: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import hmac
import importlib
import json
import sys
import threading
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from base.logger import logger

# ── 延迟导入（数字开头的目录不能用 from ... import）──
_rag_system = None
_vector_store = None
_mysql_system = None
_memory_layer = None
_upload_lock = asyncio.Lock()
_chat_slots = threading.BoundedSemaphore(2)  # R2:云端 deepseek 限流 2 并发(实测 4 并发易触发 APIConnectionError);同时覆盖 /chat 与 /chat/stream
_mysql_lock = threading.Lock()
_rag_init_lock = threading.Lock()
_memory_lock = threading.Lock()
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_LLM_TIMEOUT_SECONDS = 120


def _memory_update(session_id: str, user_query: str) -> None:
    """优化二:记忆在线写入——答案返回后异步萃取"新事实/新偏好"双写。

    后台 daemon 线程执行,不阻塞主链路;失败静默(记忆不写不影响问答)。
    单机场景用线程替代 MQ 解耦(避免为一个功能引入消息队列依赖)。
    """
    global _memory_layer
    try:
        memory_mod = importlib.import_module("2Milvus_RAG_Qa.core.memory")
        with _memory_lock:
            if _memory_layer is None:
                rag, _vs = _get_rag()
                _memory_layer = memory_mod.MemoryLayer(rag)
            _memory_layer.update_from_turn(session_id, user_query)
    except Exception:
        logger.exception("记忆在线写入失败(不影响主链路)")


def _get_rag():
    global _rag_system, _vector_store
    if _rag_system is None:
        with _rag_init_lock:
            if _rag_system is None:
                rag_main = importlib.import_module("2Milvus_RAG_Qa.core.rag_main")
                rag_mod = importlib.import_module("2Milvus_RAG_Qa.core.rag_system")
                _vector_store = rag_main.init_knowledge_base()
                _rag_system = rag_mod.RAGSystem(vector_store=_vector_store)
                logger.info("RAG 系统初始化完成")
    return _rag_system, _vector_store


_adaptive_agent = None
_agent_lock = threading.Lock()


def _get_adaptive_agent():
    """懒加载 Adaptive RAG 循环(架构主线)。

    自省/改写走本地 Qwen2.5-0.5B(失败自动回退云端);LocalLLM 加载失败时
    返回 None,调用方回退经典 RAGSystem.query。
    """
    global _adaptive_agent
    if _adaptive_agent is None:
        with _agent_lock:
            if _adaptive_agent is None:
                try:
                    rag, _vs = _get_rag()
                    agent_mod = importlib.import_module("2Milvus_RAG_Qa.core.agent_graph")
                    local_mod = importlib.import_module("2Milvus_RAG_Qa.core.local_llm")
                    local = local_mod.LocalLLM()
                    _adaptive_agent = agent_mod.AdaptiveRAG(rag, local_llm=local)
                    logger.info("Adaptive RAG 循环初始化完成(自省/改写=本地 0.5B)")
                except Exception:
                    logger.exception("Adaptive RAG 初始化失败,回退经典路径")
                    _adaptive_agent = None
    return _adaptive_agent


def _memory_hint(session_id: str | None, query: str) -> str:
    """记忆读取注入:画像 + 按当前问题语义召回的历史事实。

    记忆层不可用时返回空串(不阻塞主链路);写入侧见 _memory_update。
    """
    global _memory_layer
    if not session_id:
        return ""
    try:
        memory_mod = importlib.import_module("2Milvus_RAG_Qa.core.memory")
        with _memory_lock:
            if _memory_layer is None:
                rag, _vs = _get_rag()
                _memory_layer = memory_mod.MemoryLayer(rag)
            profile = _memory_layer.profile_hint(session_id) or ""
            facts = _memory_layer.facts_hint(session_id, query) or ""
        hint = (profile + " " + facts).strip()
        return f"【用户画像与历史事实,仅供参考,不得作为技术知识依据】{hint}" if hint else ""
    except Exception:
        logger.exception("记忆读取注入失败(不影响主链路)")
        return ""


def _get_mysql():
    global _mysql_system
    if _mysql_system is None:
        # MySQL QA 模块在 1MySQL_qa/ 下，该目录内有自己的 sys.path 逻辑
        mod = importlib.import_module("1MySQL_qa.mysql_qa_main")
        _mysql_system = mod.MySQLQaSystem()
        logger.info("MySQL QA 系统初始化完成")
    return _mysql_system


app = FastAPI(
    title="EdeRAG 智慧问答系统",
    description="语义 FAQ + Milvus 混合检索 + DeepSeek 大模型 RAG",
    version="2.1.0",
)


@app.middleware("http")
async def limit_upload_body(request: Request, call_next):
    """在 multipart 解析前拒绝已声明的超大上传请求。"""
    if request.url.path == "/upload":
        content_length = request.headers.get("content-length")
        try:
            if content_length and int(content_length) > _MAX_UPLOAD_BYTES + 1024 * 1024:
                return HTMLResponse("请求体不能超过 11MB", status_code=413)
        except ValueError:
            return HTMLResponse("非法 Content-Length", status_code=400)
    return await call_next(request)


# ── 聊天页面 HTML ──

CHAT_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EdeRAG 智慧问答</title>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#0a0a0a;
  --surface:#141414;
  --surface2:#1a1a1a;
  --text:#f5f5f5;
  --text2:#a0a0a0;
  --text3:#666;
  --accent:#0072f5;
  --accent-glow:rgba(0,114,245,0.15);
  --border:rgba(255,255,255,0.06);
  --shadow:0px 0px 0px 1px rgba(255,255,255,0.06);
  --radius:10px;
}
*{margin:0;padding:0;box-sizing:border-box}
body{
  font-family:'Geist',system-ui,-apple-system,sans-serif;
  background:var(--bg);color:var(--text);height:100vh;
  display:flex;flex-direction:column;overflow:hidden;
}
/* ── header ── */
.header{
  padding:14px 24px;font-size:15px;font-weight:500;
  letter-spacing:-0.3px;color:var(--text2);
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:10px;
  background:var(--surface);flex-shrink:0;
}
.header .dot{width:8px;height:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 8px var(--accent-glow)}
.header .title{color:var(--text);font-weight:600;margin-right:8px}
.header .badge{font-size:11px;padding:2px 8px;border-radius:99px;background:rgba(0,114,245,0.1);color:var(--accent);font-weight:500}
/* ── messages ── */
.messages{
  flex:1;overflow-y:auto;padding:24px 20px;
  display:flex;flex-direction:column;gap:20px;
  scroll-behavior:smooth;
}
.msg{max-width:76%;line-height:1.65;animation:in .25s ease}
.msg.user{align-self:flex-end}
.msg.bot{align-self:flex-start}
.msg .bubble{
  padding:12px 16px;font-size:15px;
  box-shadow:var(--shadow);
}
.msg.user .bubble{
  background:var(--accent);color:#fff;
  border-radius:var(--radius) 4px var(--radius) var(--radius);
}
.msg.bot .bubble{
  background:var(--surface2);color:var(--text);
  border-radius:4px var(--radius) var(--radius) var(--radius);
}
.msg .meta{
  font-size:11px;color:var(--text3);margin-top:6px;padding:0 4px;
  font-family:'Geist Mono',ui-monospace,monospace;letter-spacing:-0.2px;
}
/* ── thumbs 反馈 ── */
.thumbs{display:flex;gap:6px;margin-top:6px;padding:0 4px}
.thumbs button{
  background:var(--surface2);border:none;border-radius:6px;padding:3px 9px;
  color:var(--text2);cursor:pointer;font-size:13px;box-shadow:var(--shadow);
  transition:.12s;
}
.thumbs button:hover{color:var(--text);transform:translateY(-1px)}
.thumbs button.picked{outline:2px solid var(--accent);color:var(--text)}
/* ── sources 引用块 ── */
.sources{margin-top:8px;display:flex;flex-direction:column;gap:6px}
.source-item{
  background:var(--surface2);border-radius:8px;overflow:hidden;
  box-shadow:var(--shadow);
}
.source-head{
  padding:6px 12px;font-size:12px;color:var(--accent);cursor:pointer;
  user-select:none;font-weight:500;
}
.source-head:hover{background:rgba(0,114,245,0.08)}
.source-body{
  padding:8px 12px;font-size:13px;color:var(--text2);line-height:1.6;
  border-top:1px solid var(--border);white-space:pre-wrap;word-break:break-word;
}
/* ── input ── */
.input-area{
  display:flex;gap:10px;padding:16px 20px;background:var(--surface);
  border-top:1px solid var(--border);flex-shrink:0;
}
.input-area input{
  flex:1;padding:12px 16px;background:var(--surface2);
  border:none;border-radius:var(--radius);color:var(--text);
  font-size:15px;font-family:inherit;outline:none;
  box-shadow:0px 0px 0px 1px var(--border);
}
.input-area input::placeholder{color:var(--text3)}
.input-area input:focus{box-shadow:0px 0px 0px 2px var(--accent)}
.input-area button{
  padding:12px 20px;background:var(--accent);color:#fff;
  border:none;border-radius:var(--radius);font-size:14px;
  font-weight:500;font-family:inherit;cursor:pointer;
  transition:opacity .15s;white-space:nowrap;
}
.input-area button:hover{opacity:.88}
/* ── welcome ── */
.welcome{text-align:center;padding:40px 20px;color:var(--text2)}
.welcome h2{font-size:20px;font-weight:600;color:var(--text);margin-bottom:8px;letter-spacing:-0.4px}
.welcome p{font-size:14px;line-height:1.7;max-width:520px;margin:0 auto}
.welcome .hints{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:20px}
.welcome .hint{
  padding:6px 14px;border-radius:99px;font-size:13px;
  background:var(--surface2);color:var(--text2);cursor:pointer;
  box-shadow:var(--shadow);transition:.15s;
}
.welcome .hint:hover{color:var(--text);box-shadow:0 0 0 2px var(--border)}
/* ── typing ── */
.typing{display:flex;gap:5px;padding:12px 16px}
.typing span{width:6px;height:6px;border-radius:50%;background:var(--text3);animation:typing 1.4s infinite ease-in-out}
.typing span:nth-child(2){animation-delay:.2s}
.typing span:nth-child(3){animation-delay:.4s}
@keyframes typing{0%,60%,100%{transform:translateY(0);opacity:.4}30%{transform:translateY(-6px);opacity:1}}
@keyframes in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
/* ── scrollbar ── */
.messages::-webkit-scrollbar{width:4px}
.messages::-webkit-scrollbar-thumb{background:var(--border);border-radius:99px}
</style>
</head>
<body>
<div class="header">
  <div class="dot"></div>
  <span class="title">EdeRAG</span> 智慧问答
  <span class="badge">bge-small FAQ + 混合检索 + DeepSeek</span>
</div>
<div class="messages" id="messages">
  <div class="welcome" id="welcome">
    <h2>有什么可以帮你的？</h2>
    <p>我是 EdeRAG 智能助手，融合了关键词检索、向量搜索和大模型生成。课程、技术、就业相关问题都可以问我。</p>
    <div class="hints">
      <span class="hint" onclick="ask(this)">Python 数据分析课程有哪些</span>
      <span class="hint" onclick="ask(this)">AI 工程师薪资多少</span>
      <span class="hint" onclick="ask(this)">什么是 RAG 检索增强生成</span>
      <span class="hint" onclick="ask(this)">零基础怎么学 Python</span>
    </div>
  </div>
</div>
<div class="input-area">
  <input id="query" placeholder="输入你的问题…" onkeydown="if(event.key==='Enter')send()" autofocus>
  <button onclick="send()">发送</button>
</div>
<script>
const sessionId=crypto.randomUUID()
let lastQuery=''
function ask(el){document.getElementById('query').value=el.textContent;send()}
function addMsg(text,role,meta='',sources=[]){
  const w=document.getElementById('welcome');if(w)w.remove()
  const d=document.createElement('div');d.className='msg '+role
  const bubble=document.createElement('div');bubble.className='bubble';bubble.textContent=text
  d.appendChild(bubble)
  if(meta){const m=document.createElement('div');m.className='meta';m.textContent=meta;d.appendChild(m)}
  if(sources&&sources.length){
    const swrap=document.createElement('div');swrap.className='sources'
    sources.forEach((s,i)=>{
      const item=document.createElement('div');item.className='source-item'
      const head=document.createElement('div');head.className='source-head';head.textContent='来源 '+(i+1)
      const body=document.createElement('div');body.className='source-body';body.textContent=s;body.style.display='none'
      head.onclick=()=>{body.style.display=body.style.display==='none'?'block':'none'}
      item.appendChild(head);item.appendChild(body);swrap.appendChild(item)
    })
    d.appendChild(swrap)
  }
  if(role==='bot'){
    const tw=document.createElement('div');tw.className='thumbs'
    const up=document.createElement('button');up.textContent='👍';up.title='回答有帮助'
    const down=document.createElement('button');down.textContent='👎';down.title='回答有问题'
    up.onclick=()=>rate('up',up,down,bubble.textContent)
    down.onclick=()=>rate('down',down,up,bubble.textContent)
    tw.appendChild(up);tw.appendChild(down);d.appendChild(tw)
  }
  document.getElementById('messages').appendChild(d)
  document.getElementById('messages').scrollTop=document.getElementById('messages').scrollHeight
}
async function rate(v,btn,other,answer){
  try{
    const r=await fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({query:lastQuery,answer:answer,rating:v,session_id:sessionId})})
    const d=await r.json()
    if(d.status==='ok'){btn.classList.add('picked');other.classList.remove('picked')}
  }catch(e){}
}
function showTyping(){
  const w=document.getElementById('welcome');if(w)w.remove()
  const d=document.createElement('div');d.className='typing';d.id='typing'
  d.innerHTML='<span></span><span></span><span></span>'
  document.getElementById('messages').appendChild(d)
  document.getElementById('messages').scrollTop=document.getElementById('messages').scrollHeight
}
function hideTyping(){const t=document.getElementById('typing');if(t)t.remove()}
async function send(){
  const q=document.getElementById('query').value.trim();if(!q)return
  lastQuery=q
  addMsg(q,'user');document.getElementById('query').value='';showTyping()
  try{
    const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,use_rag:true,session_id:sessionId})})
    const d=await r.json();hideTyping()
    addMsg(d.answer,'bot',`${d.intent} · ${d.strategy} · ${d.latency_ms}ms`+(d.from_cache?' · 缓存':''),d.sources||[])
  }catch(e){hideTyping();addMsg('请求失败：'+e.message,'bot')}
}
</script>
</body>
</html>"""


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    use_rag: bool = Field(default=True)
    session_id: str | None = Field(default=None, max_length=128)


class FeedbackRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    answer: str = Field(default="", max_length=20000)
    rating: str = Field(..., pattern="^(up|down)$")
    session_id: str | None = Field(default=None, max_length=128)
    sources: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    intent: str = ""
    strategy: str = ""
    sources: list[str] = []
    latency_ms: int = 0
    from_cache: bool = False


@app.get("/", response_class=HTMLResponse)
async def index():
    """聊天页面"""
    return CHAT_HTML


# ── API 路由 ──

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    def locked() -> ChatResponse:
        # slot 在线程真正结束前始终占用，防止超时线程绕过并发上限。
        with _chat_slots:
            return _chat_sync(req)

    return await asyncio.to_thread(locked)


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    """E4 反馈闭环:👍/👎 落库,点踩自动写负样本 JSONL(供回灌评测集)。"""
    try:
        feedback_mod = importlib.import_module("2Milvus_RAG_Qa.core.feedback")
        result = await asyncio.to_thread(
            feedback_mod.record_feedback,
            req.session_id or "",
            req.query,
            req.answer,
            req.rating,
            req.sources,
        )
        return {"status": "ok", **result}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception:
        logger.exception("反馈记录失败")
        raise HTTPException(500, "反馈记录失败") from None


@app.get("/feedback/stats")
async def feedback_stats():
    """反馈统计(总量/好评/差评)。"""
    try:
        feedback_mod = importlib.import_module("2Milvus_RAG_Qa.core.feedback")
        return await asyncio.to_thread(feedback_mod.feedback_stats)
    except Exception:
        logger.exception("反馈统计失败")
        raise HTTPException(500, "反馈统计失败") from None


def _sse(payload: dict) -> str:
    """把事件 dict 序列化为 SSE data 行。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式问答（SSE）：逐 token 输出答案，末尾附 meta 元数据。

    同步生成器交由 Starlette 线程池迭代，不阻塞事件循环。
    """

    def event_stream():
        with _chat_slots:
            # 先试 MySQL 快筛（命中则一次性返回）
            if req.use_rag:
                try:
                    with _mysql_lock:
                        mysql_qa = _get_mysql()
                        answer, _msg = mysql_qa.search(req.query)
                    if answer:
                        yield _sse({"type": "token", "content": answer})
                        yield _sse({
                            "type": "meta", "answer": answer, "intent": "mysql_faq",
                            "strategy": "direct", "sources": [], "latency_ms": 0,
                            "from_cache": True,
                        })
                        return
                except Exception:
                    logger.exception("MySQL 快筛不可用，继续尝试 RAG 流式")
            try:
                rag, _vs = _get_rag()
                hint = _memory_hint(req.session_id, req.query)
                agent = _get_adaptive_agent()
                if agent is not None:
                    events = agent.query_stream(
                        req.query, session_id=req.session_id, memory_hint=hint
                    )
                else:
                    events = rag.query_stream(
                        req.query, session_id=req.session_id, memory_hint=hint
                    )
                for event in events:
                    yield _sse(event)
            except Exception:
                logger.exception("流式问答失败")
                yield _sse({"type": "error", "content": "抱歉，服务暂时不可用，请稍后重试。"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _chat_sync(req: ChatRequest) -> ChatResponse:
    answer = None
    try:
        if req.use_rag:
            # 先试 MySQL BM25 快速问答
            try:
                with _mysql_lock:
                    mysql_qa = _get_mysql()
                    answer, _msg = mysql_qa.search(req.query)
                if answer:
                    return ChatResponse(answer=answer, intent="mysql_faq", from_cache=True)
            except Exception:
                logger.exception("MySQL 快筛不可用，继续尝试 RAG")

            # 再试 Milvus RAG(含拒答/缓存命中等结果,均直接采纳)
            try:
                rag, vs = _get_rag()
                # 架构主线:Adaptive RAG 循环(检索→自省→改写→生成),
                # 记忆(画像+事实)请求前注入;循环任何异常内部降级单次检索,
                # LocalLLM 不可用时回退经典 RAGSystem.query
                hint = _memory_hint(req.session_id, req.query)
                agent = _get_adaptive_agent()
                if agent is not None:
                    result = agent.query(
                        req.query, memory_hint=hint, session_id=req.session_id
                    )
                    result.setdefault("intent", "rag_adaptive")
                    result.setdefault("strategy", "agent")
                    result.setdefault("latency_ms", 0)
                    result.setdefault("from_cache", False)
                else:
                    result = rag.query(
                        req.query, session_id=req.session_id, memory_hint=hint
                    )
                # 优化二:答案返回后异步记忆写入(不阻塞应答)
                if req.session_id:
                    threading.Thread(
                        target=_memory_update,
                        args=(req.session_id, req.query),
                        daemon=True,
                    ).start()
                return ChatResponse(**result)
            except Exception:
                logger.exception("Milvus RAG 不可用，继续尝试 LLM")

        # 最后：真正的纯 LLM 兜底，不初始化 Milvus/BGE/BERT。
        return ChatResponse(
            answer=_direct_llm(req.query),
            intent="llm_fallback",
            strategy="direct",
        )
    except Exception:
        logger.exception("聊天请求处理失败")

    # 全挂了，用纯 LLM
    fallback = _direct_llm(req.query)
    return ChatResponse(answer=fallback, intent="llm_fallback", strategy="direct")


def _direct_llm(query: str) -> str:
    """绕过所有数据库，直接调 DeepSeek API。"""
    try:
        from openai import OpenAI
        from base.config import cfg
        client = OpenAI(
            api_key=cfg.LLM_API_KEY,
            base_url=cfg.LLM_BASE_URL,
            timeout=_LLM_TIMEOUT_SECONDS,
        )
        resp = client.chat.completions.create(
            model=cfg.LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是EdeRAG智能助手，专业回答AI教育、编程技术、就业相关的问题。用中文简洁回答。"},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=1024,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        logger.exception("DeepSeek 直接调用失败")
        return "抱歉，服务暂时不可用，请稍后重试。"


@app.post("/upload")
async def upload_knowledge(
    request: Request,
    file: UploadFile = File(...),
    x_api_key: str = Header(default="", alias="X-API-Key"),
):
    from base.config import cfg

    content_length = request.headers.get("content-length")
    try:
        if content_length and int(content_length) > _MAX_UPLOAD_BYTES + 1024 * 1024:
            raise HTTPException(413, "请求体不能超过 11MB")
    except ValueError as exc:
        raise HTTPException(400, "非法 Content-Length") from exc
    client_host = request.client.host if request.client else ""
    if request.url.scheme != "https" and client_host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(400, "远程上传必须使用 HTTPS")
    if not cfg.UPLOAD_API_KEY or not hmac.compare_digest(x_api_key, cfg.UPLOAD_API_KEY):
        raise HTTPException(401, "未授权")
    safe_name = Path(file.filename or "").name
    if not safe_name or Path(safe_name).suffix.lower() not in {".md", ".txt"}:
        raise HTTPException(400, "仅支持 .md / .txt")

    content = bytearray()
    while chunk := await file.read(1024 * 1024):
        content.extend(chunk)
        if len(content) > _MAX_UPLOAD_BYTES:
            raise HTTPException(413, "文件不能超过 10MB")

    async with _upload_lock:
        save_dir = _PROJECT_ROOT / "2Milvus_RAG_Qa" / "data" / "ai_data"
        save_dir.mkdir(parents=True, exist_ok=True)
        target = save_dir / safe_name
        previous_content = target.read_bytes() if target.exists() else None
        target.write_bytes(content)
        loader = None
        vs = None
        try:
            loader = importlib.import_module("2Milvus_RAG_Qa.core.document_loader")
            parents, children = loader.process_documents(str(save_dir))
            if not children:
                raise ValueError("上传后未生成有效文档块")
            _, vs = _get_rag()
            target_path = target.resolve()
            matching_parents = [p for p in parents if Path(p["file_path"]).resolve() == target_path]
            matching_children = [c for c in children if Path(c["file_path"]).resolve() == target_path]
            if not matching_children:
                raise ValueError("上传文件未生成有效文档块")
            vs.replace_document(safe_name, matching_parents, matching_children)
        except Exception as exc:
            if previous_content is None:
                target.unlink(missing_ok=True)
                if vs is not None:
                    try:
                        vs.delete_document(safe_name)
                    except Exception:
                        logger.exception("新增文档的残留索引清理失败，需要执行 MAIN.py rebuild")
            else:
                target.write_bytes(previous_content)
                try:
                    if loader is None or vs is None:
                        raise RuntimeError("向量存储尚未初始化")
                    old_parents, old_children = loader.process_documents(str(save_dir))
                    target_path = target.resolve()
                    old_parents = [p for p in old_parents if Path(p["file_path"]).resolve() == target_path]
                    old_children = [c for c in old_children if Path(c["file_path"]).resolve() == target_path]
                    vs.replace_document(safe_name, old_parents, old_children)
                except Exception:
                    logger.exception("旧知识库索引恢复失败，需要执行 MAIN.py rebuild")
            logger.exception("知识库上传处理失败")
            raise HTTPException(500, "知识库更新失败") from exc
    return {"status": "ok", "file": safe_name, "chunks": len(matching_children)}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    _, vs = _get_rag()
    return {"vector_chunks": vs.count_chunks()}
