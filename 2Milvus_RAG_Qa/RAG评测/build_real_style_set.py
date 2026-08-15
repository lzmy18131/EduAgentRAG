# -*- coding: utf-8 -*-
"""构建"真实口语风格"评测子集(整改 B6:评测集分布偏移治理)。

方法:
  - 从既有黄金集 eval_golden_500.json 中选取条目,以该条目的**标准答案**
    (自然语言,非 chunk 原文)为知识来源,人工重写为学员口语化问法
    (碎片化/错别字/口语语气/省略主语),expected_chunk 沿用原条目(仅作
    评分标签,问题侧不接触 chunk 文本,避免反向生成泄漏)。
  - 求职类条目不依赖 chunk,以结构化条件 conditions 作为评分依据
    (复用 S4 同口径条件满足判定)。
  - source 全部标注 human_written;与 LLM 生成条目分文件、分指标报告。

产物:eval_real_style_56.json(55 技术/混合 + 10 求职,实际以构建结果为准)
"""
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_GOLDEN = _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_golden_500.json"
_OUT = _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_real_style_56.json"

# (匹配子串, 口语化问题) —— 问题只依据原条目的标准答案改写,不抄 chunk 词汇
_TECH_ITEMS = [
    ("时间复杂度和空间复杂度分别是多少",
     "这题用二叉搜索树做的话 时间复杂度和空间复杂度分别是啥啊"),
    ("Lighthouse性能分数",
     "lighthouse 那个性能分 到底是按哪几个指标算出来的"),
    ("动量方法在优化中为什么能起作用",
     "梯度下降里加动量 为啥就能收敛得更稳 原理是啥"),
    ("什么是回归测试",
     "啥叫回归测试 代码改完之后为啥还要把老用例再跑一遍"),
    ("React 里，key 是用来干什么的",
     "react 列表里那个 key 属性 到底干嘛用的 不写会出啥问题"),
    ("选择排序和冒泡排序",
     "选择排序跟冒泡排序 碰到相等元素的时候 处理上有啥不一样"),
    ("Maurice Wilkes",
     "Maurice Wilkes 是不是有句关于找 bug 的名言 具体咋说的"),
    ("力扣 662 题里二叉树的宽度",
     "力扣 662 二叉树的宽度 中间那些空节点算不算 到底怎么数"),
    ("InceptionV3做DeepDream",
     "inception v3 做 deepdream 的时候 一般关注哪些层 效果差别大吗"),
    ("怎么检查我的环境是否已经正确安装好了",
     "刚装完 咋检查环境装没装对啊"),
    ("深度学习现在到底成功在哪些方面",
     "深度学习这两年这么火 到底在哪些地方算真正成功了"),
    ("AlexNet这个名字是怎么来的",
     "alexnet 这名字咋来的 有故事吗"),
    ("RPA-Python是什么",
     "rpa-python 是个啥东西 能帮我自动开网页截图不"),
    ("Scoop新手",
     "scoop 咋上手 新手应该先看啥资料"),
    ("密码哈希算法",
     "现在存密码 哈希算法推荐用哪个 argon2 还是 bcrypt"),
    ("vhr这个项目",
     "vhr 这个项目是干嘛的 练手合适不"),
    ("适合想成为专业程序员的人看的资源合集",
     "想往专业程序员这条路走 有没有啥资料合集能推荐"),
    ("训练过程中回报的变化趋势",
     "训练的时候 reward 曲线一般是啥走向 一路涨吗"),
    ("GIF图片在较小尺寸时看不到锯齿",
     "gif 放大了有锯齿 那我用大图缩小来展示行不行"),
    ("覆盖了哪些技术领域",
     "这项目都涉及哪些技术方向啊"),
    ("LeetCode 1022",
     "力扣 1022 这题难不难 都考些啥知识点"),
    ("TF_CONFIG 变量怎么在本地配置两个 worker",
     "本地想跑两个 worker tf_config 这个变量咋配"),
    ("mmap 的拷贝次数是固定的吗",
     "mmap 那个零拷贝 拷贝次数是固定的吗 到底省了啥"),
    ("LeetCode 1203 题的英文版",
     "力扣 1203 的英文题解在哪能看"),
    ("芝加哥出租车数据集",
     "那个用芝加哥出租车数据的教程 我想换成自己的数据 放哪都行吗"),
    ("bert_qa 和 mobilebert_qa_squad",
     "bert_qa 跟 mobilebert_qa_squad 这俩模型到底啥区别"),
    ("MRI扫描这类体数据",
     "MRI 那种体数据 该用啥 CNN 模型处理"),
    ("generator 对象创建一个 tf.data.Dataset",
     "怎么用 generator 建一个 tf.data 的数据集来着"),
    ("压缩字符串迭代器",
     "压缩字符串迭代器那道题 初始化和 next 的复杂度是多少"),
    ("统计满足条件的小时和分钟数量",
     "二进制手表那道题 小时和分钟数怎么统计"),
    ("Interval Cancellation 这道题难度怎么样",
     "力扣 2725 这题难不难 主要考啥语言"),
    ("这个页面提供了哪些操作入口",
     "这页面都能干点啥 有哪些入口"),
    ("TF-Agents库在Cartpole环境中训练DQN",
     "tf-agents 那个 cartpole 例子 是训练啥的"),
    ("descriptions = [[20,15,1]",
     "descriptions 构造二叉树 输出序列是啥"),
    ("QCNN模型的层是怎么定义的",
     "QCNN 这个模型的层 是怎么定义的"),
    ("bandit 相关的教程",
     "tf-agents 有没有 bandit 相关的教程"),
    ("TensorFlow Agents 提供了哪些 bandit",
     "tensorflow agents 里 bandit 的教程都有哪些"),
    ("这门课用什么教材",
     "这门课用的啥教材"),
    ("Scipy 有特殊矩阵的实现",
     "scipy 有没有现成的特殊矩阵 在哪能看"),
    ("为什么这个例子中输出是1",
     "这题为啥输出是 1 没太看懂"),
    ("JavaScript 开源框架",
     "这例子用 debugger 打开的是个啥应用"),
    ("梯度实现的示例",
     "梯度实现的示例代码 在哪儿能看"),
    ("TAPL",
     "这课教材是用的 TAPL 吗 后半部分还有教材吗"),
    ("选择不同的层会决定生成的DeepDream",
     "deepdream 选不同的层 生成的图会不一样吗"),
    ("这个项目做了哪些开发相关的配置",
     "这项目都配了些啥开发工具 有 eslint 吗"),
    ("它在哪些方面",
     "深度学习离人类水平的 AI 还差多远"),
    ("如何用前面定义好的 generator",
     "用之前定义好的 generator 建 tf.data.Dataset 咋写"),
    ("二叉树某一层的宽度",
     "力扣那题 二叉树一层的宽度 是数节点还是数位置"),
    ("LeetCode 2725",
     "力扣 2725 interval cancellation 难度咋样"),
    ("如何检查我的环境",
     "环境装完咋验证 输个啥命令能看出来装没装好"),
    ("CNN 3D para classificar vídeos",
     "3d cnn 做视频分类 官方有现成教程吗"),
    ("在文件读取并发送到 TCP 的场景",
     "文件发给 tcp 的场景里 mmap 省的是哪部分拷贝"),
    ("这个示例演示了什么",
     "tf-agents 那个 cartpole 例子 是训练啥的"),
    ("Debugger 打开什么样的应用程序",
     "这例子用 debugger 打开的是个啥应用"),
    ("Kubeflow Pipelinesをデプロイ",
     "kubeflow pipelines 部署的时候 那个 api 和 zone 咋选"),
    ("wordvec_dimを大きく",
     "wordvec_dim 想调大重新训 咋操作"),
    ("GIF 파일을 tf.Tensor로 읽는 방법",
     "gif 文件怎么读成 tf tensor 视频形状是啥"),
]

# 求职类口语化条目:以结构化条件为评分依据(不依赖 chunk)
_JOB_ITEMS = [
    ("北京有没有 15k 以上的 java 岗", {"city": "北京", "tech": "java", "salary_min": 15}),
    ("想在上海找个 python 的活 20k 左右", {"city": "上海", "tech": "python", "salary_min": 20}),
    ("杭州前端 10k 有没有", {"city": "杭州", "tech": "web", "salary_min": 10}),
    ("深圳运维 linux 12k 的岗位", {"city": "深圳", "tech": "linux", "salary_min": 12}),
    ("成都 C# 10k", {"city": "成都", "tech": "C#", "salary_min": 10}),
    ("武汉 python 15k 有吗", {"city": "武汉", "tech": "python", "salary_min": 15}),
    ("西安 web 8k 的工作", {"city": "西安", "tech": "web", "salary_min": 8}),
    ("广州 java 8k 到 12k 之间", {"city": "广州", "tech": "java", "salary_min": 8, "salary_max": 12}),
    ("南京有 java 应届生岗位吗", {"city": "南京", "tech": "java"}),
    ("北京 linux 25k 的岗位", {"city": "北京", "tech": "linux", "salary_min": 25}),
]


def main() -> None:
    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    out = []
    used: set[str] = set()
    for match, new_q in _TECH_ITEMS:
        hits = [i for i, x in enumerate(golden) if match in str(x.get("question", ""))]
        if not hits:
            print(f"[skip] 未匹配到黄金条目: {match[:30]}")
            continue
        idx = hits[0]
        if idx in used:
            continue
        used.add(idx)
        src = golden[idx]
        out.append({
            "question": new_q,
            "answer": src.get("answer", ""),
            "expected_chunk": src.get("expected_chunk", ""),
            "source": "human_written",
            "kind": "tech_qa",
            "gold_id": idx,
            "build_method": "question rephrased from gold answer only; expected_chunk reused as grading label",
        })
    for q, cond in _JOB_ITEMS:
        out.append({
            "question": q,
            "answer": "",
            "expected_chunk": "",
            "conditions": cond,
            "source": "human_written",
            "kind": "job_condition",
            "build_method": "hand-written colloquial job query; graded by S4 condition-satisfaction",
        })
    _OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"构建完成: {len(out)} 条(技术 {sum(1 for x in out if x['kind']=='tech_qa')} / "
          f"求职 {sum(1 for x in out if x['kind']=='job_condition')}) → {_OUT}")


if __name__ == "__main__":
    main()
