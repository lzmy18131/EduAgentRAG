"""FastAPI 启动入口。

请先激活项目虚拟环境，再运行 ``python run.py``。不在运行时重写
``sys.path``，避免 PyCharm、系统 Python 和项目依赖互相污染。
"""

import uvicorn


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
