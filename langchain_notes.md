# LangChain 学习笔记

本文档记录了 LangChain 中 `@dataclass`、`@tool`、`checkpointer` 等核心概念的学习笔记。

---

## 1. @dataclass - 数据容器

`@dataclass` 是 Python 标准库的装饰器，用于快速创建数据容器类。

```python
from dataclasses import dataclass

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    punny_response: str
    weather_conditions: str | None = None
```

**作用**：
- 自动生成 `__init__`、`__repr__` 等方法
- 定义数据结构模板
- 用于存储运行时上下文或返回格式

---

## 2. @tool - 将函数变成 AI 可调用的工具

`@tool` 是 LangChain 的装饰器，将普通 Python 函数转换成 AI 可以理解和调用的工具。

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"
```

**关键要素**：

| 要素 | 说明 |
|---|---|
| 函数名 | 变成工具名称 |
| 参数类型 | AI 知道要传什么参数 |
| 文档字符串 | AI 通过它理解工具用途 |
| 返回值 | 工具执行后的结果返回给 AI |

---

## 3. ToolRuntime[Context] 的含义

```python
@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"
```

**为什么不直接传 Context？**

`ToolRuntime` 是一个包装器，包含了：
- `context`: 你的自定义上下文数据
- 框架提供的其他运行时信息（未来可扩展）

```python
class ToolRuntime:
    context: T           # 你的自定义上下文
    # 框架可能添加的其他运行时信息
```

**优势**：
- 灵活可扩展
- 框架可以无缝添加新功能
- 保持类型安全

---

## 4. context_schema vs context

### context_schema=Context

告诉 Agent 上下文的数据结构：

```python
agent = create_agent(
    context_schema=Context,  # 声明类型
    ...
)
```

### context=Context(user_id="1")

调用时提供实际的数据：

```python
response = agent.invoke(
    {"messages": [...]},
    context=Context(user_id="1")  # 实际数据
)
```

### 数据流

```
定义类型: context_schema=Context
         ↓
传入数据: context=Context(user_id="1")
         ↓
调用工具: runtime.context.user_id → "1"
```

---

## 5. config 与 thread_id

```python
config = {"configurable": {"thread_id": "1"}}
```

### config vs context

| 参数 | 作用 | 类比 |
|---|---|---|
| `config` (thread_id) | 管理对话会话，区分不同对话 | 聊天窗口编号 |
| `context` (user_id) | 业务数据，传递给工具使用 | 用户身份信息 |

### thread_id 的作用

配合 `checkpointer` 实现：
- 保存对话历史
- 支持多轮对话记忆
- 不同会话互不干扰

```
thread_id="1" → 用户A的对话
thread_id="2" → 用户B的对话
thread_id="3" → 用户C的对话
```

---

## 6. AI 如何自动推理调用工具

当用户问 "外面的天气怎样？" 时：

```
用户问: "外面的天气怎样？"
    ↓
AI分析: 我需要天气，但不知道用户在哪
    ↓
AI看到工具:
    - get_weather_for_location(city: str)  ← 需要参数
    - get_user_location()                  ← 可以获取位置
    ↓
AI决定: 先调用 get_user_location()
    ↓
返回: "Florida"
    ↓
AI调用: get_weather_for_location("Florida")
    ↓
返回: "It's always sunny in Florida!"
```

**关键点**：AI 有自然语言理解能力，能从返回值中提取关键信息。

---

## 7. Checkpointer - 对话记忆

```python
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
```

### 作用

保存对话状态：
- 对话历史（所有消息）
- Agent 的内部状态
- 工具调用的记录

### 有无 Checkpointer 对比

**没有 Checkpointer**：
```
用户: 北京天气怎样？
AI:  北京天气晴朗

用户: 那上海呢？
AI:  请问您问的是上海的什么？  ← 忘记了
```

**有 Checkpointer**：
```
用户: 北京天气怎样？
AI:  北京天气晴朗

用户: 那上海呢？
AI:  上海天气多云  ← 记得之前在聊天气
```

### 持久化方案

| 类型 | 存储 | 特点 |
|---|---|---|
| `InMemorySaver` | 内存 | 快速，重启后丢失 |
| `SqliteSaver` | SQLite | 持久化到本地 |
| `RedisSaver` | Redis | 分布式，适合生产 |
| `PostgresSaver` | PostgreSQL | 企业级 |

---

## 8. 完整数据流示意图

```
┌─────────────────────────────────────────────────────────┐
│  1. 定义类型                                              │
│     context_schema=Context                               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  2. 调用时传入数据                                        │
│     context=Context(user_id="1")                         │
│     config={"configurable": {"thread_id": "1"}}          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  3. AI 推理并调用工具                                     │
│     - get_user_location() → "Florida"                    │
│     - get_weather_for_location("Florida") → "Sunny!"     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  4. 返回结构化响应                                        │
│     ResponseFormat(punny_response=..., weather=...)      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  5. Checkpointer 保存对话历史                             │
│     下次调用同一 thread_id 可继续对话                     │
└─────────────────────────────────────────────────────────┘
```

---

## 总结表

| 概念 | 作用 |
|---|---|
| `@dataclass` | 快速创建数据容器类 |
| `@tool` | 将函数变成 AI 可调用的工具 |
| `ToolRuntime[T]` | 包装器，包含 context 和运行时信息 |
| `context_schema` | 声明上下文的类型结构 |
| `context` | 提供实际的上下文数据 |
| `thread_id` | 区分不同对话会话 |
| `checkpointer` | 保存对话状态，实现记忆 |

---

*学习时间：2026/04/06*
