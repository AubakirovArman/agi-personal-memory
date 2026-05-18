# AGI Personal Memory — 累积验证记忆基底，用于语言模型

[English](../../README.md) | [Русский](../ru/README.md) | [中文](README.md) | [Қазақша](../kk/README.md)

## 这是什么

当今的AI系统是**无状态的**。每次对话从零开始。你无法教会模型一个事实并让它记住。微调是批量的、破坏性的。RAG是检索，不是学习。

**AGI Personal Memory** 通过两个独立路径研究这个问题：用于持久事实的
retrieval memory，以及用于研究实验的 WAL-backed 权重编辑。

## 快速开始

```bash
pip install -e .
agim teach "巴黎是法国的首都"
agim ask "法国的首都是什么？"
agim correct "不对，拿破仑出生于1769年，不是1768年"
agim history
agim stats
agim webui --port 8720
```

## 与现有系统的区别

| | RAG | 微调 | LoRA | AGI Personal Memory |
|---|---|---|---|---|
| 改变模型 | 否 | 是（破坏性） | 是（叠加性） | **实验性** |
| 增量式 | 是 | 否 | 否 | **是** |
| 可逆 | 是 | 否 | 部分 | **是（回滚任意提交）** |
| 可审计 | 否 | 否 | 部分 | **是（完整JSONL记录）** |
| 非目标差异 | N/A | ~25% | 中等 | **WAL diagnostics 中为 0%** |

## 工作原理

```
用户输入 → 意图路由器(LLM+正则) → 记忆提取器 → MemoryCandidate
                                                  ↓
                                             验证（12项原则）
                                                  ↓
                                         记忆编译器（5个层级）
                                                  ↓
                              ┌─ WAL配方 → 模型权重编辑
                              ├─ 检索     → 键值存储
                              ├─ LoRA     → 正交适配器
                              └─ 拒绝     → 策略模式
                                                  ↓
                                             提交 + 审计追踪
```

## 接口

| 接口 | 命令 | 描述 |
|------|------|------|
| CLI | `agim teach/ask/correct/forget` | 命令行内存操作 |
| Shell | `agim shell` | 交互式REPL |
| REST API | `agim api --port 8720` | 11个端点 |
| Web仪表板 | `agim webui --port 8720` | JS仪表板，5个标签页 |
| MCP | `MCPServer` | 模型上下文协议 |
| A2A | `A2AServer` | 智能体间协议 |
| GraphQL | `GraphQLResolver` | GraphQL查询 |
| 导出 | `agim export memories.json` | 导出为JSON |
| 导入 | `agim import memories.json` | 从JSON导入 |

## 当前状态

- 当前 EasyEdit-compatible 结果见 `../../BENCHMARK.md`
- 历史本地 CounterFact 结果已分离到 `../../results/local_protocol/`
- 本地完整 pytest：`119 passed, 13 skipped`
- skipped 测试为 Gemma E2E，当当前 Transformers 不支持 `gemma4` 时跳过

## 许可证

MIT
