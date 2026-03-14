# Digital Human Backend

## 1. 项目是什么

这个项目是一个面向“数字人心理陪伴/会话支持”场景的后端原型，核心目标是把下面几条链路串起来：

- 会话输入 -> 心理状态推断 -> 检索上下文 -> LLM 输出 -> 写回记忆
- 知识库管理 -> 文档入库 -> Milvus / PageIndex 检索
- 默认可本地 mock 测试，按 `.env` 开关可切到真实 DeepSeek + 真实 Milvus

当前仓库已经具备两种运行模式：

- 测试模式：强制 mock，保证 `pytest` 稳定、可重复
- 开发/联调模式：从根目录 `.env` 读取配置，按开关接真实服务

---

## 2. 目录怎么理解

根目录主要看这几块：

- [backend/app](H:/Digital_human/backend/app)
  - 真正的应用代码
- [tests](H:/Digital_human/tests)
  - 单元测试 / 集成测试 / e2e 占位测试
- [Reference/PageIndex-main](H:/Digital_human/Reference/PageIndex-main)
  - 参考实现，主要拿来借鉴 Milvus collection 设计和 BGE 模型加载方式
- [models](H:/Digital_human/models)
  - 本地模型缓存，当前已经有 `bge-m3`
- [data/knowledge](H:/Digital_human/data/knowledge)
  - 知识文件目录，已创建，后续你把真实文档放这里
- [backend/.runtime](H:/Digital_human/backend/.runtime)
  - 运行时缓存目录，例如 PageIndex cache

---

## 3. 我建议你怎么读代码

按下面顺序读，最快：

1. [config.py](H:/Digital_human/backend/app/config.py)
   - 看所有运行时开关、`.env` 入口、路径和外部服务配置

2. [dependencies.py](H:/Digital_human/backend/app/dependencies.py)
   - 看应用启动时把哪些 service 组装到容器里

3. [main.py](H:/Digital_human/backend/app/main.py)
   - 看 FastAPI 怎么挂载路由

4. [routes_sessions.py](H:/Digital_human/backend/app/api/routes_sessions.py)
   - 看会话主链路的 API 入口

5. [routes_knowledge.py](H:/Digital_human/backend/app/api/routes_knowledge.py)
   - 看知识库创建、入库、查询怎么走

6. [nodes.py](H:/Digital_human/backend/app/graph/nodes.py)
   - 这是最核心的业务编排逻辑
   - 会话状态、检索、输出、风控、写回都在这里

7. [retrieval_orchestrator.py](H:/Digital_human/backend/app/services/retrieval_orchestrator.py)
   - 看 Milvus 和 PageIndex 怎么路由、怎么合并结果

8. [milvus_adapter.py](H:/Digital_human/backend/app/services/milvus_adapter.py)
   - 看 mock 模式和真实 Milvus 模式怎么共存

9. [real_milvus_backend.py](H:/Digital_human/backend/app/services/real_milvus_backend.py)
   - 看真实向量库接入、collection 建表、入库、搜索

10. [deepseek_client.py](H:/Digital_human/backend/app/services/deepseek_client.py)
   - 看真实 DeepSeek 调用和 mock fallback

11. [embedding_runtime.py](H:/Digital_human/backend/app/services/embedding_runtime.py)
   - 看本地 BGE-M3 / reranker 的加载策略，重点是“不联网，只走本地缓存”

---

## 4. 运行架构

### 4.1 会话链路

`/api/v1/sessions/{session_id}/turns` 进入后，流程是：

1. 读取 session 上下文
2. 生成 perception 结果
3. 做多模态融合，得到当前心理状态
4. 做 reassessment
5. 调用检索层拿上下文
6. 调用 DeepSeek 产出 `output_a`
7. 调用 DeepSeek 产出 `output_b`
8. 做安全重写
9. 写回 session 和记忆

对应代码主要在 [nodes.py](H:/Digital_human/backend/app/graph/nodes.py)。

### 4.2 知识检索链路

知识查询从 `/api/v1/knowledge/query` 进来后：

1. 先进入 [retrieval_orchestrator.py](H:/Digital_human/backend/app/services/retrieval_orchestrator.py)
2. 根据 `mode / knowledge_type / document_scope` 决定走：
   - Milvus
   - PageIndex
   - 或 hybrid
3. 最终返回：
   - `memory_hits`
   - `knowledge_hits`
   - `pageindex_hits`
   - `trace`
   - `source_breakdown`

---

## 5. 为什么拆成 3 个 collection

当前真实 Milvus 设计是：

- `dh_sections`
- `dh_chunks`
- `dh_memory`

这套设计参考了 [Reference/PageIndex-main/Milvus/script/stores/milvus_store.py](H:/Digital_human/Reference/PageIndex-main/Milvus/script/stores/milvus_store.py) 的 section/chunk 分层思路，但结合当前数字人项目额外加了一个 memory collection。

### 5.1 `dh_sections`

用途：

- 存长文档的章节级节点
- 适合做粗召回 / 结构级定位

为什么要有它：

- 长文档直接 chunk 检索容易命中碎片，不知道大结构在哪
- section 层能帮助后续路由、调试和定位

### 5.2 `dh_chunks`

用途：

- 存章节下的细粒度内容块
- 主要承担真正的召回和最终返回

为什么要有它：

- LLM 真正需要的是可引用的小块文本
- 只做 section 检索会太粗，不够回答细节问题

### 5.3 `dh_memory`

用途：

- 存用户长期记忆
- 包括：
  - `episodic_memory`
  - `profile_memory`
  - `risk_history`

为什么单独拆：

- 用户记忆和知识库文档不是同一种数据
- 过滤条件完全不同，记忆一定要按 `user_id` 查
- 生命周期也不同，记忆会持续追加，知识库更多是文档级 upsert

---

## 6. 检索设计

### 6.1 Milvus 检索

真实 Milvus 模式下，当前做法是：

- 用本地 `BGE-M3` 生成 dense 向量
- 如果可用，也生成 sparse 向量
- dense search 和 sparse search 分别查
- 再用 RRF 做融合
- 最后可选 rerank

对应代码：

- [embedding_runtime.py](H:/Digital_human/backend/app/services/embedding_runtime.py)
- [real_milvus_backend.py](H:/Digital_human/backend/app/services/real_milvus_backend.py)

### 6.2 PageIndex 检索

当前项目里的 PageIndex 还是 mock-first 版本：

- 入库时构建一个本地树缓存
- 查询时按标题/摘要匹配

对应代码：

- [pageindex_adapter.py](H:/Digital_human/backend/app/services/pageindex_adapter.py)

这意味着：

- 现在它适合做“结构化长文 mock/原型”
- 还不是完整 PDF -> 真 PageIndex pipeline

### 6.3 命中逻辑为什么改过

之前的检索匹配太依赖“空格切词 + 完全相等”，这对中文和带标点文本不友好。  
现在统一走 [text_match.py](H:/Digital_human/backend/app/services/text_match.py) 做文本规范化和包含匹配，因此：

- `睡眠` 可以命中 `睡眠卫生`
- `sleep_problem` 可以命中 `sleep_problem: True`

---

## 7. DeepSeek 设计

DeepSeek 现在是“两层策略”：

1. 默认 mock
2. 关闭 `DEEPSEEK_MOCK_ONLY` 后，走真实 `/chat/completions`

为了避免真实模型少字段把主流程打崩，当前策略是：

- 先生成一份稳定 mock 结果当 fallback
- 再调用 DeepSeek
- 如果返回了合法 JSON，就把真实结果 merge 到 fallback 上
- 如果失败，直接回退 mock

对应代码：

- [deepseek_client.py](H:/Digital_human/backend/app/services/deepseek_client.py)

这个设计的好处是：

- 自动化测试稳定
- 联调时也能逐步切真，不会因为一次模型输出漂移就全线崩掉

---

## 8. 配置方式

当前项目已经支持根目录 `.env`，文件在：

- [\.env](H:/Digital_human/.env)
- [\.env.example](H:/Digital_human/.env.example)

主要配置项分 4 类：

- DeepSeek
- Milvus
- Path / runtime
- Embedding / reranker

注意：

- 测试不会直接吃 `.env` 的真服务配置
- [tests/conftest.py](H:/Digital_human/tests/conftest.py) 会强制把测试环境切回 mock

这就是为什么：

- 你平时开发运行可以连真服务
- `pytest` 依然保持稳定可重复

---

## 9. 知识文件怎么放

当前建议把知识文件放在：

- [data/knowledge](H:/Digital_human/data/knowledge)

现在这个目录只是预留好，还没有固定 file type pipeline。  
后续你确定要支持：

- `md`
- `txt`
- `pdf`
- `json`

中的哪几种后，再把真正的 ingest parser 补上最合适。

---

## 10. 怎么跑测试

### 10.1 用 conda 环境

```powershell
conda activate milvus_test
```

或者直接用该环境里的 Python：

```powershell
C:\Users\JAQ\anaconda3\envs\milvus_test\python.exe -m pytest -ra
```

### 10.2 跑默认测试

```powershell
python -m pytest -ra
```

当前结果应为：

- `9 passed, 1 skipped`

因为 e2e 测试默认有开关保护。

### 10.3 跑全部测试

```powershell
$env:RUN_E2E='1'
python -m pytest -ra
```

当前结果应为：

- `10 passed`

### 10.4 建议测试顺序

先跑失败点，再跑全量：

```powershell
python -m pytest tests/unit/test_retrieval_orchestrator.py tests/integration/test_knowledge_api.py -ra
python -m pytest -ra
```

原理：

- 第一条命令反馈最快，适合改检索逻辑时快速回归
- 第二条命令做全局验收，防止局部修复引入回归

---

## 11. 已完成的真实联调验证

这次已经实际验证过两件事：

1. 所有自动化测试通过
2. 真 Milvus 烟测通过

烟测内容是：

- 用真实 `.env` 创建 `MilvusAdapter`
- 确认 `real_backend=True`
- 入库一篇简单文档
- 查询 `睡眠 压力`
- 成功返回命中结果

DeepSeek 真实网络调用代码已经接好，但自动化测试没有直接打真实 API。  
这是刻意的，目的是避免测试过程消耗外部额度并引入不稳定性。

---

## 12. 当前已知限制

1. `KnowledgeBase` 本身还是进程内内存态
   - 也就是说 base_id 映射不会跨进程持久化

2. PageIndex 仍然是 mock-first
   - 还没有接完整 PDF/树构建真实流程

3. 音频/视频输入目前还是 stub
   - schema 已有字段，但真实 ASR / 视觉分析尚未接入

4. DeepSeek 真实调用已接好，但 prompt 仍偏原型
   - 后续可以继续做结构化 prompt 和输出约束增强

---

## 13. 如果你下一步继续做，建议优先级

建议按这个顺序往下推进：

1. 明确知识文件类型
2. 把 `data/knowledge` 的 ingest pipeline 定下来
3. 完整接 PageIndex 真流程
4. 把 KnowledgeBase 持久化
5. 再补真实 ASR / 视频理解

如果你继续让我接着做，最值得优先推进的是：

- “把 `data/knowledge` 里的真实文档做成可批量入库和可查询”
