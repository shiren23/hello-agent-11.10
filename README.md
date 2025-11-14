# hello-agent-11.10

# hello agent 学习笔记

## 先贴一个目录，方便梳理

1. 1. 什么是智能体
   2. 智能体的构成与运行原理
   3. 5分钟实现第一个智能体
   4. 智能体应用的协作模式

## 什么是智能体

### 从三个角度对智能体进行分类

- 基于内部决策架构的分类
- 基于时间与反应性的分类
- 基于知识表示的分类

## 智能体的构成与运行原理

### PASE模型

- 性能度量（performance）
- 环境(environment)
- 执行器(actuators)
- 传感器(sensors)

### 智能体的运行机制

1. 感知(perception)
2. 思考(thought) 
   - 规划(planning)
   - 工具选择(tool selection)
3. 行动(action)
4. 根据行动引起的环境变化产生新的观察，进入下一个循环

## 智能旅行助手

0. 安装库：requests、tavily-python、openai

   ```bash
   pip install requests tavily-python openai
   ```

1. 构建指令模版：

   ```
   AGENT_SYSTEM_PROMPT = """
   你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。
   
   # 可用工具:
   - `get_weather(city: str)`: 查询指定城市的实时天气。
   - `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。
   
   # 行动格式:
   你的回答必须严格遵循以下格式。首先是你的思考过程，然后是你要执行的具体行动，每次回复只输出一对Thought-Action：
   Thought: [这里是你的思考过程和下一步计划]
   Action: [这里是你要调用的工具，格式为 function_name(arg_name="arg_value")]
   
   # 任务完成:
   当你收集到足够的信息，能够回答用户的最终问题时，你必须在`Action:`字段后使用 `finish(answer="...")` 来输出最终答案。
   
   请开始吧！
   """
   ```

2. 工具1：查询真实天气（免费天气查询服务wttr.in）

   ```python
   import requests
   import json
   
   def get_weather(city: str) -> str:
       """
       通过调用 wttr.in API 查询真实的天气信息。
       """
       # API端点，我们请求JSON格式的数据
       url = f"https://wttr.in/{city}?format=j1"
       
       try:
           # 发起网络请求
           response = requests.get(url)
           # 检查响应状态码是否为200 (成功)
           response.raise_for_status() 
           # 解析返回的JSON数据
           data = response.json()
           
           # 提取当前天气状况
           current_condition = data['current_condition'][0]
           weather_desc = current_condition['weatherDesc'][0]['value']
           temp_c = current_condition['temp_C']
           
           # 格式化成自然语言返回
           return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"
           
       except requests.exceptions.RequestException as e:
           # 处理网络错误
           return f"错误:查询天气时遇到网络问题 - {e}"
       except (KeyError, IndexError) as e:
           # 处理数据解析错误
           return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"
   ```

3. 工具2：搜索并推荐旅游景点（新工具search_attraction）

   ```python
   import os
   from tavily import TavilyClient
   
   def get_attraction(city: str, weather: str) -> str:
       """
       根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
       """
       # 1. 从环境变量中读取API密钥
       api_key = os.environ.get("TAVILY_API_KEY")
       if not api_key:
           return "错误:未配置TAVILY_API_KEY环境变量。"
   
       # 2. 初始化Tavily客户端
       tavily = TavilyClient(api_key=api_key)
       
       # 3. 构造一个精确的查询
       query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"
       
       try:
           # 4. 调用API，include_answer=True会返回一个综合性的回答
           response = tavily.search(query=query, search_depth="basic", include_answer=True)
           
           # 5. Tavily返回的结果已经非常干净，可以直接使用
           # response['answer'] 是一个基于所有搜索结果的总结性回答
           if response.get("answer"):
               return response["answer"]
           
           # 如果没有综合性回答，则格式化原始结果
           formatted_results = []
           for result in response.get("results", []):
               formatted_results.append(f"- {result['title']}: {result['content']}")
           
           if not formatted_results:
                return "抱歉，没有找到相关的旅游景点推荐。"
   
           return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)
   
       except Exception as e:
           return f"错误:执行Tavily搜索时出现问题 - {e}"
   ```

​	将所有工具函数放入一个字典，供主循环调用

```python
# 将所有工具函数放入一个字典，方便后续调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}
```

4. 接入大语言模型

5. ```python
   from openai import OpenAI
   
   class OpenAICompatibleClient:
       """
       一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
       """
       def __init__(self, model: str, api_key: str, base_url: str):
           self.model = model
           self.client = OpenAI(api_key=api_key, base_url=base_url)
   
       def generate(self, prompt: str, system_prompt: str) -> str:
           """调用LLM API来生成回应。"""
           print("正在调用大语言模型...")
           try:
               messages = [
                   {'role': 'system', 'content': system_prompt},
                   {'role': 'user', 'content': prompt}
               ]
               response = self.client.chat.completions.create(
                   model=self.model,
                   messages=messages,
                   stream=False
               )
               answer = response.choices[0].message.content
               print("大语言模型响应成功。")
               return answer
           except Exception as e:
               print(f"调用LLM API时发生错误: {e}")
               return "错误:调用语言模型服务时出错。"
   ```

5. 执行行动循环

6. ```python
   import re
   
   # --- 1. 配置LLM客户端 ---
   # 请根据您使用的服务，将这里替换成对应的凭证和地址
   API_KEY = "YOUR_API_KEY"
   BASE_URL = "YOUR_BASE_URL"
   MODEL_ID = "YOUR_MODEL_ID"
   TAVILY_API_KEY="YOUR_Tavily_KEY"
   os.environ['TAVILY_API_KEY'] = "YOUR_TAVILY_API_KEY"
   
   llm = OpenAICompatibleClient(
       model=MODEL_ID,
       api_key=API_KEY,
       base_url=BASE_URL
   )
   
   # --- 2. 初始化 ---
   user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
   prompt_history = [f"用户请求: {user_prompt}"]
   
   print(f"用户输入: {user_prompt}\n" + "="*40)
   
   # --- 3. 运行主循环 ---
   for i in range(5): # 设置最大循环次数
       print(f"--- 循环 {i+1} ---\n")
       
       # 3.1. 构建Prompt
       full_prompt = "\n".join(prompt_history)
       
       # 3.2. 调用LLM进行思考
       llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
       # 模型可能会输出多余的Thought-Action，需要截断
       match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
       if match:
           truncated = match.group(1).strip()
           if truncated != llm_output.strip():
               llm_output = truncated
               print("已截断多余的 Thought-Action 对")
       print(f"模型输出:\n{llm_output}\n")
       prompt_history.append(llm_output)
       
       # 3.3. 解析并执行行动
       action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
       if not action_match:
           print("解析错误:模型输出中未找到 Action。")
           break
       action_str = action_match.group(1).strip()
   
       if action_str.startswith("finish"):
           final_answer = re.search(r'finish\(answer="(.*)"\)', action_str).group(1)
           print(f"任务完成，最终答案: {final_answer}")
           break
       
       tool_name = re.search(r"(\w+)\(", action_str).group(1)
       args_str = re.search(r"\((.*)\)", action_str).group(1)
       kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
   
       if tool_name in available_tools:
           observation = available_tools[tool_name](**kwargs)
       else:
           observation = f"错误:未定义的工具 '{tool_name}'"
   
       # 3.4. 记录观察结果
       observation_str = f"Observation: {observation}"
       print(f"{observation_str}\n" + "="*40)
       prompt_history.append(observation_str)
   ```

## workflow和agent的差异

workflow是让AI按部就班地执行指令，而agent则是赋予AI自由度去自主达成目标

workflow的核心是对一系列任务或步骤进行预先定义的、结构话的编排。本质上是一个精准的、静态的流程图

agent是一个具备自主性的、以目标导向的系统。它不仅仅是执行预设指令，而是能够在一定程度上理解环境、进行推理、制定计划，动态地采取行动以达成最终目标

# 大语言模型基础

语言模型（LM）的根本任务是计算一个词序列（即一个句子）出现的概率

## 语言模型与transformer架构

### 发展历程

1. 统计语言模型与N-GRAM思想
2. 神经网络语言模型与词嵌入
3. 循环神经网络(RNN)与长时记忆网络(LSTM)

### transformer架构解析

1. Encoder-Decoder整体结构:编码器的任务是"理解"输入的整个句子,解码器的任务是"生成"目标句子
2. 从自注意力到多头注意力:query,key,value
3. 前馈神经网络
4. 残差链接与层归一化

### Decoder-only架构

语言的核心任务,是预测下一个最有可能出现的词

工作模式:自回归(掩码自注意力机制)

优势:

- 训练目标统一
- 结构简单,易于扩展
- 天然适合生成任务

## 与大语言模型交互

### 提示词工程

模型采样参数:temperature,Top-k,Top-p

零样本,单样本,少样本提示

指令调优:极大地简化了我们与模型交互的方式,使得直接,清晰的自然语言指令成为可能

基础提示技巧:角色扮演,上下文示例

思维链(COT):添加一句引导语,如"请逐步思考"或"Let's think step by step"

### 文本分词

分词（tokenization）：计算机本质上只能理解数字，所以在将自然语言喂给大语言模型之前，必须先将其转换成模型能够处理的数字格式。这个将文本序列转化成数字序列的过程，称为分词

分词器：定义一套规则，将原始文本切分成一个个最小的单元，即**词元**（Token）

分词器对开发者的意义：

- 上下文窗口限制：精确管理输入长度，避免超出上下文限制
- API成本：节省token，预估和控制智能体运行成本
- 模型表现的异常：优势模型的奇怪表现根源在于分词

### 调用开源大语言模型

### 模型的选择

关键考量因素：性能与能力，成本，速度（延迟），上下文窗口，部署方式，生态与工具链，可微调与定制化，安全性与伦理

## 缩放法则与局限性

### 模型幻觉

根据幻觉的表现形式可以分为多个种类，如：

- 事实性幻觉：生成与现实世界事实不符的信息
- 忠实性幻觉：在文本摘要、翻译等任务中，生成的内容未能忠实地反应源文本的含义
- 内在幻觉：模型生成的内容与输入信息相矛盾

幻觉的产生是多方面的因素：

- 首先，训练数据中可能包含错误或矛盾的信息。

- 其次，模型的自回归生成机制决定了它只是在预测下一个最可能的词元，而没有内置的事实核查模块。
- 最后，在面对需要复杂推理的任务时，模型可能会在逻辑链条中出错，从而“编造”出错误的结论

大语言模型还面临知识时效性不足和训练数据中存在的偏见等挑战。大语言模型的能力来源于其训练数据。这意味着模型所掌握的知识是其训练数据收集时的最新材料。对于在此日期之后发生的事情、新出现的概念或最新的事实，模型将无法感知或正确回答。于此同时训练数据往往包含了人类社会的各种偏见和刻板印象。

检测和缓解幻觉的方法：

- 数据层面：通过高质量数据清洗、引入事实性知识以及强化学习和人类反馈（RLHF）等方式，从源头减少幻觉
- 模型层面：探索新的模型架构，或让模型能够表达其对生成内容的不确定性
- 推理与生成层面：
  1. 检索增强生成（RAG）：这是目前缓解幻觉的有效方法之一。RAG系统通过在生成之前从外部知识库（如文档数据库、网页）中检索相关信息，然后将检索到的信息作为上下文，引导模型生成基于事实的回答
  2. 多步推理与验证：引导模型进行多步推理，并在每一步进行自我检查或外部验证
  3. 引入外部工具：允许模型调用外部工具（如搜索引擎、计算器、代码解释器）来获取实时信息或进行精确计算

------

# 第五章 基于低代码平台的智能体搭建  
## —— 零代码到生产级落地的完整路径

---

## 5.1 平台化构建的兴起

### 5.1.1 为何需要低代码平台
- 降门槛：产品经理、运营人员无需写代码即可参与构建。  
- 提效率：拖拽式编排把“天”级开发压缩到“小时”级。  
- 可视化：运行轨迹、数据流、异常节点一目了然，调试成本骤降。  
- 沉淀最佳实践：官方模板、插件市场、提示词库直接复用，避免重复踩坑。  

### 5.1.2 主流平台速览
1. **Coze**：字节跳动出品，零代码+丰富插件+一键发布多平台（微信、飞书、抖音）。  
2. **Dify**：开源企业级，全栈 LLMOps，支持私有化与 MCP 生态。  
3. **n8n**：开源自动化工作流，数百个 SaaS 节点，把 LLM 当成“高级函数”嵌入现有业务。  

---

## 5.2 平台一：Coze —— 零代码极速验证

### 5.2.1 核心模块
- 插件商店：RSS、GitHub、Arxiv、天气、股票等 1000+ 即插即用。  
- 对话流 / 工作流：可视化连线，支持条件分支、循环、变量。  
- 发布渠道：同一 Bot 可一键同步到 12 个终端，含微信客服、抖音小程序。  

### 5.2.2 实战：每日 AI 简报助手
1. 信息源  
   - RSS 插件拉取 36Kr、虎嗅、IT 之家 24 h 新闻。  
   - GitHub 插件按“AI”关键词取最新 10 个仓库。  
   - Arxiv 插件取 5 篇最新摘要。  
2. 提示词工程  
   - System：设定“资深科技编辑”角色，强制 Emoji 开头、附原文链接、排除广告。  
   - User：规定 10 条新闻 + 5 篇论文 + 5 个开源项目，总量可控。  
3. 结果  
   - 3 min 完成编排，生成格式统一、可直接推送到飞书群的日报。  

### 5.2.3 优势与局限
- 优势：真正零代码、插件多、发布快。  
- 局限：暂不支持 MCP 协议；导出为 zip 而非标准 JSON，跨平台迁移难。  

---

## 5.3 平台二：Dify —— 开源企业级全栈

### 5.3.1 架构与生态
- Marketplace：8677+ 插件，含 Google Search、Notion、高德 MCP 等。  
- 模型中立：OpenAI、Anthropic、DeepSeek、Llama 统一接口，一键切换。  
- 部署灵活：官方 SaaS / Docker Compose 私有化，支持 RBAC、审计日志、AES-256。  

### 5.3.2 实战：超级智能体个人助手
1. 多智能体路由  
   - 问题分类器 → 日常助手 / 文案优化 / 生图 / 数据分析 / MCP 工具 五大子 Agent。  
2. 关键能力  
   - 文案：OpenAI 报告数据，60% 用户用于文本优化，提示词强制>500 字且给出可操作建议。  
   - 生图：豆包 Seedream 插件，支持 1:1/16:9 多比例、多模型版本。  
   - 数据分析：rookie-text2data 插件 → 自然语言转 SQL → 结果可视化（饼/柱/折线）。  
   - MCP 工具链：高德地图（路线规划）、饮食推荐（食谱+热量）、新闻（实时头条）。  
3. 效果  
   - 一人 30 min 搭建出覆盖“衣食住行+工作”的私人助理，可本地私有化部署。  

### 5.3.3 优势与局限
- 优势：全栈、插件海量、企业安全合规。  
- 局限：Python 后端性能瓶颈；高并发需额外优化；学习曲线高于 Coze。  

---

## 5.4 平台三：n8n —— 把 AI 嵌入自动化流程

### 5.4.1 核心概念
- 节点：触发器（Gmail、Webhook、定时）+ 操作节点（LLM、数据库、Slack 等）。  
- 工作流：JSON 描述的 DAG，数据以结构化格式在节点间传递。  
- AI Agent 节点：统合模型、记忆、工具，一次配置即可完成“思考-决策-调用-回复”闭环。  

### 5.4.2 实战：智能邮件助手
1. 私有知识库  
   - Code 节点写入个人工作时间、离岗策略。  
   - Embeddings → Simple Vector Store，内存级，服务重启即失。  
2. 主流程  
   - Gmail 触发 → AI Agent 节点（Gemini）→ 工具：SerpAPI（公开搜索）+ Vector Store（私有知识）。  
   - 非工作时间自动加前缀告知“我会在下一个工作日 9-17 点回复”。  
   - 最终 Gmail 节点 Send，JSON 格式输出，HTML 换行用 `<br>`。  
3. 结果  
   - 10 分钟搭建 7×24 小时客服，可私有化运行，敏感数据不出内网。  

### 5.4.3 优势与局限
- 优势：节点生态庞大、私有化简单、把 LLM 当“函数”嵌入任意流程。  
- 局限：内存存储非持久；复杂调试靠逐节点检查；多人协作无 Git 级版本控制。  

---

## 5.5 本章小结与选型指南

### 一句话记忆
- **原型验证**选 Coze：零代码、最快、插件多。  
- **企业落地**选 Dify：开源、全栈、可私有化、Marketplace 生态成熟。  
- **业务自动化**选 n8n：把 AI 当节点，串起现有 SaaS 与数据库。  
- **深度定制**回第四章：手写框架，100% 控制。

### 混合开发思维
1. 用平台跑通 MVP，验证价值。  
2. 遇到“插件缺失 / 性能瓶颈 / 深度定制”时，用代码补位。  
3. 同一系统内，平台负责标准流程，代码负责特殊逻辑，二者通过 API 或 Webhook 无缝对接。
