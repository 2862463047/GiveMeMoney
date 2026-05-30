# GiveMeMoney - 项目结构与代码分析

## 一、项目概述

**GiveMeMoney** 是一个基于 AI 的自动化长视频生成平台，核心功能是将文字创意/剧本转换为完整的视频。项目使用 Python 异步编程 + LangChain 框架 + Flask Web 服务，通过调用多种 LLM 模型（DeepSeek、Gemini、Qwen-VL 等）和图像/视频生成 API 来实现从创意到视频的全流程自动化。

- **项目名称**: autolongvideogeneration
- **Python 版本**: >=3.12
- **包管理**: uv (pyproject.toml)
- **核心框架**: LangChain、Flask、Pydantic
- **视频处理**: moviepy、opencv-python、scenedetect

---

## 二、目录结构总览

```
GiveMeMoney/
├── app.py                          # Flask Web 应用入口
├── main_idea2video.py               # Idea→视频 命令行入口
├── main_script2video.py             # 剧本→视频 命令行入口
├── pyproject.toml                   # 项目配置与依赖
├── uv.lock                          # 依赖锁定文件
│
├── configs/                         # 配置文件目录
│   ├── idea2video.yaml              # Idea2Video 配置 (Gemini)
│   ├── idea2video_deepseek.yaml     # Idea2Video 配置 (DeepSeek)
│   └── script2video.yaml            # Script2Video 配置
│
├── pipelines/                       # 核心管道（业务流程编排）
│   ├── __init__.py
│   ├── idea2video_pipeline.py       # Idea→视频 主管道
│   ├── idea2video_pipeline_deprecated.py  # 废弃版本
│   ├── script2video_pipeline.py     # 剧本→视频 主管道
│   └── novel2movie_pipeline.py     # 小说→电影 管道（未完成）
│
├── agents/                          # AI Agent（功能模块）
│   ├── __init__.py
│   ├── screenwriter.py              # 编剧：创意→故事→剧本
│   ├── character_extractor.py       # 角色提取器
│   ├── character_portraits_generator.py  # 角色肖像图生成
│   ├── storyboard_artist.py         # 分镜设计师
│   ├── camera_image_generator.py    # 相机树/过渡视频生成
│   ├── reference_image_selector.py  # 参考图选择器（多模态）
│   ├── best_image_selector.py       # 最佳图像选择器
│   ├── script_planner.py            # 剧本规划器（叙事/动作/蒙太奇）
│   ├── script_enhancer.py           # 剧本增强器
│   ├── novel_compressor.py          # 小说压缩器
│   ├── scene_extractor.py           # 场景提取器
│   ├── event_extractor.py           # 事件提取器
│   └── global_information_planner.py # 全局信息规划器（角色合并）
│
├── interfaces/                      # Pydantic 数据模型
│   ├── __init__.py
│   ├── character.py                 # 角色模型（场景/事件/小说级）
│   ├── scene.py                     # 场景模型
│   ├── camera.py                    # 相机模型
│   ├── shot_description.py          # 镜头描述模型
│   ├── frame.py                     # 帧模型
│   ├── image_output.py              # 图像输出封装
│   ├── video_output.py              # 视频输出封装
│   ├── event.py                     # 事件模型
│   └── environment.py               # 环境/场景设定模型
│
├── tools/                           # 外部 API 工具封装
│   ├── __init__.py
│   ├── image_generator_nanobanana_google_api.py    # Google NanoBanana 图像生成
│   ├── image_generator_nanobanana_yunwu_api.py     # 云雾 NanoBanana 图像生成
│   ├── image_generator_doubao_seedream_yunwu_api.py # 豆包 Seedream 图像生成
│   ├── wuyinkeji_nanoBanana_api.py                 # 无际科技 NanoBanana 图像生成
│   ├── video_generator_veo_google_api.py           # Google Veo 视频生成
│   ├── video_generator_veo_yunwu_api.py            # 云雾 Veo 视频生成
│   ├── video_generator_doubao_seedance_yunwu_api.py # 豆包 Seedance 视频生成
│   ├── wuyinkeji_sora2_api.py                      # 无际科技 Sora2 视频生成
│   └── reranker_bge_silicon_api.py                 # BGE Reranker (RAG)
│
├── utils/                           # 工具函数
│   ├── __init__.py
│   ├── image.py                     # 图像工具（下载/Base64转换）
│   ├── video.py                     # 视频下载工具
│   ├── log_handler.py               # Web 日志处理器
│   ├── rate_limiter.py              # API 速率限制器
│   ├── retry.py                     # 重试工具
│   └── timer.py                     # 计时器（装饰器/上下文）
│
├── templates/                       # 前端模板
│   └── index.html                   # 主页面（科技风格 UI）
│
├── static/                          # 前端静态资源
│   ├── style.css                    # 样式表（赛博朋克/科技风格）
│   └── script.js                    # 前端交互逻辑 (jQuery)
│
├── working_dir_idea2video/          # 工作输出目录
│   ├── ebeda45d.../                 # UUID 命名的项目目录
│   │   ├── story.txt                # 生成的完整故事
│   │   ├── characters.json          # 角色提取结果
│   │   ├── script.json              # 分场景剧本
│   │   ├── character_portraits_registry.json  # 角色肖像注册表
│   │   └── scene_0/                 # 场景目录
│   │       ├── storyboard.json      # 分镜设计
│   │       ├── camera_tree.json     # 相机依赖树
│   │       └── shots/0/             # 镜头目录
│   │           ├── shot_description.json  # 镜头详细描述
│   │           ├── first_frame_selector_output.json  # 首帧参考图选择
│   │           ├── first_frame.png  # 生成的首帧图
│   │           └── video.mp4        # 生成的视频
│   └── idea2video/                  # 固定名称的项目目录
│
├── assets/                          # 资源文件
│   └── GiveMeMoney_technical_report.pdf
└── generated_images/                # 示例生成图片存放目录
```

---

## 三、核心管道 (Pipelines) 详解

### 3.1 Idea2VideoPipeline（创意→视频）

**文件**: `pipelines/idea2video_pipeline.py`

**流程**:
1. **develop_story()**: 调用 Screenwriter 将创意扩展为完整故事 (story.txt)
2. **extract_characters()**: 调用 CharacterExtractor 从故事提取角色 (characters.json)
3. **generate_character_portraits()**: 为每个角色生成前/侧/背三视角肖像图
4. **write_script_based_on_story()**: 将故事改写为分场景剧本 (script.json)
5. **foreach scene**: 调用 Script2VideoPipeline 为每个场景生成视频
6. **concatenate_videoclips**: 拼接所有场景视频为 final_video.mp4

**工厂方法 `init_from_config()`**:
- 从 YAML 配置文件读取 chat_model、mllm_model、image_generator、video_generator 的配置
- 使用 `importlib` 动态加载图像/视频生成器类
- 为每个服务创建独立的 RateLimiter

### 3.2 Script2VideoPipeline（剧本→视频）

**文件**: `pipelines/script2video_pipeline.py`

**流程**:
1. **extract_characters()**: 从剧本提取角色
2. **generate_character_portraits()**: 生成角色肖像（如果父管道未完成）
3. **design_storyboard()**: StoryboardArtist 设计分镜 (storyboard.json)
4. **decompose_visual_descriptions()**: 将镜头分解为首帧/末帧/运动描述
5. **construct_camera_tree()**: 构建相机依赖树（父子相机关系）
6. **并行帧生成**: 为每个相机生成首帧和末帧
7. **并行视频生成**: 为每个镜头生成视频
8. **concatenate_videoclips**: 拼接所有镜头视频

**关键数据结构**:
- `frame_events`: `Dict[int, Dict[str, asyncio.Event]` — 控制帧生成的异步事件
- `shot_desc_events`: 镜头描述完成事件
- `character_portrait_events`: 角色肖像完成事件

### 3.3 Novel2MoviePipeline（小说→电影，未完成）

**文件**: `pipelines/novel2movie_pipeline.py`

**流程**:
1. 压缩小说文本（分块→压缩→聚合）
2. 提取事件
3. RAG 检索相关文本块
4. 为每个事件提取场景
5. 合并角色（场景级→事件级→小说级）
6. 生成角色肖像（全局+场景特定）
7. 为每个场景生成视频

---

## 四、Agents（AI Agent 功能模块）

| Agent | 文件 | 功能 | 使用的模型 |
|-------|------|------|-----------|
| **Screenwriter** | `screenwriter.py` | 创意→完整故事；故事→分场景剧本 | chat_model |
| **CharacterExtractor** | `character_extractor.py` | 从剧本/故事提取角色（含静态/动态特征） | chat_model |
| **CharacterPortraitsGenerator** | `character_portraits_generator.py` | 生成角色正面/侧面/背面肖像图 | image_generator |
| **StoryboardArtist** | `storyboard_artist.py` | 设计分镜；将镜头描述分解为首帧/末帧/运动 | chat_model |
| **CameraImageGenerator** | `camera_image_generator.py` | 构建相机树；生成过渡视频；提取新相机视角 | chat_model + video_generator |
| **ReferenceImageSelector** | `reference_image_selector.py` | 两阶段选择参考图（文本过滤+多模态精选） | chat_model + mllm_model |
| **BestImageSelector** | `best_image_selector.py` | 从候选图中选出与参考图/文本最一致的图像 | mllm_model |
| **ScriptPlanner** | `script_planner.py` | 将简单创意扩展为剧本（叙事/动作/蒙太奇三种风格） | chat_model |
| **ScriptEnhancer** | `script_enhancer.py` | 增强剧本细节、连续性、对话自然度 | chat_model |
| **NovelCompressor** | `novel_compressor.py` | 分块压缩小说文本 | chat_model |
| **EventExtractor** | `event_extractor.py` | 从小说提取事件链 | chat_model |
| **SceneExtractor** | `scene_extractor.py` | 将事件适配为剧本场景 | chat_model |
| **GlobalInformationPlanner** | `global_information_planner.py` | 合并角色（场景→事件→小说级） | chat_model |

---

## 五、Interfaces（Pydantic 数据模型）

| 模型 | 文件 | 说明 |
|------|------|------|
| **CharacterInScene** | `character.py` | 场景中的角色：idx, identifier, 静态/动态特征, 可见性 |
| **CharacterInEvent** | `character.py` | 事件中的角色：含 active_scenes 映射 |
| **CharacterInNovel** | `character.py` | 小说中的角色：含 active_events 映射 |
| **Scene** | `scene.py` | 场景：idx, environment, characters, script |
| **Event** | `event.py` | 事件：index, description, process_chain |
| **Camera** | `camera.py` | 相机：idx, active_shot_idxs, 父子关系, missing_info |
| **ShotBriefDescription** | `shot_description.py` | 镜头简要描述：idx, cam_idx, visual_desc, audio_desc |
| **ShotDescription** | `shot_description.py` | 镜头详细描述：含首帧/末帧/运动描述, variation_type |
| **Frame** | `frame.py` | 帧：shot_idx, frame_type, cam_idx, vis_char_idxs |
| **ImageOutput** | `image_output.py` | 图像输出封装：支持 b64/url/pil/np 四种格式 |
| **VideoOutput** | `video_output.py` | 视频输出封装：支持 url/bytes 两种格式 |
| **EnvironmentInScene** | `environment.py` | 场景环境：slugline (INT./EXT.), description |

---

## 六、Tools（外部 API 封装）

### 6.1 图像生成器

| 类名 | 文件 | API 提供商 | 模型 |
|------|------|-----------|------|
| `ImageGeneratorNanobananaGoogleAPI` | `image_generator_nanobanana_google_api.py` | Google AI | gemini-2.5-flash-image |
| `ImageGeneratorNanobananaYunwuAPI` | `image_generator_nanobanana_yunwu_api.py` | 云雾 AI | gemini-2.5-flash-image-preview |
| `ImageGeneratorDoubaoSeedreamYunwuAPI` | `image_generator_doubao_seedream_yunwu_api.py` | 云雾 AI | doubao-seedream-4-0 |
| `ImageGeneratorNanobananaWuYinAPI` | `wuyinkeji_nanoBanana_api.py` | 无际科技 | nano-banana |

### 6.2 视频生成器

| 类名 | 文件 | API 提供商 | 模型 |
|------|------|-----------|------|
| `VideoGeneratorVeoGoogleAPI` | `video_generator_veo_google_api.py` | Google AI | veo-3.1-generate-preview |
| `VideoGeneratorVeoYunwuAPI` | `video_generator_veo_yunwu_api.py` | 云雾 AI | veo3.1-fast / veo2-fast-frames |
| `VideoGeneratorDoubaoSeedanceYunwuAPI` | `video_generator_doubao_seedance_yunwu_api.py` | 云雾 AI | doubao-seedance-1-0-lite |
| `VideoGeneratorSora2API` | `wuyinkeji_sora2_api.py` | 无际科技 | Sora2 |

### 6.3 其他工具

| 类名 | 文件 | 功能 |
|------|------|------|
| `RerankerBgeSiliconapi` | `reranker_bge_silicon_api.py` | 文本重排序（用于 RAG） |

---

## 七、Utils（工具函数）

| 模块 | 功能 |
|------|------|
| `image.py` | `download_image()`, `image_path_to_b64()`, `pil_to_b64()`, `save_base64_image()` |
| `video.py` | `download_video()` — 从 URL 下载视频 |
| `log_handler.py` | `WebLogHandler` — 将 logging 输出转发到 queue 供 Web 前端 SSE 推送 |
| `rate_limiter.py` | `RateLimiter` — 异步 API 速率限制（每分钟/每天） |
| `retry.py` | `after_func()` — tenacity 重试回调，记录警告日志 |
| `timer.py` | `Timer` — 装饰器/上下文管理器计时器 |

---

## 八、Web 前端（Flask + SSE）

**文件**: `app.py` + `templates/index.html` + `static/script.js` + `static/style.css`

### 后端 API 端点

| 路由 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 主页面 |
| `/api/generate` | POST | 开始生成视频（idea, user_requirement, style, work_dir） |
| `/api/logs` | GET | SSE 流式日志推送 |
| `/api/task_status` | GET | 查询任务运行状态 |
| `/api/files` | GET | 列出工作目录文件树 |
| `/api/work_dirs` | GET | 列出所有历史工作目录 |
| `/api/file/<path>` | GET | 获取文件内容（文本/JSON 等） |
| `/api/preview/<path>` | GET | 预览文件（图片/视频） |
| `/api/stats` | GET | 获取项目统计信息 |
| `/download/<path>` | GET | 下载文件 |

### 前端架构

- **布局**: 左中右三栏布局（工作目录+文件树 / 创意输入+日志 / 视频预览+统计）
- **风格**: 赛博朋克/科技风格，深色主题，霓虹光效
- **技术**: jQuery + SSE (Server-Sent Events) 实时日志推送
- **功能**:
  - 创意输入（Markdown 支持）+ 用户要求 + 风格预设
  - 实时日志查看（支持筛选、导出、复制）
  - 文件树浏览（递归展开/折叠）
  - 视频预览播放器
  - 项目统计面板
  - 历史项目列表

---

## 九、配置文件说明

配置文件位于 `configs/` 目录，YAML 格式：

### 9.1 `idea2video_deepseek.yaml`
```yaml
chat_model: deepseek-chat (OpenAI 兼容接口)
mllm_model: Qwen/Qwen3-VL-32B-Instruct (SiliconFlow)
image_generator: ImageGeneratorNanobananaWuYinAPI (无际科技)
video_generator: VideoGeneratorSora2API (无际科技)
```

### 9.2 `idea2video.yaml`
```yaml
chat_model: google/gemini-2.5-flash-lite (OpenRouter)
image_generator: ImageGeneratorNanobananaGoogleAPI
video_generator: VideoGeneratorVeoGoogleAPI
```

### 9.3 `script2video.yaml`
```yaml
chat_model: google/gemini-2.5-flash-lite (OpenRouter)
image_generator: ImageGeneratorNanobananaGoogleAPI
video_generator: VideoGeneratorVeoGoogleAPI
```

配置项支持：
- `class_path`: 动态加载工具类（`module.ClassName` 格式）
- `max_requests_per_minute` / `max_requests_per_day`: API 速率限制
- `init_args`: 传递给类构造函数的参数

---

## 十、核心设计模式与特点

### 10.1 管道模式 (Pipeline Pattern)
- `Idea2VideoPipeline` 嵌套调用 `Script2VideoPipeline`
- 每个管道步骤都有**断点续传**机制（检查已有文件，跳过已完成步骤）

### 10.2 动态加载 (Plugin Pattern)
- 图像/视频生成器通过 `importlib` 动态加载
- YAML 配置 `class_path` 指定具体实现类

### 10.3 异步事件同步
- 使用 `asyncio.Event` 协调并行任务间的依赖
- 例如：必须先生成首帧，才能生成视频

### 10.4 相机树 (Camera Tree)
- 多个镜头可能共享同一相机位
- 通过 LLM 分析镜头描述，构建父子相机依赖关系
- 子相机通过过渡视频+场景检测提取新视角

### 10.5 两阶段参考图选择
- **阶段一（文本过滤）**: 使用文本模型从大量参考图中筛选 ≤8 张
- **阶段二（多模态精选）**: 使用多模态 MLLM 模型精选最优参考图

### 10.6 内容安全处理
- 图像/视频生成失败时自动检测安全策略违规
- 自动优化提示词并重试（添加安全修饰词）
- 最终备用方案降级为通用安全提示词

### 10.7 速率限制 (Rate Limiting)
- 支持每分钟和每天两个维度的限制
- 使用异步锁确保线程安全

---

## 十一、数据流图

```
用户输入 (Idea/要求/风格)
    │
    ▼
Screenwriter.develop_story()         → story.txt
    │
    ▼
CharacterExtractor.extract_characters() → characters.json
    │
    ▼
CharacterPortraitsGenerator            → character_portraits/
    │
    ▼
Screenwriter.write_script_based_on_story() → script.json (分场景)
    │
    ▼ (foreach scene)
Script2VideoPipeline
    │
    ├── StoryboardArtist.design_storyboard()    → storyboard.json
    ├── StoryboardArtist.decompose_visual_description() → shot_description.json
    ├── CameraImageGenerator.construct_camera_tree() → camera_tree.json
    │
    ├── (parallel) ReferenceImageSelector → select → ImageGenerator → 首帧/末帧
    └── (parallel) VideoGenerator → video.mp4
    │
    ▼
moviepy.concatenate_videoclips()      → final_video.mp4
```

---

## 十二、关键依赖

| 包 | 用途 |
|----|------|
| `langchain` + `langchain-openai` | LLM 调用框架 |
| `langchain-community` | FAISS 向量存储 |
| `openai` | OpenAI 兼容 API 客户端 |
| `google-genai` | Google Gemini/Veo API |
| `flask` | Web 服务框架 |
| `moviepy` | 视频拼接 |
| `opencv-python` | 图像/视频处理 |
| `scenedetect` | 视频场景检测 |
| `pydantic` | 数据模型验证 |
| `tenacity` | 重试机制 |
| `aiohttp` | 异步 HTTP 请求 |
| `faiss-cpu` | 向量检索 |
| `Pillow` | 图像处理 |
| `chardet` | 字符编码检测 |

---

## 十三、待完成功能 (TODO)

1. **Novel2MoviePipeline** 标记为 `NOT IMPLEMENTED YET`，代码框架已搭建但未调试完成
2. **镜头音频描述** 目前仅生成文本描述，未实际合成音频
3. **角色一致性** 依赖 LLM 的参考图选择，尚无专门的 IP-Adapter/FaceID 等一致性方案
4. **相机树完整实现**: TODO 中提到需要 `State` 保存相机轨迹
5. **Reranker** 已实现但仅在 Novel2Movie 管道中使用
6. **ScriptPlanner/Enhancer** 已实现但未集成到主管道

---

*文件生成时间: 2026-05-27*
*分析文件数量: 60+ 源文件*
