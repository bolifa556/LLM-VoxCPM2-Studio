# TTS 请求格式扩展说明

当前项目已经把“调用 TTS 的请求格式”从页面逻辑里拆出来，放在 `app/tts_adapters/`。

## 当前已实现

- `voxcpm_local`
  - 文件：`app/tts_adapters/voxcpm_local.py`
  - 作用：调用本地 `VoxCPM2` 模型
  - LLM 会读取它对应的 `request_format_markdown()`，从而针对当前 TTS 格式生成更合适的分段和 control

## 当前页面交互约定

- 配置使用齿轮弹窗承载
- 弹窗内修改会立即参与当前请求
- 只有用户点击“保存并退出”时，配置才会写入 `config/app_config.json`
- 生成音频默认只用于页面预览，不自动保存到项目目录
- 用户可手动选择将音频保存到本地
- 用户也可以把当前生成结果保存成 `voice/usr` 下的新音色

## 以后怎么加新的 TTS

1. 在 `app/tts_adapters/` 下新增一个适配器文件。
2. 继承 `BaseTTSAdapter`。
3. 实现两个核心方法：
   - `request_format_markdown()`
   - `synthesize(...)`
4. 在 `app/tts_adapters/registry.py` 注册。
5. 在 `config/app_config.json` 里增加新的 adapter 配置。

## 为什么要这样拆

- 不同 TTS 的输入文本格式不一样。
- 不同 TTS 对 `control` 的支持方式不一样。
- 有的模型需要参考音频和参考文本，有的只吃文本或 API 字段。
- 让 LLM 在提示词里看到“当前请求格式”，它生成的文本和 control 会更贴近对应模型的实际要求。
- 让页面只关心用户交互，把协议差异收敛到 adapter 层。

## 推荐扩展接口

如果后续要接第三方 TTS API，建议把这些差异都放到适配器里，而不是直接改页面：

- 鉴权方式
- 请求 URL
- 请求体字段
- 文本预处理规则
- control 映射规则
- 输出音频格式
- 是否支持参考音频 / 参考文本 / 音色 ID
