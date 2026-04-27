# TTS Runtime Template

下面是系统运行时上下文。你要基于分段规则，把文本整理成适合当前 TTS 请求格式的分段结果。

{{SEGMENTATION_RULES}}

只返回一个 JSON 对象，不要输出解释。

```json
{
  "segments": [
    {
      "text": "这一段最终要送去 TTS 的正文",
      "control": "这一段的说话方式",
      "pause_ms": 320,
      "emotion": "neutral",
      "cfg_value": 1.55,
      "inference_timesteps": 8
    }
  ]
}
```

## 当前 TTS 请求格式

名称：`{{REQUEST_FORMAT_NAME}}`

{{REQUEST_FORMAT_MARKDOWN}}

## 角色设定

{{SOUL_MARKDOWN}}

## 当前上下文

- 是否启用参考：`{{REFERENCE_MODE}}`
- 参考文本：`{{REFERENCE_TEXT}}`
- 用户补充 control：`{{CONTROL_HINT}}`

## 待处理文本

{{USER_TEXT}}
