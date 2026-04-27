let appConfig = null;
let configDraft = null;
let voices = [];
let generatedAudio = null;
let currentSegments = [];
let currentSegmentIndex = 0;
let hiddenReferenceText = null;
let lastPresetIndex = -1;
let statusToastTimer = null;

const RANDOM_PRESETS = [
  {
    label: "夜色独白",
    text: "夜已经很深了，窗外的风从树梢上慢慢滑下来，像有人把一天的喧闹轻轻折好，放回了远处的灯光里。",
    control: "低声、靠近麦克风、慢节奏、尾音轻收、带一点夜晚的松弛感",
  },
  {
    label: "温柔旁白",
    text: "如果你愿意，就把今天暂时放下，让呼吸先慢一点，再把注意力交给此刻最安静的那一束光。",
    control: "温柔叙述、呼吸平稳、节奏舒展、重点词轻柔强调",
  },
  {
    label: "故事开场",
    text: "那天黄昏来得比平时更早一些，街道还没完全暗下去，第一盏路灯已经悄悄亮了。",
    control: "有画面感的故事开场、中速、语气自然、层次清楚",
  },
  {
    label: "英文口播",
    text: "Some voices do not need to be loud. They just need enough space to land gently and stay with you for a while.",
    control: "soft spoken English, warm tone, medium slow pace, clear phrasing, intimate delivery",
  },
];

const player = document.getElementById("player");
const configDialog = document.getElementById("configDialog");
const promptDialog = document.getElementById("promptDialog");
const statusToast = document.getElementById("statusToast");
const segmentLoader = document.getElementById("segmentLoader");
const audioLoader = document.getElementById("audioLoader");
const segmentEmpty = document.getElementById("segmentEmpty");
const segmentViewer = document.getElementById("segmentViewer");
const audioEmpty = document.getElementById("audioEmpty");
const audioReady = document.getElementById("audioReady");
const voiceAudioPreview = document.getElementById("voiceAudioPreview");
const voiceAudioPreviewWrap = document.getElementById("voiceAudioPreviewWrap");

function setStatus(message) {
  statusToast.textContent = message;
  statusToast.classList.add("visible");
  window.clearTimeout(statusToastTimer);
  statusToastTimer = window.setTimeout(() => {
    statusToast.classList.remove("visible");
  }, 2600);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

function selectedVoice() {
  return voices.find((voice) => voice.id === document.getElementById("voiceSelect").value) || null;
}

function setSegmentLoading(loading) {
  segmentLoader.classList.toggle("hidden", !loading);
}

function setAudioLoading(loading) {
  audioLoader.classList.toggle("hidden", !loading);
}

function setAudioState(hasAudio) {
  audioEmpty.classList.toggle("hidden", hasAudio);
  audioReady.classList.toggle("hidden", !hasAudio);
}

function syncSegmentFormToState() {
  if (!currentSegments.length) return;
  currentSegments[currentSegmentIndex] = {
    ...currentSegments[currentSegmentIndex],
    text: document.getElementById("segmentText").value,
    control: document.getElementById("segmentControl").value,
    emotion: document.getElementById("segmentEmotion").value,
  };
}

function renderSegmentPage() {
  const hasSegments = currentSegments.length > 0;
  segmentEmpty.classList.toggle("hidden", hasSegments);
  segmentViewer.classList.toggle("hidden", !hasSegments);
  if (!hasSegments) return;

  const seg = currentSegments[currentSegmentIndex];
  document.getElementById("segmentPageInfo").textContent = `第 ${currentSegmentIndex + 1} 段 / 共 ${currentSegments.length} 段`;
  document.getElementById("segmentEmotion").value = seg.emotion || "neutral";
  document.getElementById("segmentControl").value = seg.control || "";
  document.getElementById("segmentText").value = seg.text || "";
  document.getElementById("prevSegmentBtn").disabled = currentSegmentIndex === 0;
  document.getElementById("nextSegmentBtn").disabled = currentSegmentIndex === currentSegments.length - 1;
}

function setSegments(segments) {
  currentSegments = segments.map((item, index) => ({
    index: index + 1,
    text: item.text || "",
    control: item.control || "",
    pause_ms: item.pause_ms || 300,
    emotion: item.emotion || "neutral",
    cfg_value: item.cfg_value ?? null,
    inference_timesteps: item.inference_timesteps ?? null,
  }));
  currentSegmentIndex = 0;
  renderSegmentPage();
}

function collectSegments() {
  syncSegmentFormToState();
  return currentSegments.map((segment, index) => ({
    ...segment,
    index: index + 1,
  }));
}

async function fileToDataUrl(file) {
  if (!file) return null;
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function fileToText(file) {
  if (!file) return "";
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsText(file, "utf-8");
  });
}

function cloneConfig(value) {
  return JSON.parse(JSON.stringify(value));
}

function hydrateConfig(config) {
  appConfig = config;
  configDraft = cloneConfig(config);
  fillConfigForm(configDraft);
}

function fillConfigForm(config) {
  const llmProviderId = config.llm.active_provider;
  const ttsAdapterId = config.tts.active_adapter;
  const llmSelect = document.getElementById("llmProvider");
  const ttsSelect = document.getElementById("ttsAdapter");

  llmSelect.innerHTML = Object.keys(config.llm.providers).map((key) => `<option value="${key}">${key}</option>`).join("");
  ttsSelect.innerHTML = Object.keys(config.tts.adapters).map((key) => `<option value="${key}">${key}</option>`).join("");
  llmSelect.value = llmProviderId;
  ttsSelect.value = ttsAdapterId;

  const llm = config.llm.providers[llmProviderId];
  const tts = config.tts.adapters[ttsAdapterId];
  document.getElementById("llmBaseUrl").value = llm.base_url;
  document.getElementById("llmApiKey").value = llm.api_key;
  document.getElementById("llmModel").value = llm.model || "";
  document.getElementById("ttsModelPath").value = tts.model_path;
  document.getElementById("ttsCfg").value = tts.default_cfg_value;
  document.getElementById("ttsSteps").value = tts.default_inference_timesteps;
  updateModelPathHint(tts.model_path);
}

function syncDraftFromForm() {
  if (!configDraft) return null;
  const llmProviderId = document.getElementById("llmProvider").value;
  const ttsAdapterId = document.getElementById("ttsAdapter").value;
  configDraft.llm.active_provider = llmProviderId;
  configDraft.tts.active_adapter = ttsAdapterId;
  configDraft.llm.providers[llmProviderId].base_url = document.getElementById("llmBaseUrl").value.trim();
  configDraft.llm.providers[llmProviderId].api_key = document.getElementById("llmApiKey").value.trim();
  configDraft.llm.providers[llmProviderId].model = document.getElementById("llmModel").value.trim();
  configDraft.tts.adapters[ttsAdapterId].model_path = document.getElementById("ttsModelPath").value.trim();
  configDraft.tts.adapters[ttsAdapterId].default_cfg_value = Number(document.getElementById("ttsCfg").value || 1.45);
  configDraft.tts.adapters[ttsAdapterId].default_inference_timesteps = Number(document.getElementById("ttsSteps").value || 8);
  return configDraft;
}

function voiceAudioUrl(path) {
  if (!path) return null;
  const trimmed = String(path).replace(/\\/g, "/").replace(/^\/+/, "");
  const normalized = trimmed.startsWith("voice/") ? trimmed.slice("voice/".length) : trimmed;
  return `/voice-files/${normalized}`;
}

async function loadVoices() {
  const payload = await fetchJson("/api/voices");
  voices = payload.voices;
  const select = document.getElementById("voiceSelect");
  select.innerHTML = `<option value="">不使用预设音色</option>` + voices.map((voice) => (
    `<option value="${voice.id}">[${voice.scope}] ${voice.name}</option>`
  )).join("");
}

function resetVoicePreview() {
  voiceAudioPreview.pause();
  voiceAudioPreview.removeAttribute("src");
  voiceAudioPreview.load();
  voiceAudioPreviewWrap.classList.add("hidden");
}

function updateReferenceState() {
  const voice = selectedVoice();
  const customFile = document.getElementById("referenceAudio").files?.[0];
  const summary = document.getElementById("referenceSummary");
  const fileName = document.getElementById("referenceAudioName");
  const controlNote = document.getElementById("controlNote");
  const referenceTextInput = document.getElementById("referenceTextInput");

  if (customFile) {
    summary.textContent = "已上传参考音频，请填写与之对应的参考文字。";
    fileName.textContent = customFile.name;
    referenceTextInput.value = referenceTextInput.value || "";
    hiddenReferenceText = referenceTextInput.value.trim() || null;
    resetVoicePreview();
    controlNote.textContent = "启用参考音频和参考文字时，这里的 control 不会直接拼进 VoxCPM 正文。";
    return;
  }

  if (voice) {
    summary.textContent = `当前使用音色“${voice.name}”的参考材料。`;
    referenceTextInput.value = voice.reference_text || "";
    hiddenReferenceText = referenceTextInput.value.trim() || null;
    controlNote.textContent = "启用参考音频和参考文字时，这里的 control 不会直接拼进 VoxCPM 正文。";

    if (voice.reference_audio_path) {
      fileName.textContent = voice.reference_audio_path.split("/").pop() || "reference.wav";
      const audioUrl = voiceAudioUrl(voice.reference_audio_path);
      if (audioUrl) {
        voiceAudioPreview.src = audioUrl;
        voiceAudioPreviewWrap.classList.remove("hidden");
      } else {
        resetVoicePreview();
      }
    } else {
      fileName.textContent = "该音色没有参考音频文件";
      resetVoicePreview();
    }
    return;
  }

  summary.textContent = "也可以直接上传参考音频，并手动填写对应参考文字。";
  fileName.textContent = "未选择文件";
  referenceTextInput.value = "";
  hiddenReferenceText = null;
  resetVoicePreview();
  controlNote.textContent = "启用参考音频和参考文字时，这里的 control 不会直接拼进 VoxCPM 正文。";
}

function updateModelPathHint(value) {
  const hint = document.getElementById("ttsModelPathHint");
  const trimmed = (value || "").trim();
  if (!trimmed) {
    hint.textContent = "例如：openbmb/VoxCPM2 或 E:\\models\\VoxCPM2";
    return;
  }
  if (/^[a-zA-Z]:\\|^\//.test(trimmed)) {
    hint.textContent = `当前会直接读取本地模型目录：${trimmed}`;
    return;
  }
  hint.textContent = `当前会把它当作 Hugging Face 仓库名：${trimmed}`;
}

function applyRandomPreset() {
  let index = Math.floor(Math.random() * RANDOM_PRESETS.length);
  if (RANDOM_PRESETS.length > 1) {
    while (index === lastPresetIndex) {
      index = Math.floor(Math.random() * RANDOM_PRESETS.length);
    }
  }
  lastPresetIndex = index;
  const preset = RANDOM_PRESETS[index];
  document.getElementById("sourceText").value = preset.text;
  document.getElementById("controlHint").value = preset.control;
  document.getElementById("presetPreview").textContent = `${preset.label} · ${preset.control}`;
  setStatus("已填入一组随机灵感。");
}

function dataUrlToBlob(dataUrl) {
  const [header, base64Part] = dataUrl.split(",");
  const mime = /data:(.*?);base64/.exec(header)?.[1] || "audio/wav";
  const binary = atob(base64Part);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: mime });
}

async function saveAudioToDisk() {
  if (!generatedAudio?.audio_data_url) {
    setStatus("还没有可保存的音频。");
    return;
  }
  const blob = dataUrlToBlob(generatedAudio.audio_data_url);
  const filename = `voxcpm2_${Date.now()}.wav`;
  if ("showSaveFilePicker" in window) {
    const handle = await window.showSaveFilePicker({
      suggestedName: filename,
      types: [{ description: "WAV Audio", accept: { "audio/wav": [".wav"] } }],
    });
    const writable = await handle.createWritable();
    await writable.write(blob);
    await writable.close();
    document.getElementById("savedAudioPath").textContent = `已保存：${filename}`;
    setStatus("音频已保存到本地。");
    return;
  }
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
  document.getElementById("savedAudioPath").textContent = `已触发下载：${filename}`;
  setStatus("已触发浏览器下载。");
}

document.getElementById("randomPresetBtn").addEventListener("click", applyRandomPreset);

document.getElementById("soulFile").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  document.getElementById("soulOverride").value = await fileToText(file);
  setStatus("角色设定已载入。");
});

document.getElementById("voiceSelect").addEventListener("change", () => {
  document.getElementById("referenceAudio").value = "";
  updateReferenceState();
});

document.getElementById("referenceAudioPicker").addEventListener("click", () => {
  document.getElementById("referenceAudio").click();
});

document.getElementById("referenceAudio").addEventListener("change", () => {
  if (document.getElementById("referenceAudio").files?.length) {
    document.getElementById("voiceSelect").value = "";
  }
  updateReferenceState();
});

document.getElementById("referenceTextInput").addEventListener("input", () => {
  hiddenReferenceText = document.getElementById("referenceTextInput").value.trim() || null;
});

document.getElementById("editPromptBtn").addEventListener("click", async () => {
  const payload = await fetchJson("/api/prompt");
  document.getElementById("promptEditor").value = payload.content || "";
  promptDialog.showModal();
});

document.getElementById("cancelPromptBtn").addEventListener("click", () => {
  promptDialog.close();
});

document.getElementById("savePromptBtn").addEventListener("click", async () => {
  await fetchJson("/api/prompt", {
    method: "PUT",
    body: JSON.stringify({ content: document.getElementById("promptEditor").value }),
  });
  promptDialog.close();
  setStatus("分段规则已保存。");
});

document.getElementById("openConfigBtn").addEventListener("click", () => {
  fillConfigForm(configDraft || appConfig);
  configDialog.showModal();
});

document.getElementById("cancelConfigBtn").addEventListener("click", () => {
  configDialog.close();
});

document.getElementById("saveConfigBtn").addEventListener("click", async () => {
  syncDraftFromForm();
  appConfig = (await fetchJson("/api/config", {
    method: "PUT",
    body: JSON.stringify(configDraft),
  })).config;
  configDraft = cloneConfig(appConfig);
  configDialog.close();
  setStatus("设置已保存。");
});

document.getElementById("llmProvider").addEventListener("change", () => {
  syncDraftFromForm();
  fillConfigForm(configDraft);
});

document.getElementById("ttsAdapter").addEventListener("change", () => {
  syncDraftFromForm();
  fillConfigForm(configDraft);
});

document.getElementById("ttsModelPath").addEventListener("input", (event) => {
  updateModelPathHint(event.target.value);
});

document.getElementById("pickModelFolderBtn").addEventListener("click", async () => {
  try {
    const payload = await fetchJson("/api/system/pick-model-folder", {
      method: "POST",
      body: JSON.stringify({ title: "选择 VoxCPM 模型文件夹" }),
    });
    if (payload.cancelled || !payload.selected_path) {
      setStatus("已取消选择模型文件夹。");
      return;
    }
    document.getElementById("ttsModelPath").value = payload.selected_path;
    updateModelPathHint(payload.selected_path);
    setStatus("模型文件夹已选中。");
  } catch (error) {
    console.error(error);
    setStatus(`打开文件夹选择器失败：${error.message}`);
  }
});

document.getElementById("downloadModelBtn").addEventListener("click", async () => {
  try {
    syncDraftFromForm();
    const payload = await fetchJson("/api/system/download-model", {
      method: "POST",
      body: JSON.stringify({
        tts_adapter_id: configDraft.tts.active_adapter,
        config_override: configDraft,
        repo_id: document.getElementById("ttsModelPath").value.trim(),
      }),
    });
    if (payload.local_model_path) {
      document.getElementById("ttsModelPath").value = payload.local_model_path;
      updateModelPathHint(payload.local_model_path);
    }
    setStatus("模型下载完成，已回填本地路径。");
  } catch (error) {
    console.error(error);
    setStatus(`下载模型失败：${error.message}`);
  }
});

document.getElementById("prevSegmentBtn").addEventListener("click", () => {
  syncSegmentFormToState();
  if (currentSegmentIndex > 0) {
    currentSegmentIndex -= 1;
    renderSegmentPage();
  }
});

document.getElementById("nextSegmentBtn").addEventListener("click", () => {
  syncSegmentFormToState();
  if (currentSegmentIndex < currentSegments.length - 1) {
    currentSegmentIndex += 1;
    renderSegmentPage();
  }
});

document.getElementById("clearSegmentsBtn").addEventListener("click", () => {
  setSegments([]);
  setStatus("当前分段已清空。");
});

async function runSegmentation() {
  const text = document.getElementById("sourceText").value.trim();
  if (!text) {
    setStatus("请先输入文本。");
    return;
  }
  if (document.getElementById("referenceAudio").files?.[0] && !document.getElementById("referenceTextInput").value.trim()) {
    setStatus("上传参考音频时，请同时填写参考文字。");
    return;
  }
  syncDraftFromForm();
  setSegmentLoading(true);
  try {
    const payload = await fetchJson("/api/segment", {
      method: "POST",
      body: JSON.stringify({
        text,
        llm_provider_id: configDraft.llm.active_provider,
        tts_adapter_id: configDraft.tts.active_adapter,
        config_override: configDraft,
        soul_override: document.getElementById("soulOverride").value.trim() || null,
        selected_voice_id: selectedVoice()?.id || null,
        reference_text: hiddenReferenceText,
        control_hint: document.getElementById("controlHint").value.trim(),
        reference_mode: Boolean(selectedVoice()?.reference_audio_path || document.getElementById("referenceAudio").files?.[0]),
      }),
    });
    setSegments(payload.segments);
    setStatus("分段完成。");
  } finally {
    setSegmentLoading(false);
  }
}

async function runGenerate() {
  const text = document.getElementById("sourceText").value.trim();
  const segments = currentSegments.length
    ? collectSegments()
    : [{
        index: 1,
        text,
        control: document.getElementById("controlHint").value.trim(),
        pause_ms: 300,
        emotion: "neutral",
        cfg_value: null,
        inference_timesteps: null,
      }];
  if (!text) {
    setStatus("请先输入文本。");
    return;
  }
  if (document.getElementById("referenceAudio").files?.[0] && !document.getElementById("referenceTextInput").value.trim()) {
    setStatus("上传参考音频时，请同时填写参考文字。");
    return;
  }
  syncDraftFromForm();
  setAudioLoading(true);
  try {
    const payload = await fetchJson("/api/generate", {
      method: "POST",
      body: JSON.stringify({
        text,
        segments,
        llm_provider_id: configDraft.llm.active_provider,
        tts_adapter_id: configDraft.tts.active_adapter,
        config_override: configDraft,
        selected_voice_id: selectedVoice()?.id || null,
        reference_text: hiddenReferenceText,
        reference_audio_data_url: await fileToDataUrl(document.getElementById("referenceAudio").files?.[0]),
        control_hint: document.getElementById("controlHint").value.trim(),
        cfg_value: Number(document.getElementById("ttsCfg").value || 1.45),
        inference_timesteps: Number(document.getElementById("ttsSteps").value || 8),
      }),
    });
    generatedAudio = payload;
    player.src = payload.audio_data_url;
    setAudioState(true);
    document.getElementById("savedAudioPath").textContent = "音频已就绪，你可以试听或手动保存。";
    setStatus("音频生成完成。");
  } finally {
    setAudioLoading(false);
  }
}

document.getElementById("segmentBtn").addEventListener("click", runSegmentation);
document.getElementById("generateBtn").addEventListener("click", runGenerate);
document.getElementById("generateDirectBtn").addEventListener("click", runGenerate);

document.getElementById("saveAudioBtn").addEventListener("click", async () => {
  try {
    await saveAudioToDisk();
  } catch (error) {
    console.error(error);
    setStatus(`保存音频失败：${error.message}`);
  }
});

document.getElementById("saveVoiceBtn").addEventListener("click", async () => {
  if (!generatedAudio?.audio_data_url) {
    setStatus("还没有可保存的音色。");
    return;
  }
  const name = window.prompt("给这个音色起个名字");
  if (!name) return;
  const payload = await fetchJson("/api/voices/save", {
    method: "POST",
    body: JSON.stringify({
      name,
      audio_data_url: generatedAudio.audio_data_url,
      reference_text: hiddenReferenceText || document.getElementById("sourceText").value.trim(),
      control_hint: document.getElementById("controlHint").value.trim(),
      description: "由当前页面试听结果保存",
    }),
  });
  await loadVoices();
  setStatus(`新音色已保存：${payload.voice.name}`);
});

async function init() {
  const config = await fetchJson("/api/config");
  hydrateConfig(config);
  await loadVoices();
  updateReferenceState();
  setSegments([]);
  setAudioState(false);
}

init().catch((error) => {
  console.error(error);
  setStatus(`初始化失败：${error.message}`);
});
