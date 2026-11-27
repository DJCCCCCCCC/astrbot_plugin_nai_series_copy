const bridge = window.AstrBotPluginView;
const form = document.querySelector('#form');
const button = document.querySelector('#submit');
const status = document.querySelector('#status');
let image;
const download = document.querySelector('#download');
await bridge.ready();
const model = document.querySelector('#model');
const preset = document.querySelector('#preset');
const translator = document.querySelector('#translator');
const raw = document.querySelector('#raw');
const modeButtons = [...document.querySelectorAll('[data-request-mode]')];
const modeStatus = document.querySelector('#mode-status');
const referenceType = document.querySelector('#reference-type');
const referenceImage = document.querySelector('#reference-image');
const referenceStrength = document.querySelector('#reference-strength');
const referenceHint = document.querySelector('#reference-hint');
const clearReference = document.querySelector('#clear-reference');
const referenceNames = {character: '角色', style: '风格', 'character&style': '角色＋风格', costume: '服装'};
let options;
let requestMode;
let busy = false;
const updateReference = () => {
  const family = options.families[model.value];
  const types = family.reference_types || [];
  const selected = referenceType.value;
  const enabled = requestMode === 'native' && options.precise_reference_available && types.length > 0;
  referenceType.replaceChildren(new Option('不使用参考', ''));
  for (const type of types) referenceType.add(new Option(referenceNames[type] || type, type));
  if (selected && !types.includes(selected)) referenceType.add(new Option(`${referenceNames[selected] || selected}（当前模型不可用）`, selected));
  referenceType.value = selected;
  referenceType.disabled = !enabled || busy;
  referenceImage.disabled = !enabled || busy;
  referenceStrength.disabled = !enabled || busy || !selected;
  clearReference.disabled = busy;
  referenceStrength.max = family.reference_strength_max ?? 5;
  if (enabled) {
    const maxMiB = Math.floor(options.reference_max_bytes / 1048576);
    referenceHint.textContent = `支持 PNG、JPEG、WebP，单张不超过 ${maxMiB} MB。`;
  } else if (requestMode !== 'native') {
    referenceHint.textContent = '切换到 NovelAI 原生后可使用精准参考。';
  } else {
    referenceHint.textContent = '请在插件设置中启用支持精准参考的线路，并选择支持的模型。';
  }
};
const updateMode = () => {
  for (const modeButton of modeButtons) {
    modeButton.setAttribute('aria-pressed', String(modeButton.dataset.requestMode === requestMode));
    modeButton.disabled = busy;
  }
  updateReference();
};
const updatePresets = (selection = options.families[model.value].default_preset) => {
  const presets = options.families[model.value].presets;
  preset.replaceChildren(new Option('不使用预设', ''));
  for (const item of presets) preset.add(new Option(item.name, item.id));
  const selected = presets.find(item => item.id === selection || item.name === selection);
  preset.value = selected?.id || '';
};
try {
  options = await bridge.apiGet('canvas/options');
  requestMode = options.request_mode;
  if (!['openai', 'native'].includes(requestMode)) throw new Error('服务端未返回有效的请求模式');
  if (!options.translator_enabled) translator.options[0].text = '翻译未启用';
  for (const name of options.translators) translator.add(new Option(name, name));
  const updateModel = () => {
    const family = options.families[model.value];
    updatePresets();
    for (const key of ['steps', 'scale', 'cfg_rescale', 'sampler', 'noise_schedule']) document.getElementById(key).value = family[key];
    updateReference();
  };
  model.addEventListener('change', updateModel);
  referenceType.addEventListener('change', updateReference);
  const updateTranslator = () => { translator.disabled = raw.checked || !options.translator_enabled; };
  raw.addEventListener('change', updateTranslator);
  updateTranslator();
  updateModel();
  updateMode();
  document.addEventListener('presets-changed', async () => {
    try {
      const refreshed = await bridge.apiGet('canvas/options');
      const selection = preset.value;
      options.families = refreshed.families;
      updatePresets(selection);
      updateReference();
    } catch {
      console.warn('Could not refresh canvas presets.');
    }
  });
} catch (error) {
  status.textContent = `加载画布设置失败：${error.message || '请刷新页面'}`;
  button.disabled = true;
}
clearReference.addEventListener('click', () => {
  referenceImage.value = '';
  referenceType.value = '';
  referenceStrength.value = '1';
  if (options) updateReference();
  status.textContent = '已清除精准参考。';
});
for (const modeButton of modeButtons) {
  modeButton.addEventListener('click', async () => {
    const requested = modeButton.dataset.requestMode;
    if (busy || requested === requestMode) return;
    busy = true; button.disabled = true; updateMode();
    modeStatus.textContent = '正在切换全局请求模式…';
    try {
      const result = await bridge.apiPost('canvas/request-mode', {mode: requested});
      if (!['openai', 'native'].includes(result.request_mode)) throw new Error('服务端返回了无效的请求模式');
      requestMode = result.request_mode;
      options.precise_reference_available = result.precise_reference_available;
      modeStatus.textContent = `已切换为 ${requestMode === 'native' ? 'NovelAI 原生' : 'OpenAI Images'}。`;
    } catch (error) { modeStatus.textContent = `切换失败：${error.message || '请稍后重试'}`; }
    finally { busy = false; button.disabled = false; updateMode(); }
  });
}
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!options || !requestMode || busy) return;
  const selectedType = referenceType.value;
  const file = referenceImage.files[0];
  const hasReference = Boolean(selectedType || file);
  if (hasReference && (requestMode !== 'native' || !options.precise_reference_available)) {
    status.textContent = '当前模式不支持所选参考图；请切换到可用的原生模式，或清除参考类型和图片。'; return;
  }
  if (hasReference && (!selectedType || !file)) {
    status.textContent = '请同时选择参考类型和参考图片，或清除两者。'; return;
  }
  if (hasReference && !options.families[model.value].reference_types.includes(selectedType)) {
    status.textContent = '当前模型不支持所选参考类型，请重新选择。'; return;
  }
  if (hasReference && file.size > options.reference_max_bytes) {
    status.textContent = '参考图片超过大小限制，请选择较小的图片。'; return;
  }
  const strength = referenceStrength.valueAsNumber;
  if (hasReference && (!Number.isFinite(strength) || strength < 0 || strength > Number(referenceStrength.max))) {
    status.textContent = `参考强度须在 0 到 ${referenceStrength.max} 之间。`; return;
  }
  const payload = {
    prompt: document.querySelector('#prompt').value,
    model: model.value,
    size: document.querySelector('#size').value,
    preset: preset.value,
    translator: raw.checked ? '' : translator.value,
    raw: raw.checked,
    request_mode: requestMode,
    steps: document.querySelector('#steps').valueAsNumber,
    scale: document.querySelector('#scale').valueAsNumber,
    cfg_rescale: document.querySelector('#cfg_rescale').valueAsNumber,
    sampler: document.querySelector('#sampler').value,
    noise_schedule: document.querySelector('#noise_schedule').value,
  };
  busy = true; button.disabled = true; updateMode();
  status.textContent = '正在提交任务…';
  try {
    if (hasReference) {
      payload.reference_type = selectedType;
      payload.reference_strength = strength;
      payload.reference_image = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const result = String(reader.result);
          const comma = result.indexOf(',');
          if (comma < 0) reject(new Error('参考图片读取失败'));
          else resolve(result.slice(comma + 1));
        };
        reader.onerror = () => reject(new Error('参考图片读取失败'));
        reader.readAsDataURL(file);
      });
    }
    const task = await bridge.apiPost('canvas/generate', payload);
    for (;;) {
      await new Promise(resolve => setTimeout(resolve, 2000));
      const result = await bridge.apiGet(`canvas/task/${task.id}`);
      if (result.image) {
        if (!image) {
          image = document.createElement('img');
          image.alt = '生成的图片';
          document.querySelector('.preview').insertBefore(image, download);
        }
        image.src = `data:image/png;base64,${result.image}`;
        download.href = image.src;
        image.hidden = false;
        download.hidden = false;
        document.querySelector('#empty').hidden = true;
        status.textContent = ({not_applied: '图片已生成，但结果元数据显示精准参考未生效。', parameters_present: '图片已生成，结果元数据包含精准参考参数。', unknown: '图片已生成，无法从元数据确认精准参考是否生效。'})[result.precise_reference_status] || '生成完成';
        document.dispatchEvent(new Event('canvas-generated'));
        break;
      }
      if (['failed', 'interrupted', 'cancelled', 'send_unknown'].includes(result.state)) throw new Error(result.error);
      status.textContent = ({queued:'排队中…', translating:'翻译中…', running:'生成中…'})[result.state] || '处理中…';
    }
  } catch (error) { status.textContent = error.message || '操作失败，请稍后重试'; }
  finally { busy = false; button.disabled = false; updateMode(); }
});
