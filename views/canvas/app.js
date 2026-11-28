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
const imageEndpoint = document.querySelector('#image-endpoint');
const referenceType = document.querySelector('#reference-type');
const referenceImage = document.querySelector('#reference-image');
const referenceStrength = document.querySelector('#reference-strength');
const referenceHint = document.querySelector('#reference-hint');
const clearReference = document.querySelector('#clear-reference');
const visionReferenceImage = document.querySelector('#vision-reference-image');
const clearVisionReference = document.querySelector('#clear-vision-reference');
const visionReferenceHint = document.querySelector('#vision-reference-hint');
const referenceNames = {character: '角色', style: '风格', 'character&style': '角色＋风格', costume: '服装'};
let options;
let busy = false;
const canUsePreciseReference = () => options.image_endpoints.some(site =>
  (!imageEndpoint.value || site.id === imageEndpoint.value) && site.supports_precise_reference
);
const updateReference = () => {
  const family = options.families[model.value];
  const types = family.reference_types || [];
  const selected = referenceType.value;
  const enabled = canUsePreciseReference() && types.length > 0;
  referenceType.replaceChildren(new Option('不使用参考', ''));
  for (const type of types) referenceType.add(new Option(referenceNames[type] || type, type));
  if (selected && !types.includes(selected)) referenceType.add(new Option(`${referenceNames[selected] || selected}（当前模型不可用）`, selected));
  referenceType.value = selected;
  referenceType.disabled = !enabled || busy;
  referenceImage.disabled = !enabled || busy;
  referenceStrength.disabled = !enabled || busy || !selected;
  clearReference.disabled = busy;
  const visionEnabled = options.translator_enabled && !raw.checked;
  visionReferenceImage.disabled = !visionEnabled || busy;
  clearVisionReference.disabled = busy;
  visionReferenceHint.textContent = visionEnabled
    ? `支持 PNG、JPEG、WebP，与精准参考图合计不超过 ${Math.floor(options.reference_max_bytes / 1048576)} MiB；图片会发送至翻译服务。`
    : '需要启用支持图片输入的翻译模型，并关闭“使用原文”。';
  referenceStrength.max = family.reference_strength_max ?? 5;
  if (enabled) {
    const maxMiB = Math.floor(options.reference_max_bytes / 1048576);
    referenceHint.textContent = `支持 PNG、JPEG、WebP，单张不超过 ${maxMiB} MB。`;
  } else if (types.length === 0) {
    referenceHint.textContent = '当前模型不支持精准参考。';
  } else {
    referenceHint.textContent = '请选择支持精准参考的生图站。';
  }
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
  for (const site of options.image_endpoints) {
    const mode = site.request_mode === 'native' ? 'NovelAI 原生' : 'OpenAI Images';
    imageEndpoint.add(new Option(`${site.id}（${mode}）`, site.id));
  }
  if (options.image_endpoints.length) {
    imageEndpoint.disabled = false;
    imageEndpoint.addEventListener('change', updateReference);
  } else {
    button.disabled = true;
    status.textContent = '请先在插件设置中添加生图站。';
  }
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
  const updateTranslator = () => { translator.disabled = raw.checked || !options.translator_enabled; updateReference(); };
  raw.addEventListener('change', updateTranslator);
  updateTranslator();
  updateModel();
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
clearVisionReference.addEventListener('click', () => {
  visionReferenceImage.value = '';
  status.textContent = '已清除普通参考图。';
});
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!options?.image_endpoints?.length || busy) return;
  const selectedType = referenceType.value;
  const file = referenceImage.files[0];
  const visionFile = visionReferenceImage.files[0];
  const hasReference = Boolean(selectedType || file);
  if (visionFile && (raw.checked || !options.translator_enabled)) {
    status.textContent = '普通参考图需要启用翻译并关闭“使用原文”；请调整设置或清除普通参考图。'; return;
  }
  if (hasReference && !canUsePreciseReference()) {
    status.textContent = '请选择支持精准参考的生图站，或清除参考类型和图片。'; return;
  }
  if (hasReference && (!selectedType || !file)) {
    status.textContent = '请同时选择参考类型和参考图片，或清除两者。'; return;
  }
  if (hasReference && !options.families[model.value].reference_types.includes(selectedType)) {
    status.textContent = '当前模型不支持所选参考类型，请重新选择。'; return;
  }
  if ((file?.size || 0) + (visionFile?.size || 0) > options.reference_max_bytes) {
    status.textContent = '两张参考图合计超过大小限制，请选择较小的图片。'; return;
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
    image_endpoint: imageEndpoint.value,
    steps: document.querySelector('#steps').valueAsNumber,
    scale: document.querySelector('#scale').valueAsNumber,
    cfg_rescale: document.querySelector('#cfg_rescale').valueAsNumber,
    sampler: document.querySelector('#sampler').value,
    noise_schedule: document.querySelector('#noise_schedule').value,
  };
  busy = true; button.disabled = true; imageEndpoint.disabled = true; updateReference();
  status.textContent = '正在提交任务…';
  try {
    if (hasReference) {
      payload.reference_type = selectedType;
      payload.reference_strength = strength;
    }
    for (const [key, source] of [['reference_image', hasReference ? file : null], ['vision_reference_image', visionFile]]) {
      if (!source) continue;
      payload[key] = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const result = String(reader.result);
          const comma = result.indexOf(',');
          if (comma < 0) reject(new Error('参考图片读取失败'));
          else resolve(result.slice(comma + 1));
        };
        reader.onerror = () => reject(new Error('参考图片读取失败'));
        reader.readAsDataURL(source);
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
        status.textContent = ({not_applied: '图片已生成，但结果元数据显示精准参考未生效。', unknown: '图片已生成，无法从元数据确认精准参考是否生效。'})[result.precise_reference_status] || '生成完成';
        document.dispatchEvent(new Event('canvas-generated'));
        break;
      }
      if (['failed', 'interrupted', 'cancelled', 'send_unknown'].includes(result.state)) throw new Error(result.error);
      status.textContent = ({queued:'排队中…', translating:'翻译中…', running:'生成中…'})[result.state] || '处理中…';
    }
  } catch (error) { status.textContent = error.message || '操作失败，请稍后重试'; }
  finally { busy = false; button.disabled = false; imageEndpoint.disabled = false; updateReference(); }
});
