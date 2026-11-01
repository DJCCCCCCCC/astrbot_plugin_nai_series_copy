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
let options;
try {
  options = await bridge.apiGet('canvas/options');
  if (!options.translator_enabled) translator.options[0].text = '翻译未启用';
  for (const name of options.translators) translator.add(new Option(name, name));
  const updateModel = () => {
    const family = options.families[model.value];
    preset.replaceChildren(new Option('不使用预设', ''));
    for (const item of family.presets) preset.add(new Option(item.name, item.id));
    const selected = family.presets.find(item => item.id === family.default_preset || item.name === family.default_preset);
    preset.value = selected?.id || '';
    for (const key of ['steps', 'scale', 'cfg_rescale', 'sampler', 'noise_schedule']) document.getElementById(key).value = family[key];
  };
  model.addEventListener('change', updateModel);
  const updateTranslator = () => { translator.disabled = raw.checked || !options.translator_enabled; };
  raw.addEventListener('change', updateTranslator);
  updateTranslator();
  updateModel();
} catch (error) {
  status.textContent = `加载画布设置失败：${error.message || '请刷新页面'}`;
  button.disabled = true;
}
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  button.disabled = true;
  status.textContent = '正在提交任务…';
  try {
    const task = await bridge.apiPost('canvas/generate', {prompt: document.querySelector('#prompt').value, model: model.value, size: document.querySelector('#size').value, preset: preset.value, translator: raw.checked ? '' : translator.value, raw: raw.checked, steps: Number(document.querySelector('#steps').value), scale: Number(document.querySelector('#scale').value), cfg_rescale: Number(document.querySelector('#cfg_rescale').value), sampler: document.querySelector('#sampler').value, noise_schedule: document.querySelector('#noise_schedule').value});
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
        status.textContent = '生成完成';
        break;
      }
      if (['failed', 'interrupted', 'cancelled', 'send_unknown'].includes(result.state)) throw new Error(result.error);
      status.textContent = ({queued:'排队中…', translating:'翻译中…', running:'生成中…'})[result.state] || '处理中…';
    }
  } catch (error) { status.textContent = error.message || '操作失败，请稍后重试'; }
  finally { button.disabled = false; }
});
