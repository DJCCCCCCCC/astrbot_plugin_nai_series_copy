const bridge = window.AstrBotPluginView;
const tablist = document.querySelector('[role="tablist"]');
const tabs = Array.from(tablist.querySelectorAll('[role="tab"]'));
const compactLayout = matchMedia('(max-width: 760px)');
const loaded = new Set();
const loading = new Set();
const gallery = document.querySelector('#panel-gallery');
const loadMore = gallery.querySelector('.load-more');
const previewDialog = document.querySelector('#library-preview');
const previewImage = document.createElement('img');
const previewDownload = previewDialog.querySelector('a');
const presetPanel = document.querySelector('#panel-presets');
const presetForm = document.querySelector('#preset-editor');
const presetNew = document.querySelector('#preset-new');
const presetCancel = document.querySelector('#preset-cancel');
const presetSave = presetForm.querySelector('button[type="submit"]');
const presetFields = presetForm.querySelector('fieldset');
const presetStatus = presetForm.querySelector('.panel-status');
const presetFeedback = document.querySelector('#preset-feedback');
const presetModel = presetForm.elements.namedItem('model');
const inheritNegative = presetForm.elements.namedItem('inherit_negative');
const presetNegative = presetForm.elements.namedItem('negative_prompt');
let editedPreset;
let editorLauncher;
let galleryPage = 0;

previewDialog.querySelector('button').addEventListener('click', () => previewDialog.close());
previewDownload.addEventListener('click', downloadImage);
previewDialog.addEventListener('close', () => {
  if (previewDialog.open) return;
  previewImage.remove();
  previewImage.removeAttribute('src');
  previewDownload.removeAttribute('href');
});

function downloadImage(event) {
  event.preventDefault();
  const link = event.currentTarget;
  const [header, base64] = link.getAttribute('href').split(',');
  const bytes = Uint8Array.from(atob(base64), character => character.charCodeAt(0));
  const url = URL.createObjectURL(new Blob([bytes], { type: header.slice(5, -7) }));
  const file = document.createElement('a');
  file.href = url;
  file.download = link.download;
  file.hidden = true;
  link.after(file);
  file.click();
  file.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function renderRanking(panel, period, rows) {
  panel.querySelectorAll('[data-period]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.period === period));
  });
  const list = panel.querySelector('[data-results]');
  list.replaceChildren();
  panel.querySelector('.panel-status').textContent = rows.length ? '' : '暂无生图记录';

  for (const [index, row] of rows.entries()) {
    const item = document.createElement('li');
    const rank = document.createElement('strong');
    rank.className = 'rank';
    rank.textContent = index + 1;
    const details = document.createElement('div');
    details.className = 'rank-details';
    const username = document.createElement('span');
    username.textContent = row.nickname ? `${row.nickname} · ${row.user}` : row.user;
    const track = document.createElement('div');
    track.className = 'rank-track';
    track.setAttribute('aria-hidden', 'true');
    const bar = document.createElement('div');
    bar.style.width = `${row.count / rows[0].count * 100}%`;
    track.append(bar);
    details.append(username, track);
    const count = document.createElement('strong');
    count.className = 'rank-count';
    count.textContent = `${row.count} 张`;
    item.append(rank, details, count);
    list.append(item);
  }
}

function appendCards(grid, rows) {
  for (const row of rows) {
    const card = document.createElement('article');
    const title = document.createElement('strong');
    title.textContent = row.name || row.prompt || '无描述';
    const caption = document.createElement('div');
    caption.className = 'card-caption';
    caption.append(title);
    const actions = document.createElement('div');
    actions.className = 'library-actions';
    const description = row.description || row.model;
    if (description) {
      const detail = document.createElement('p');
      detail.textContent = description;
      caption.append(detail);
    }
    if (row.image) {
      // Gallery images retain the provider's format. Saved preset previews are PNG.
      let format = 'png';
      if (row.image.startsWith('/9j/')) format = 'jpeg';
      if (row.image.startsWith('UklGR')) format = 'webp';
      const image = document.createElement('img');
      image.src = `data:image/${format};base64,${row.image}`;
      image.alt = row.name || row.prompt || '生成图片';
      image.loading = 'lazy';
      card.append(image);

      const enlarge = document.createElement('button');
      enlarge.type = 'button';
      enlarge.textContent = '放大预览';
      const download = document.createElement('a');
      download.textContent = '下载';
      download.href = image.src;
      const filename = row.name ? 'nai-preset-preview' : 'nai-image';
      download.download = `${filename}.${format === 'jpeg' ? 'jpg' : format}`;
      download.addEventListener('click', downloadImage);
      enlarge.addEventListener('click', () => {
        previewImage.src = image.src;
        previewImage.alt = image.alt;
        previewDownload.href = download.href;
        previewDownload.download = download.download;
        previewDialog.append(previewImage);
        previewDialog.showModal();
      });
      actions.append(enlarge, download);
    } else {
      const placeholder = document.createElement('p');
      placeholder.className = 'card-placeholder';
      placeholder.textContent = '暂无展示图';
      card.append(placeholder);
    }
    if (row.id && (row.family === 'nai4' || row.family === 'nai5')) {
      const edit = document.createElement('button');
      edit.type = 'button';
      edit.textContent = '编辑';
      edit.setAttribute('aria-controls', 'preset-editor');
      edit.setAttribute('aria-expanded', 'false');
      edit.addEventListener('click', () => openPresetEditor(row, edit));
      actions.append(edit);
    }
    if (actions.childElementCount) caption.append(actions);
    card.append(caption);
    grid.append(card);
  }
}

async function loadView(view, panel) {
  if (loaded.has(view) || loading.has(view)) return;
  loading.add(view);
  const status = panel.querySelector('[data-library-status]');
  status.textContent = '正在加载…';
  try {
    await bridge.ready();
    const data = await bridge.apiGet('library', { view });
    if (view === 'ranking') {
      const periods = panel.querySelector('.periods');
      periods.onclick = event => {
        const period = event.target.closest('[data-period]')?.dataset.period;
        if (period) renderRanking(panel, period, data.rankings[period]);
      };
      renderRanking(panel, 'total', data.rankings.total);
    } else {
      const grid = panel.querySelector('[data-results]');
      const rows = view === 'gallery' ? data.images : data.presets;
      grid.replaceChildren();
      appendCards(grid, rows);
      status.textContent = rows.length ? '' : view === 'gallery' ? '暂无保存的图片' : '暂无预设';
      if (view === 'gallery') {
        galleryPage = 0;
        loadMore.hidden = !data.has_more;
      } else {
        presetNew.disabled = false;
      }
    }
    loaded.add(view);
  } catch {
    const retry = document.createElement('button');
    retry.type = 'button';
    retry.textContent = '重试';
    retry.addEventListener('click', () => loadView(view, panel));
    status.replaceChildren('加载失败。', retry);
  } finally {
    loading.delete(view);
  }
}

function openPresetEditor(preset, launcher) {
  if (!presetForm.hidden) {
    presetForm.elements.namedItem('name').focus();
    return;
  }
  editedPreset = preset;
  editorLauncher = launcher;
  presetForm.reset();
  presetForm.querySelector('details').open = false;
  presetModel.querySelector('[data-existing-model]')?.remove();
  if (preset) {
    for (const name of ['name', 'artist_prompt', 'negative_prompt', 'description', 'preview_prompt', 'preview_size']) {
      presetForm.elements.namedItem(name).value = preset[name] ?? '';
    }
    if (![...presetModel.options].some(option => option.value === preset.model)) {
      const existing = new Option(preset.model, preset.model);
      existing.dataset.existingModel = '';
      presetModel.add(existing);
    }
    presetModel.value = preset.model;
  }
  presetModel.disabled = Boolean(preset);
  inheritNegative.checked = preset?.negative_prompt == null;
  presetNegative.disabled = inheritNegative.checked;
  document.querySelector('#preset-editor-title').textContent = preset ? '编辑预设' : '新增预设';
  presetStatus.textContent = '';
  presetFeedback.textContent = '';
  launcher.setAttribute('aria-expanded', 'true');
  presetForm.hidden = false;
  presetForm.elements.namedItem('name').focus();
}

function closePresetEditor() {
  presetForm.hidden = true;
  editorLauncher.setAttribute('aria-expanded', 'false');
  editorLauncher.focus();
}

presetNew.addEventListener('click', () => openPresetEditor(null, presetNew));
presetCancel.addEventListener('click', closePresetEditor);
inheritNegative.addEventListener('change', () => {
  presetNegative.disabled = inheritNegative.checked;
});
presetForm.addEventListener('submit', async event => {
  event.preventDefault();
  if (presetSave.disabled) return;
  const payload = Object.fromEntries(new FormData(presetForm));
  delete payload.inherit_negative;
  payload.model = presetModel.value;
  payload.family = editedPreset?.family ?? (payload.model.startsWith('nai-diffusion-4-') ? 'nai4' : 'nai5');
  if (editedPreset) payload.id = editedPreset.id;
  if (inheritNegative.checked) payload.negative_prompt = null;
  presetFields.disabled = true;
  presetSave.disabled = true;
  presetCancel.disabled = true;
  presetStatus.textContent = '正在保存…';
  try {
    await bridge.apiPost('library/presets/save', payload);
    closePresetEditor();
    presetNew.focus();
    presetFeedback.textContent = '预设已保存';
    document.dispatchEvent(new Event('presets-changed'));
    loaded.delete('presets');
    await loadView('presets', presetPanel);
  } catch (error) {
    presetStatus.textContent = error.message || '保存失败，请重试。';
  } finally {
    presetFields.disabled = false;
    presetSave.disabled = false;
    presetCancel.disabled = false;
  }
});

function selectTab(selected) {
  for (const tab of tabs) {
    const active = tab === selected;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    const panel = document.getElementById(tab.getAttribute('aria-controls'));
    panel.hidden = !active;
    if (active && tab.id !== 'tab-canvas') loadView(tab.id.slice(4), panel);
  }
}

const syncOrientation = () => {
  tablist.setAttribute('aria-orientation', compactLayout.matches ? 'horizontal' : 'vertical');
};
compactLayout.addEventListener('change', syncOrientation);
syncOrientation();

for (const tab of tabs) tab.addEventListener('click', () => selectTab(tab));
document.addEventListener('canvas-generated', () => {
  for (const view of ['ranking', 'gallery']) {
    loaded.delete(view);
    const panel = document.getElementById(`panel-${view}`);
    if (!panel.hidden) loadView(view, panel);
  }
});
tablist.addEventListener('keydown', event => {
  const index = tabs.indexOf(event.target);
  if (index === -1) return;
  const previous = compactLayout.matches ? 'ArrowLeft' : 'ArrowUp';
  const next = compactLayout.matches ? 'ArrowRight' : 'ArrowDown';
  let target;
  if (event.key === previous) target = tabs[(index + tabs.length - 1) % tabs.length];
  if (event.key === next) target = tabs[(index + 1) % tabs.length];
  if (event.key === 'Home') target = tabs[0];
  if (event.key === 'End') target = tabs.at(-1);
  if (!target) return;
  event.preventDefault();
  target.focus();
  selectTab(target);
});

loadMore.addEventListener('click', async () => {
  loadMore.disabled = true;
  loadMore.textContent = '正在加载…';
  const status = gallery.querySelector('.panel-status');
  try {
    const data = await bridge.apiGet('library', { view: 'gallery', page: galleryPage + 1 });
    appendCards(gallery.querySelector('[data-results]'), data.images);
    galleryPage++;
    loadMore.hidden = !data.has_more;
    status.textContent = '';
  } catch {
    status.textContent = '加载失败，请重试。';
  } finally {
    loadMore.disabled = false;
    loadMore.textContent = '加载更多';
  }
});
