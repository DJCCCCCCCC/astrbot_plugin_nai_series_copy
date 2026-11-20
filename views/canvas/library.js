const bridge = window.AstrBotPluginView;
const tablist = document.querySelector('[role="tablist"]');
const tabs = Array.from(tablist.querySelectorAll('[role="tab"]'));
const compactLayout = matchMedia('(max-width: 760px)');
const loaded = new Set();
const loading = new Set();
const gallery = document.querySelector('#panel-gallery');
const loadMore = gallery.querySelector('.load-more');
let galleryPage = 0;

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
    if (row.image) {
      const image = document.createElement('img');
      image.src = `data:image/png;base64,${row.image}`;
      image.alt = row.name || row.prompt || '生成图片';
      image.loading = 'lazy';
      card.append(image);
    } else {
      const placeholder = document.createElement('p');
      placeholder.className = 'card-placeholder';
      placeholder.textContent = '暂无展示图';
      card.append(placeholder);
    }
    const caption = document.createElement('div');
    caption.className = 'card-caption';
    caption.append(title);
    const description = row.description || row.model;
    if (description) {
      const detail = document.createElement('p');
      detail.textContent = description;
      caption.append(detail);
    }
    card.append(caption);
    grid.append(card);
  }
}

async function loadView(view, panel) {
  if (loaded.has(view) || loading.has(view)) return;
  loading.add(view);
  const status = panel.querySelector('.panel-status');
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
