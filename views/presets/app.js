const bridge = window.AstrBotPluginView;
const content = document.querySelector('#content');
const view = document.body.dataset.view;
await bridge.ready();
try {
  const data = await bridge.apiGet('library', { view });
  if (view === 'ranking') {
    const show = period => {
      const rows = data.rankings[period];
      content.replaceChildren();
      if (!rows.length) { content.textContent = '暂无生图记录'; return; }
      const max = rows[0].count;
      rows.forEach((row, index) => {
        const item = document.createElement('div'); item.className = 'rank-row';
        const rank = document.createElement('strong'); rank.className = 'rank'; rank.textContent = index + 1;
        const body = document.createElement('div'); body.className = 'rank-body';
        const label = document.createElement('div'); label.textContent = row.nickname ? `${row.nickname} · ${row.user}` : row.user;
        const bar = document.createElement('div'); bar.className = 'track'; const fill = document.createElement('div'); fill.className = 'fill'; fill.style.width = `${row.count / max * 100}%`; bar.append(fill); body.append(label, bar);
        const count = document.createElement('strong'); count.textContent = `${row.count} 张`; item.append(rank, body, count); content.append(item);
      });
    };
    document.querySelectorAll('button[data-period]').forEach(button => button.addEventListener('click', () => { document.querySelectorAll('button[data-period]').forEach(b => b.setAttribute('aria-pressed', String(b === button))); show(button.dataset.period); }));
    show('total');
  } else {
    const rows = view === 'gallery' ? data.images : data.presets;
    content.replaceChildren(); content.className = 'grid';
    if (!rows.length) { content.textContent = view === 'gallery' ? '暂无保存的图片' : '暂无预设'; }
    rows.forEach(row => {
      const item = document.createElement('article');
      if (row.image) { const image = document.createElement('img'); image.src = `data:image/png;base64,${row.image}`; image.alt = row.name || row.prompt || '生成图片'; image.loading = 'lazy'; item.append(image); }
      else { const empty = document.createElement('div'); empty.className = 'placeholder'; empty.textContent = '暂无展示图'; item.append(empty); }
      const title = document.createElement('strong'); title.textContent = row.name || row.prompt || '无描述'; const detail = document.createElement('p'); detail.textContent = row.description || row.model || ''; item.append(title, detail); content.append(item);
    });
    if (view === 'gallery' && data.has_more) {
      let page = 0;
      const more = document.createElement('button'); more.textContent = '加载更多'; content.after(more);
      more.addEventListener('click', async () => {
        more.disabled = true;
        try {
          const next = await bridge.apiGet('library', { view, page: page + 1 });
          for (const row of next.images) {
            const item = document.createElement('article'); const image = document.createElement('img');
            image.src = `data:image/png;base64,${row.image}`; image.alt = row.prompt || '生成图片'; image.loading = 'lazy';
            const title = document.createElement('strong'); title.textContent = row.prompt || '无描述';
            const detail = document.createElement('p'); detail.textContent = row.model || ''; item.append(image, title, detail); content.append(item);
          }
          page++; if (!next.has_more) more.remove();
        } catch (error) { more.textContent = `加载失败，点击重试`; }
        finally { more.disabled = false; }
      });
    }
  }
} catch (error) { content.textContent = `加载失败：${error.message || '请刷新页面'}`; }
