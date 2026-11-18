const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const icon = name => `<svg aria-hidden="true"><use href="/icons.svg#${name}"/></svg>`;
const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const state = { metadata: null, model: 'segformer_b0', stage: 'adapt', mode: 'single', view: 'overlay',
  opacity: 0.55, image: null, results: {}, busy: false, generation: 0 };
const percent = value => value == null ? '—' : `${(100 * value).toFixed(2)}%`;
const activeModel = () => state.metadata?.models.find(m => m.id === state.model);
const result = () => state.results[state.model];

function notice(message = '', type = '') {
  $('#notice').textContent = message;
  $('#notice').className = `notice ${type}`;
  $('#notice').hidden = !message;
}

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || 'Không kết nối được studio.');
  return body;
}

function showBusy(busy, message = 'Đang phân đoạn ảnh…') {
  state.busy = busy;
  $('#busyLayer').hidden = !busy;
  $('#busyTitle').textContent = message;
  for (const button of $$('#modelOptions button, #stageSelect, #uploadButton, #emptyUpload, #samples button, nav button')) {
    button.disabled = busy || button.dataset.unavailable === 'true';
  }
  renderRunButton();
}

function renderRunButton() {
  const available = state.mode === 'compare'
    ? ['segformer_b0', 'segformer_b2'].every(id => state.metadata?.models.find(m => m.id === id)?.stages[state.stage])
    : activeModel()?.stages[state.stage];
  $('#runButton').disabled = state.busy || !state.image || !available;
  $('#runLabel').textContent = state.busy ? 'Đang xử lý…' : state.mode === 'compare' ? 'Chạy cả hai mô hình' : 'Phân đoạn ảnh';
}

function renderModels() {
  const models = state.metadata.models;
  $('#modelOptions').innerHTML = models.map(model => {
    const available = model.stages[state.stage];
    if (model.id.startsWith('deeplab') && !available) return '<div class="pending-model">DeepLabV3+ ResNet50<span>CHỜ TRAIN</span></div>';
    return `<button class="model-option ${model.id === state.model ? 'active' : ''}" role="radio" aria-checked="${model.id === state.model}" data-model="${model.id}" data-unavailable="${!available}" ${!available || state.busy ? 'disabled' : ''}>
      <span class="model-initial">${model.id.endsWith('b0') ? 'B0' : model.id.endsWith('b2') ? 'B2' : 'R50'}</span><span class="model-name">${escape(model.name)}<small>${available ? escape(model.detail) : 'Chưa có checkpoint local'}</small></span><span class="radio-dot"></span></button>`;
  }).join('');
  $('#modelOptions').setAttribute('role', 'radiogroup');
  $('#modelOptions').setAttribute('aria-label', 'Chọn mô hình');
  for (const button of $$('#modelOptions [data-model]')) button.addEventListener('click', () => {
    if (state.busy) return;
    state.model = button.dataset.model;
    renderModels(); renderResults(); renderRunButton();
  });
  const model = activeModel();
  const score = state.stage === 'adapt' ? model?.dev_miou : model?.source_dev_miou;
  $('#modelScore').innerHTML = `${percent(score)}<small>mIoU</small>`;
  $('#modelTable').innerHTML = `<table class="model-table"><thead><tr><th>Mô hình</th><th>Tham số</th><th>ACDC dev</th><th>Local</th></tr></thead><tbody>${models.map(m => `<tr><td>${escape(m.name)}</td><td>${m.parameters ? `${(m.parameters / 1e6).toFixed(2)} M` : '—'}</td><td>${percent(m.dev_miou)}</td><td class="${m.stages.adapt ? 'available' : ''}">${m.stages.adapt ? 'Có checkpoint' : 'Chưa có'}</td></tr>`).join('')}</tbody></table>`;
  $('#checkpointHelp').textContent = models.some(m => !m.stages.adapt && !m.id.startsWith('deeplab'))
    ? 'Thiếu checkpoint? Đặt biến SEGFORMER_B0_CHECKPOINT hoặc SEGFORMER_B2_CHECKPOINT đến thư mục best/, rồi khởi động lại server. Xem web/README.md trong repo.' : '';
}

function renderSamples() {
  const samples = state.metadata.samples;
  const icons = { clear: 'sun', fog: 'fog', rain: 'rain', snow: 'snow', night: 'moon' };
  $('#samples').innerHTML = samples.length ? samples.map(sample => `<button class="sample-card" data-sample="${sample.id}" aria-label="Chọn ảnh mẫu ${escape(sample.label)}"><img src="/api/samples/${sample.id}" alt="${escape(sample.label)}" loading="lazy"><span class="sample-tag">${escape(sample.dataset.toUpperCase())}</span><span class="sample-caption">${icon(icons[sample.id])}${escape(sample.label)}</span></button>`).join('') : '<p class="muted">Chưa có dataset local. Bạn vẫn có thể tải ảnh từ máy.</p>';
  for (const button of $$('#samples button')) button.addEventListener('click', async () => {
    if (state.busy) return;
    const sample = samples.find(s => s.id === button.dataset.sample);
    try {
      const response = await fetch(`/api/samples/${sample.id}`);
      if (!response.ok) throw new Error('Không tải được ảnh mẫu.');
      const file = new File([await response.blob()], `${sample.dataset}_${sample.id}.png`, { type: 'image/png' });
      await selectFile(file, `${sample.dataset.toUpperCase()} · ${sample.label.toUpperCase()}`, sample.id);
    } catch (error) { notice(error.message); }
  });
}

const readFile = file => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.onload = () => resolve(reader.result);
  reader.onerror = () => reject(new Error('Chưa đọc được tệp ảnh.'));
  reader.readAsDataURL(file);
});
const loadImage = src => new Promise((resolve, reject) => {
  const image = new Image();
  image.onload = () => resolve(image);
  image.onerror = () => reject(new Error('Tệp không phải ảnh hợp lệ. Chọn PNG, JPG, WebP hoặc BMP.'));
  image.src = src;
});

async function selectFile(file, label = 'ẢNH CỦA BẠN', sampleId = null) {
  if (state.busy || !file) return;
  try {
    if (file.size > 10 * 1024 * 1024) throw new Error('Ảnh tối đa 10 MB. Hãy chọn tệp nhỏ hơn.');
    if (!/\.(png|jpe?g|webp|bmp)$/i.test(file.name)) throw new Error('Chọn ảnh PNG, JPG, WebP hoặc BMP.');
    const dataUrl = await readFile(file);
    const image = await loadImage(dataUrl);
    if (image.naturalWidth * image.naturalHeight > 8_000_000) throw new Error('Ảnh tối đa 8 triệu pixel. Hãy giảm kích thước rồi thử lại.');
    state.image = { dataUrl, name: file.name, width: image.naturalWidth, height: image.naturalHeight, label };
    state.generation++;
    state.results = {};
    $('#fileName').textContent = file.name;
    $('#resolution').textContent = `${image.naturalWidth} × ${image.naturalHeight}`;
    $('#originalImage').src = dataUrl;
    $('#originalImage').hidden = false;
    $('#emptyState').hidden = true;
    $('#frameBadge').hidden = false;
    $('#frameBadgeText').textContent = label;
    for (const card of $$('#samples button')) card.classList.toggle('active', card.dataset.sample === sampleId);
    notice(); renderResults(); renderRunButton();
  } catch (error) { notice(error.message); }
  $('#fileInput').value = '';
}

function applyView() {
  for (const tab of $$('[data-view]')) tab.setAttribute('aria-selected', String(tab.dataset.view === state.view));
  const current = result();
  $('#maskImage').hidden = !current || state.view === 'original';
  if (current) {
    $('#maskImage').src = current.images.mask;
    $('#maskImage').style.opacity = state.view === 'mask' ? 1 : state.opacity;
  }
  for (const img of $$('.compare-mask')) {
    img.hidden = state.view === 'original';
    img.style.opacity = state.view === 'mask' ? 1 : state.opacity;
  }
  $('#opacity').disabled = state.view !== 'overlay';
}

function renderResults() {
  const current = result();
  const compare = state.mode === 'compare';
  $('#imageFrame').hidden = compare;
  $('#compareGrid').hidden = !compare;
  if (compare) {
    $('#compareGrid').innerHTML = ['segformer_b0', 'segformer_b2'].map(id => {
      const r = state.results[id];
      const name = state.metadata?.models.find(m => m.id === id)?.name || id;
      return `<div class="compare-frame">${state.image ? `<img alt="Ảnh đầu vào ${escape(name)}" src="${r ? r.images.original : state.image.dataUrl}">` : ''}${r ? `<img class="compare-mask" alt="Mask ${escape(name)}" src="${r.images.mask}">` : '<div class="compare-wait">Chọn cùng một ảnh<br>để đối chiếu hai góc nhìn.</div>'}<span class="compare-label">${escape(name)}${r ? ` · ${(r.inference_ms / 1000).toFixed(2)}s` : ''}</span></div>`;
    }).join('');
  }
  if (current) {
    $('#originalImage').src = current.images.original;
    $('#classCount').textContent = current.classes.length;
    $('#classList').innerHTML = current.classes.map(c => `<div class="class-row"><span class="class-swatch" style="background:rgb(${c.color.join(',')})"></span><span class="class-name">${escape(c.label)}</span><span class="percent">${c.percent.toFixed(1)}%</span><div class="class-bar"><i style="width:${c.percent}%;background:rgb(${c.color.join(',')})"></i></div></div>`).join('');
    $('#resultStrip').hidden = false;
    $('#resultTitle').textContent = `${activeModel().name} · ${current.classes.length} lớp xuất hiện`;
    $('#resultDetail').textContent = `${current.width} × ${current.height} · ${(current.inference_ms / 1000).toFixed(2)}s xử lý · CPU${current.cold_load ? ' · Có nạp model lần đầu' : ''}`;
  } else {
    if (state.image) $('#originalImage').src = state.image.dataUrl;
    $('#classCount').textContent = '—';
    $('#classList').innerHTML = `<div class="legend-empty">${icon('layers')}<p>Chạy phân đoạn để khám phá<br>các lớp trong khung hình.</p></div>`;
    $('#resultStrip').hidden = true;
  }
  applyView();
}

async function predict() {
  if (state.busy || !state.image) return;
  const generation = state.generation;
  const snapshot = { image: state.image.dataUrl.split(',')[1], stage: state.stage };
  const models = state.mode === 'compare' ? ['segformer_b0', 'segformer_b2'] : [state.model];
  notice();
  showBusy(true, state.mode === 'compare' ? 'Đang đối chiếu B0 và B2…' : 'Đang phân đoạn ảnh…');
  try {
    for (const model of models) {
      if (state.results[model]) continue;
      const prediction = await api('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...snapshot, model }) });
      if (generation !== state.generation) return;
      state.results[model] = prediction;
      renderResults();
    }
  } catch (error) { if (generation === state.generation) notice(error.message); }
  finally { showBusy(false); }
}

async function download(kind) {
  const current = result();
  if (!current) return;
  const link = document.createElement('a');
  link.download = `${state.model}_${state.stage}_${kind}.png`;
  if (kind !== 'overlay') link.href = current.images[kind];
  else {
    const [base, mask] = await Promise.all([loadImage(current.images.original), loadImage(current.images.mask)]);
    const canvas = document.createElement('canvas');
    canvas.width = current.width; canvas.height = current.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(base, 0, 0); ctx.globalAlpha = state.opacity; ctx.drawImage(mask, 0, 0);
    link.href = canvas.toDataURL('image/png');
  }
  link.click();
}

for (const button of $$('#uploadButton, #emptyUpload')) button.addEventListener('click', () => $('#fileInput').click());
$('#fileInput').addEventListener('change', e => selectFile(e.target.files[0]));
$('#runButton').addEventListener('click', predict);
for (const button of $$('[data-mode]')) button.addEventListener('click', () => {
  if (state.busy) return;
  state.mode = button.dataset.mode;
  for (const item of $$('[data-mode]')) item.classList.toggle('active', item === button);
  $('#pageCrumb').textContent = state.mode === 'compare' ? 'So sánh mô hình' : 'Phân đoạn ảnh';
  $('#pageTitle').innerHTML = state.mode === 'compare' ? 'Cùng một ảnh. <span>Hai góc nhìn khác nhau.</span>' : 'Một khung hình. <span>Nhiều lớp ý nghĩa.</span>';
  $('#pageDescription').textContent = state.mode === 'compare' ? 'Đối chiếu B0 và B2 trên cùng đầu vào. Chọn model ở bên phải để xem diện tích lớp và tải kết quả.' : 'Khám phá cách mô hình nhìn đường phố, từ trời quang đến những điều kiện khó nhất.';
  renderResults(); renderRunButton();
});
for (const button of $$('[data-view]')) button.addEventListener('click', () => { state.view = button.dataset.view; applyView(); });
$('#opacity').addEventListener('input', e => { state.opacity = Number(e.target.value) / 100; $('#opacityValue').textContent = `${e.target.value}%`; applyView(); });
$('#stageSelect').addEventListener('change', e => {
  if (!state.metadata) return;
  state.stage = e.target.value; state.generation++; state.results = {};
  renderModels(); renderResults(); renderRunButton();
});
for (const button of $$('[data-download]')) button.addEventListener('click', () => download(button.dataset.download).catch(error => notice(error.message)));
for (const [button, dialog] of [['#helpButton', '#helpDialog'], ['#modelsButton', '#modelsDialog']]) $(button).addEventListener('click', () => $(dialog).showModal());
for (const button of $$('[data-close]')) button.addEventListener('click', () => button.closest('dialog').close());
for (const dialog of $$('dialog')) dialog.addEventListener('click', e => { if (e.target === dialog) dialog.close(); });
$('#expandButton').addEventListener('click', async () => {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await $('.canvas-card').requestFullscreen(); }
  catch { notice('Trình duyệt chưa hỗ trợ toàn màn hình. Bạn có thể phóng to trang.'); }
});
let dragDepth = 0;
$('#dropZone').addEventListener('dragenter', e => { e.preventDefault(); dragDepth++; if (!state.busy) $('#dropZone').classList.add('dragging'); });
$('#dropZone').addEventListener('dragover', e => e.preventDefault());
$('#dropZone').addEventListener('dragleave', () => { if (--dragDepth <= 0) $('#dropZone').classList.remove('dragging'); });
$('#dropZone').addEventListener('drop', e => { e.preventDefault(); dragDepth = 0; $('#dropZone').classList.remove('dragging'); selectFile(e.dataTransfer.files[0]); });
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => e.preventDefault());

async function boot() {
  try {
    state.metadata = await api('/api/metadata');
    if (!activeModel()?.stages.adapt) state.model = state.metadata.models.find(m => m.stages.adapt)?.id || 'segformer_b0';
    renderModels(); renderSamples(); renderResults(); renderRunButton();
    if (!state.metadata.models.some(m => m.stages.adapt)) notice('Chưa có checkpoint trên máy. Mở thư viện mô hình để xem cách cài. Bạn có thể xem giao diện và thử tải ảnh.', 'info');
  } catch (error) {
    notice(error.message);
    $('#modelOptions').textContent = 'Chưa kết nối được inference worker.';
    $('#samples').textContent = 'Bạn vẫn có thể tải ảnh để xem trước.';
  }
}
boot();
