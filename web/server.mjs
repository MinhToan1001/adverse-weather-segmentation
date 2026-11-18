import http from 'node:http';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PUBLIC = path.join(ROOT, 'web', 'public');
const MAX_BODY = 14 * 1024 * 1024;

export class InferenceWorker {
  constructor({ python, timeout = 180_000 } = {}) {
    const local = path.join(ROOT, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
    this.pending = new Map();
    this.next = 0;
    this.timeout = timeout;
    this.child = spawn(python || process.env.PYTHON_BIN || (existsSync(local) ? local : 'python'), ['-u', '-m', 'web.worker'], {
      cwd: ROOT, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', USE_TF: '0', USE_TORCH: '1' },
    });
    this.child.stderr.on('data', data => process.stderr.write(data));
    createInterface({ input: this.child.stdout }).on('line', line => {
      try {
        const reply = JSON.parse(line);
        if (reply.event === 'ready') return;
        const pending = this.pending.get(reply.id);
        if (!pending) return;
        clearTimeout(pending.timer);
        this.pending.delete(reply.id);
        if (reply.ok) pending.resolve(reply.result);
        else pending.reject(Object.assign(new Error(reply.error), { status: 422 }));
      } catch (error) { console.error('Worker protocol:', error.message); }
    });
    this.child.on('error', () => this.fail('Không mở được Python. Đặt PYTHON_BIN đến Python đã cài requirements.txt.'));
    this.child.on('exit', () => this.fail('Worker đã dừng. Khởi động lại npm start.'));
    this.child.stdin.on('error', () => this.fail('Kết nối inference đã đóng. Khởi động lại npm start.'));
  }

  fail(message) {
    this.failure = message;
    for (const { reject, timer } of this.pending.values()) {
      clearTimeout(timer);
      reject(Object.assign(new Error(message), { status: 503 }));
    }
    this.pending.clear();
  }

  request(data) {
    if (this.failure) return Promise.reject(Object.assign(new Error(this.failure), { status: 503 }));
    if (this.pending.size >= 12 || (data.action === 'predict' && [...this.pending.values()].filter(p => p.action === 'predict').length >= 2)) {
      return Promise.reject(Object.assign(new Error('Đang xử lý ảnh khác. Hãy thử lại sau.'), { status: 429 }));
    }
    return new Promise((resolve, reject) => {
      const id = ++this.next;
      const timer = setTimeout(() => {
        // Stop a stuck worker rather than let subsequent requests pile up.
        this.fail('Inference quá thời gian. Khởi động lại npm start và thử ảnh nhỏ hơn.');
        this.child.kill();
      }, this.timeout);
      this.pending.set(id, { resolve, reject, timer, action: data.action });
      this.child.stdin.write(JSON.stringify({ ...data, id }) + '\n');
    });
  }

  close() { this.fail('Studio đã đóng.'); this.child.kill(); }
}

async function readJson(request) {
  if (Number(request.headers['content-length']) > MAX_BODY) {
    request.resume();
    throw Object.assign(new Error('Ảnh tối đa 10 MB.'), { status: 413 });
  }
  let size = 0;
  const chunks = [];
  for await (const chunk of request) {
    size += chunk.length;
    if (size > MAX_BODY) throw Object.assign(new Error('Ảnh tối đa 10 MB.'), { status: 413 });
    chunks.push(chunk);
  }
  try { return JSON.parse(Buffer.concat(chunks).toString('utf8')); }
  catch { throw Object.assign(new Error('Dữ liệu gửi lên không hợp lệ.'), { status: 400 }); }
}

export function createStudioServer(worker) {
  const sampleCache = new Map();
  const assets = new Map([['/', ['index.html', 'text/html; charset=utf-8']],
    ['/styles.css', ['styles.css', 'text/css; charset=utf-8']],
    ['/app.js', ['app.js', 'text/javascript; charset=utf-8']],
    ['/icons.svg', ['icons.svg', 'image/svg+xml']], ['/favicon.svg', ['favicon.svg', 'image/svg+xml']]]);
  return http.createServer(async (request, response) => {
    response.setHeader('X-Content-Type-Options', 'nosniff');
    response.setHeader('Referrer-Policy', 'no-referrer');
    response.setHeader('Cache-Control', 'no-store');
    response.setHeader('Content-Security-Policy', "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'");
    const json = (status, data) => {
      response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
      response.end(JSON.stringify(data));
    };
    try {
      const url = new URL(request.url, 'http://localhost');
      if (request.method === 'GET' && assets.has(url.pathname)) {
        const [file, type] = assets.get(url.pathname);
        response.writeHead(200, { 'Content-Type': type });
        response.end(await readFile(path.join(PUBLIC, file)));
      } else if (request.method === 'GET' && url.pathname === '/api/metadata') {
        json(200, await worker.request({ action: 'metadata' }));
      } else if (request.method === 'GET' && /^\/api\/samples\/[a-z]+$/.test(url.pathname)) {
        const sample = url.pathname.split('/').at(-1);
        if (!sampleCache.has(sample)) sampleCache.set(sample, await worker.request({ action: 'sample', sample }));
        const result = sampleCache.get(sample);
        response.writeHead(200, { 'Content-Type': result.mime });
        response.end(Buffer.from(result.content, 'base64'));
      } else if (request.method === 'POST' && url.pathname === '/api/predict') {
        const origin = request.headers.origin;
        if (origin && new URL(origin).host !== request.headers.host) return json(403, { error: 'Origin không hợp lệ.' });
        if (!request.headers['content-type']?.startsWith('application/json')) return json(415, { error: 'Yêu cầu JSON.' });
        const body = await readJson(request);
        if (!body || typeof body !== 'object' || typeof body.image !== 'string' || body.image.length > MAX_BODY - 1024) {
          return json(400, { error: 'Thiếu dữ liệu ảnh hợp lệ.' });
        }
        json(200, await worker.request({ action: 'predict', model: body.model, stage: body.stage || 'adapt', image: body.image }));
      } else json(404, { error: 'Không tìm thấy.' });
    } catch (error) {
      if (!response.headersSent) json(error.status || 500, { error: error.status ? error.message : 'Studio chưa xử lý được yêu cầu. Thử lại sau.' });
      else response.end();
    }
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const worker = new InferenceWorker();
  const server = createStudioServer(worker);
  const port = Number(process.env.PORT || 3000);
  server.listen(port, '127.0.0.1', () => console.log(`\n  WeatherLens Studio → http://127.0.0.1:${port}\n  Node.js + local CPU inference · Ctrl+C to stop\n`));
  server.on('error', error => { console.error(error.message); worker.close(); process.exitCode = 1; });
  for (const event of ['SIGINT', 'SIGTERM']) process.on(event, () => { worker.close(); server.close(); });
  process.on('exit', () => worker.child.kill());
}
