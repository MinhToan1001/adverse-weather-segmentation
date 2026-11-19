import { test } from 'node:test';
import assert from 'node:assert/strict';
import { once } from 'node:events';
import { createStudioServer } from '../server.mjs';

test('HTTP boundary validates uploads, origin and static paths; inference errors stay readable', async t => {
  const calls = [];
  const worker = { async request(data) {
    calls.push(data);
    if (data.action === 'metadata') return { models: [], samples: [] };
    if (data.action === 'sample') {
      if (data.sample !== 'clear') throw Object.assign(new Error('Không tìm thấy ảnh mẫu.'), { status: 422 });
      return { mime: 'image/png', content: Buffer.from('image-fixture').toString('base64') };
    }
    throw Object.assign(new Error('Tệp ảnh không hợp lệ.'), { status: 422 });
  } };
  const server = createStudioServer(worker);
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => { server.closeAllConnections(); server.close(); });
  const base = `http://127.0.0.1:${server.address().port}`;
  const get = (path, options) => fetch(base + path, options);
  const post = body => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body });
  const page = await get('/');
  assert.equal(page.status, 200);
  assert.match(await page.text(), /WeatherLens/);
  assert.match(page.headers.get('content-security-policy'), /frame-ancestors 'none'/);
  assert.equal((await get('/api/metadata')).status, 200);
  for (const path of ['/server.mjs', '/..%2f..%2f.env', '/api/samples/../../package.json', '/outputs/reviews/model.safetensors']) {
    assert.equal((await get(path)).status, 404, path);
  }
  assert.equal((await get('/api/predict', { method: 'POST' })).status, 415);
  assert.equal((await get('/api/predict', post('{'))).status, 400);
  assert.equal((await get('/api/predict', post('null'))).status, 400);
  assert.equal((await get('/api/predict', post('{}'))).status, 400);
  const crossSite = post('{}'); crossSite.headers.Origin = 'https://example.com';
  assert.equal((await get('/api/predict', crossSite)).status, 403);
  const error = await get('/api/predict', post(JSON.stringify({ image: 'invalid', model: 'segformer_b2' })));
  assert.equal(error.status, 422);
  assert.equal((await error.json()).error, 'Tệp ảnh không hợp lệ.');
  assert.equal(calls.at(-1).stage, 'adapt');
  const oversized = await get('/api/predict', post(' '.repeat(14 * 1024 * 1024 + 1)));
  assert.equal(oversized.status, 413);
  assert.equal((await get('/api/samples/clear')).status, 200);
  assert.equal((await get('/api/samples/clear')).status, 200);
  assert.equal(calls.filter(c => c.action === 'sample').length, 1, 'Sample response is cached');
  assert.equal((await get('/api/samples/missing')).status, 422);
});
