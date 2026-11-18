import { test, expect } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const output = path.resolve('outputs/web-studio');

test('real B0/B2 inference, same-image comparison, mask download and responsive upload flow', async ({ page, request }) => {
  await mkdir(output, { recursive: true });
  const errors = [];
  const screenshot = async file => {
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: path.join(output, file), fullPage: true });
  };
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  const metadata = await (await request.get('/api/metadata')).json();
  expect(metadata.models.filter(m => m.stages.adapt)).toHaveLength(2);
  await page.goto('/');
  await expect(page.locator('[data-model="segformer_b0"]')).toBeVisible();
  await expect(page.locator('#samples button')).toHaveCount(5);
  await expect(page.locator('#runButton')).toBeDisabled();
  await screenshot('desktop-empty.png');
  await page.getByLabel('Chọn ảnh mẫu Trời quang').click();
  await expect(page.locator('#resolution')).toContainText('2048');
  await expect(page.locator('#runButton')).toBeEnabled();
  await page.locator('#runButton').click();
  await expect(page.locator('#resultStrip')).toBeVisible({ timeout: 150_000 });
  await expect(page.locator('#resultTitle')).toContainText('SegFormer B0');
  const classRows = await page.locator('.class-row').count();
  expect(classRows).toBeGreaterThan(3);
  await page.locator('[data-mode="compare"]').click();
  await page.locator('#runButton').click();
  await expect(page.locator('.compare-mask')).toHaveCount(2, { timeout: 150_000 });
  await expect(page.locator('#busyLayer')).toBeHidden({ timeout: 150_000 });
  await page.locator('[data-model="segformer_b2"]').click();
  await expect(page.locator('#resultTitle')).toContainText('SegFormer B2');
  await screenshot('desktop-comparison.png');
  await page.locator('[data-mode="single"]').click();
  await page.locator('#opacity').evaluate(el => { el.value = '75'; el.dispatchEvent(new Event('input', { bubbles: true })); });
  await expect(page.locator('#opacityValue')).toHaveText('75%');
  await expect(page.locator('#maskImage')).toHaveCSS('opacity', '0.75');
  await page.locator('[data-view="mask"]').click();
  await expect(page.locator('#maskImage')).toHaveCSS('opacity', '1');
  await page.locator('[data-view="original"]').click();
  await expect(page.locator('#maskImage')).toBeHidden();
  await page.locator('[data-view="overlay"]').click();
  const downloaded = page.waitForEvent('download');
  await page.locator('[data-download="labels"]').click();
  const labels = await downloaded;
  expect(labels.suggestedFilename()).toBe('segformer_b2_adapt_labels.png');
  await labels.saveAs(path.join(output, 'download-labels.png'));
  for (const kind of ['mask', 'overlay']) {
    const download = page.waitForEvent('download');
    await page.locator(`[data-download="${kind}"]`).click();
    await (await download).saveAs(path.join(output, `download-${kind}.png`));
  }
  await screenshot('desktop-b2.png');
  // Loading another image must invalidate previous inference and downloads.
  await page.getByLabel('Chọn ảnh mẫu Ban đêm').click();
  await expect(page.locator('#resultStrip')).toBeHidden();
  await expect(page.locator('#maskImage')).toBeHidden();
  await page.locator('#runButton').click();
  await expect(page.locator('#resultStrip')).toBeVisible({ timeout: 150_000 });
  await screenshot('desktop-night.png');
  // Switching checkpoint stage also invalidates old results.
  await page.locator('#stageSelect').selectOption('source');
  await expect(page.locator('#resultStrip')).toBeHidden();
  await expect(page.locator('#modelScore')).toContainText('48.44%');
  await page.locator('#runButton').click();
  await expect(page.locator('#resultStrip')).toBeVisible({ timeout: 150_000 });
  await expect(page.locator('#resultTitle')).toContainText('SegFormer B2');
  await page.locator('#stageSelect').selectOption('adapt');
  await expect(page.locator('#resultStrip')).toBeHidden();
  // Invalid files produce a readable error and never trigger inference.
  await page.locator('#fileInput').setInputFiles({ name: 'broken.png', mimeType: 'image/png', buffer: Buffer.from('not-an-image') });
  await expect(page.locator('#notice')).toContainText('hợp lệ');
  // Real upload flow uses the same raw local sample but passes through FileReader.
  const imageResponse = await request.get('/api/samples/fog');
  await page.locator('#fileInput').setInputFiles({ name: 'my-fog.png', mimeType: 'image/png', buffer: await imageResponse.body() });
  await expect(page.locator('#fileName')).toHaveText('my-fog.png');
  await expect(page.locator('#notice')).toBeHidden();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator('body')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await screenshot('mobile.png');
  await page.getByLabel('Hướng dẫn sử dụng').click();
  await expect(page.locator('#helpDialog')).toBeVisible();
  await page.getByLabel('Đóng hướng dẫn').click();
  expect(errors).toEqual([]);
  await writeFile(path.join(output, 'browser-verification.json'), JSON.stringify({ status: 'PASS',
    actualLocalCheckpoints: ['segformer_b0', 'segformer_b2'], realSampleInputs: ['clear', 'night', 'fog'],
    modelComparison: true, uploadAndInvalidFile: true, stageInvalidation: true, sourceInference: true,
    maskDownload: true, colorAndOpacityOverlayDownloads: true,
    mobileViewport: 390, browserErrors: errors, accuracyClaims: false }, null, 2));
});
