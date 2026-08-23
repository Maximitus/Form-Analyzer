import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const distDir = path.dirname(require.resolve('onnxruntime-web'));
const destDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../public/ort');

fs.mkdirSync(destDir, { recursive: true });

// Cloudflare Workers assets cap files at 25 MiB. Keep the SIMD WASM backend
// (WebGPU uses JSEP ~26MB, which cannot be hosted same-origin on Workers).
const allow = new Set([
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.mjs',
]);
const files = fs.readdirSync(distDir).filter((name) => allow.has(name));

if (files.length === 0) {
  console.warn(`[copy-ort-wasm] no ort-wasm binaries found in ${distDir}`);
  process.exit(0);
}

for (const file of files) {
  fs.copyFileSync(path.join(distDir, file), path.join(destDir, file));
}

console.log(`[copy-ort-wasm] copied ${files.length} files to public/ort`);
