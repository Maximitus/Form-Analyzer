import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const destDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../public/models');
fs.mkdirSync(destDir, { recursive: true });

const models = {
  'rtmpose-s.onnx':
    'https://huggingface.co/bukuroo/RTMPose-ONNX/resolve/main/rtmpose-s.onnx',
  'rtmpose-m.onnx':
    'https://huggingface.co/bukuroo/RTMPose-ONNX/resolve/main/rtmpose-m.onnx',
};

const wanted = process.argv.includes('--m') ? ['rtmpose-m.onnx'] : ['rtmpose-s.onnx'];
if (process.argv.includes('--all')) {
  wanted.push('rtmpose-m.onnx');
}

for (const name of wanted) {
  const dest = path.join(destDir, name);
  if (fs.existsSync(dest) && fs.statSync(dest).size > 1_000_000) {
    console.log(`[download-models] ${name} already present`);
    continue;
  }
  console.log(`[download-models] fetching ${name}…`);
  try {
    const response = await fetch(models[name]);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const buffer = Buffer.from(await response.arrayBuffer());
    if (buffer.length < 1_000_000 || buffer[0] === 0x3c || buffer[0] === 0x7b) {
      throw new Error(`unexpected payload (${buffer.length} bytes)`);
    }
    fs.writeFileSync(dest, buffer);
    console.log(`[download-models] wrote ${name} (${buffer.length} bytes)`);
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    console.warn(`[download-models] ${name} skipped (${reason}); runtime will fetch from Hugging Face`);
  }
}
