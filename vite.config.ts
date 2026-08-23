import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {createRequire} from 'node:module';
import {defineConfig, loadEnv} from 'vite';
import {viteStaticCopy} from 'vite-plugin-static-copy';

const require = createRequire(import.meta.url);
const ortDist = path.dirname(require.resolve('onnxruntime-web'));

const coopCoepHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
} as const;

export default defineConfig(({mode}) => {
  const env = loadEnv(mode, '.', '');
  return {
    base: '/formanalyzer/',
    plugins: [
      react(),
      tailwindcss(),
      viteStaticCopy({
        targets: [
          {
            src: path.join(ortDist, 'ort-wasm*.wasm'),
            dest: 'ort',
          },
          {
            src: path.join(ortDist, 'ort-wasm*.mjs'),
            dest: 'ort',
          },
        ],
      }),
    ],
    define: {
      'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
      conditions: ['onnxruntime-web-use-extern-wasm'],
    },
    optimizeDeps: {
      exclude: ['onnxruntime-web'],
    },
    worker: {
      format: 'es',
      rollupOptions: {
        output: {
          inlineDynamicImports: true,
        },
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      hmr: process.env.DISABLE_HMR !== 'true',
      headers: coopCoepHeaders,
      fs: {
        allow: ['.', ortDist],
      },
    },
    preview: {
      headers: coopCoepHeaders,
    },
  };
});
