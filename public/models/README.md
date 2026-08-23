Place RTMPose ONNX checkpoints here for offline use:

- `rtmpose-s.onnx`
- `rtmpose-m.onnx`

These are SimCC body models (17 COCO keypoints, 256×192). If a local file is
missing, the worker downloads it from Hugging Face on first run and caches it
in the browser Cache Storage.

Example:

```bash
curl -L -o public/models/rtmpose-s.onnx \
  https://huggingface.co/bukuroo/RTMPose-ONNX/resolve/main/rtmpose-s.onnx
```
