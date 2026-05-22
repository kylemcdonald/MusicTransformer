# Music Transformer WebGPU

Browser port for the local Google/Magenta Music Transformer export.

```sh
npm install
npm run export:model
npm run dev
```

The model assets are generated into `public/manual-model/` from
`checkpoint/unconditional_model_16.ckpt`. They are intentionally git-ignored
because the raw float32 weight file is about 194 MB.

The browser runtime does not use TF.js. It loads the checkpoint tensors directly,
runs the 16-layer decoder with WebGPU compute shaders, samples performance tokens
one at a time, and streams completed notes into the synth, piano roll, and MIDI
writer. Generation defaults to an 8192-token song sequence with a 1024-token KV
attention cache.
