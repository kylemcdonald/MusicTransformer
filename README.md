# Music Transformer WebGPU

Browser frontend for running Google/Magenta's Music Transformer with WebGPU.

Live app: https://kylemcdonald.github.io/MusicTransformer/

The original Music Transformer project was announced by Magenta in
[Music Transformer: Generating Music with Long-Term Structure](https://magenta.withgoogle.com/music-transformer).
This repository keeps the present working tree focused on the browser app.

## Run Locally

```sh
npm install
npm run dev
```

The app loads the fp16 manual model from `public/manual-model/`, generates MIDI
events in the browser, plays them with Tone.js, visualizes them in a piano roll,
and can import, edit, branch, replay, and download MIDI files.

## Build

```sh
npm run build
npm run preview
```

GitHub Pages deploys the Vite build from `.github/workflows/deploy.yml`.
