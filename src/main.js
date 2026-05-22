import './styles.css';

import { Circle, CircleDot, createIcons, Download, Pause, Play, RotateCcw, Square } from 'lucide';

import { backendName, generateTokenStream, loadMusicTransformer } from './model.js';
import {
  encodeNotesToPerformanceTokens,
  PerformanceStreamDecoder
} from './performance_codec.js';
import { writeMidiBlob } from './midi.js';
import { MidiPlayer } from './player.js';
import { PianoRoll } from './piano_roll.js';

const MAX_TOKENS = 8192;
const TOKENS_PER_FRAME = 16;
const PREFIX_TOKENS_PER_FRAME = 256;
const FIRST_NOTE_LEAD_IN = 1;
const PAD_SEED_TOKEN = 0;
const KEYBOARD_NOTE_VELOCITY = 108;

const KEYBOARD_PITCHES = new Map([
  ['KeyA', 60],
  ['KeyW', 61],
  ['KeyS', 62],
  ['KeyE', 63],
  ['KeyD', 64],
  ['KeyF', 65],
  ['KeyT', 66],
  ['KeyG', 67],
  ['KeyY', 68],
  ['KeyH', 69],
  ['KeyU', 70],
  ['KeyJ', 71],
  ['KeyK', 72],
  ['KeyO', 73],
  ['KeyL', 74],
  ['KeyP', 75]
]);

const generateButton = document.querySelector('#generate');
const recordButton = document.querySelector('#record');
const regenerateButton = document.querySelector('#regenerate');
const stopButton = document.querySelector('#stop');
const playButton = document.querySelector('#play');
const downloadButton = document.querySelector('#download');
const synthSelect = document.querySelector('#synth-select');
const previousRunsSelect = document.querySelector('#previous-runs');
const statusText = document.querySelector('#status');
const pianoRoll = new PianoRoll(document.querySelector('#piano-roll'));
const player = new MidiPlayer();

const GHOST_NOTE_OPACITY = 0.2;
const MAX_PREVIOUS_RUNS = 12;

let model = null;
let decoder = new PerformanceStreamDecoder();
let generatedTokens = [];
let generatedNotes = [];
let visibleNotes = [];
let userNotes = [];
let ghostNotes = [];
let previousRuns = [];
let renderedNotes = [];
let midiUrl = null;
let generating = false;
let pendingGeneration = null;
let currentAbortController = null;
let timeOffset = 0;
let hasTimeOffset = false;
let streamingStarted = false;
let cursorTime = 0;
let userNoteId = 1;
let previousRunId = 1;
let selectedPreviousRunId = '';
let generationCursorTime = null;
let recording = false;
let recordedSinceLastAutoGenerate = false;
let recordAnchorClock = 0;
let recordAnchorTime = 0;
let recordGenerationTimer = null;
const activeKeyboardNotes = new Map();

function setStatus(text) {
  statusText.textContent = text;
}

function renderLucideIcons() {
  createIcons({
    icons: { Circle, CircleDot, Download, Pause, Play, RotateCcw, Square }
  });
}

function setIconButton(button, icon, label) {
  if (button.dataset.icon === icon && button.title === label) return;
  button.dataset.icon = icon;
  button.innerHTML = `<i data-lucide="${icon}"></i>`;
  button.setAttribute('aria-label', label);
  button.title = label;
  renderLucideIcons();
}

function setRecordButtonState() {
  const icon = recording ? 'circle-dot' : 'circle';
  const label = recording ? 'Recording' : 'Record';
  recordButton.classList.toggle('is-recording', recording);
  recordButton.setAttribute('aria-pressed', String(recording));
  if (recordButton.dataset.icon === icon && recordButton.dataset.label === label) return;

  recordButton.dataset.icon = icon;
  recordButton.dataset.label = label;
  recordButton.innerHTML = `<i data-lucide="${icon}"></i><span>${label}</span>`;
  recordButton.title = label;
  renderLucideIcons();
}

function noteDuration(notes) {
  return notes.reduce((max, note) => Math.max(max, note.end), 0);
}

function latestNoteEnd(notes) {
  if (!notes.length) return null;
  return notes.reduce((max, note) => Math.max(max, note.end), 0);
}

function cloneNotes(notes) {
  return notes.map((note) => ({ ...note }));
}

function renderPreviousRunOptions() {
  previousRunsSelect.innerHTML = '';

  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = previousRuns.length ? 'Select version' : 'No saved versions';
  previousRunsSelect.append(placeholder);

  for (const run of previousRuns) {
    const option = document.createElement('option');
    option.value = run.id;
    option.textContent = run.label;
    previousRunsSelect.append(option);
  }

  previousRunsSelect.disabled = previousRuns.length === 0;
  previousRunsSelect.value = selectedPreviousRunId;
}

function versionNotes(notes, id) {
  return notes
    .filter((note) => note.end > note.start)
    .map((note, index) => ({
      ...note,
      id: `version-${id}-${index}`,
      ghost: false,
      opacity: undefined,
      provisional: false
    }))
    .sort((a, b) => a.start - b.start || a.pitch - b.pitch);
}

function ghostNotesAfterCursor(notes, cutoff, id) {
  return notes
    .filter((note) => note.end > cutoff)
    .map((note, index) => ({
      ...note,
      id: `ghost-${id}-${index}`,
      start: Math.max(cutoff, note.start),
      ghost: true,
      opacity: GHOST_NOTE_OPACITY,
      provisional: false
    }))
    .filter((note) => note.end > note.start);
}

function savePreviousRun(notes, {
  cutoff = cursorTime,
  select = false,
  showGhostTail = false,
  label
} = {}) {
  const id = String(previousRunId++);
  const playable = versionNotes(notes, id);
  if (!playable.length) return null;

  const run = {
    id,
    label: label || (cutoff > 0
      ? `Version ${id} @ ${cutoff.toFixed(2)}s (${playable.length} notes)`
      : `Version ${id} (${playable.length} notes)`),
    cursorTime: cutoff,
    notes: playable,
    ghostNotes: ghostNotesAfterCursor(playable, cutoff, id)
  };

  previousRuns = [run, ...previousRuns].slice(0, MAX_PREVIOUS_RUNS);
  if (select) selectedPreviousRunId = id;
  if (showGhostTail) ghostNotes = cloneNotes(run.ghostNotes);
  renderPreviousRunOptions();
  return run;
}

function loadPreviousRun(run) {
  player.stop();
  generatedTokens = [];
  decoder = new PerformanceStreamDecoder();
  generatedNotes = cloneNotes(run.notes);
  visibleNotes = generatedNotes.slice();
  userNotes = [];
  ghostNotes = [];
  timeOffset = 0;
  hasTimeOffset = false;
  streamingStarted = false;
  generationCursorTime = null;
  cursorTime = 0;
  selectedPreviousRunId = run.id;
  renderPreviousRunOptions();
  drawRoll(noteDuration(generatedNotes));
  updateDownload();
  setStatus(`${generatedNotes.length} notes - ${run.label}`);
}

function outputNotes() {
  return [...generatedNotes, ...userNotes]
    .filter((note) => note.end > note.start)
    .sort((a, b) => a.start - b.start || a.pitch - b.pitch);
}

function playableNotes() {
  return [...displayBaseNotes(), ...userNotes]
    .filter((note) => note.end > note.start)
    .sort((a, b) => a.start - b.start || a.pitch - b.pitch);
}

function displayBaseNotes() {
  return generating ? visibleNotes : generatedNotes;
}

function updateDownload() {
  if (midiUrl) {
    URL.revokeObjectURL(midiUrl);
  }

  const notes = outputNotes();
  const blob = writeMidiBlob(notes);
  midiUrl = URL.createObjectURL(blob);
  downloadButton.disabled = notes.length === 0;
}

function updateControls() {
  const notes = playableNotes();
  const hasNotes = notes.length > 0;
  setRecordButtonState();
  stopButton.disabled = !generating;
  playButton.disabled = !hasNotes;
  regenerateButton.disabled = !hasNotes;
  downloadButton.disabled = notes.length === 0;
  previousRunsSelect.disabled = generating || previousRuns.length === 0;

  if (player.playing) {
    setIconButton(playButton, 'pause', 'Pause');
  } else if (hasNotes && cursorTime <= 0.01) {
    setIconButton(playButton, 'rotate-ccw', 'Replay');
  } else {
    setIconButton(playButton, 'play', 'Play');
  }
}

function updateTimeOffset() {
  if (hasTimeOffset || decoder.firstNoteTime === null) return;
  timeOffset = decoder.firstNoteTime;
  hasTimeOffset = true;
}

function startStreamingWhenReady(notes, offset = 0) {
  if (streamingStarted || !notes.length) return;
  player.startStreaming(offset, { leadIn: FIRST_NOTE_LEAD_IN });
  streamingStarted = true;
}

function nowSeconds() {
  return performance.now() / 1000;
}

function resetRecordClock(anchorTime = cursorTime) {
  recordAnchorClock = nowSeconds();
  recordAnchorTime = Math.max(0, anchorTime);
}

function recordTime() {
  if (!recordAnchorClock) {
    resetRecordClock();
  }
  return recordAnchorTime + nowSeconds() - recordAnchorClock;
}

function cancelRecordGenerationTimer() {
  if (recordGenerationTimer) {
    clearTimeout(recordGenerationTimer);
    recordGenerationTimer = null;
  }
}

function keyboardEventTargetAllowsRecording(event) {
  if (event.metaKey || event.ctrlKey || event.altKey) return false;
  const tagName = event.target?.tagName;
  return !['INPUT', 'TEXTAREA', 'SELECT'].includes(tagName);
}

function updateActiveKeyboardNoteEnds() {
  if (!activeKeyboardNotes.size) return false;
  const time = recordTime();
  for (const entry of activeKeyboardNotes.values()) {
    entry.note.end = Math.max(entry.note.start + 0.05, time);
  }
  cursorTime = Math.max(cursorTime, time);
  return true;
}

function armRecordGeneration() {
  cancelRecordGenerationTimer();
  if (!recording || activeKeyboardNotes.size || !recordedSinceLastAutoGenerate) return;

  recordGenerationTimer = setTimeout(() => {
    recordGenerationTimer = null;
    generateFromRecordedInput();
  }, 0);
}

function prepareForKeyboardInput() {
  cancelRecordGenerationTimer();

  if (generating) {
    stopCurrentGeneration('Recording');
  } else if (player.playing) {
    cursorTime = player.pause();
    pianoRoll.setPlayhead(cursorTime);
    updateControls();
  }

  if (!activeKeyboardNotes.size) {
    resetRecordClock(cursorTime);
  }
}

async function startKeyboardNote(code, pitch) {
  prepareForKeyboardInput();
  const start = recordTime();
  const note = {
    id: `user-${userNoteId++}`,
    pitch,
    velocity: KEYBOARD_NOTE_VELOCITY,
    start,
    end: start + 0.05,
    provisional: true
  };

  activeKeyboardNotes.set(code, { note });
  userNotes.push(note);
  recordedSinceLastAutoGenerate = true;
  cursorTime = Math.max(cursorTime, note.end);
  drawRoll();

  await player.ensureStarted();
  if (activeKeyboardNotes.get(code)?.note === note) {
    player.triggerAttack(pitch, KEYBOARD_NOTE_VELOCITY / 127);
  }
}

function finishKeyboardNote(code) {
  const entry = activeKeyboardNotes.get(code);
  if (!entry) return;

  activeKeyboardNotes.delete(code);
  const end = Math.max(entry.note.start + 0.05, recordTime());
  entry.note.end = end;
  entry.note.provisional = false;
  cursorTime = Math.max(cursorTime, end);
  player.triggerRelease(entry.note.pitch);
  drawRoll();
  updateDownload();
  setStatus(`${outputNotes().length} notes`);
  armRecordGeneration();
}

function releaseKeyboardNotes({ finalize = true } = {}) {
  for (const code of [...activeKeyboardNotes.keys()]) {
    if (finalize) {
      finishKeyboardNote(code);
    } else {
      const entry = activeKeyboardNotes.get(code);
      activeKeyboardNotes.delete(code);
      player.triggerRelease(entry.note.pitch);
    }
  }
  cancelRecordGenerationTimer();
}

function generateFromRecordedInput() {
  if (!recording || activeKeyboardNotes.size || !recordedSinceLastAutoGenerate) return;

  const notes = playableNotes();
  if (!notes.length) return;

  cursorTime = Math.max(cursorTime, noteDuration(notes));
  const prefixTokens = encodeNotesToPerformanceTokens(notes, cursorTime);
  if (!prefixTokens.length) return;

  recordedSinceLastAutoGenerate = false;
  setStatus('Generating from recording');
  requestGeneration({
    prefixTokens,
    streamOffset: cursorTime
  });
}

function setRecording(nextRecording) {
  if (recording === nextRecording) return;
  recording = nextRecording;
  if (recording) {
    recordedSinceLastAutoGenerate = false;
    resetRecordClock(cursorTime);
    setStatus('Recording');
  } else {
    releaseKeyboardNotes();
    recordedSinceLastAutoGenerate = false;
    setStatus(outputNotes().length ? `${outputNotes().length} notes` : 'Ready');
  }
  updateControls();
}

function normalizeNote(note) {
  const offset = hasTimeOffset ? timeOffset : 0;
  return {
    ...note,
    start: Math.max(0, note.start - offset),
    end: Math.max(0, note.end - offset)
  };
}

function normalizeNotes(notes) {
  return notes
    .map(normalizeNote)
    .filter((note) => note.end > note.start);
}

function visibleDuration() {
  const offset = hasTimeOffset ? timeOffset : 0;
  return Math.max(0, decoder.currentTime - offset);
}

function drawRoll(duration = visibleDuration()) {
  renderedNotes = [...ghostNotes, ...displayBaseNotes(), ...userNotes]
    .filter((note) => note.end > note.start)
    .sort((a, b) => Number(Boolean(b.ghost)) - Number(Boolean(a.ghost))
      || a.start - b.start
      || a.pitch - b.pitch);
  pianoRoll.setNotes(renderedNotes, Math.max(duration, noteDuration(renderedNotes), 8));
  pianoRoll.setPlayhead(cursorTime);
  pianoRoll.setGenerationCursor(generationCursorTime);
  updateControls();
}

function refreshVisibleNotes({ useDecoderProgress = false } = {}) {
  updateTimeOffset();
  visibleNotes = normalizeNotes(decoder.snapshot({ includeOpen: true }));
  if (generating) {
    const noteEnd = latestNoteEnd(visibleNotes);
    generationCursorTime = useDecoderProgress
      ? Math.max(noteEnd ?? 0, visibleDuration())
      : noteEnd;
  } else {
    generationCursorTime = null;
  }
  drawRoll();
}

function finishGenerationNotes({ preserveCursor = false, stopPlayback = true } = {}) {
  updateTimeOffset();
  generatedNotes = normalizeNotes(decoder.finalizedNotes());
  visibleNotes = generatedNotes.slice();
  generationCursorTime = null;
  if (streamingStarted) {
    if (stopPlayback) {
      player.stop();
    }
    streamingStarted = false;
    if (stopPlayback && !preserveCursor) {
      cursorTime = 0;
    }
  }
  drawRoll();
  updateDownload();
}

function resetGeneration({ keepUserNotes = false, keepGhostNotes = false } = {}) {
  generatedTokens = [];
  generatedNotes = [];
  visibleNotes = [];
  if (!keepUserNotes) userNotes = [];
  if (!keepGhostNotes) {
    ghostNotes = [];
    selectedPreviousRunId = '';
    renderPreviousRunOptions();
  }
  decoder = new PerformanceStreamDecoder();
  timeOffset = 0;
  hasTimeOffset = false;
  streamingStarted = false;
  cursorTime = 0;
  generationCursorTime = null;
  player.stop();
  drawRoll(8);
  if (midiUrl) {
    URL.revokeObjectURL(midiUrl);
    midiUrl = null;
  }
  updateControls();
}

async function primeDecoder(prefixTokens, signal) {
  if (!prefixTokens.length) return;

  generationCursorTime = 0;
  drawRoll(Math.max(visibleDuration(), cursorTime));

  let index = 0;
  for (const token of prefixTokens) {
    if (signal?.aborted) throw new DOMException('Generation stopped', 'AbortError');
    generatedTokens.push(token);
    decoder.pushToken(token);

    if (index % PREFIX_TOKENS_PER_FRAME === 0) {
      timeOffset = 0;
      hasTimeOffset = true;
      refreshVisibleNotes({ useDecoderProgress: true });
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }

    index += 1;
  }

  timeOffset = 0;
  hasTimeOffset = true;
  visibleNotes = normalizeNotes(decoder.snapshot({ includeOpen: true }));
  generatedNotes = normalizeNotes(decoder.finalizedNotes());
  generationCursorTime = Math.max(latestNoteEnd(visibleNotes) ?? 0, visibleDuration(), cursorTime);
  drawRoll(Math.max(visibleDuration(), cursorTime));
}

function stopCurrentGeneration(status = 'Stopping') {
  if (!generating) return;
  currentAbortController?.abort();
  if (player.playing) {
    cursorTime = player.pause();
    pianoRoll.setPlayhead(cursorTime);
  } else {
    player.stop();
  }
  setStatus(status);
  updateControls();
}

async function ensureModel() {
  if (model) return model;

  model = await loadMusicTransformer((fraction) => {
    const percent = Math.round(fraction * 100);
    setStatus(`Loading model ${percent}%`);
  });
  setStatus(`WebGPU backend: ${backendName()}`);
  return model;
}

function requestGeneration(options = {}) {
  if (generating) {
    pendingGeneration = options;
    stopCurrentGeneration(options.prefixTokens?.length ? 'Regenerating' : 'Restarting');
    return;
  }

  startGeneration(options);
}

async function startGeneration({ prefixTokens = [], streamOffset = 0, preserveGhostNotes = false } = {}) {
  generating = true;
  currentAbortController = new AbortController();
  const { signal } = currentAbortController;

  try {
    resetGeneration({ keepGhostNotes: preserveGhostNotes });
    cursorTime = Math.max(0, streamOffset);
    await primeDecoder(prefixTokens, signal);
    if (!prefixTokens.length) {
      generationCursorTime = null;
    }
    drawRoll(Math.max(visibleDuration(), cursorTime));
    updateControls();

    await player.ensureStarted();
    if (signal.aborted) throw new DOMException('Generation stopped', 'AbortError');

    const musicTransformer = await ensureModel();
    if (signal.aborted) throw new DOMException('Generation stopped', 'AbortError');

    setStatus(prefixTokens.length
      ? `Regenerating from ${cursorTime.toFixed(2)}s`
      : `Generating 0/${MAX_TOKENS}`);

    const seedTokens = prefixTokens.length ? [PAD_SEED_TOKEN, ...prefixTokens] : undefined;
    let index = 0;
    for await (const token of generateTokenStream(musicTransformer, {
      maxTokens: MAX_TOKENS,
      seedTokens,
      signal
    })) {
      generatedTokens.push(token);
      const completedNotes = decoder.pushToken(token);
      updateTimeOffset();
      const normalizedCompleted = normalizeNotes(completedNotes);
      const latestCompletedEnd = latestNoteEnd(normalizedCompleted);
      if (latestCompletedEnd !== null) {
        generationCursorTime = Math.max(generationCursorTime ?? 0, latestCompletedEnd);
      }
      startStreamingWhenReady(normalizedCompleted, cursorTime);
      player.scheduleStreamingNotes(normalizedCompleted);

      if (index % TOKENS_PER_FRAME === 0) {
        refreshVisibleNotes();
        setStatus(`Generating ${generatedTokens.length}/${MAX_TOKENS} - ${renderedNotes.length} notes`);
        await new Promise((resolve) => requestAnimationFrame(resolve));
      }

      index += 1;
    }

    finishGenerationNotes({ preserveCursor: true, stopPlayback: false });
    ghostNotes = [];
    savePreviousRun(outputNotes(), { cutoff: 0, select: true });
    drawRoll(noteDuration(outputNotes()));
    setStatus(`${outputNotes().length} notes, ${generatedTokens.length} tokens`);
  } catch (error) {
    finishGenerationNotes({ preserveCursor: error.name === 'AbortError' });
    if (error.name === 'AbortError') {
      setStatus(generatedTokens.length
        ? `Stopped - ${outputNotes().length} notes, ${generatedTokens.length} tokens`
        : 'Stopped');
    } else {
      console.error(error);
      setStatus(error.message || 'Generation failed');
    }
  } finally {
    generating = false;
    currentAbortController = null;
    updateControls();
    if (pendingGeneration) {
      const next = pendingGeneration;
      pendingGeneration = null;
      queueMicrotask(() => requestGeneration(next));
    }
  }
}

function notesLeftOfCursor(notes = outputNotes()) {
  return notes
    .filter((note) => note.start < cursorTime)
    .map((note) => ({
      ...note,
      end: Math.max(note.start + 0.01, Math.min(note.end, cursorTime))
    }));
}

function regenerateFromCursor() {
  const sourceNotes = playableNotes();
  const prefixTokens = encodeNotesToPerformanceTokens(sourceNotes, cursorTime);
  const previousRun = savePreviousRun(sourceNotes, {
    cutoff: cursorTime,
    showGhostTail: true
  });
  selectedPreviousRunId = '';
  renderPreviousRunOptions();
  if (!prefixTokens.length && cursorTime <= 0) {
    requestGeneration({ preserveGhostNotes: Boolean(previousRun) });
    return;
  }

  userNotes = [];
  generatedNotes = notesLeftOfCursor(sourceNotes);
  visibleNotes = generatedNotes.slice();
  updateDownload();
  requestGeneration({
    prefixTokens,
    streamOffset: prefixTokens.length ? cursorTime : 0,
    preserveGhostNotes: Boolean(previousRun)
  });
}

async function togglePlayback() {
  const notes = playableNotes();
  if (!notes.length) return;

  if (player.playing) {
    cursorTime = player.pause();
    pianoRoll.setPlayhead(cursorTime);
    updateControls();
    return;
  }

  await player.ensureStarted();
  const duration = noteDuration(notes);
  const offset = cursorTime >= duration - 0.05 ? 0 : cursorTime;
  cursorTime = offset;
  player.play(notes, { offset });
  pianoRoll.setPlayhead(cursorTime);
  updateControls();
}

generateButton.addEventListener('click', () => requestGeneration());
recordButton.addEventListener('click', () => setRecording(!recording));
regenerateButton.addEventListener('click', regenerateFromCursor);
stopButton.addEventListener('click', () => stopCurrentGeneration('Stopping'));
playButton.addEventListener('click', togglePlayback);

document.addEventListener('keydown', (event) => {
  if (!recording) return;
  const pitch = KEYBOARD_PITCHES.get(event.code);
  if (pitch === undefined || !keyboardEventTargetAllowsRecording(event)) return;

  event.preventDefault();
  if (event.repeat || activeKeyboardNotes.has(event.code)) return;

  startKeyboardNote(event.code, pitch).catch((error) => {
    console.error(error);
    setStatus(error.message || 'Could not play note');
  });
});

document.addEventListener('keyup', (event) => {
  if (!recording || !activeKeyboardNotes.has(event.code)) return;
  event.preventDefault();
  finishKeyboardNote(event.code);
});

document.addEventListener('keydown', (event) => {
  if (event.code !== 'Space') return;
  const tagName = event.target?.tagName;
  if (['BUTTON', 'SELECT', 'INPUT', 'TEXTAREA'].includes(tagName)) return;

  event.preventDefault();
  togglePlayback();
});

window.addEventListener('blur', () => {
  if (!recording || !activeKeyboardNotes.size) return;
  for (const code of [...activeKeyboardNotes.keys()]) {
    finishKeyboardNote(code);
  }
});

synthSelect.addEventListener('change', async () => {
  synthSelect.disabled = true;
  const previousStatus = statusText.textContent;
  try {
    setStatus(synthSelect.value === 'piano' ? 'Loading piano' : previousStatus);
    await player.setInstrument(synthSelect.value);
    setStatus(previousStatus);
  } catch (error) {
    console.error(error);
    setStatus(error.message || 'Could not load selected synth');
    synthSelect.value = 'synth';
    await player.setInstrument('synth');
  } finally {
    synthSelect.disabled = false;
  }
});

previousRunsSelect.addEventListener('change', () => {
  selectedPreviousRunId = previousRunsSelect.value;
  const run = previousRuns.find((entry) => entry.id === selectedPreviousRunId);
  if (!run) {
    renderPreviousRunOptions();
    return;
  }
  loadPreviousRun(run);
});

pianoRoll.setSeekHandler((seconds) => {
  cursorTime = seconds;
  pianoRoll.setPlayhead(cursorTime);
  if (player.playing) {
    player.play(playableNotes(), { offset: cursorTime });
  }
  updateControls();
});

pianoRoll.setNoteDrawHandler((note) => {
  const created = {
    ...note,
    id: `user-${userNoteId++}`,
    provisional: false
  };
  userNotes.push(created);
  cursorTime = created.end;
  drawRoll();
  updateDownload();
  setStatus(`${outputNotes().length} notes`);
});

downloadButton.addEventListener('click', () => {
  if (!midiUrl) return;
  const link = document.createElement('a');
  link.href = midiUrl;
  link.download = `music-transformer-${Date.now()}.mid`;
  link.click();
});

function animate() {
  const playhead = player.playhead();
  if (playhead !== null) {
    cursorTime = playhead;
  }
  if (recording && updateActiveKeyboardNoteEnds()) {
    drawRoll();
  } else {
    pianoRoll.setPlayhead(cursorTime);
    updateControls();
  }
  requestAnimationFrame(animate);
}

setIconButton(stopButton, 'square', 'Stop');
setIconButton(downloadButton, 'download', 'Download MIDI');
setRecordButtonState();
renderPreviousRunOptions();
updateControls();
animate();
