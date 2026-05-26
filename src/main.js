import './styles.css';

import {
  Circle,
  CircleDot,
  createIcons,
  Download,
  Pause,
  Play,
  RotateCcw,
  SlidersHorizontal,
  Square,
  Upload,
  X
} from 'lucide';

import { backendName, generateTokenStream, loadMusicTransformer } from './model.js';
import {
  encodeNotesToPerformanceTokens,
  PerformanceStreamDecoder,
  STEPS_PER_SECOND,
  tokenEvent
} from './performance_codec.js';
import { writeMidiBlob } from './midi.js';
import { readMidiFile } from './midi_import.js';
import { MidiPlayer } from './player.js';
import { PianoRoll } from './piano_roll.js';

const MAX_TOKENS = 8192;
const TOKENS_PER_FRAME = 16;
const PREFIX_TOKENS_PER_FRAME = 256;
const FIRST_NOTE_LEAD_IN = 1;
const PAD_SEED_TOKEN = 0;
const KEYBOARD_NOTE_VELOCITY = 108;
const CURSOR_JUMP_EPSILON = 0.03;
const SETTINGS_STORAGE_KEY = 'music-transformer-generation-settings';

const DEFAULT_GENERATION_SETTINGS = Object.freeze({
  temperature: 1,
  topK: 64,
  maxNoteDuration: 1,
  minPitch: 21,
  maxPitch: 108,
  minTimeShift: 0.01,
  maxTimeShift: 1,
  minVelocity: 1,
  maxVelocity: 127,
  minTokens: 256,
  maxTokens: MAX_TOKENS
});

const GENERATION_SETTING_LIMITS = Object.freeze({
  temperature: { min: 0.05, max: 2, step: 0.05, digits: 2 },
  topK: { min: 1, max: 300, step: 1, digits: 0 },
  maxNoteDuration: { min: 0.05, max: 8, step: 0.05, digits: 2, unit: 's' },
  minPitch: { min: 21, max: 108, step: 1, digits: 0 },
  maxPitch: { min: 21, max: 108, step: 1, digits: 0 },
  minTimeShift: { min: 0.01, max: 1, step: 0.01, digits: 2, unit: 's' },
  maxTimeShift: { min: 0.01, max: 1, step: 0.01, digits: 2, unit: 's' },
  minVelocity: { min: 1, max: 127, step: 1, digits: 0 },
  maxVelocity: { min: 1, max: 127, step: 1, digits: 0 },
  minTokens: { min: 0, max: 2048, step: 1, digits: 0 },
  maxTokens: { min: 256, max: MAX_TOKENS, step: 256, digits: 0 }
});

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
const uploadMidiButton = document.querySelector('#upload-midi');
const midiFileInput = document.querySelector('#midi-file');
const regenerateButton = document.querySelector('#regenerate');
const stopButton = document.querySelector('#stop');
const playButton = document.querySelector('#play');
const downloadButton = document.querySelector('#download');
const synthSelect = document.querySelector('#synth-select');
const previousRunsSelect = document.querySelector('#previous-runs');
const settingsButton = document.querySelector('#settings');
const settingsDialog = document.querySelector('#settings-dialog');
const settingsResetButton = document.querySelector('#settings-reset');
const statusText = document.querySelector('#status');
const rollPanel = document.querySelector('.roll-panel');
const pianoRoll = new PianoRoll(document.querySelector('#piano-roll'));
const player = new MidiPlayer();

const settingFields = {
  temperature: {
    slider: document.querySelector('#temperature'),
    number: document.querySelector('#temperature-number'),
    output: document.querySelector('#temperature-value')
  },
  topK: {
    slider: document.querySelector('#top-k'),
    number: document.querySelector('#top-k-number'),
    output: document.querySelector('#top-k-value')
  },
  maxNoteDuration: {
    slider: document.querySelector('#max-note-duration'),
    number: document.querySelector('#max-note-duration-number'),
    output: document.querySelector('#max-note-duration-value')
  },
  minTokens: {
    slider: document.querySelector('#min-tokens'),
    number: document.querySelector('#min-tokens-number'),
    output: document.querySelector('#min-tokens-value')
  },
  maxTokens: {
    slider: document.querySelector('#max-tokens'),
    number: document.querySelector('#max-tokens-number'),
    output: document.querySelector('#max-tokens-value')
  }
};

const rangeSettingFields = {
  pitch: {
    minName: 'minPitch',
    maxName: 'maxPitch',
    sliderMin: document.querySelector('#min-pitch'),
    sliderMax: document.querySelector('#max-pitch'),
    numberMin: document.querySelector('#min-pitch-number'),
    numberMax: document.querySelector('#max-pitch-number'),
    output: document.querySelector('#pitch-range-value')
  },
  timeShift: {
    minName: 'minTimeShift',
    maxName: 'maxTimeShift',
    sliderMin: document.querySelector('#min-time-shift'),
    sliderMax: document.querySelector('#max-time-shift'),
    numberMin: document.querySelector('#min-time-shift-number'),
    numberMax: document.querySelector('#max-time-shift-number'),
    output: document.querySelector('#time-shift-range-value')
  },
  velocity: {
    minName: 'minVelocity',
    maxName: 'maxVelocity',
    sliderMin: document.querySelector('#min-velocity'),
    sliderMax: document.querySelector('#max-velocity'),
    numberMin: document.querySelector('#min-velocity-number'),
    numberMax: document.querySelector('#max-velocity-number'),
    output: document.querySelector('#velocity-range-value')
  }
};

const GHOST_NOTE_OPACITY = 0.2;
const MAX_PREVIOUS_RUNS = 12;

let model = null;
let generationSettings = loadGenerationSettings();
let decoder = createPerformanceDecoder();
let generatedTokens = [];
let generatedNotes = [];
let visibleNotes = [];
let userNotes = [];
let ghostNotes = [];
let previousRuns = [];
let renderedNotes = [];
let selectedNoteIds = new Set();
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
    icons: {
      Circle,
      CircleDot,
      Download,
      Pause,
      Play,
      RotateCcw,
      SlidersHorizontal,
      Square,
      Upload,
      X
    }
  });
}

function normalizeSettingValue(name, value, { maxOverride } = {}) {
  const limits = GENERATION_SETTING_LIMITS[name];
  const fallback = DEFAULT_GENERATION_SETTINGS[name];
  const max = maxOverride ?? limits.max;
  const min = Math.min(limits.min, max);
  let next = Number(value);
  if (!Number.isFinite(next)) next = fallback;
  next = Math.min(max, Math.max(min, next));
  next = Math.round(next / limits.step) * limits.step;
  next = Math.min(max, Math.max(min, next));
  return limits.digits > 0 ? Number(next.toFixed(limits.digits)) : Math.round(next);
}

function normalizeSettingRange(settings, minName, maxName) {
  const minValue = normalizeSettingValue(minName, settings[minName]);
  const maxValue = normalizeSettingValue(maxName, settings[maxName]);
  return minValue <= maxValue ? [minValue, maxValue] : [maxValue, minValue];
}

function normalizeGenerationSettings(settings = {}) {
  const maxTokens = normalizeSettingValue('maxTokens', settings.maxTokens);
  const [minPitch, maxPitch] = normalizeSettingRange(settings, 'minPitch', 'maxPitch');
  const [minTimeShift, maxTimeShift] = normalizeSettingRange(
    settings,
    'minTimeShift',
    'maxTimeShift'
  );
  const [minVelocity, maxVelocity] = normalizeSettingRange(
    settings,
    'minVelocity',
    'maxVelocity'
  );

  return {
    temperature: normalizeSettingValue('temperature', settings.temperature),
    topK: normalizeSettingValue('topK', settings.topK),
    maxNoteDuration: normalizeSettingValue('maxNoteDuration', settings.maxNoteDuration),
    minPitch,
    maxPitch,
    minTimeShift,
    maxTimeShift,
    minVelocity,
    maxVelocity,
    minTokens: normalizeSettingValue('minTokens', settings.minTokens, {
      maxOverride: Math.min(GENERATION_SETTING_LIMITS.minTokens.max, maxTokens)
    }),
    maxTokens
  };
}

function loadGenerationSettings() {
  try {
    const savedSettings = JSON.parse(localStorage.getItem(SETTINGS_STORAGE_KEY) || '{}');
    return normalizeGenerationSettings(savedSettings);
  } catch {
    return { ...DEFAULT_GENERATION_SETTINGS };
  }
}

function saveGenerationSettings() {
  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(generationSettings));
  } catch {
    // Local storage is optional; the current session still uses the setting.
  }
}

function formatSettingValue(name, value) {
  return GENERATION_SETTING_LIMITS[name].digits > 0 ? value.toFixed(2) : String(value);
}

function formatSettingOutput(name, value) {
  return `${formatSettingValue(name, value)}${GENERATION_SETTING_LIMITS[name].unit || ''}`;
}

function formatRangeOutput(name, minValue, maxValue) {
  if (name === 'timeShift') {
    return `${formatSettingOutput('minTimeShift', minValue)}-${formatSettingOutput('maxTimeShift', maxValue)}`;
  }
  return `${formatSettingValue(rangeSettingFields[name].minName, minValue)}-${formatSettingValue(rangeSettingFields[name].maxName, maxValue)}`;
}

function updateRangeSliderFill(field, minValue, maxValue) {
  const limits = GENERATION_SETTING_LIMITS[field.minName];
  const range = limits.max - limits.min;
  const start = range > 0 ? ((minValue - limits.min) / range) * 100 : 0;
  const end = range > 0 ? ((maxValue - limits.min) / range) * 100 : 100;
  field.sliderMin.parentElement.style.setProperty('--range-start', `${start}%`);
  field.sliderMin.parentElement.style.setProperty('--range-end', `${end}%`);
}

function renderGenerationSettings() {
  generationSettings = normalizeGenerationSettings(generationSettings);

  const minTokensMax = Math.min(
    GENERATION_SETTING_LIMITS.minTokens.max,
    generationSettings.maxTokens
  );
  settingFields.minTokens.slider.max = String(minTokensMax);
  settingFields.minTokens.number.max = String(minTokensMax);

  for (const [name, field] of Object.entries(settingFields)) {
    const value = generationSettings[name];
    const formatted = formatSettingValue(name, value);
    field.slider.value = String(value);
    field.number.value = formatted;
    field.output.value = formatted;
    field.output.textContent = formatSettingOutput(name, value);
  }

  for (const [name, field] of Object.entries(rangeSettingFields)) {
    const minValue = generationSettings[field.minName];
    const maxValue = generationSettings[field.maxName];
    const formattedMin = formatSettingValue(field.minName, minValue);
    const formattedMax = formatSettingValue(field.maxName, maxValue);
    field.sliderMin.value = String(minValue);
    field.sliderMax.value = String(maxValue);
    field.numberMin.value = formattedMin;
    field.numberMax.value = formattedMax;
    field.output.value = `${formattedMin}-${formattedMax}`;
    field.output.textContent = formatRangeOutput(name, minValue, maxValue);
    updateRangeSliderFill(field, minValue, maxValue);
  }
}

function setGenerationSetting(name, value) {
  generationSettings = normalizeGenerationSettings({
    ...generationSettings,
    [name]: value
  });
  renderGenerationSettings();
  saveGenerationSettings();
}

function setRangeGenerationSetting(name, side, value) {
  const field = rangeSettingFields[name];
  const settingName = side === 'min' ? field.minName : field.maxName;
  const nextValue = normalizeSettingValue(settingName, value);
  const nextSettings = { ...generationSettings };
  if (side === 'min') {
    nextSettings[field.minName] = Math.min(nextValue, generationSettings[field.maxName]);
  } else {
    nextSettings[field.maxName] = Math.max(nextValue, generationSettings[field.minName]);
  }
  generationSettings = normalizeGenerationSettings(nextSettings);
  renderGenerationSettings();
  saveGenerationSettings();
}

function createPerformanceDecoder({ maxNoteDuration = Infinity } = {}) {
  return new PerformanceStreamDecoder({ maxNoteDuration });
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

function prefixTokenTimes(tokens, maxTime) {
  const times = [0];
  let step = 0;
  for (const token of tokens) {
    const event = tokenEvent(token, step);
    if (event.type === 'eos') break;
    if (event.type === 'time_shift') {
      step = event.endStep;
    }
    times.push(Math.min(maxTime, step / STEPS_PER_SECOND));
  }
  return times;
}

function cloneNotes(notes) {
  return notes.map((note) => ({ ...note }));
}

function clearSelection() {
  selectedNoteIds = new Set();
  pianoRoll.setSelectedNoteIds(selectedNoteIds);
}

function setSelection(ids) {
  selectedNoteIds = new Set(ids);
  pianoRoll.setSelectedNoteIds(selectedNoteIds);
  updateControls();
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

function versionLabel(id, cutoff, count) {
  return cutoff > 0
    ? `Version ${id} @ ${cutoff.toFixed(2)}s (${count} notes)`
    : `Version ${id} (${count} notes)`;
}

function savePreviousRun(notes, {
  cutoff = cursorTime,
  id,
  select = false,
  showGhostTail = false,
  label
} = {}) {
  const runId = id || String(previousRunId++);
  const playable = versionNotes(notes, runId);
  if (!playable.length) return null;

  const run = {
    id: runId,
    label: label || versionLabel(runId, cutoff, playable.length),
    cursorTime: cutoff,
    notes: playable,
    ghostNotes: ghostNotesAfterCursor(playable, cutoff, runId)
  };

  const existingIndex = previousRuns.findIndex((entry) => entry.id === runId);
  if (existingIndex >= 0) {
    previousRuns = previousRuns.map((entry) => (entry.id === runId ? run : entry));
  } else {
    previousRuns = [run, ...previousRuns].slice(0, MAX_PREVIOUS_RUNS);
  }

  if (select) selectedPreviousRunId = runId;
  if (showGhostTail) ghostNotes = cloneNotes(run.ghostNotes);
  renderPreviousRunOptions();
  return run;
}

function activeVersion() {
  return previousRuns.find((entry) => entry.id === selectedPreviousRunId) || null;
}

function updateActiveVersionSnapshot() {
  const run = activeVersion();
  if (!run) return null;
  return savePreviousRun(outputNotes(), {
    id: run.id,
    cutoff: run.cursorTime,
    select: true
  });
}

function setGhostTailFromRun(run, cutoff) {
  ghostNotes = run ? ghostNotesAfterCursor(run.notes, cutoff, run.id) : [];
}

function loadPreviousRun(run) {
  updateActiveVersionSnapshot();
  player.stop();
  generatedTokens = [];
  decoder = createPerformanceDecoder();
  generatedNotes = cloneNotes(run.notes);
  visibleNotes = generatedNotes.slice();
  userNotes = [];
  ghostNotes = [];
  timeOffset = 0;
  hasTimeOffset = false;
  streamingStarted = false;
  generationCursorTime = null;
  cursorTime = 0;
  clearSelection();
  selectedPreviousRunId = run.id;
  renderPreviousRunOptions();
  pianoRoll.fitToContent();
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
  if (settingsDialog.open) return false;
  if (event.metaKey || event.ctrlKey || event.altKey) return false;
  const tagName = event.target?.tagName;
  return !['BUTTON', 'INPUT', 'TEXTAREA', 'SELECT'].includes(tagName);
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
  updateActiveVersionSnapshot();
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

function reschedulePlaybackIfRunning() {
  if (!player.playing) return;
  cursorTime = player.pause();
  const notes = playableNotes();
  if (notes.length) {
    player.play(notes, { offset: cursorTime });
  }
}

function forEachEditableNote(callback) {
  const seen = new Set();
  for (const list of [generatedNotes, visibleNotes, userNotes]) {
    for (const note of list) {
      if (seen.has(note.id)) continue;
      seen.add(note.id);
      callback(note);
    }
  }
}

function moveNotes(ids, deltaSeconds, deltaPitch) {
  const idSet = new Set(ids);
  if (!idSet.size) return;

  forEachEditableNote((note) => {
    if (!idSet.has(note.id)) return;
    const duration = Math.max(0.05, note.end - note.start);
    note.start = Math.max(0, note.start + deltaSeconds);
    note.end = note.start + duration;
    note.pitch = Math.max(21, Math.min(108, note.pitch + deltaPitch));
  });

  reschedulePlaybackIfRunning();
  drawRoll(noteDuration(outputNotes()));
  updateDownload();
  updateActiveVersionSnapshot();
}

function deleteSelectedNotes() {
  if (!selectedNoteIds.size) return;
  const keep = (note) => !selectedNoteIds.has(note.id);
  generatedNotes = generatedNotes.filter(keep);
  visibleNotes = visibleNotes.filter(keep);
  userNotes = userNotes.filter(keep);
  clearSelection();
  reschedulePlaybackIfRunning();
  drawRoll(noteDuration(outputNotes()));
  updateDownload();
  updateActiveVersionSnapshot();
  setStatus(`${outputNotes().length} notes`);
}

async function waitForGenerationToStop() {
  if (!generating) return;
  pendingGeneration = null;
  currentAbortController?.abort();
  player.stop();
  const started = performance.now();
  while (generating && performance.now() - started < 5000) {
    await new Promise((resolve) => setTimeout(resolve, 16));
  }
}

async function replaceWithNotes(notes, label) {
  await waitForGenerationToStop();
  releaseKeyboardNotes({ finalize: false });
  recording = false;
  player.stop();
  generatedTokens = [];
  decoder = createPerformanceDecoder();
  generatedNotes = cloneNotes(notes);
  visibleNotes = generatedNotes.slice();
  userNotes = [];
  ghostNotes = [];
  previousRuns = [];
  selectedPreviousRunId = '';
  timeOffset = 0;
  hasTimeOffset = false;
  streamingStarted = false;
  generationCursorTime = null;
  cursorTime = 0;
  clearSelection();
  pianoRoll.fitToContent();
  renderPreviousRunOptions();
  drawRoll(noteDuration(generatedNotes));
  updateDownload();
  setStatus(`${label}: ${generatedNotes.length} notes`);
}

async function importMidiFile(file) {
  if (!file) return;
  try {
    setStatus(`Importing ${file.name}`);
    const notes = readMidiFile(await file.arrayBuffer());
    await replaceWithNotes(notes, file.name);
  } catch (error) {
    console.error(error);
    setStatus(error.message || 'Could not import MIDI');
  } finally {
    midiFileInput.value = '';
    rollPanel.classList.remove('is-dragover');
  }
}

function findMidiFile(fileList) {
  return [...(fileList || [])].find((file) => /\.(mid|midi)$/i.test(file.name));
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
  selectedNoteIds = new Set([...selectedNoteIds].filter((id) => renderedNotes.some((note) => note.id === id && !note.ghost)));
  pianoRoll.setNotes(renderedNotes, Math.max(duration, noteDuration(renderedNotes), 8));
  pianoRoll.setSelectedNoteIds(selectedNoteIds);
  pianoRoll.setPlayhead(cursorTime);
  pianoRoll.setGenerationCursor(generationCursorTime);
  updateControls();
}

function refreshVisibleNotes({ useDecoderProgress = false, updateGenerationCursor = true } = {}) {
  updateTimeOffset();
  visibleNotes = normalizeNotes(decoder.snapshot({ includeOpen: true }));
  if (!updateGenerationCursor) {
    drawRoll();
    return;
  }

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
  decoder = createPerformanceDecoder();
  timeOffset = 0;
  hasTimeOffset = false;
  streamingStarted = false;
  cursorTime = 0;
  generationCursorTime = null;
  clearSelection();
  player.stop();
  pianoRoll.fitToContent();
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
      refreshVisibleNotes({ updateGenerationCursor: false });
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }

    index += 1;
  }

  timeOffset = 0;
  hasTimeOffset = true;
  visibleNotes = normalizeNotes(decoder.snapshot({ includeOpen: true }));
  generatedNotes = normalizeNotes(decoder.finalizedNotes());
  generationCursorTime = 0;
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

  updateActiveVersionSnapshot();
  startGeneration(options);
}

async function startGeneration({ prefixTokens = [], streamOffset = 0, preserveGhostNotes = false } = {}) {
  generating = true;
  currentAbortController = new AbortController();
  const { signal } = currentAbortController;
  const runSettings = normalizeGenerationSettings(generationSettings);

  try {
    selectedPreviousRunId = '';
    renderPreviousRunOptions();
    resetGeneration({ keepGhostNotes: preserveGhostNotes });
    cursorTime = Math.max(0, streamOffset);
    await primeDecoder(prefixTokens, signal);
    decoder.setMaxNoteDuration(runSettings.maxNoteDuration);
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
      : `Generating 0/${runSettings.maxTokens}`);

    const seedTokens = prefixTokens.length ? [PAD_SEED_TOKEN, ...prefixTokens] : undefined;
    const prefixTimes = prefixTokens.length ? prefixTokenTimes(prefixTokens, cursorTime) : [];
    let lastPrimeStatusAt = 0;
    let index = 0;
    for await (const token of generateTokenStream(musicTransformer, {
      maxTokens: runSettings.maxTokens,
      minTokens: runSettings.minTokens,
      temperature: runSettings.temperature,
      topK: runSettings.topK,
      minPitch: runSettings.minPitch,
      maxPitch: runSettings.maxPitch,
      minTimeShift: runSettings.minTimeShift,
      maxTimeShift: runSettings.maxTimeShift,
      minVelocity: runSettings.minVelocity,
      maxVelocity: runSettings.maxVelocity,
      seedTokens,
      signal,
      onSeedProgress: prefixTokens.length
        ? ({ completed, total }) => {
            const processedPrefixTokens = completed === total
              ? prefixTimes.length - 1
              : Math.max(0, completed - 2);
            const progressIndex = Math.min(processedPrefixTokens, prefixTimes.length - 1);
            generationCursorTime = prefixTimes[progressIndex] ?? 0;

            const now = performance.now();
            if (now - lastPrimeStatusAt > 100 || completed === total) {
              lastPrimeStatusAt = now;
              drawRoll(Math.max(visibleDuration(), cursorTime));
              setStatus(
                `Priming model ${completed}/${total} tokens - ${generationCursorTime.toFixed(2)}s/${cursorTime.toFixed(2)}s`
              );
            }
          }
        : undefined
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
        setStatus(`Generating ${generatedTokens.length}/${runSettings.maxTokens} - ${renderedNotes.length} notes`);
        await new Promise((resolve) => requestAnimationFrame(resolve));
      }

      index += 1;
    }

    finishGenerationNotes({ preserveCursor: true, stopPlayback: false });
    ghostNotes = [];
    savePreviousRun(outputNotes(), { cutoff: streamOffset, select: true });
    drawRoll(noteDuration(outputNotes()));
    setStatus(`${outputNotes().length} notes, ${generatedTokens.length} tokens`);
  } catch (error) {
    finishGenerationNotes({ preserveCursor: error.name === 'AbortError' });
    if (error.name === 'AbortError') {
      const savedRun = savePreviousRun(outputNotes(), { cutoff: streamOffset, select: true });
      setStatus(generatedTokens.length
        ? `Stopped - ${outputNotes().length} notes, ${generatedTokens.length} tokens${savedRun ? ` - ${savedRun.label}` : ''}`
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
  const previousRun = updateActiveVersionSnapshot()
    || savePreviousRun(sourceNotes, {
      cutoff: cursorTime,
      select: true
    });
  setGhostTailFromRun(previousRun, cursorTime);
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

function seekPlayback(seconds) {
  cursorTime = Math.max(0, seconds);
  pianoRoll.setPlayhead(cursorTime);
  if (player.playing) {
    player.play(playableNotes(), { offset: cursorTime });
  }
  updateControls();
}

function intersectingNotesAtCursor(notes) {
  return notes.filter((note) => (
    note.start <= cursorTime + CURSOR_JUMP_EPSILON &&
    note.end >= cursorTime - CURSOR_JUMP_EPSILON
  ));
}

function jumpToPreviousNoteStart() {
  const notes = playableNotes();
  if (!notes.length) return;

  const intersecting = intersectingNotesAtCursor(notes);
  if (intersecting.length) {
    const start = Math.min(...intersecting.map((note) => note.start));
    if (cursorTime - start > CURSOR_JUMP_EPSILON) {
      seekPlayback(start);
      return;
    }
  }

  const previousStarts = notes
    .map((note) => note.start)
    .filter((start) => start < cursorTime - CURSOR_JUMP_EPSILON);
  seekPlayback(previousStarts.length ? Math.max(...previousStarts) : 0);
}

function jumpToNextNoteEnd() {
  const notes = playableNotes();
  if (!notes.length) return;

  const intersecting = intersectingNotesAtCursor(notes);
  if (intersecting.length) {
    const end = Math.max(...intersecting.map((note) => note.end));
    if (end - cursorTime > CURSOR_JUMP_EPSILON) {
      seekPlayback(end);
      return;
    }
  }

  const nextEnds = notes
    .map((note) => note.end)
    .filter((end) => end > cursorTime + CURSOR_JUMP_EPSILON);
  if (nextEnds.length) {
    seekPlayback(Math.min(...nextEnds));
  }
}

generateButton.addEventListener('click', () => requestGeneration());
recordButton.addEventListener('click', () => setRecording(!recording));
regenerateButton.addEventListener('click', regenerateFromCursor);
stopButton.addEventListener('click', () => stopCurrentGeneration('Stopping'));
playButton.addEventListener('click', togglePlayback);
uploadMidiButton.addEventListener('click', () => midiFileInput.click());
settingsButton.addEventListener('click', () => {
  renderGenerationSettings();
  settingsDialog.showModal();
});
settingsDialog.addEventListener('click', (event) => {
  if (event.target === settingsDialog) {
    settingsDialog.close();
  }
});
settingsResetButton.addEventListener('click', () => {
  generationSettings = { ...DEFAULT_GENERATION_SETTINGS };
  renderGenerationSettings();
  saveGenerationSettings();
});
for (const [name, field] of Object.entries(settingFields)) {
  field.slider.addEventListener('input', () => setGenerationSetting(name, field.slider.value));
  field.number.addEventListener('change', () => setGenerationSetting(name, field.number.value));
}
for (const [name, field] of Object.entries(rangeSettingFields)) {
  field.sliderMin.addEventListener('input', () => setRangeGenerationSetting(name, 'min', field.sliderMin.value));
  field.sliderMax.addEventListener('input', () => setRangeGenerationSetting(name, 'max', field.sliderMax.value));
  field.numberMin.addEventListener('change', () => setRangeGenerationSetting(name, 'min', field.numberMin.value));
  field.numberMax.addEventListener('change', () => setRangeGenerationSetting(name, 'max', field.numberMax.value));
}
midiFileInput.addEventListener('change', () => {
  importMidiFile(midiFileInput.files?.[0]);
});

rollPanel.addEventListener('dragenter', (event) => {
  event.preventDefault();
  rollPanel.classList.add('is-dragover');
});

rollPanel.addEventListener('dragover', (event) => {
  event.preventDefault();
  rollPanel.classList.add('is-dragover');
});

rollPanel.addEventListener('dragleave', (event) => {
  if (!event.relatedTarget || !rollPanel.contains(event.relatedTarget)) {
    rollPanel.classList.remove('is-dragover');
  }
});

rollPanel.addEventListener('drop', (event) => {
  event.preventDefault();
  event.stopPropagation();
  rollPanel.classList.remove('is-dragover');
  importMidiFile(findMidiFile(event.dataTransfer?.files));
});

document.addEventListener('dragover', (event) => {
  event.preventDefault();
});

document.addEventListener('drop', (event) => {
  event.preventDefault();
  rollPanel.classList.remove('is-dragover');
  importMidiFile(findMidiFile(event.dataTransfer?.files));
});

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
  if (settingsDialog.open) return;

  if (event.code === 'Space') {
    event.preventDefault();
    event.stopPropagation();
    if (!event.repeat) togglePlayback();
    return;
  }

  if (event.key === 'Escape' && selectedNoteIds.size) {
    event.preventDefault();
    clearSelection();
    updateControls();
    return;
  }

  const tagName = event.target?.tagName;
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tagName)) return;

  if ((event.key === 'Delete' || event.key === 'Backspace') && selectedNoteIds.size) {
    event.preventDefault();
    deleteSelectedNotes();
    return;
  }

  if (event.key === 'ArrowLeft') {
    event.preventDefault();
    jumpToPreviousNoteStart();
    return;
  }

  if (event.key === 'ArrowRight') {
    event.preventDefault();
    jumpToNextNoteEnd();
  }
}, { capture: true });

document.addEventListener('keyup', (event) => {
  if (event.code !== 'Space') return;
  event.preventDefault();
  event.stopPropagation();
}, { capture: true });

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
  const nextRunId = previousRunsSelect.value;
  if (!nextRunId || nextRunId === selectedPreviousRunId) {
    renderPreviousRunOptions();
    return;
  }

  updateActiveVersionSnapshot();
  const run = previousRuns.find((entry) => entry.id === nextRunId);
  if (!run) {
    renderPreviousRunOptions();
    return;
  }
  loadPreviousRun(run);
});

pianoRoll.setSeekHandler((seconds) => {
  seekPlayback(seconds);
});

pianoRoll.setSelectionChangeHandler((ids) => {
  setSelection(ids);
});

pianoRoll.setNoteMoveHandler((ids, deltaSeconds, deltaPitch) => {
  moveNotes(ids, deltaSeconds, deltaPitch);
});

pianoRoll.setNoteDrawHandler((note) => {
  const created = {
    ...note,
    id: `user-${userNoteId++}`,
    provisional: false
  };
  userNotes.push(created);
  cursorTime = created.end;
  setSelection([created.id]);
  drawRoll();
  updateDownload();
  updateActiveVersionSnapshot();
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
setIconButton(uploadMidiButton, 'upload', 'Upload MIDI');
setIconButton(downloadButton, 'download', 'Download MIDI');
setRecordButtonState();
renderGenerationSettings();
renderPreviousRunOptions();
updateControls();
animate();
