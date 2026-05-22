export const EOS_ID = 1;
export const NUM_RESERVED_IDS = 2;
export const STEPS_PER_SECOND = 100;
export const MIN_PITCH = 21;
export const MAX_PITCH = 108;
export const NUM_VELOCITY_BINS = 32;
export const MIN_VELOCITY = 1;
export const MAX_VELOCITY = 127;

const NOTE_COUNT = MAX_PITCH - MIN_PITCH + 1;
const NOTE_ON_START = NUM_RESERVED_IDS;
const NOTE_OFF_START = NOTE_ON_START + NOTE_COUNT;
const TIME_SHIFT_START = NOTE_OFF_START + NOTE_COUNT;
const VELOCITY_START = TIME_SHIFT_START + STEPS_PER_SECOND;
const VOCAB_SIZE = VELOCITY_START + NUM_VELOCITY_BINS;

function velocityBinToVelocity(bin) {
  const binSize = Math.ceil((MAX_VELOCITY - MIN_VELOCITY + 1) / NUM_VELOCITY_BINS);
  return MIN_VELOCITY + (bin - 1) * binSize;
}

function velocityToBin(velocity) {
  const binSize = Math.ceil((MAX_VELOCITY - MIN_VELOCITY + 1) / NUM_VELOCITY_BINS);
  return Math.max(1, Math.min(NUM_VELOCITY_BINS, Math.floor((velocity - MIN_VELOCITY) / binSize) + 1));
}

function pushTimeShiftTokens(tokens, steps) {
  let remaining = Math.max(0, steps);
  while (remaining > 0) {
    const shift = Math.min(STEPS_PER_SECOND, remaining);
    tokens.push(TIME_SHIFT_START + shift - 1);
    remaining -= shift;
  }
}

export function splitAtEos(tokens) {
  const index = tokens.indexOf(EOS_ID);
  if (index === -1) {
    return { tokens, eos: false };
  }
  return { tokens: tokens.slice(0, index), eos: true };
}

export function newTokensFromModelOutput(prefix, outputTokens) {
  if (
    outputTokens.length >= prefix.length &&
    prefix.every((token, index) => outputTokens[index] === token)
  ) {
    return outputTokens.slice(prefix.length);
  }

  return outputTokens;
}

export function tokenEvent(token, step) {
  if (token === EOS_ID) return { type: 'eos', step };
  if (token < NUM_RESERVED_IDS || token >= VOCAB_SIZE) return { type: 'invalid', step };
  if (token < NOTE_OFF_START) {
    return { type: 'note_on', step, pitch: MIN_PITCH + token - NOTE_ON_START };
  }
  if (token < TIME_SHIFT_START) {
    return { type: 'note_off', step, pitch: MIN_PITCH + token - NOTE_OFF_START };
  }
  if (token < VELOCITY_START) {
    const steps = token - TIME_SHIFT_START + 1;
    return { type: 'time_shift', step, endStep: step + steps, steps };
  }
  const bin = token - VELOCITY_START + 1;
  return { type: 'velocity', step, velocity: velocityBinToVelocity(bin) };
}

export function tokensUpToTime(tokens, cutoffSeconds) {
  const cutoffStep = Math.max(0, Math.round(cutoffSeconds * STEPS_PER_SECOND));
  const prefix = [];
  let step = 0;

  for (const token of tokens) {
    const event = tokenEvent(token, step);
    if (event.type === 'eos') break;
    if (event.type === 'invalid') continue;

    if (event.type === 'time_shift') {
      if (step >= cutoffStep) break;
      if (event.endStep <= cutoffStep) {
        prefix.push(token);
        step = event.endStep;
        continue;
      }
      pushTimeShiftTokens(prefix, cutoffStep - step);
      break;
    }

    if (step > cutoffStep) break;
    prefix.push(token);
  }

  return prefix;
}

export function encodeNotesToPerformanceTokens(notes, cutoffSeconds = Infinity) {
  const cutoffStep = Number.isFinite(cutoffSeconds)
    ? Math.max(0, Math.round(cutoffSeconds * STEPS_PER_SECOND))
    : Infinity;
  const events = [];

  for (const note of notes) {
    const pitch = Math.max(MIN_PITCH, Math.min(MAX_PITCH, Math.round(note.pitch)));
    const startStep = Math.max(0, Math.round(note.start * STEPS_PER_SECOND));
    const endStep = Math.max(startStep + 1, Math.round(note.end * STEPS_PER_SECOND));
    if (startStep > cutoffStep) continue;

    const velocity = Math.max(MIN_VELOCITY, Math.min(MAX_VELOCITY, Math.round(note.velocity || 100)));
    events.push({ step: startStep, order: 1, type: 'velocity', velocity });
    events.push({ step: startStep, order: 2, type: 'note_on', pitch });
    if (endStep <= cutoffStep) {
      events.push({ step: endStep, order: 0, type: 'note_off', pitch });
    }
  }

  events.sort((a, b) => a.step - b.step || a.order - b.order || (a.pitch || 0) - (b.pitch || 0));

  const tokens = [];
  let step = 0;
  let velocityBin = null;
  for (const event of events) {
    if (event.step > step) {
      pushTimeShiftTokens(tokens, event.step - step);
      step = event.step;
    }

    if (event.type === 'velocity') {
      const nextBin = velocityToBin(event.velocity);
      if (nextBin !== velocityBin) {
        tokens.push(VELOCITY_START + nextBin - 1);
        velocityBin = nextBin;
      }
      continue;
    }

    if (event.type === 'note_on') {
      tokens.push(NOTE_ON_START + event.pitch - MIN_PITCH);
      continue;
    }

    tokens.push(NOTE_OFF_START + event.pitch - MIN_PITCH);
  }

  if (Number.isFinite(cutoffStep) && cutoffStep > step) {
    pushTimeShiftTokens(tokens, cutoffStep - step);
  }

  return tokens;
}

export class PerformanceStreamDecoder {
  constructor() {
    this.reset();
  }

  reset() {
    this.step = 0;
    this.velocity = 100;
    this.notes = [];
    this.active = new Map();
    this.nextNoteId = 1;
    this.eos = false;
    this.firstNoteStep = null;
  }

  get currentTime() {
    return this.step / STEPS_PER_SECOND;
  }

  get firstNoteTime() {
    return this.firstNoteStep === null ? null : this.firstNoteStep / STEPS_PER_SECOND;
  }

  pushToken(token) {
    const completed = [];

    if (token === EOS_ID) {
      this.eos = true;
      return completed;
    }

    if (token < NUM_RESERVED_IDS || token >= VOCAB_SIZE) {
      return completed;
    }

    if (token < NOTE_OFF_START) {
      const pitch = MIN_PITCH + token - NOTE_ON_START;
      if (this.firstNoteStep === null) {
        this.firstNoteStep = this.step;
      }
      const stack = this.active.get(pitch) || [];
      stack.push({ step: this.step, velocity: this.velocity });
      this.active.set(pitch, stack);
      return completed;
    }

    if (token < TIME_SHIFT_START) {
      const pitch = MIN_PITCH + token - NOTE_OFF_START;
      const stack = this.active.get(pitch) || [];
      const start = stack.shift();
      if (stack.length) {
        this.active.set(pitch, stack);
      } else {
        this.active.delete(pitch);
      }

      if (start && this.step > start.step) {
        const note = this.createNote(pitch, start.step, this.step, start.velocity, false);
        this.notes.push(note);
        completed.push(note);
      }
      return completed;
    }

    if (token < VELOCITY_START) {
      this.step += token - TIME_SHIFT_START + 1;
      return completed;
    }

    const velocityBin = token - VELOCITY_START + 1;
    this.velocity = velocityBinToVelocity(velocityBin);
    return completed;
  }

  createNote(pitch, startStep, endStep, velocity, provisional) {
    return {
      id: provisional ? `active-${pitch}-${startStep}` : this.nextNoteId++,
      pitch,
      velocity,
      start: startStep / STEPS_PER_SECOND,
      end: endStep / STEPS_PER_SECOND,
      provisional
    };
  }

  snapshot({ includeOpen = true } = {}) {
    if (!includeOpen) {
      return this.notes.slice();
    }

    const notes = this.notes.slice();
    for (const [pitch, starts] of this.active.entries()) {
      for (const start of starts) {
        if (this.step > start.step) {
          notes.push(this.createNote(pitch, start.step, this.step, start.velocity, true));
        }
      }
    }
    return notes;
  }

  finalizedNotes() {
    return this.snapshot({ includeOpen: true })
      .filter((note) => note.end > note.start)
      .map((note) => ({ ...note, provisional: false }));
  }
}
