import * as Tone from 'tone';
import { Piano } from '@tonejs/piano/build/piano/Piano';

function pitchToFrequency(pitch) {
  return 440 * Math.pow(2, (pitch - 69) / 12);
}

function clampVelocity(velocity) {
  return Math.max(0.08, Math.min(0.98, (velocity || 100) / 127));
}

export class MidiPlayer {
  constructor() {
    this.instrumentType = 'synth';
    this.scheduledIds = new Set();
    this.timers = new Set();
    this.heldPianoNotes = new Set();
    this.activeSynthPitches = new Set();
    this.playing = false;
    this.startedAt = 0;
    this.duration = 0;
    this.offset = 0;
    this.pianoPromise = null;
    this.createSynth();
  }

  createSynth() {
    this.synth = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: 'triangle' },
      envelope: { attack: 0.006, decay: 0.18, sustain: 0.34, release: 0.8 }
    }).toDestination();
    this.synth.volume.value = -10;
  }

  async createPiano() {
    if (!this.pianoPromise) {
      this.piano = new Piano({
        velocities: 5,
        minNote: 21,
        maxNote: 108,
        pedal: false,
        release: true,
        maxPolyphony: 48,
        volume: {
          strings: -8,
          pedal: -24,
          keybed: -24,
          harmonics: -20
        }
      }).toDestination();
      this.pianoPromise = this.piano.load();
    }

    await this.pianoPromise;
  }

  async setInstrument(type) {
    if (type === this.instrumentType) return;
    if (type === 'piano') {
      await this.createPiano();
    }
    this.releaseHeldNotes();
    this.instrumentType = type;
  }

  async ensureStarted() {
    await Tone.start();
    if (this.instrumentType === 'piano') {
      await this.createPiano();
    }
  }

  clearTimers() {
    for (const timer of this.timers) {
      clearTimeout(timer);
    }
    this.timers.clear();
  }

  releaseHeldNotes() {
    if (this.piano?.loaded) {
      this.piano.stopAll();
    }
    this.heldPianoNotes.clear();
    this.synth.releaseAll?.(Tone.now());
    for (const pitch of this.activeSynthPitches) {
      this.synth.triggerRelease(pitchToFrequency(pitch));
    }
    this.activeSynthPitches.clear();
  }

  stop() {
    this.playing = false;
    this.scheduledIds.clear();
    this.clearTimers();
    this.releaseHeldNotes();
  }

  pause() {
    const seconds = this.playhead() ?? this.offset;
    this.playing = false;
    this.scheduledIds.clear();
    this.clearTimers();
    this.releaseHeldNotes();
    return seconds;
  }

  startStreaming(offset = 0, { leadIn = 0.1 } = {}) {
    this.stop();
    this.playing = true;
    this.offset = Math.max(0, offset);
    const startAt = Tone.now() + leadIn;
    this.startedAt = startAt - this.offset;
    this.duration = this.offset;
  }

  scheduleStreamingNotes(notes) {
    if (!this.playing) return;
    for (const note of notes) {
      this.scheduleNote(note, this.startedAt, this.offset);
      this.duration = Math.max(this.duration, note.end);
    }
  }

  play(notes, { offset = 0, leadIn = 0.08 } = {}) {
    this.stop();
    this.playing = true;
    this.offset = Math.max(0, offset);
    const startAt = Tone.now() + leadIn;
    this.startedAt = startAt - this.offset;
    this.duration = notes.reduce((max, note) => Math.max(max, note.end), 0);

    for (const note of notes) {
      if (note.end > this.offset) {
        this.scheduleNote(note, this.startedAt, this.offset);
      }
    }
  }

  scheduleNote(note, zeroTime, offset = 0) {
    if (this.scheduledIds.has(note.id)) return;
    this.scheduledIds.add(note.id);

    const noteStart = Math.max(note.start, offset);
    const startAt = Math.max(zeroTime + noteStart, Tone.now() + 0.01);
    const duration = Math.max(0.035, note.end - noteStart);
    const velocity = clampVelocity(note.velocity);

    const attackDelay = Math.max(0, (startAt - Tone.now()) * 1000);
    const attackTimer = setTimeout(() => {
      this.timers.delete(attackTimer);
      if (!this.playing) return;
      this.triggerAttack(note.pitch, velocity);

      const releaseTimer = setTimeout(() => {
        this.timers.delete(releaseTimer);
        this.triggerRelease(note.pitch);
      }, duration * 1000);
      this.timers.add(releaseTimer);
    }, attackDelay);

    this.timers.add(attackTimer);
  }

  triggerAttack(pitch, velocity) {
    if (this.instrumentType === 'piano' && this.piano?.loaded) {
      this.heldPianoNotes.add(pitch);
      this.piano.keyDown({ midi: pitch, velocity });
      return;
    }

    this.synth.triggerAttack(pitchToFrequency(pitch), undefined, velocity);
    this.activeSynthPitches.add(pitch);
  }

  triggerRelease(pitch) {
    if (this.instrumentType === 'piano' && this.piano?.loaded) {
      if (this.heldPianoNotes.has(pitch)) {
        this.heldPianoNotes.delete(pitch);
        this.piano.keyUp({ midi: pitch, velocity: 0.7 });
      }
      return;
    }

    this.synth.triggerRelease(pitchToFrequency(pitch));
    this.activeSynthPitches.delete(pitch);
  }

  playhead() {
    if (!this.playing) return null;
    const seconds = Tone.now() - this.startedAt;
    if (this.duration && seconds > this.duration + 0.5) {
      this.playing = false;
      return null;
    }
    return Math.max(0, seconds);
  }
}
