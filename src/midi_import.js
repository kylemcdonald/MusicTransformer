const DEFAULT_TEMPO_US_PER_QUARTER = 500000;
const MIN_PITCH = 21;
const MAX_PITCH = 108;

class MidiReader {
  constructor(buffer) {
    this.view = new DataView(buffer);
    this.offset = 0;
  }

  remaining() {
    return this.view.byteLength - this.offset;
  }

  readUint8() {
    const value = this.view.getUint8(this.offset);
    this.offset += 1;
    return value;
  }

  readUint16() {
    const value = this.view.getUint16(this.offset, false);
    this.offset += 2;
    return value;
  }

  readUint32() {
    const value = this.view.getUint32(this.offset, false);
    this.offset += 4;
    return value;
  }

  readBytes(length) {
    const bytes = new Uint8Array(this.view.buffer, this.view.byteOffset + this.offset, length);
    this.offset += length;
    return bytes;
  }

  readChunkId() {
    return String.fromCharCode(...this.readBytes(4));
  }

  readVariableLength() {
    let value = 0;
    for (let i = 0; i < 4; i += 1) {
      const byte = this.readUint8();
      value = (value << 7) | (byte & 0x7f);
      if ((byte & 0x80) === 0) return value;
    }
    throw new Error('Invalid MIDI variable-length quantity.');
  }
}

function channelDataLength(status) {
  const type = status & 0xf0;
  if (type === 0xc0 || type === 0xd0) return 1;
  if (type >= 0x80 && type <= 0xe0) return 2;
  throw new Error(`Unsupported MIDI status byte 0x${status.toString(16)}.`);
}

function tempoMap(tempoEvents) {
  const sorted = [{ tick: 0, tempo: DEFAULT_TEMPO_US_PER_QUARTER }, ...tempoEvents]
    .sort((a, b) => a.tick - b.tick);
  const unique = [];

  for (const event of sorted) {
    if (unique.length && unique[unique.length - 1].tick === event.tick) {
      unique[unique.length - 1] = event;
    } else {
      unique.push(event);
    }
  }

  return unique;
}

function createTickConverter(tempoEvents, ticksPerQuarter) {
  const events = tempoMap(tempoEvents);
  const segments = [];
  let seconds = 0;
  let lastTick = 0;
  let tempo = DEFAULT_TEMPO_US_PER_QUARTER;

  for (const event of events) {
    if (event.tick > lastTick) {
      seconds += ((event.tick - lastTick) * tempo) / ticksPerQuarter / 1000000;
      lastTick = event.tick;
    }
    tempo = event.tempo;
    segments.push({ tick: lastTick, seconds, tempo });
  }

  return (tick) => {
    let segment = segments[0];
    for (const next of segments) {
      if (next.tick > tick) break;
      segment = next;
    }
    return segment.seconds + ((tick - segment.tick) * segment.tempo) / ticksPerQuarter / 1000000;
  };
}

function parseTrack(bytes, noteEvents, tempoEvents, orderStart) {
  const reader = new MidiReader(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  let tick = 0;
  let runningStatus = null;
  let order = orderStart;

  while (reader.remaining() > 0) {
    tick += reader.readVariableLength();
    let status = reader.readUint8();

    if (status < 0x80) {
      if (runningStatus === null) {
        throw new Error('MIDI running status appeared before a status byte.');
      }
      reader.offset -= 1;
      status = runningStatus;
    } else if (status < 0xf0) {
      runningStatus = status;
    }

    if (status === 0xff) {
      runningStatus = null;
      const metaType = reader.readUint8();
      const length = reader.readVariableLength();
      const data = reader.readBytes(length);
      if (metaType === 0x51 && length === 3) {
        tempoEvents.push({
          tick,
          tempo: (data[0] << 16) | (data[1] << 8) | data[2]
        });
      }
      if (metaType === 0x2f) break;
      continue;
    }

    if (status === 0xf0 || status === 0xf7) {
      runningStatus = null;
      reader.offset += reader.readVariableLength();
      continue;
    }

    const length = channelDataLength(status);
    const data1 = reader.readUint8();
    const data2 = length === 2 ? reader.readUint8() : 0;
    const type = status & 0xf0;
    const channel = status & 0x0f;

    if (type === 0x90 && data2 > 0) {
      noteEvents.push({ type: 'on', tick, order: order++, channel, pitch: data1, velocity: data2 });
    } else if (type === 0x80 || type === 0x90) {
      noteEvents.push({ type: 'off', tick, order: order++, channel, pitch: data1, velocity: data2 });
    }
  }

  return order;
}

function noteEventRank(event) {
  return event.type === 'off' ? 0 : 1;
}

export function readMidiFile(buffer) {
  const reader = new MidiReader(buffer);
  if (reader.readChunkId() !== 'MThd') {
    throw new Error('Not a standard MIDI file.');
  }

  const headerLength = reader.readUint32();
  const format = reader.readUint16();
  const trackCount = reader.readUint16();
  const division = reader.readUint16();
  reader.offset += Math.max(0, headerLength - 6);

  if (format > 1) {
    throw new Error('MIDI format 2 is not supported.');
  }
  if (division & 0x8000) {
    throw new Error('SMPTE-time MIDI files are not supported.');
  }

  const noteEvents = [];
  const tempoEvents = [];
  let order = 0;

  for (let track = 0; track < trackCount; track += 1) {
    const chunkId = reader.readChunkId();
    const length = reader.readUint32();
    const bytes = reader.readBytes(length);
    if (chunkId !== 'MTrk') continue;
    order = parseTrack(bytes, noteEvents, tempoEvents, order);
  }

  const ticksToSeconds = createTickConverter(tempoEvents, division);
  const active = new Map();
  const notes = [];

  noteEvents.sort((a, b) => a.tick - b.tick || noteEventRank(a) - noteEventRank(b) || a.order - b.order);

  for (const event of noteEvents) {
    const key = `${event.channel}:${event.pitch}`;
    if (event.type === 'on') {
      const stack = active.get(key) || [];
      stack.push(event);
      active.set(key, stack);
      continue;
    }

    const stack = active.get(key) || [];
    const start = stack.shift();
    if (stack.length) {
      active.set(key, stack);
    } else {
      active.delete(key);
    }
    if (!start || event.tick <= start.tick) continue;

    if (start.pitch < MIN_PITCH || start.pitch > MAX_PITCH) continue;
    notes.push({
      id: `midi-${notes.length + 1}`,
      pitch: start.pitch,
      velocity: Math.max(1, Math.min(127, start.velocity || 100)),
      start: ticksToSeconds(start.tick),
      end: ticksToSeconds(event.tick),
      provisional: false
    });
  }

  const playable = notes
    .filter((note) => note.end > note.start && note.pitch >= MIN_PITCH && note.pitch <= MAX_PITCH)
    .sort((a, b) => a.start - b.start || a.pitch - b.pitch);

  const firstStart = playable.length ? playable[0].start : 0;
  return playable.map((note, index) => ({
    ...note,
    id: `midi-${index + 1}`,
    start: Math.max(0, note.start - firstStart),
    end: Math.max(0.01, note.end - firstStart)
  }));
}
