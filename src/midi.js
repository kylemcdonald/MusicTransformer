const PPQ = 220;
const BPM = 120;
const TICKS_PER_SECOND = PPQ * BPM / 60;

function textBytes(text) {
  return Array.from(text, (char) => char.charCodeAt(0));
}

function uint32(value) {
  return [
    (value >>> 24) & 0xff,
    (value >>> 16) & 0xff,
    (value >>> 8) & 0xff,
    value & 0xff
  ];
}

function uint16(value) {
  return [(value >>> 8) & 0xff, value & 0xff];
}

function variableLength(value) {
  let buffer = value & 0x7f;
  const bytes = [];
  while ((value >>= 7)) {
    buffer <<= 8;
    buffer |= ((value & 0x7f) | 0x80);
  }
  while (true) {
    bytes.push(buffer & 0xff);
    if (buffer & 0x80) {
      buffer >>= 8;
    } else {
      break;
    }
  }
  return bytes;
}

export function writeMidiBlob(notes) {
  const firstStart = notes.reduce(
    (min, note) => (note.end > note.start ? Math.min(min, note.start) : min),
    Infinity
  );
  const offset = Number.isFinite(firstStart) ? firstStart : 0;
  const events = [
    { tick: 0, order: 0, bytes: [0xff, 0x51, 0x03, 0x07, 0xa1, 0x20] },
    { tick: 0, order: 1, bytes: [0xc0, 0x00] }
  ];

  for (const note of notes) {
    const startSeconds = Math.max(0, note.start - offset);
    const endSeconds = Math.max(startSeconds, note.end - offset);
    const start = Math.max(0, Math.round(startSeconds * TICKS_PER_SECOND));
    const end = Math.max(start + 1, Math.round(endSeconds * TICKS_PER_SECOND));
    const velocity = Math.max(1, Math.min(127, Math.round(note.velocity || 100)));
    events.push({ tick: start, order: 3, bytes: [0x90, note.pitch & 0x7f, velocity] });
    events.push({ tick: end, order: 2, bytes: [0x80, note.pitch & 0x7f, 0x40] });
  }

  events.sort((a, b) => a.tick - b.tick || a.order - b.order);

  const track = [];
  let lastTick = 0;
  for (const event of events) {
    track.push(...variableLength(event.tick - lastTick), ...event.bytes);
    lastTick = event.tick;
  }
  track.push(...variableLength(0), 0xff, 0x2f, 0x00);

  const bytes = [
    ...textBytes('MThd'),
    ...uint32(6),
    ...uint16(0),
    ...uint16(1),
    ...uint16(PPQ),
    ...textBytes('MTrk'),
    ...uint32(track.length),
    ...track
  ];

  return new Blob([new Uint8Array(bytes)], { type: 'audio/midi' });
}
