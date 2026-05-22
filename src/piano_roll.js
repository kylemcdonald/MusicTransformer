const MIN_PITCH = 21;
const MAX_PITCH = 108;
const PITCHES = MAX_PITCH - MIN_PITCH + 1;

export class PianoRoll {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d');
    this.notes = [];
    this.duration = 8;
    this.playhead = null;
    this.generationCursor = null;
    this.onSeek = null;
    this.onDrawNote = null;
    this.pointerStart = null;
    this.draftNote = null;
    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(canvas);
    this.canvas.addEventListener('pointerdown', (event) => this.handlePointerDown(event));
    this.canvas.addEventListener('pointermove', (event) => this.handlePointerMove(event));
    this.canvas.addEventListener('pointerup', (event) => this.handlePointerUp(event));
    this.canvas.addEventListener('pointercancel', () => this.cancelDraft());
  }

  setNotes(notes, duration) {
    this.notes = notes;
    this.duration = Math.max(8, duration || 0, ...notes.map((note) => note.end));
    this.draw();
  }

  setPlayhead(seconds) {
    this.playhead = seconds;
    this.draw();
  }

  setGenerationCursor(seconds) {
    this.generationCursor = seconds;
    this.draw();
  }

  setSeekHandler(handler) {
    this.onSeek = handler;
  }

  setNoteDrawHandler(handler) {
    this.onDrawNote = handler;
  }

  handlePointerDown(event) {
    const point = this.eventToPoint(event);
    this.pointerStart = point;
    this.draftNote = null;
    this.canvas.setPointerCapture(event.pointerId);
  }

  handlePointerMove(event) {
    if (!this.pointerStart || !this.onDrawNote) return;

    const point = this.eventToPoint(event);
    const distance = Math.hypot(point.clientX - this.pointerStart.clientX, point.clientY - this.pointerStart.clientY);
    if (distance < 8 && !this.draftNote) return;

    const start = Math.min(this.pointerStart.seconds, point.seconds);
    const end = Math.max(this.pointerStart.seconds, point.seconds);
    this.draftNote = {
      id: 'draft',
      pitch: this.pointerStart.pitch,
      velocity: 100,
      start,
      end: Math.max(start + 0.05, end),
      provisional: true
    };
    this.draw();
  }

  handlePointerUp(event) {
    if (!this.pointerStart) return;

    const point = this.eventToPoint(event);
    const draft = this.draftNote;
    this.pointerStart = null;
    this.draftNote = null;
    this.canvas.releasePointerCapture(event.pointerId);

    if (draft && this.onDrawNote) {
      const start = Math.min(draft.start, draft.end);
      const end = Math.max(draft.start, draft.end);
      this.onDrawNote({
        pitch: draft.pitch,
        velocity: 100,
        start,
        end: Math.max(start + 0.1, end)
      });
      return;
    }

    this.onSeek?.(Math.max(0, Math.min(this.duration, point.seconds)));
  }

  cancelDraft() {
    this.pointerStart = null;
    this.draftNote = null;
    this.draw();
  }

  eventToPoint(event) {
    const rect = this.canvas.getBoundingClientRect();
    const labelWidth = rect.width < 640 ? 34 : 46;
    const rollWidth = Math.max(1, rect.width - labelWidth);
    const x = Math.max(labelWidth, Math.min(rect.width, event.clientX - rect.left));
    const y = Math.max(0, Math.min(rect.height, event.clientY - rect.top));
    const seconds = ((x - labelWidth) / rollWidth) * this.duration;
    const pitch = Math.max(
      MIN_PITCH,
      Math.min(MAX_PITCH, MAX_PITCH - Math.floor((y / Math.max(1, rect.height)) * PITCHES))
    );
    return {
      clientX: event.clientX,
      clientY: event.clientY,
      seconds: Math.max(0, Math.min(this.duration, seconds)),
      pitch
    };
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(rect.width * ratio));
    const height = Math.max(1, Math.floor(rect.height * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return rect;
  }

  draw() {
    const rect = this.resize();
    const ctx = this.context;
    const width = rect.width;
    const height = rect.height;
    const labelWidth = width < 640 ? 34 : 46;
    const rollWidth = Math.max(1, width - labelWidth);
    const rowHeight = height / PITCHES;
    const pxPerSecond = rollWidth / this.duration;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#111318';
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = '#1a2029';
    ctx.fillRect(0, 0, labelWidth, height);

    for (let pitch = MIN_PITCH; pitch <= MAX_PITCH; pitch += 1) {
      const y = this.pitchToY(pitch, height, rowHeight);
      const isBlack = [1, 3, 6, 8, 10].includes(pitch % 12);
      ctx.fillStyle = isBlack ? '#171d27' : '#222936';
      ctx.fillRect(labelWidth, y, rollWidth, Math.ceil(rowHeight));
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.035)';
      ctx.beginPath();
      ctx.moveTo(labelWidth, y);
      ctx.lineTo(width, y);
      ctx.stroke();
      if (pitch % 12 === 0) {
        ctx.fillStyle = '#c0c7d2';
        ctx.font = '11px system-ui, sans-serif';
        ctx.fillText(`C${Math.floor(pitch / 12) - 1}`, 8, y + Math.max(11, rowHeight));
      }
    }

    for (const note of [...this.notes, ...(this.draftNote ? [this.draftNote] : [])]) {
      const x = labelWidth + note.start * pxPerSecond;
      const y = this.pitchToY(note.pitch, height, rowHeight) + 1;
      const w = Math.max(2, (note.end - note.start) * pxPerSecond);
      const h = Math.max(2, rowHeight - 2);
      const velocity = Math.max(0.05, Math.min(1, (note.velocity || 100) / 127));
      const opacity = note.opacity ?? (note.ghost ? 0.2 : 1);
      ctx.fillStyle = this.velocityColor(velocity, note.provisional, opacity);
      ctx.fillRect(x, y, w, h);
      ctx.fillStyle = `rgba(255, 255, 255, ${(0.08 + velocity * 0.28) * opacity})`;
      ctx.fillRect(x, y, Math.min(w, 3), h);
      ctx.fillStyle = `rgba(255, 255, 255, ${(0.08 + velocity * 0.18) * opacity})`;
      ctx.fillRect(x, y, w, Math.max(1, Math.min(3, h * velocity)));
    }

    if (this.generationCursor !== null) {
      const playheadX = this.playhead === null ? null : labelWidth + this.playhead * pxPerSecond;
      let x = labelWidth + this.generationCursor * pxPerSecond;
      if (playheadX !== null && Math.abs(x - playheadX) < 2) {
        x = Math.max(labelWidth, x - 4);
      }
      if (x >= labelWidth && x <= width) {
        ctx.strokeStyle = '#47bda8';
        ctx.lineWidth = 2;
        ctx.setLineDash([7, 5]);
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    if (this.playhead !== null) {
      const x = labelWidth + this.playhead * pxPerSecond;
      if (x >= labelWidth && x <= width) {
        ctx.strokeStyle = '#ff6b5f';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
    }
  }

  pitchToY(pitch, height, rowHeight) {
    return height - (pitch - MIN_PITCH + 1) * rowHeight;
  }

  velocityColor(velocity, provisional, opacity = 1) {
    const low = [84, 154, 214];
    const mid = [71, 189, 168];
    const high = [245, 177, 66];
    const t = velocity < 0.5 ? velocity * 2 : (velocity - 0.5) * 2;
    const from = velocity < 0.5 ? low : mid;
    const to = velocity < 0.5 ? mid : high;
    const rgb = from.map((value, index) => Math.round(value + (to[index] - value) * t));
    const alpha = provisional ? 0.36 + velocity * 0.24 : 0.52 + velocity * 0.42;
    return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha * opacity})`;
  }
}
