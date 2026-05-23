const MIN_PITCH = 21;
const MAX_PITCH = 108;
const PITCHES = MAX_PITCH - MIN_PITCH + 1;
const MIN_VIEW_DURATION = 1;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export class PianoRoll {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d');
    this.notes = [];
    this.duration = 8;
    this.viewStart = 0;
    this.viewDuration = 8;
    this.autoFit = true;
    this.playhead = null;
    this.generationCursor = null;
    this.selectedIds = new Set();
    this.onSeek = null;
    this.onDrawNote = null;
    this.onSelectionChange = null;
    this.onMoveNotes = null;
    this.pointerStart = null;
    this.interaction = null;
    this.draftNote = null;
    this.selectionRect = null;
    this.movePreview = null;
    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(canvas);
    this.canvas.addEventListener('pointerdown', (event) => this.handlePointerDown(event));
    this.canvas.addEventListener('pointermove', (event) => this.handlePointerMove(event));
    this.canvas.addEventListener('pointerup', (event) => this.handlePointerUp(event));
    this.canvas.addEventListener('pointercancel', () => this.cancelInteraction());
    this.canvas.addEventListener('pointerleave', () => this.resetCursor());
    this.canvas.addEventListener('wheel', (event) => this.handleWheel(event), { passive: false });
  }

  setNotes(notes, duration) {
    this.notes = notes;
    this.duration = Math.max(8, duration || 0, ...notes.map((note) => note.end));
    if (this.autoFit) {
      this.viewStart = 0;
      this.viewDuration = this.duration;
    } else {
      this.clampViewport();
    }
    this.draw();
  }

  setSelectedNoteIds(ids) {
    const next = new Set(ids);
    if (this.sameSelection(next)) return;
    this.selectedIds = next;
    this.draw();
  }

  fitToContent() {
    this.autoFit = true;
    this.viewStart = 0;
    this.viewDuration = Math.max(8, this.duration);
    this.draw();
  }

  setViewport(start, duration) {
    this.autoFit = false;
    this.viewDuration = clamp(duration, MIN_VIEW_DURATION, Math.max(MIN_VIEW_DURATION, this.duration));
    this.viewStart = start;
    this.clampViewport();
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

  setSelectionChangeHandler(handler) {
    this.onSelectionChange = handler;
  }

  setNoteMoveHandler(handler) {
    this.onMoveNotes = handler;
  }

  sameSelection(next) {
    if (next.size !== this.selectedIds.size) return false;
    for (const id of next) {
      if (!this.selectedIds.has(id)) return false;
    }
    return true;
  }

  clampViewport() {
    const maxDuration = Math.max(MIN_VIEW_DURATION, this.duration);
    this.viewDuration = clamp(this.viewDuration, MIN_VIEW_DURATION, maxDuration);
    this.viewStart = clamp(this.viewStart, 0, Math.max(0, this.duration - this.viewDuration));
  }

  handlePointerDown(event) {
    const point = this.eventToPoint(event);
    this.pointerStart = point;
    this.draftNote = null;
    this.selectionRect = null;
    this.movePreview = null;
    this.canvas.setPointerCapture(event.pointerId);

    if (event.shiftKey) {
      this.interaction = { type: 'select', start: point, current: point };
      this.selectionRect = this.interaction;
      this.resetCursor();
      this.draw();
      return;
    }

    const hitNote = this.noteAtPoint(point);
    if (hitNote) {
      const ids = this.selectedIds.has(hitNote.id)
        ? [...this.selectedIds]
        : [hitNote.id];
      this.selectedIds = new Set(ids);
      this.onSelectionChange?.(ids);
      const originals = new Map();
      for (const note of this.notes) {
        if (this.selectedIds.has(note.id) && !note.ghost) {
          originals.set(note.id, {
            start: note.start,
            end: note.end,
            pitch: note.pitch
          });
        }
      }
      this.interaction = {
        type: 'move',
        start: point,
        current: point,
        moved: false,
        ids,
        originals
      };
      this.canvas.style.cursor = 'move';
      this.draw();
      return;
    }

    this.interaction = { type: 'draw', start: point, current: point };
    this.resetCursor();
  }

  handlePointerMove(event) {
    if (!this.pointerStart || !this.interaction) {
      this.updateHoverCursor(event);
      return;
    }

    const point = this.eventToPoint(event);
    this.interaction.current = point;

    if (this.interaction.type === 'select') {
      this.selectionRect = this.interaction;
      this.draw();
      return;
    }

    const distance = Math.hypot(point.clientX - this.pointerStart.clientX, point.clientY - this.pointerStart.clientY);
    if (this.interaction.type === 'move') {
      if (distance < 3 && !this.interaction.moved) return;
      this.interaction.moved = true;
      this.movePreview = this.constrainMovePreview(this.interaction, point);
      this.canvas.style.cursor = 'move';
      this.draw();
      return;
    }

    if (!this.onDrawNote || distance < 8) return;

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
    if (!this.pointerStart || !this.interaction) return;

    const point = this.eventToPoint(event);
    const interaction = this.interaction;
    const draft = this.draftNote;
    this.pointerStart = null;
    this.interaction = null;
    this.draftNote = null;
    this.selectionRect = null;
    this.canvas.releasePointerCapture(event.pointerId);
    this.updateHoverCursor(event);

    if (interaction.type === 'select') {
      const ids = this.notesInSelection(interaction.start, point).map((note) => note.id);
      this.selectedIds = new Set(ids);
      this.onSelectionChange?.(ids);
      this.draw();
      return;
    }

    if (interaction.type === 'move') {
      const preview = this.movePreview;
      this.movePreview = null;
      if (interaction.moved && preview) {
        this.onMoveNotes?.(preview.ids, preview.deltaSeconds, preview.deltaPitch);
      } else {
        this.draw();
      }
      return;
    }

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

  cancelInteraction() {
    this.pointerStart = null;
    this.interaction = null;
    this.draftNote = null;
    this.selectionRect = null;
    this.movePreview = null;
    this.resetCursor();
    this.draw();
  }

  resetCursor() {
    this.canvas.style.cursor = 'crosshair';
  }

  updateHoverCursor(event) {
    const point = this.eventToPoint(event);
    this.canvas.style.cursor = this.noteAtPoint(point) ? 'move' : 'crosshair';
  }

  handleWheel(event) {
    event.preventDefault();
    const point = this.eventToPoint(event);
    const rect = this.canvas.getBoundingClientRect();
    const labelWidth = rect.width < 640 ? 34 : 46;
    const rollWidth = Math.max(1, rect.width - labelWidth);

    if (event.ctrlKey || event.metaKey) {
      const factor = Math.exp(event.deltaY * 0.002);
      const nextDuration = clamp(
        this.viewDuration * factor,
        MIN_VIEW_DURATION,
        Math.max(MIN_VIEW_DURATION, this.duration)
      );
      const ratio = (point.seconds - this.viewStart) / this.viewDuration;
      this.setViewport(point.seconds - ratio * nextDuration, nextDuration);
      return;
    }

    const delta = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
    const seconds = (delta / rollWidth) * this.viewDuration;
    this.setViewport(this.viewStart + seconds, this.viewDuration);
  }

  constrainMovePreview(interaction, point) {
    let deltaSeconds = point.seconds - interaction.start.seconds;
    let deltaPitch = point.pitch - interaction.start.pitch;
    let minStart = Infinity;
    let minPitch = Infinity;
    let maxPitch = -Infinity;

    for (const original of interaction.originals.values()) {
      minStart = Math.min(minStart, original.start);
      minPitch = Math.min(minPitch, original.pitch);
      maxPitch = Math.max(maxPitch, original.pitch);
    }

    deltaSeconds = Math.max(deltaSeconds, -minStart);
    deltaPitch = clamp(deltaPitch, MIN_PITCH - minPitch, MAX_PITCH - maxPitch);
    return { ids: interaction.ids, deltaSeconds, deltaPitch };
  }

  noteAtPoint(point) {
    for (let index = this.notes.length - 1; index >= 0; index -= 1) {
      const note = this.notes[index];
      if (note.ghost || note.provisional) continue;
      if (point.pitch !== note.pitch) continue;
      if (point.seconds >= note.start && point.seconds <= note.end) return note;
    }
    return null;
  }

  notesInSelection(startPoint, endPoint) {
    const start = Math.min(startPoint.seconds, endPoint.seconds);
    const end = Math.max(startPoint.seconds, endPoint.seconds);
    const minPitch = Math.min(startPoint.pitch, endPoint.pitch);
    const maxPitch = Math.max(startPoint.pitch, endPoint.pitch);

    return this.notes.filter((note) => (
      !note.ghost &&
      !note.provisional &&
      note.end >= start &&
      note.start <= end &&
      note.pitch >= minPitch &&
      note.pitch <= maxPitch
    ));
  }

  eventToPoint(event) {
    const rect = this.canvas.getBoundingClientRect();
    const labelWidth = rect.width < 640 ? 34 : 46;
    const rollWidth = Math.max(1, rect.width - labelWidth);
    const x = Math.max(labelWidth, Math.min(rect.width, event.clientX - rect.left));
    const y = Math.max(0, Math.min(rect.height, event.clientY - rect.top));
    const seconds = this.viewStart + ((x - labelWidth) / rollWidth) * this.viewDuration;
    const pitch = Math.max(
      MIN_PITCH,
      Math.min(MAX_PITCH, MAX_PITCH - Math.floor((y / Math.max(1, rect.height)) * PITCHES))
    );
    return {
      clientX: event.clientX,
      clientY: event.clientY,
      x,
      y,
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
    const pxPerSecond = rollWidth / this.viewDuration;

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

    for (const note of this.renderNotes()) {
      this.drawNote(ctx, note, labelWidth, width, height, rowHeight, pxPerSecond);
    }

    if (this.selectionRect) {
      const x1 = labelWidth + (this.selectionRect.start.seconds - this.viewStart) * pxPerSecond;
      const x2 = labelWidth + (this.selectionRect.current.seconds - this.viewStart) * pxPerSecond;
      const y1 = this.pitchToY(this.selectionRect.start.pitch, height, rowHeight);
      const y2 = this.pitchToY(this.selectionRect.current.pitch, height, rowHeight);
      const x = Math.max(labelWidth, Math.min(x1, x2));
      const y = Math.min(y1, y2);
      const w = Math.min(width, Math.max(x1, x2)) - x;
      const h = Math.abs(y2 - y1) + rowHeight;
      ctx.fillStyle = 'rgba(245, 177, 66, 0.16)';
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = 'rgba(245, 177, 66, 0.9)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, Math.max(0, w - 1), Math.max(0, h - 1));
    }

    this.drawCursor(ctx, this.generationCursor, labelWidth, width, height, pxPerSecond, true);
    this.drawCursor(ctx, this.playhead, labelWidth, width, height, pxPerSecond, false);
  }

  renderNotes() {
    const notes = this.notes.map((note) => {
      if (!this.movePreview?.ids.includes(note.id)) return note;
      return {
        ...note,
        start: note.start + this.movePreview.deltaSeconds,
        end: note.end + this.movePreview.deltaSeconds,
        pitch: note.pitch + this.movePreview.deltaPitch
      };
    });
    return [...notes, ...(this.draftNote ? [this.draftNote] : [])];
  }

  drawNote(ctx, note, labelWidth, width, height, rowHeight, pxPerSecond) {
    if (note.end < this.viewStart || note.start > this.viewStart + this.viewDuration) return;

    const x = labelWidth + (note.start - this.viewStart) * pxPerSecond;
    const y = this.pitchToY(note.pitch, height, rowHeight) + 1;
    const w = Math.max(2, (note.end - note.start) * pxPerSecond);
    const h = Math.max(2, rowHeight - 2);
    const velocity = Math.max(0.05, Math.min(1, (note.velocity || 100) / 127));
    const opacity = note.opacity ?? (note.ghost ? 0.2 : 1);
    const selected = this.selectedIds.has(note.id) && !note.ghost;
    ctx.fillStyle = this.velocityColor(velocity, note.provisional, opacity);
    ctx.fillRect(x, y, w, h);
    ctx.fillStyle = `rgba(255, 255, 255, ${(0.08 + velocity * 0.28) * opacity})`;
    ctx.fillRect(x, y, Math.min(w, 3), h);
    ctx.fillStyle = `rgba(255, 255, 255, ${(0.08 + velocity * 0.18) * opacity})`;
    ctx.fillRect(x, y, w, Math.max(1, Math.min(3, h * velocity)));

    if (selected) {
      ctx.strokeStyle = '#f5b142';
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 1, y + 1, Math.max(1, w - 2), Math.max(1, h - 2));
    }
  }

  drawCursor(ctx, seconds, labelWidth, width, height, pxPerSecond, generation) {
    if (seconds === null) return;

    const playheadX = this.playhead === null ? null : labelWidth + (this.playhead - this.viewStart) * pxPerSecond;
    let x = labelWidth + (seconds - this.viewStart) * pxPerSecond;
    if (generation && playheadX !== null && Math.abs(x - playheadX) < 2) {
      x = Math.max(labelWidth, x - 4);
    }
    if (x < labelWidth || x > width) return;

    ctx.strokeStyle = generation ? '#47bda8' : '#ff6b5f';
    ctx.lineWidth = 2;
    if (generation) ctx.setLineDash([7, 5]);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
    if (generation) ctx.setLineDash([]);
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
