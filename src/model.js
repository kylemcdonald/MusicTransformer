const MANIFEST_URL = '/manual-model/manifest.json';
const WEIGHTS_URL = '/manual-model/weights.bin';

const CONTEXT_LENGTH = 1024;
const DEFAULT_MAX_TOKENS = 8192;
const DEFAULT_MIN_TOKENS = 256;
const DEFAULT_TEMPERATURE = 1.0;
const DEFAULT_TOP_K = 64;

const PAD_ID = 0;
const EOS_ID = 1;
const RESERVED_ID = 2;

const FLOAT_BYTES = 4;

let modelPromise = null;

const embedShader = /* wgsl */ `
const HIDDEN: u32 = 512u;
const HALF_HIDDEN: u32 = 256u;
const VOCAB: u32 = 310u;
const EMBEDDING_SCALE: f32 = 22.627416997969522;
const LOG_TIMESCALE: f32 = 9.210340371976184;

struct StepParams {
  token: u32,
  position: u32,
  history_count: u32,
  start_position: u32,
  context_length: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<storage, read> embedding: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: StepParams;

fn timing_signal(channel: u32, position: u32) -> f32 {
  let index = channel % HALF_HIDDEN;
  let inv_timescale = exp(-f32(index) * LOG_TIMESCALE / f32(HALF_HIDDEN - 1u));
  let scaled_time = f32(position) * inv_timescale;
  if (channel < HALF_HIDDEN) {
    return sin(scaled_time);
  }
  return cos(scaled_time);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= HIDDEN) {
    return;
  }

  var value = 0.0;
  if (params.token != 0u && params.token < VOCAB) {
    value = embedding[params.token * HIDDEN + i] * EMBEDDING_SCALE;
  }

  output[i] = value + timing_signal(i, params.position);
}
`;

const layerNormShader = /* wgsl */ `
const HIDDEN: u32 = 512u;
const EPSILON: f32 = 0.000001;

@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_values: array<f32>;

@compute @workgroup_size(1)
fn main() {
  var mean = 0.0;
  for (var i = 0u; i < HIDDEN; i = i + 1u) {
    mean = mean + input_values[i];
  }
  mean = mean / f32(HIDDEN);

  var variance = 0.0;
  for (var i = 0u; i < HIDDEN; i = i + 1u) {
    let centered = input_values[i] - mean;
    variance = variance + centered * centered;
  }
  let inv_std = inverseSqrt(variance / f32(HIDDEN) + EPSILON);

  for (var i = 0u; i < HIDDEN; i = i + 1u) {
    output_values[i] = (input_values[i] - mean) * inv_std * scale[i] + bias[i];
  }
}
`;

const matVecShader = /* wgsl */ `
struct MatVecParams {
  input_size: u32,
  output_size: u32,
  activation: u32,
  pad0: u32,
};

@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(4) var<uniform> params: MatVecParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let out_index = id.x;
  if (out_index >= params.output_size) {
    return;
  }

  var sum = bias[out_index];
  for (var i = 0u; i < params.input_size; i = i + 1u) {
    sum = sum + input_values[i] * weights[i * params.output_size + out_index];
  }

  if (params.activation == 1u && sum < 0.0) {
    sum = 0.0;
  }

  output_values[out_index] = sum;
}
`;

const addShader = /* wgsl */ `
const HIDDEN: u32 = 512u;

@group(0) @binding(0) var<storage, read_write> target_values: array<f32>;
@group(0) @binding(1) var<storage, read> delta: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i < HIDDEN) {
    target_values[i] = target_values[i] + delta[i];
  }
}
`;

const cacheShader = /* wgsl */ `
const HIDDEN: u32 = 512u;

struct StepParams {
  token: u32,
  position: u32,
  history_count: u32,
  start_position: u32,
  context_length: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<storage, read> key: array<f32>;
@group(0) @binding(1) var<storage, read> value: array<f32>;
@group(0) @binding(2) var<storage, read_write> key_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> value_cache: array<f32>;
@group(0) @binding(4) var<uniform> params: StepParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= HIDDEN) {
    return;
  }
  let slot = params.position % params.context_length;
  let offset = slot * HIDDEN + i;
  key_cache[offset] = key[i];
  value_cache[offset] = value[i];
}
`;

const attentionScoresShader = /* wgsl */ `
const HIDDEN: u32 = 512u;
const NUM_HEADS: u32 = 8u;
const HEAD_SIZE: u32 = 64u;
const SCALE: f32 = 0.125;

struct StepParams {
  token: u32,
  position: u32,
  history_count: u32,
  start_position: u32,
  context_length: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> params: StepParams;

@compute @workgroup_size(16, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let offset = id.x;
  let head = id.y;
  if (offset >= params.history_count || head >= NUM_HEADS) {
    return;
  }

  let absolute_position = params.start_position + offset;
  let slot = absolute_position % params.context_length;
  let q_base = head * HEAD_SIZE;
  let k_base = slot * HIDDEN + head * HEAD_SIZE;
  var sum = 0.0;
  for (var i = 0u; i < HEAD_SIZE; i = i + 1u) {
    sum = sum + query[q_base + i] * key_cache[k_base + i];
  }
  scores[head * params.context_length + offset] = sum * SCALE;
}
`;

const attentionContextShader = /* wgsl */ `
const HIDDEN: u32 = 512u;
const NUM_HEADS: u32 = 8u;
const HEAD_SIZE: u32 = 64u;

struct StepParams {
  token: u32,
  position: u32,
  history_count: u32,
  start_position: u32,
  context_length: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read> value_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> context: array<f32>;
@group(0) @binding(3) var<uniform> params: StepParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let depth = id.x;
  let head = id.y;
  if (depth >= HEAD_SIZE || head >= NUM_HEADS) {
    return;
  }

  var max_score = -3.4028234663852886e38;
  for (var offset = 0u; offset < params.history_count; offset = offset + 1u) {
    max_score = max(max_score, scores[head * params.context_length + offset]);
  }

  var denominator = 0.0;
  var weighted_sum = 0.0;
  for (var offset = 0u; offset < params.history_count; offset = offset + 1u) {
    let weight = exp(scores[head * params.context_length + offset] - max_score);
    let absolute_position = params.start_position + offset;
    let slot = absolute_position % params.context_length;
    let value_index = slot * HIDDEN + head * HEAD_SIZE + depth;
    denominator = denominator + weight;
    weighted_sum = weighted_sum + weight * value_cache[value_index];
  }

  context[head * HEAD_SIZE + depth] = weighted_sum / max(denominator, 1.0e-20);
}
`;

function createPipeline(device, code, label) {
  return device.createComputePipeline({
    label,
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code, label: `${label}-module` }),
      entryPoint: 'main'
    }
  });
}

function createBuffer(device, size, usage, label) {
  return device.createBuffer({
    label,
    size: Math.max(4, size),
    usage
  });
}

function createAndWriteBuffer(device, data, usage, label) {
  const buffer = createBuffer(device, data.byteLength, usage | GPUBufferUsage.COPY_DST, label);
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function storageUsage() {
  return GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
}

function createZeroBuffer(device, length, label) {
  return createAndWriteBuffer(
    device,
    new Float32Array(length),
    storageUsage(),
    label
  );
}

function createUniformBuffer(device, values, label) {
  const buffer = createBuffer(device, values.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label);
  device.queue.writeBuffer(buffer, 0, values);
  return buffer;
}

function alignTo(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}

function dispatch1d(pass, pipeline, bindGroup, count, workgroupSize = 64) {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / workgroupSize));
}

function sampleFromLogits(logits, position, options) {
  const temperature = Math.max(0.05, options.temperature ?? DEFAULT_TEMPERATURE);
  const minTokens = options.minTokens ?? DEFAULT_MIN_TOKENS;
  const topK = options.topK ?? DEFAULT_TOP_K;
  const candidates = [];

  for (let token = 0; token < logits.length; token += 1) {
    if (token === PAD_ID || token === RESERVED_ID) continue;
    if (token === EOS_ID && position + 1 < minTokens) continue;
    const logit = logits[token];
    if (Number.isFinite(logit)) {
      candidates.push({ token, logit: logit / temperature });
    }
  }

  if (!candidates.length) {
    throw new Error('Model produced no finite logits.');
  }

  candidates.sort((a, b) => b.logit - a.logit);
  const kept = topK > 0 ? candidates.slice(0, topK) : candidates;
  const maxLogit = kept[0].logit;
  let total = 0;
  for (const candidate of kept) {
    candidate.weight = Math.exp(candidate.logit - maxLogit);
    total += candidate.weight;
  }

  let draw = Math.random() * total;
  for (const candidate of kept) {
    draw -= candidate.weight;
    if (draw <= 0) {
      return candidate.token;
    }
  }
  return kept[kept.length - 1].token;
}

function throwIfAborted(signal) {
  if (!signal?.aborted) return;
  throw new DOMException('Generation stopped', 'AbortError');
}

async function fetchArrayBufferWithProgress(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.status}`);
  }

  const total = Number(response.headers.get('content-length')) || 0;
  if (!response.body || !total) {
    const buffer = await response.arrayBuffer();
    onProgress?.(1);
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onProgress?.(Math.min(received / total, 1));
  }

  const output = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return output.buffer;
}

export class ManualMusicTransformer {
  constructor(device, manifest, weightsBuffer, onProgress) {
    this.device = device;
    this.manifest = manifest;
    this.config = manifest.config;
    this.contextLength = CONTEXT_LENGTH;
    this.weightsBuffer = weightsBuffer;
    this.onProgress = onProgress;

    this.pipelines = {
      embed: createPipeline(device, embedShader, 'embed'),
      layerNorm: createPipeline(device, layerNormShader, 'layer-norm'),
      matVec: createPipeline(device, matVecShader, 'mat-vec'),
      add: createPipeline(device, addShader, 'add'),
      cache: createPipeline(device, cacheShader, 'cache'),
      attentionScores: createPipeline(device, attentionScoresShader, 'attention-scores'),
      attentionContext: createPipeline(device, attentionContextShader, 'attention-context')
    };

    this.weights = new Map();
    this.layers = [];
    this.initializeBuffers();
  }

  initializeBuffers() {
    const { device, config } = this;
    const hiddenBytes = config.hiddenSize * FLOAT_BYTES;
    const filterBytes = config.filterSize * FLOAT_BYTES;
    const vocabBytes = config.vocabSize * FLOAT_BYTES;
    const scoreBytes = config.numHeads * this.contextLength * FLOAT_BYTES;
    const cacheBytes = this.contextLength * config.hiddenSize * FLOAT_BYTES;

    const tensorEntries = Object.entries(this.manifest.tensors);
    tensorEntries.forEach(([name, info], index) => {
      const view = new Uint8Array(this.weightsBuffer, info.offset, info.bytes);
      this.weights.set(name, createAndWriteBuffer(this.device, view, storageUsage(), name));
      this.onProgress?.(0.75 + ((index + 1) / tensorEntries.length) * 0.25);
    });
    this.weightsBuffer = null;

    this.x = createBuffer(device, hiddenBytes, storageUsage(), 'x');
    this.norm = createBuffer(device, hiddenBytes, storageUsage(), 'norm');
    this.query = createBuffer(device, hiddenBytes, storageUsage(), 'query');
    this.key = createBuffer(device, hiddenBytes, storageUsage(), 'key');
    this.value = createBuffer(device, hiddenBytes, storageUsage(), 'value');
    this.context = createBuffer(device, hiddenBytes, storageUsage(), 'context');
    this.attentionOutput = createBuffer(device, hiddenBytes, storageUsage(), 'attention-output');
    this.ffnHidden = createBuffer(device, filterBytes, storageUsage(), 'ffn-hidden');
    this.ffnOutput = createBuffer(device, hiddenBytes, storageUsage(), 'ffn-output');
    this.logits = createBuffer(device, vocabBytes, storageUsage() | GPUBufferUsage.COPY_SRC, 'logits');
    this.readback = createBuffer(
      device,
      alignTo(vocabBytes, 4),
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      'logits-readback'
    );

    this.zeroHidden = createZeroBuffer(device, config.hiddenSize, 'zero-hidden');
    this.zeroVocab = createZeroBuffer(device, config.vocabSize, 'zero-vocab');

    this.stepParams = createUniformBuffer(device, new Uint32Array(8), 'step-params');
    this.mat512x512 = createUniformBuffer(device, new Uint32Array([512, 512, 0, 0]), 'mat-512-512');
    this.mat512x2048Relu = createUniformBuffer(device, new Uint32Array([512, 2048, 1, 0]), 'mat-512-2048-relu');
    this.mat2048x512 = createUniformBuffer(device, new Uint32Array([2048, 512, 0, 0]), 'mat-2048-512');
    this.mat512x310 = createUniformBuffer(device, new Uint32Array([512, 310, 0, 0]), 'mat-512-310');

    for (let layer = 0; layer < config.numLayers; layer += 1) {
      this.layers.push({
        index: layer,
        keyCache: createBuffer(device, cacheBytes, storageUsage(), `layer-${layer}-key-cache`),
        valueCache: createBuffer(device, cacheBytes, storageUsage(), `layer-${layer}-value-cache`),
        scores: createBuffer(device, scoreBytes, storageUsage(), `layer-${layer}-scores`)
      });
    }
  }

  tensor(name) {
    const buffer = this.weights.get(name);
    if (!buffer) {
      throw new Error(`Missing model tensor: ${name}`);
    }
    return buffer;
  }

  bind(pipeline, entries) {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map((buffer, binding) => ({
        binding,
        resource: { buffer }
      }))
    });
  }

  encodeDispatch1d(encoder, pipeline, bindGroup, count, workgroupSize = 64) {
    const pass = encoder.beginComputePass();
    dispatch1d(pass, pipeline, bindGroup, count, workgroupSize);
    pass.end();
  }

  encodeDispatch2d(encoder, pipeline, bindGroup, x, y) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(x, y);
    pass.end();
  }

  encodeLayerNorm(encoder, input, scale, bias, output) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.layerNorm);
    pass.setBindGroup(0, this.bind(this.pipelines.layerNorm, [input, scale, bias, output]));
    pass.dispatchWorkgroups(1);
    pass.end();
  }

  encodeMatVec(encoder, input, weight, bias, output, params, outputSize) {
    this.encodeDispatch1d(
      encoder,
      this.pipelines.matVec,
      this.bind(this.pipelines.matVec, [input, weight, bias, output, params]),
      outputSize,
      64
    );
  }

  encodeAdd(encoder, target, delta) {
    this.encodeDispatch1d(
      encoder,
      this.pipelines.add,
      this.bind(this.pipelines.add, [target, delta]),
      this.config.hiddenSize,
      256
    );
  }

  async generateToken(previousToken, position) {
    const historyCount = Math.min(position + 1, this.contextLength);
    const startPosition = position + 1 - historyCount;
    this.device.queue.writeBuffer(
      this.stepParams,
      0,
      new Uint32Array([
        previousToken,
        position,
        historyCount,
        startPosition,
        this.contextLength,
        0,
        0,
        0
      ])
    );

    const encoder = this.device.createCommandEncoder();

    this.encodeDispatch1d(
      encoder,
      this.pipelines.embed,
      this.bind(this.pipelines.embed, [this.tensor('embedding'), this.x, this.stepParams]),
      this.config.hiddenSize,
      256
    );

    for (const layer of this.layers) {
      const prefix = `layers.${layer.index}`;

      this.encodeLayerNorm(
        encoder,
        this.x,
        this.tensor(`${prefix}.attn_ln_scale`),
        this.tensor(`${prefix}.attn_ln_bias`),
        this.norm
      );
      this.encodeMatVec(encoder, this.norm, this.tensor(`${prefix}.wq`), this.zeroHidden, this.query, this.mat512x512, 512);
      this.encodeMatVec(encoder, this.norm, this.tensor(`${prefix}.wk`), this.zeroHidden, this.key, this.mat512x512, 512);
      this.encodeMatVec(encoder, this.norm, this.tensor(`${prefix}.wv`), this.zeroHidden, this.value, this.mat512x512, 512);

      this.encodeDispatch1d(
        encoder,
        this.pipelines.cache,
        this.bind(this.pipelines.cache, [this.key, this.value, layer.keyCache, layer.valueCache, this.stepParams]),
        this.config.hiddenSize,
        256
      );

      this.encodeDispatch2d(
        encoder,
        this.pipelines.attentionScores,
        this.bind(this.pipelines.attentionScores, [this.query, layer.keyCache, layer.scores, this.stepParams]),
        Math.ceil(historyCount / 16),
        this.config.numHeads
      );

      this.encodeDispatch2d(
        encoder,
        this.pipelines.attentionContext,
        this.bind(this.pipelines.attentionContext, [layer.scores, layer.valueCache, this.context, this.stepParams]),
        Math.ceil(this.config.headSize / 8),
        this.config.numHeads
      );

      this.encodeMatVec(encoder, this.context, this.tensor(`${prefix}.wo`), this.zeroHidden, this.attentionOutput, this.mat512x512, 512);
      this.encodeAdd(encoder, this.x, this.attentionOutput);

      this.encodeLayerNorm(
        encoder,
        this.x,
        this.tensor(`${prefix}.ffn_ln_scale`),
        this.tensor(`${prefix}.ffn_ln_bias`),
        this.norm
      );
      this.encodeMatVec(encoder, this.norm, this.tensor(`${prefix}.w1`), this.tensor(`${prefix}.b1`), this.ffnHidden, this.mat512x2048Relu, 2048);
      this.encodeMatVec(encoder, this.ffnHidden, this.tensor(`${prefix}.w2`), this.tensor(`${prefix}.b2`), this.ffnOutput, this.mat2048x512, 512);
      this.encodeAdd(encoder, this.x, this.ffnOutput);
    }

    this.encodeLayerNorm(encoder, this.x, this.tensor('final_ln_scale'), this.tensor('final_ln_bias'), this.norm);
    this.encodeMatVec(encoder, this.norm, this.tensor('embedding_t'), this.zeroVocab, this.logits, this.mat512x310, 310);

    encoder.copyBufferToBuffer(this.logits, 0, this.readback, 0, this.config.vocabSize * FLOAT_BYTES);
    this.device.queue.submit([encoder.finish()]);

    await this.readback.mapAsync(GPUMapMode.READ);
    const logits = new Float32Array(this.readback.getMappedRange()).slice();
    this.readback.unmap();
    return logits;
  }

  async *generate(options = {}) {
    const maxTokens = options.maxTokens ?? DEFAULT_MAX_TOKENS;
    const seedTokens = options.seedTokens ?? [PAD_ID];
    const signal = options.signal;
    let previousToken = PAD_ID;
    let position = 0;

    for (const seedToken of seedTokens) {
      throwIfAborted(signal);
      await this.generateToken(previousToken, position);
      throwIfAborted(signal);
      previousToken = seedToken;
      position += 1;
    }

    for (let outputIndex = 0; outputIndex < maxTokens; outputIndex += 1, position += 1) {
      throwIfAborted(signal);
      const logits = await this.generateToken(previousToken, position);
      throwIfAborted(signal);
      const token = sampleFromLogits(logits, outputIndex, options);
      yield token;
      if (token === EOS_ID && outputIndex + 1 >= (options.minTokens ?? DEFAULT_MIN_TOKENS)) {
        break;
      }
      previousToken = token;
    }
  }
}

export async function loadMusicTransformer(onProgress) {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not available in this browser.');
  }

  if (!modelPromise) {
    modelPromise = (async () => {
      onProgress?.(0);
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) {
        throw new Error('No WebGPU adapter is available.');
      }
      const device = await adapter.requestDevice();
      const manifest = await fetch(MANIFEST_URL).then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load ${MANIFEST_URL}: ${response.status}`);
        }
        return response.json();
      });
      onProgress?.(0.05);
      const weightsBuffer = await fetchArrayBufferWithProgress(
        WEIGHTS_URL,
        (fraction) => onProgress?.(0.05 + fraction * 0.70)
      );
      return new ManualMusicTransformer(device, manifest, weightsBuffer, onProgress);
    })();
  }

  return modelPromise;
}

export async function* generateTokenStream(model, options = {}) {
  yield* model.generate(options);
}

export async function generateTokens(model, prefixTokens, decodeLength) {
  const tokens = [];
  const seedTokens = prefixTokens?.length ? prefixTokens : [PAD_ID];
  for await (const token of model.generate({ maxTokens: decodeLength, seedTokens })) {
    tokens.push(token);
  }
  return tokens;
}

export function backendName() {
  return `manual WebGPU, ${CONTEXT_LENGTH}-token KV context`;
}
