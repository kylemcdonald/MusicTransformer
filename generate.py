import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from magenta.models.score2perf import score2perf

class PianoPerformanceLanguageModelProblem(score2perf.Score2PerfProblem):
  @property
  def add_eos_symbol(self):
    return True
  
problem = PianoPerformanceLanguageModelProblem()

from tensor2tensor import problems
unconditional_encoders = problem.get_feature_encoders()

from tensor2tensor.utils import trainer_lib
hparams = trainer_lib.create_hparams(hparams_set='transformer_tpu')
trainer_lib.add_problem_hparams(hparams, problem)
hparams.num_hidden_layers = 16
hparams.sampling_method = 'random'

from tensor2tensor.utils import decoding
decode_hparams = decoding.decode_hparams()
decode_hparams.alpha = 0.0
decode_hparams.beam_size = 1
run_config = trainer_lib.create_run_config(hparams)

from tensorflow.compat.v1 import estimator as tf_estimator
from tensor2tensor.models.transformer import Transformer
def make_estimator_model_fn(model_name,
                            hparams,
                            decode_hparams=None,
                            use_tpu=False):
    
    def wrapping_model_fn(features, labels, mode, params=None, config=None):
        return Transformer.estimator_model_fn(
            hparams,
            features,
            labels,
            mode,
            config=config,
            params=params,
            decode_hparams=decode_hparams,
            use_tpu=use_tpu)

    return wrapping_model_fn

model_fn = make_estimator_model_fn(
      'transformer', hparams, decode_hparams=decode_hparams)
estimator = tf_estimator.Estimator(
    model_fn=model_fn,
    model_dir=run_config.model_dir,
    config=run_config,
)

ckpt_path = 'checkpoint/unconditional_model_16.ckpt'

def input_generator():
    global targets
    global decode_length
    while True:
        print('generating')
        yield {
            'targets': np.array([targets], dtype=np.int32),
            'decode_length': np.array(decode_length, dtype=np.int32)
        }
targets = []
decode_length = 0
input_fn = decoding.make_input_fn_from_generator(input_generator())

import six
from tensorflow.compat.v1.saved_model import simple_save
from tensorflow.python.eager import context
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def predict(self,
            input_fn,
            predict_keys=None,
            hooks=None,
            checkpoint_path=None,
            yield_single_examples=True):
    # _estimator_api_gauge.get_cell('predict').set(True)
    with context.graph_mode():
        hooks = [] # _check_hooks_type(hooks)
        # Check that model has been trained.
        with tf.Graph().as_default() as g:
            tf.compat.v1.random.set_random_seed(self._config.tf_random_seed)
            self._create_and_assert_global_step(g)
            features, input_hooks = self._get_features_from_input_fn(
                    input_fn, ModeKeys.PREDICT)
            estimator_spec = self._call_model_fn(features, None, ModeKeys.PREDICT, self.config)

            # Call to warm_start has to be after model_fn is called.
            self._maybe_warm_start(checkpoint_path)

            predictions = self._extract_keys(estimator_spec.predictions, predict_keys)
            all_hooks = [] #list(input_hooks)
            # all_hooks.extend(hooks)
            # all_hooks.extend(list(estimator_spec.prediction_hooks or []))
            with tf.compat.v1.train.MonitoredSession(
                    session_creator=tf.compat.v1.train.ChiefSessionCreator(
                            checkpoint_filename_with_path=checkpoint_path,
                            master=self._config.master,
                            scaffold=estimator_spec.scaffold,
                            config=self._session_config),
                    hooks=all_hooks) as mon_sess:
                while not mon_sess.should_stop():

                    # this doesn't work.. because the graph is finalized

                    # simple_save(mon_sess,
                    #             'saved_model',
                    #             {k:predictions[k] for k in ['inputs', 'decode_length', 'input_space_id', 'target_space_id']},
                    #             {k:predictions[k] for k in ['outputs', 'scores']})
                    
                    preds_evaluated = mon_sess.run(predictions)
                    
                    yield {
                        'outputs': preds_evaluated['outputs'][0] # only part we need
                    }
                    # for i in range(self._extract_batch_length(preds_evaluated)):
                    #     yield {
                    #             key: value[i]
                    #             for key, value in six.iteritems(preds_evaluated)
                    #     }

# unconditional_samples = estimator.predict(
#     input_fn, checkpoint_path=ckpt_path)
unconditional_samples = predict(estimator,
    input_fn, checkpoint_path=ckpt_path)
_ = next(unconditional_samples)

targets = []
decode_length = 1024
sample_ids = next(unconditional_samples)['outputs']

from tensor2tensor.data_generators import text_encoder


import note_seq
import shutil
import os

def decode(ids, encoder):
    ids = list(ids)
    if text_encoder.EOS_ID in ids:
        idx = ids.index(text_encoder.EOS_ID)
        print('found EOS idx at', idx, 'out of', len(ids))
        ids = ids[:idx]
    else:
        print('did not find EOS in stream', len(ids))
    return encoder.decode(ids)

words = open('words.txt').read().splitlines()
import random
def random_name():
    return '-'.join(random.sample(words, 3))

while True:
    try:
        targets = []
        decode_length = 8192
        sample_ids = next(unconditional_samples)['outputs']

        midi_filename = decode(
            sample_ids,
            encoder=unconditional_encoders['targets'])

        unconditional_ns = note_seq.midi_file_to_note_sequence(midi_filename)

        output_fn = 'output/' + random_name() + '.mid'
        shutil.copyfile(midi_filename, output_fn)
        print(output_fn)

    except KeyboardInterrupt:
        pass