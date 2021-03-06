#Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/

from __future__ import unicode_literals, print_function, division

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import os
import time

import torch
from torch.autograd import Variable

from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util import data, config
from model import Model
from data_util.utils import write_for_rouge, rouge_eval, rouge_log
from data_util.utils import analyze_pgen
from train_util import get_input_from_batch


use_cuda = config.use_gpu and torch.cuda.is_available()

class Beam(object):
  # Michael added arg p_gen
  def __init__(self, tokens, log_probs, state, context, coverage, 
              p_gen, vocab_dist, final_dist):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage
    # Michael added to track all p_gen of this beam
    self.p_gens = p_gen
    self.vocab_list = vocab_dist
    self.final_dist_list = final_dist

  # Michael added arg p_gen
  def extend(self, token, log_prob, state, context, coverage, 
            p_gen, vocab_dist, final_dist):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      context = context,
                      coverage = coverage,
                      p_gen = self.p_gens + [p_gen],
                      vocab_dist = self.vocab_list + [vocab_dist],
                      final_dist = self.final_dist_list + [final_dist])

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path):
        model_name = os.path.basename(model_file_path)
        self._decode_dir = os.path.join(config.log_root, 'decode_%s' % (model_name))
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
                               batch_size=config.beam_size, single_pass=True)
        time.sleep(15)

        self.model = Model(model_file_path, is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


    def decode(self,data_fn):
        start = time.time()
        counter = 0
        batch = self.batcher.next_batch()
        while batch is not None:
            # Run beam search to get best Hypothesis
            # Here try to return the p_gen values
            best_summary = self.beam_search(batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self.vocab,
                                                 (batch.art_oovs[0] if config.pointer_gen else None))

            # Extract list of p_gen values without first value
            p_gens = best_summary.p_gens[1:]
            # Extract list of contexts values without first value
            #context_list = best_summary.context_list[1:]
            #there is no start for these ones
            final_dist_list = best_summary.final_dist_list[:]
            vocab_list = best_summary.vocab_list[:]

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
                p_gens = p_gens[:fst_stop_idx]
                #context_list = context_list [:fst_stop_idx]
                output_ids_nostop = output_ids[:fst_stop_idx]
                final_dist_list = final_dist_list [:fst_stop_idx]
                vocab_list = vocab_list [:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_abstract_sents = batch.original_abstracts_sents[0]

            write_for_rouge(original_abstract_sents, decoded_words, counter,
                            self._rouge_ref_dir, self._rouge_dec_dir)

            # analyze relation of pgen to the decoded words
            analyze_pgen(data_fn=data_fn,
                        vocab=self.vocab,
                        reference_sents=original_abstract_sents, 
                        input_art_ids=batch.enc_batch, 
                        oov_ids=batch.art_oovs[0],
                        decoded_word_ids=output_ids_nostop, 
                        decoded_words=decoded_words, 
                        final_dist=final_dist_list,
                        vocab_dist=vocab_list,
                        p_gens=p_gens)

            counter += 1
            if counter % 1000 == 0:
                print('%d example in %d sec'%(counter, time.time() - start))
                start = time.time()

            batch = self.batcher.next_batch()

        print("Decoder has finished reading dataset for single_pass.")
        print("Now starting ROUGE eval...")
        results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._decode_dir)


    def beam_search(self, batch):
        #batch should have only one example
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
            get_input_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        #decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None),
                      p_gen = [0.0],
                      vocab_dist=[],
                      final_dist=[])
                 for _ in xrange(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                y_t_1 = y_t_1.cuda()
            all_state_h =[]
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in xrange(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)
                p_gen_i = p_gen[i].item()
                vocab_dist_i = self.model.decoder.get_vocab_dist()[i]
                final_dist_i = final_dist[i]

                for j in xrange(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i,
                                   p_gen=p_gen_i,
                                   vocab_dist=vocab_dist_i,
                                   final_dist=final_dist_i)
                    all_beams.append(new_beam)

            # filter top k best beams
            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

if __name__ == '__main__':
    model_filename = sys.argv[1]
    beam_Search_processor = BeamSearch(model_filename)
    beam_Search_processor.decode(sys.argv[2])


