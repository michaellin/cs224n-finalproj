#Content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/
import os
import pyrouge
import logging
import tensorflow as tf
import torch
from data_util import config
import numpy as np
import tables

def print_results(article, abstract, decoded_output):
  print ("")
  print('ARTICLE:  %s', article)
  print('REFERENCE SUMMARY: %s', abstract)
  print('GENERATED SUMMARY: %s', decoded_output)
  print( "")


def make_html_safe(s):
  s.replace("<", "&lt;")
  s.replace(">", "&gt;")
  return s


def rouge_eval(ref_dir, dec_dir):
  r = pyrouge.Rouge155()
  r.model_filename_pattern = '#ID#_reference.txt'
  r.system_filename_pattern = '(\d+)_decoded.txt'
  r.model_dir = ref_dir
  r.system_dir = dec_dir
  logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
  rouge_results = r.convert_and_evaluate()
  return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
  log_str = ""
  for x in ["1","2","l"]:
    log_str += "\nROUGE-%s:\n" % x
    for y in ["f_score", "recall", "precision"]:
      key = "rouge_%s_%s" % (x,y)
      key_cb = key + "_cb"
      key_ce = key + "_ce"
      val = results_dict[key]
      val_cb = results_dict[key_cb]
      val_ce = results_dict[key_ce]
      log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
  print(log_str)
  results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
  print("Writing final ROUGE results to %s..."%(results_file))
  with open(results_file, "w") as f:
    f.write(log_str)


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
  if running_avg_loss == 0:  # on the first iteration just take the loss
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)  # clip
  loss_sum = tf.Summary()
  tag_name = 'running_avg_loss/decay=%f' % (decay)
  loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  return running_avg_loss


def write_for_rouge(reference_sents, decoded_words, ex_index,
                    _rouge_ref_dir, _rouge_dec_dir):
  decoded_sents = []
  while len(decoded_words) > 0:
    try:
      fst_period_idx = decoded_words.index(".")
    except ValueError:
      fst_period_idx = len(decoded_words)
    sent = decoded_words[:fst_period_idx + 1]
    decoded_words = decoded_words[fst_period_idx + 1:]
    decoded_sents.append(' '.join(sent))

  # pyrouge calls a perl script that puts the data into HTML files.
  # Therefore we need to make our output HTML safe.
  decoded_sents = [make_html_safe(w) for w in decoded_sents]
  reference_sents = [make_html_safe(w) for w in reference_sents]

  ref_file = os.path.join(_rouge_ref_dir, "%06d_reference.txt" % ex_index)
  decoded_file = os.path.join(_rouge_dec_dir, "%06d_decoded.txt" % ex_index)

  with open(ref_file, "w") as f:
    for idx, sent in enumerate(reference_sents):
      f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
  with open(decoded_file, "w") as f:
    for idx, sent in enumerate(decoded_sents):
      f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")



example_num = 0
def analyze_pgen(data_fn, vocab, reference_sents, input_art_ids, oov_ids,
          decoded_word_ids, decoded_words, final_dist, vocab_dist, p_gens):
    """ Compare number of novel n-grams to the values of p-gen.
      Is there a correlation between p-gen values and n-gram
      novelty?
        @param vocab
        @param reference_sents (str)
        @param input_art_ids (List[int])
        @param oov_ids (List[int])
        @param decoded_word_ids (List[int])
        @param decoded_words (List[str])
        @param final_dist
        @param vocab_dist
        @param p_gens (List[float])

        Notes:

        Ideas
        1. gather top p_gen values and their associated words
        2. is attn_dist overtaking vocab_dist because one dist
        is more spread than the other? compare their max values
    """
    global example_num

    pad_size = config.max_enc_steps - input_art_ids.shape[1]
    if (pad_size != 0):
        input_art_ids_padded = np.pad(input_art_ids[0,:], (0,pad_size),'constant')
    else:
        input_art_ids_padded = input_art_ids[0,:]
    # definition of pytable to store processed data
    if (os.path.exists(data_fn)):
        f = tables.open_file(data_fn, mode="a", title="Processed data from model {}".format(data_fn))
        dec_arr = f.root.dec
        point_arr = f.root.point
        pgen_arr = f.root.pgen
        copy_arr = f.root.copy
        dec_word_arr = f.root.dec_word
        dec_ids_arr = f.root.dec_ids 
        ex_num_arr = f.root.ex_num
        in_ids_arr = f.root.in_ids
    else:
        f = tables.open_file(data_fn, mode="w", title="Processed data from model {}".format(data_fn))
        dec_out  = tables.Float32Col()     # float  (single-precision)
        point_out  = tables.Float32Col()   # float  (single-precision)
        pgen_out  = tables.Float32Col()    # float  (single-precision)
        inart_id = tables.UInt32Col()      # int  (unsigned 32-bit)
        ex_num = tables.UInt32Col()        # int  (unsigned 32-bit)
        dec_id = tables.UInt32Col()        # int  (unsigned 32-bit)
        copy = tables.UInt8Col()           # int  (unsigned 8-bit)
        dec_word = tables.StringCol(100)
        dec_arr = f.create_earray(f.root,'dec',dec_out,(1,0))
        point_arr = f.create_earray(f.root,'point',point_out,(1,0))
        pgen_arr = f.create_earray(f.root,'pgen',pgen_out,(1,0))
        copy_arr = f.create_earray(f.root,'copy',copy,(1,0))
        dec_word_arr = f.create_earray(f.root,'dec_word',dec_word,(1,0))
        dec_ids_arr = f.create_earray(f.root,'dec_ids',dec_id,(1,0))
        ex_num_arr = f.create_earray(f.root,'ex_num',ex_num,(1,0))
        in_ids_arr = f.create_earray(f.root,'in_ids',inart_id,(config.max_enc_steps,0))

    # add the input article ids to the table
    in_ids_arr.append(input_art_ids_padded.reshape((config.max_enc_steps,1)))

    for wi, word_id in enumerate(decoded_word_ids):
        if word_id < vocab.size():
            word_from_input = word_id in input_art_ids
        else:
            # if it is an oov then we know it was copied
            word_from_input = True
        
        #word_in_source_temp += [word_from_input]
        copy_arr.append(np.array([[int(word_from_input)]]))
        decode_out = vocab_dist[wi][word_id]*p_gens[wi]
        dec_arr.append(np.array([[decode_out.item()]]))
        point_arr.append(np.array([[(final_dist[wi][word_id]-decode_out).item()]]))
        pgen_arr.append(np.array([[p_gens[wi]]]))
        dec_word_arr.append(np.array([[decoded_words[wi]]]))
        dec_ids_arr.append(np.array([[word_id]]))
        ex_num_arr.append(np.array([[example_num]]))
    example_num += 1
    f.close()
