import os
import sys

from data_util.utils import write_for_rouge, rouge_eval, rouge_log


if __name__ == '__main__':
    model_filename = sys.argv[1]
    print("ROUGE calculation for decoded text.")
    results_dict = rouge_eval("log/" + model_filename + "/rouge_ref", "log/" + model_filename + "/rouge_dec_dir")
    rouge_log(results_dict, "log/" + model_filename + "/rouge_calc")
