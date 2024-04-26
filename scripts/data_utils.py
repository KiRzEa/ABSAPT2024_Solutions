from __future__ import absolute_import, division, print_function

import csv
import logging
import numpy as np
import os
import pdb
import sys

from sklearn.metrics import matthews_corrcoef, f1_score, hamming_loss, precision_score, recall_score

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_cate=None, text_senti=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, tokens_len, aspect_input_ids, aspect_input_mask, aspect_ids, aspect_segment_ids, aspect_labels):
        self.tokens_len = tokens_len
        self.aspect_input_ids=aspect_input_ids
        self.aspect_input_mask=aspect_input_mask
        self.aspect_ids=aspect_ids
        self.aspect_segment_ids=aspect_segment_ids
        self.aspect_labels=aspect_labels


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";", quotechar=quotechar)
            lines = []
            for line in reader[1:]:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class ATEProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, domain_type):
        """See base class."""
        string = domain_type
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train2024.csv")))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train2024.csv")), "train")

    def get_valid_examples(self, data_dir, domain_type):
        """See base class."""
        string = domain_type
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "task2_test.csv")))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "task2_test.csv"), "valid"))

    def get_test_examples(self, data_dir):
        """See base class."""
        string = domain_type
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "task1_test.csv")))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "task1_test.csv")), "test")

    def get_labels(self, domain_type, cls_token):
        """See base class."""

        seqlabs = [cls_token, 'B', 'I', 'O']
        label_list = []
        
        label_list.append(sentiment)
        label_list.append(seqlabs)
        return label_list

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for line in lines:
            guid = line[0]
            try:
                text_a = line[1]
            except:
                pdb.set_trace()
            labels = line[1:]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=labels))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode, cls_token='[CLS]'):
    """Loads a data file into a list of `InputBatch`s."""

    label_map_senti = {label : i for i, label in enumerate(label_list[0])}
    label_map_seq = {label : i for i, label in enumerate(label_list[1])}

    features = []

    for (ex_index, example) in enumerate(examples):
        # pdb.set_trace()
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        orig_tokens = example.text_a.strip().split()
        labels = example.label


        bert_tokens_a = orig_tokens

        aspect_labels = ['O' for ele in range(len(orig_tokens))]
        for quad in labels:
            cur_aspect = quad.split(' ')[0]; cur_opinion = quad.split(' ')[-1]
            a_st = int(cur_aspect.split(',')[0]); a_ed = int(cur_aspect.split(',')[1])
            if a_ed != -1:
                aspect_labels[a_st] = 'B'
                for i in range(a_st+1, a_ed):
                    aspect_labels[i] = 'I'
            else:
                exist_imp_aspect = 1
            o_st = int(cur_opinion.split(',')[0]); o_ed = int(cur_opinion.split(',')[1])

        _truncate_seq_pair(bert_tokens_a, aspect_labels, max_seq_length - 2)

        aspect_ids = []

        aspect_tokens = []
        aspect_segment_ids = []

        aspect_tokens.append(cls_token)
        aspect_ids.append(label_map_seq[cls_token])
        aspect_segment_ids.append(0)

        for i, token in enumerate(bert_tokens_a):
            aspect_tokens.append(token)
            aspect_ids.append(label_map_seq[aspect_labels[i]])
            aspect_segment_ids.append(0)
            
        aspect_tokens.append(cls_token)
        tokens_len = len(aspect_tokens)

        aspect_ids.append(label_map_seq[cls_token])
        aspect_segment_ids.append(0)

        aspect_input_ids = tokenizer.convert_tokens_to_ids(aspect_tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        aspect_input_mask = [1] * len(aspect_input_ids)
        # if example.text_a.startswith('it has all the features that we'):
        #   pdb.set_trace()

        # Zero-pad up to the sequence length.
        while len(aspect_input_ids) < max_seq_length:
            aspect_input_ids.append(0)
            aspect_input_mask.append(0)
            aspect_ids.append(label_map_seq["O"])
            aspect_segment_ids.append(0)

        assert len(aspect_input_ids) == max_seq_length
        assert len(aspect_input_mask) == max_seq_length
        assert len(aspect_ids) == max_seq_length
        assert len(aspect_segment_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens_len: %s" % (tokens_len))
            logger.info("guid: %s" % (exist_imp_aspect))
            logger.info("guid: %s" % (exist_imp_opinion))

            logger.info("aspect tokens: %s" % " ".join(
                    [str(x) for x in aspect_tokens]))
            logger.info("aspect_input_ids: %s" % " ".join([str(x) for x in aspect_input_ids]))
            logger.info("aspect_input_mask: %s" % " ".join([str(x) for x in aspect_input_mask]))
            logger.info("aspect_ids: %s" % " ".join([str(x) for x in aspect_ids]))
            logger.info(
                    "aspect_segment_ids: %s" % " ".join([str(x) for x in aspect_segment_ids]))

        features.append(
                InputFeatures(tokens_len,
                    aspect_input_ids=aspect_input_ids,
                    aspect_input_mask=aspect_input_mask,
                    aspect_ids=aspect_ids,
                    aspect_segment_ids=aspect_segment_ids,
                    aspect_labels=aspect_labels,
                    exist_imp_aspect=exist_imp_aspect,
                    exist_imp_opinion=exist_imp_opinion))
    return features


def _truncate_seq_pair(bert_tokens_a, aspect_labels, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(bert_tokens_a)
    if total_length <= max_length:
        break
    bert_tokens_a.pop()
    aspect_labels.pop()


def convert_examples_to_features2nd(examples, label_list, max_seq_length,
                                 tokenizer, output_mode, cls_token='[CLS]'):
    """Loads a data file into a list of `InputBatch`s."""

    category_senti_map = {label : i for i, label in enumerate(label_list[0])}

    features = []

    for (ex_index, example) in enumerate(examples):
        # pdb.set_trace()
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        orig_tokens, ao_tags = example.text_a.strip().split('####')
        # label for examples with negative samples
        # labels = example.label[:-1]
        orig_tokens = orig_tokens.split()
        labels = example.label

        bert_tokens_a = orig_tokens
        bert_tokens_b = None

        _truncate_seq_pair2nd(bert_tokens_a, max_seq_length - 2)

        aspect_tokens = []
        aspect_segment_ids = []

        aspect_tokens.append(cls_token)
        aspect_segment_ids.append(0)

        for i, token in enumerate(bert_tokens_a):
            aspect_tokens.append(token)
            aspect_segment_ids.append(0)
        aspect_tokens.append(cls_token)
        tokens_len = len(aspect_tokens)
        aspect_segment_ids.append(0)

        aspect_input_ids = tokenizer.convert_tokens_to_ids(aspect_tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        aspect_input_mask = [1] * len(aspect_input_ids)
        imp_opinion_pos = len(aspect_input_ids)
        # if example.text_a.startswith('it has all the features that we'):
        #   pdb.set_trace()

        # Zero-pad up to the sequence length.
        while len(aspect_input_ids) < max_seq_length:
            aspect_input_ids.append(0)
            aspect_input_mask.append(0)
            aspect_segment_ids.append(0)

        assert len(aspect_input_ids) == max_seq_length
        assert len(aspect_input_mask) == max_seq_length
        assert len(aspect_segment_ids) == max_seq_length

        # get candidate aspect and opinion
        label_id = [0] * len(label_list[0])
        candidate_aspect = [0 for i in range(max_seq_length)]
        candidate_opinion = [0 for i in range(max_seq_length)]
        cur_aspect = ao_tags.split()[0]; cur_opinion = ao_tags.split()[1]
        a_st = int(cur_aspect.split(',')[0]); a_ed = int(cur_aspect.split(',')[1])
        o_st = int(cur_opinion.split(',')[0]); o_ed = int(cur_opinion.split(',')[1])
        if a_st == -1:
            a_ed = 0
        if o_st == -1:
            o_st = imp_opinion_pos - 2; o_ed = imp_opinion_pos - 1
        for i in range(a_st+1, a_ed+1):
            candidate_aspect[i] = 1
        for i in range(o_st+1, o_ed+1):
            candidate_opinion[i] = 1
        if len(labels) > 0:
            for ele in labels[0].split():
                label_id[category_senti_map[ele]] = 1

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens_len: %s" % (tokens_len))

            logger.info("aspect tokens: %s" % " ".join(
                    [str(x) for x in aspect_tokens]))
            logger.info("aspect_input_ids: %s" % " ".join([str(x) for x in aspect_input_ids]))
            logger.info("aspect_input_mask: %s" % " ".join([str(x) for x in aspect_input_mask]))
            logger.info(
                    "aspect_segment_ids: %s" % " ".join([str(x) for x in aspect_segment_ids]))
            logger.info(
                    "candidate_aspect: %s" % " ".join([str(x) for x in candidate_aspect]))
            logger.info(
                    "candidate_opinion: %s" % " ".join([str(x) for x in candidate_opinion]))
            logger.info(
                    "label_id: %s" % " ".join([str(x) for x in label_id]))

        features.append(
                InputFeatures2nd(
                    tokens_len=tokens_len,
                    aspect_tokens=aspect_tokens,
                    aspect_input_ids=aspect_input_ids,
                    aspect_input_mask=aspect_input_mask,
                    aspect_segment_ids=aspect_segment_ids,
                    
                    candidate_aspect=candidate_aspect,
                    candidate_opinion=candidate_opinion,
                    label_id=label_id,
                    ))
    return features


processors = {
    "quad": QuadProcessor,
    "categorysenti": CategorySentiProcessor,
}

output_modes = {
    "quad": "classification",
    "categorysenti": "classification",
}