import glob
import six
import os
import tarfile
import numpy as np
class Converter(object):
    def __init__(self, vocab, beg, end, unk, delimiter, add_beg):
        self._vocab = vocab
        self._beg = beg
        self._end = end
        self._unk = unk
        self._delimiter = delimiter
        self._add_beg = add_beg

    def _call__(self, sentence):
        return ([self._beg] if self._add_beg else []) + [
            self._vocab.get(w, self._unk)
            for w in sentence.split(self._delimiter)
        ] + [self._end]

class ComposedConverter(object):
    def __init__(self, converters):
        self._converters = converters

    def __call__(self, parallel_sentence):
        return [
            self._converters[i](parallel_sentence[i])
            for i in range(len(self._converters))
        ]

class SentenceBatchCreator(object):
    def __init__(self, batch_size):
        self.batch = []
        self._batch_size = batch_size

    def append(self, info):
        self.batch.append(info)
        if len(self.batch) == self._batch_size:
            tmp = self.batch
            self.batch = []
            return tmp

class SampleInfo(object):
    def __init__(self, i, max_len, min_len):
        self.i = i
        self.min_len = min_len
        self.max_len = max_len

class MinMaxFilter(object):
    def __init__(self, max_len, min_len, underlying_creator):
        self._min_len = min_len
        self._max_len = max_len
        self._creator = underlying_creator

    def append(self, info):
        if info.max_len > self._max_len or info.min_len < self._min_len:
            return
        else:
            return self._creator.append(info)

    @property
    def batch(self):
        return self._creator.batch

class DataReader(object):
    def __init__(self,
                 src_vocab_fpath,
                 trg_vocab_fpath,
                 fpattern,
                 batch_size,
                 pool_size,
                 sort_type=SortType.GLOBAL,
                 clip_last_batch=True,
                 tar_fname=None,
                 min_length=0,
                 max_length=100,
                 shuffle=False,
                 shuffle_batch=False,
                 use_token_batch=False,
                 field_delimiter="\t",
                 token_delimiter=" ",
                 start_mark="<s>",
                 end_mark="<e>",
                 unk_mark="<unk>",
                 seed=0):
        self._src_vocab = self.load_dict(src_vocab_fpath)
        self._trg_vocab = self.load_dict(trg_vocab_fpath)
        self._only_src = False
        self._pool_size = pool_size
        self._batch_size = batch_size
        self._use_token_batch = use_token_batch
        self._sort_type = sort_type
        self._clip_last_batch = clip_last_batch
        self._shuffle = shuffle
        self._shuffle_batch = shuffle_batch
        self._min_length = min_length
        self._max_length = max_length
        self._field_delimiter = field_delimiter
        self._token_delimiter = token_delimiter
        self.load_src_trg_ids(end_mark, fpattern, start_mark, tar_fname,unk_mark)
        self._random = np.random
        self._random.seed(seed)

    def load_src_trg_ids(self, end_mark, fpattern, start_mark, tar_fname, unk_mark):
        converters = [
            Converter(
                vocab=self._src_vocab,
                beg=self._src_vocab[start_mark],
                end=self._src_vocab[end_mark],
                unk=self._src_vocab[unk_mark],
                delimiter=self._token_delimiter,
                add_beg=False)
        ]
        converters.append(
            Converter(
                vocab=self._trg_vocab,
                beg=self._trg_vocab[start_mark],
                end=self._trg_vocab[end_mark],
                unk=self._trg_vocab[unk_mark],
                delimiter=self._token_delimiter,
                add_beg=True))
        converters = ComposedConverter(converters)

        self._src_seq_ids = []
        self._trg_seq_ids = []
        self._sample_infos = []

        for i, line in enumerate(self._load_lines(fpattern, tar_fname)):
            src_trg_ids = converters(line)
            self._src_seq_ids.append(src_trg_ids[0])
            lens = [len(src_trg_ids[0])]
            self._trg_seq_ids.append(src_trg_ids[1])
            lens.append(len(src_trg_ids[1]))
            self._sample_infos.append(SampleInfo(i, max(lens), min(lens)))

    def _load_lines(self, fpattern, tar_fname):
        fpaths = glob.glob(fpattern)

        if not os.path.isfile(fpath):
                raise IOError("Invalid file: %s" % fpath)
            with open(fpath, "rb") as f:
                for line in f:
                    if six.PY3:
                        line = line.decode()
                    fields = line.strip("\n").split(self._field_delimiter)
                    if (len(fields) == 2) or (len(fields) == 1):
                        yield fields

    @staticmethod
    def load_dict(dict_path, reverse=False):
        word_dict = {}
        with open(dict_path, "rb") as fdict:
            for idx, line in enumerate(fdict):
                if six.PY3:
                    line = line.decode()
                if reverse:
                    word_dict[idx] = line.strip("\n")
                else:
                    word_dict[line.strip("\n")] = idx
        return word_dict

    def batch_generator(self):
        infos = self._sample_infos
        # concat batch
        batches = []
        batch_creator = SentenceBatchCreator(self._batch_size)
        batch_creator = MinMaxFilter(self._max_length, self._min_length,batch_creator)
        for info in infos:
            batch = batch_creator.append(info)
            if batch is not None:
                batches.append(batch)

        if not self._clip_last_batch and len(batch_creator.batch) != 0:
            batches.append(batch_creator.batch)

        for batch in batches:
            batch_ids = [info.i for info in batch]
            yield [(self._src_seq_ids[idx], self._trg_seq_ids[idx][:-1], self._trg_seq_ids[idx][1:]) for idx in batch_ids]
