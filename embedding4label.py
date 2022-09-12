import os
import json
import codecs
import tensorflow as tf
from typing import Tuple, List
from keras_bert import Tokenizer
from tensorflow.data import Dataset
from keras_bert import load_trained_model_from_checkpoint


class LabelEmbedding(object):
    def __init__(self, bert_path: str):
        self.__bert_path = bert_path
        self.__max_seq_len = 8

    def _extract_label_feature(self, labels: List, path: str) -> List:
        if os.path.exists(path):
            with open(path, "r") as fr:
                return json.load(fr)
        else:
            features = self.__extract_bert_feature(self.__parse_label_text(labels))
            with open(path, "w") as fw:
                json.dump(features, fw)
            return features

    def __extract_bert_feature(self, label_texts: Dataset) -> List:
        ckpt_path = f"{self.__bert_path}/bert_model.ckpt"
        config_path = f"{self.__bert_path}/bert_config.json"
        bert_model = load_trained_model_from_checkpoint(
            config_path, ckpt_path, seq_len=None
        )
        for ind, seg in label_texts:
            features = tf.reduce_sum(bert_model.predict([ind, seg]), axis=1)
        return features.numpy().tolist()

    def __parse_label_text(self, labels: List) -> Dataset:
        ext_labels = []
        for label in labels:
            tmp = ""
            for c in label:
                if c >= "A" and c <= "Z":
                    tmp += f" {c}" if tmp != "" else c
                else:
                    tmp += c
            ext_labels.append(tmp)
        tokenizer = self.__load_tokenizer_from_vocab(f"{self.__bert_path}/vocab.txt")
        inds, segs = [], []
        for label in ext_labels:
            ind, seg = tokenizer.encode(label, max_len=self.__max_seq_len)
            inds.append(ind)
            segs.append(seg)
        return Dataset.from_tensor_slices((inds, segs)).batch(len(inds))

    def _load_dicts_from_file(self, path: str) -> Tuple:
        with open(path, "r") as fr:
            dicts = json.load(fr)
        return (dicts["slots"], dicts["intents"])

    def __load_tokenizer_from_vocab(self, vocab_path: str) -> Tokenizer:
        token_dict = {}

        with codecs.open(vocab_path, "r", "utf8") as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)

        return Tokenizer(token_dict)


class SlotLabelEmbedding(LabelEmbedding):
    def __init__(self, dataset: str, bert_path: str):
        super(SlotLabelEmbedding, self).__init__(bert_path=bert_path)
        self.__dict_slots, self.__dict_intents = self._load_dicts_from_file(
            f"{dataset}/dicts.json"
        )
        slot_label_embedding_path = f"{dataset}/slot_label_embedding.json"
        self.__embedding = self._extract_label_feature(
            self.__dict_slots, slot_label_embedding_path
        )

    @property
    def Embedding(self):
        return self.__embedding


class IntentLabelEmbedding(LabelEmbedding):
    def __init__(self, dataset: str, bert_path: str):
        super(IntentLabelEmbedding, self).__init__(bert_path=bert_path)
        self.__dict_slots, self.__dict_intents = self._load_dicts_from_file(
            f"{dataset}/dicts.json"
        )
        intent_label_embedding_path = f"{dataset}/intent_label_embedding.json"
        self.__embedding = self._extract_label_feature(
            self.__dict_intents, intent_label_embedding_path
        )

    @property
    def Embedding(self):
        return self.__embedding