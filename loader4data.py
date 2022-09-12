import re
import os
import json
import codecs
import numpy as np
from tensorflow.python.framework.ops import Graph
from tqdm import trange
import tensorflow as tf
from scipy import sparse
from typing import Tuple, List
from keras_bert import Tokenizer
from tensorflow.data import Dataset
from dependency_parser import DependencyParser
from keras.preprocessing.sequence import pad_sequences


class DataLoader(object):
    def __init__(
        self,
        dataset: str,
        bert_path: str,
        batch_size: int,
        max_seq_len: int,
        rm_num: bool,
        undirected_graph: int,
    ) -> None:
        self.__rm_num = rm_num
        self.__undirected_graph = undirected_graph

        self.__bert_path = bert_path
        self.__tokenizer = self.__load_tokenizer_from_vocab(
            f"{self.__bert_path}/vocab.txt"
        )

        self.__dict_slots, self.__dict_intents = self.__load_dicts_from_file(
            f"{dataset}/dicts.json"
        )
        self.__dict_entity = ["O", "B-", "I-", "X"]
        self.__max_seq_len = max_seq_len
        self.__batch_size = batch_size
        self.__dependency_parser = DependencyParser(
            undirected_graph=self.__undirected_graph
        )

        ctd_train_file_path, ctd_valid_file_path, ctd_test_file_path = [
            f"{dataset}/{ele}.json" for ele in ["train", "valid", "test"]
        ]

        if not (
            os.path.exists(ctd_train_file_path)
            and os.path.exists(ctd_valid_file_path)
            and os.path.exists(ctd_test_file_path)
        ):
            for ele in ["train", "valid", "test"]:
                self.__transform_data4model(
                    self.__read_data_from_file(f"{dataset}/{ele}.txt"),
                    f"{dataset}/{ele}.json",
                )

        self._train_data = self.__load_data(ctd_train_file_path, shuffle=True)
        self._valid_data = self.__load_data(ctd_valid_file_path)
        self._test_data = self.__load_data(ctd_test_file_path)

    @property
    def SLOTS(self):
        return self.__dict_slots

    @property
    def INTENTS(self):
        return self.__dict_intents

    @property
    def ENTITY(self):
        return self.__dict_entity

    @property
    def MAX_SEQ_LEN(self):
        return self.__max_seq_len

    def Data(self, dtype: str = "train"):
        return getattr(self, f"_{dtype}_data")

    def __load_tokenizer_from_vocab(self, vocab_path: str) -> Tokenizer:
        token_dict = {}

        with codecs.open(vocab_path, "r", "utf8") as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)

        return Tokenizer(token_dict)

    def __load_dicts_from_file(self, path: str) -> Tuple:
        with open(path, "r") as fr:
            dicts = json.load(fr)
        return (dicts["slots"], dicts["intents"])

    def __remove_digital_token(self, token) -> str:
        def istime(input):
            regexp = re.compile(
                "(24:00|24:00:00|2[0-3]:[0-5][0-9]|2[0-3]:[0-5][0-9]:[0-5][0-9]|"
                "[0-1][0-9]:[0-5][0-9]|[0-1][0-9]:[0-5][0-9]:[0-5][0-9]|"
                "([0-9][0-9]|[0-9])/([0-9][0-9]|[0-9])/([0-9][0-9][0-9][0-9]|[0-9][0-9]))"
            )
            return bool(regexp.match(input))

        if str.isdigit(token):
            return "0"
        elif istime(token):
            return "$time$"
        else:
            return token

    def __read_data_from_file(self, path: str) -> Tuple:
        sentences = []
        bio_labels = []
        intent_labels = []
        entity_type_labels = []

        with open(path, "r") as fr:
            sentence = []
            bio_label = []
            entity_type_label = []
            for line in fr.readlines():
                line = line.strip()
                if line:
                    if "\t" in line:
                        sample = line.split("\t")
                        if len(sample) == 3:
                            token, bio, entity_type = sample
                            if self.__rm_num:
                                token = self.__remove_digital_token(token)
                            sentence.append(token)
                            bio_label.append(bio)
                            entity_type_label.append(entity_type)
                    else:
                        intent_labels.append(line)
                else:
                    if len(sentence):
                        sentences.append(sentence.copy())
                        bio_labels.append(bio_label.copy())
                        entity_type_labels.append(entity_type_label.copy())
                        sentence.clear()
                        bio_label.clear()
                        entity_type_label.clear()

        return (sentences, bio_labels, entity_type_labels, intent_labels)

    def __tokenize_word(self, word: str) -> Tuple:
        ind, seg = self.__tokenizer.encode(first=word)
        CLS, SEP = ind[0], ind[-1]
        ind = ind[1:-1]
        seg = seg[1:-1]
        return (CLS, SEP, ind, seg)

    def __extend_bio_label(self, size, bio_label) -> List:
        ext_label = []
        for i in range(size):
            if bio_label != "O":
                ext_label.append(bio_label if i == 0 else "X")
            else:
                ext_label.append(bio_label)
        return ext_label

    def __extract_entity_from_sample(self, bio_label, entity_type_label) -> List:
        dict_entity = []
        tmp_entity = []
        for i, e in enumerate(bio_label):
            if e != "O":
                if len(tmp_entity):
                    if e != "B-":
                        tmp_entity[-1] += 1
                    else:
                        dict_entity.append(tmp_entity.copy())
                        tmp_entity.clear()
                        tmp_entity.extend([entity_type_label[i], i, i + 1])
                else:
                    if e == "B-":
                        tmp_entity.extend([entity_type_label[i], i, i + 1])
            else:
                if len(tmp_entity):
                    dict_entity.append(tmp_entity.copy())
                    tmp_entity.clear()
        if len(tmp_entity):
            dict_entity.append(tmp_entity.copy())
            tmp_entity.clear()
        return dict_entity

    def __extend_sample_with_entity(
        self, indice, segment, bio_label, entity_type_label, dep_graph
    ) -> Tuple:
        masks = []
        indices, segments, bio_labels, entity_labels, dep_graphs = [], [], [], [], []
        dict_entity = self.__extract_entity_from_sample(bio_label, entity_type_label)
        if len(dict_entity):
            for ele in dict_entity:
                entity_type, start_idx, end_idx = ele
                mask = [0] * len(bio_label)
                for i in range(start_idx, end_idx):
                    mask[i] = 1
                masks.append(mask)
                indices.append(indice)
                segments.append(segment)
                bio_labels.append(bio_label)
                dep_graphs.append(dep_graph)
                entity_labels.append(entity_type)
        else:
            entity_type = "O"
            mask = [0] * len(bio_label)
            masks.append(mask)
            indices.append(indice)
            segments.append(segment)
            bio_labels.append(bio_label)
            dep_graphs.append(dep_graph)
            entity_labels.append(entity_type)
        assert len(bio_labels[0]) == len(indices[0]), "Sequence length is not equal"
        return (indices, segments, bio_labels, masks, entity_labels, dep_graphs)

    def __parse_sample(
        self, sentence: List, bio_label: List, entity_type_label: List
    ) -> Tuple:
        dict_token = []
        ext_bio_label = []
        indice, segment = [], []
        ext_entity_type_label = []
        for j, word in enumerate(sentence):
            CLS, SEP, ind, seg = self.__tokenize_word(word)
            dict_token.extend(
                [
                    f"{j+1}:{i}"
                    for i in range(len(indice) + 1, len(indice) + len(ind) + 1)
                ]
            )
            indice.extend(ind)
            segment.extend(seg)
            ext_bio_label.extend(self.__extend_bio_label(len(ind), bio_label[j]))
            ext_entity_type_label.extend([entity_type_label[j]] * len(ind))
        indice = [CLS] + indice[: self.__max_seq_len - 2] + [SEP]
        segment = [0] + segment[: self.__max_seq_len - 2] + [0]
        dict_token = ["0:0"] + dict_token[: self.__max_seq_len - 1]
        dep_graph = self.__dependency_parser.parse_sentence2graph(
            " ".join(sentence), dict_token, self.__max_seq_len
        )
        ext_bio_label, ext_entity_type_label = [
            ["O"] + ele[: self.__max_seq_len - 2] + ["O"]
            for ele in [ext_bio_label, ext_entity_type_label]
        ]
        assert len(ext_bio_label) == len(
            indice
        ), "Parsed entity labels are not equal in length"
        return self.__extend_sample_with_entity(
            indice, segment, ext_bio_label, ext_entity_type_label, dep_graph
        )

    def __label2idx(self, labels, dict) -> List:
        idxs = []
        for i in range(len(labels)):
            tmp = []
            for j, ele in enumerate(labels[i]):
                tmp.append(dict.index(ele) if j != 0 else -1)
            idxs.append(tmp)
        return idxs

    def idx2biolabel(self, idxs) -> List:
        labels = []
        for idx in idxs:
            labels.append([self.__dict_entity[i] for i in idx])
        return labels

    def idx2slotlabel(self, idxs) -> List:
        return [self.__dict_slots[i] for i in idxs]

    def idx2intentlabel(self, idxs) -> List:
        return [self.__dict_intents[i] for i in idxs]

    def __load_data(self, save_path: str, shuffle: bool = False) -> Dataset:
        data = None
        with open(save_path, "r") as fr:
            data = json.load(fr)
        graph_type = "undirected" if self.__undirected_graph else "directed"
        dep_graphs = sparse.load_npz(f"{save_path}.dep_{graph_type}_graph.npz")
        dataset = Dataset.from_tensor_slices(
            (
                data["ids"],
                data["masks"],
                data["indices"],
                data["segments"],
                tf.SparseTensor(
                    indices=list(zip(*dep_graphs.nonzero())),
                    values=np.float32(dep_graphs.data),
                    dense_shape=dep_graphs.get_shape(),
                ),
                data["bio_labels"],
                data["entity_labels"],
                data["intent_labels"],
            )
        )
        if shuffle:
            dataset = dataset.shuffle(len(data["masks"]))
        dataset = dataset.batch(self.__batch_size).prefetch(
            tf.data.experimental.AUTOTUNE
        )
        return dataset

    def __transform_data4model(self, data: Tuple, save_path: str) -> None:
        sentences, bio_labels, entity_type_labels, intent_labels = data
        ids, masks, indices, segments, dep_graphs = [], [], [], [], []
        ext_bio_labels, ext_entity_labels, ext_intent_labels = [], [], []

        for i in trange(len(sentences), ascii=True):
            inds, segs, bios, ms, entity_labels, dep_gs = self.__parse_sample(
                sentences[i], bio_labels[i], entity_type_labels[i]
            )
            ids.extend([i] * len(inds))
            masks.extend(ms)
            indices.extend(inds)
            segments.extend(segs)
            dep_graphs.extend(dep_gs)
            ext_bio_labels.extend(bios)
            ext_entity_labels.extend(entity_labels)
            ext_intent_labels.extend([intent_labels[i]] * len(inds))

        ext_bio_labels = self.__label2idx(ext_bio_labels, self.__dict_entity)
        ext_entity_labels = [
            self.__dict_slots.index(ele) if ele != "O" else -1
            for ele in ext_entity_labels
        ]
        ext_intent_labels = [
            self.__dict_intents.index(ele) for ele in ext_intent_labels
        ]
        dep_graphs = sparse.vstack(dep_graphs)
        masks, indices, segments, ext_bio_labels = [
            pad_sequences(ele, self.__max_seq_len, value=0, padding="post")
            for ele in [masks, indices, segments, ext_bio_labels]
        ]
        assert len(masks) == len(indices), "Sample numbers are not equal"

        with open(save_path, "w") as fp:
            json.dump(
                {
                    "ids": ids,
                    "masks": masks.tolist(),
                    "indices": indices.tolist(),
                    "segments": segments.tolist(),
                    "bio_labels": ext_bio_labels.tolist(),
                    "entity_labels": ext_entity_labels,
                    "intent_labels": ext_intent_labels,
                },
                fp,
            )
        graph_type = "undirected" if self.__undirected_graph else "directed"
        sparse.save_npz(f"{save_path}.dep_{graph_type}_graph.npz", dep_graphs)


if __name__ == "__main__":
    dataLoader = DataLoader(
        "./dataset/snips",
        "./wwm_uncased_L-24_H-1024_A-16",
        batch_size=12,
        max_seq_len=64,
        rm_num=True,
    )
    for (
        ids,
        mask,
        indice,
        segment,
        dep_graph,
        ext_bio_label,
        ext_entity_label,
        ext_intent_label,
    ) in dataLoader.Data("train"):
        # print(ids)
        # print(mask)
        # print(indice)
        # print(segment)
        # print(tf.sparse.to_dense(tf.sparse.reshape(dep_graph, (-1, 64, 64))))
        # print(ext_bio_label)
        # print(ext_entity_label)
        # print(ext_intent_label)
        break
