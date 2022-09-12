import numpy as np
from typing import List, Dict
from scipy.sparse import csr_matrix
from stanfordcorenlp import StanfordCoreNLP


class DependencyParser(object):
    def __init__(
        self,
        undirected_graph: int = 0,
        model_path: str = "../stanford-corenlp-full-2018-02-27",
    ) -> None:
        self.__parser = StanfordCoreNLP(model_path)
        self.__undirected_graph = undirected_graph

    def __parse_dict_token(self, dict_token: List) -> Dict:
        dict_result = {}
        for t in dict_token:
            t_id, s_id = [int(e) for e in t.split(":")]
            if t_id not in dict_result:
                dict_result[t_id] = []
            dict_result[t_id].append(s_id)
        return dict_result

    def parse_sentence2graph(
        self, sentence: str, dict_token: List, max_seq_len: int
    ) -> csr_matrix:
        dep = self.__parser.dependency_parse(sentence)
        graph = np.eye(max_seq_len, dtype=np.float32).reshape((1, -1))
        dict_token = self.__parse_dict_token(dict_token)

        for _, start_idx, end_idx in dep:
            if start_idx in dict_token and end_idx in dict_token:
                for i in dict_token[start_idx]:
                    for j in dict_token[end_idx]:
                        graph[0, j * max_seq_len + i] = 1.0
                        if self.__undirected_graph:
                            graph[0, i * max_seq_len + j] = 1.0
                        graph[0, j * max_seq_len + j] = 1.0

        return csr_matrix(graph)
