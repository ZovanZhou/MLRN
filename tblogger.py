import pprint
import tensorflow as tf
from typing import Dict


class TensorboardLogger(object):
    def __init__(self, path: str, desc: str) -> None:
        self.__path_train_log = f"{path}/{desc}/train"
        self.__path_valid_log = f"{path}/{desc}/valid"
        self.__path_test_log = f"{path}/{desc}/test"
        self.__train_summary_writer = tf.summary.create_file_writer(
            self.__path_train_log
        )
        self.__valid_summary_writer = tf.summary.create_file_writer(
            self.__path_valid_log
        )
        self.__test_summary_writer = tf.summary.create_file_writer(self.__path_test_log)

        self.__best_valid_result = {}
        self.__best_test_result = {}

    def plot_best_results(self):
        pprint.pprint(
            {"Valid": self.__best_valid_result, "Test": self.__best_test_result}
        )

    def save_results(self, epoch: int, results: Dict) -> bool:
        flag = False
        if epoch == 0:
            self.__best_valid_result = results["valid"]
            self.__best_test_result = results["test"]
        else:
            for k, v in results["valid"].items():
                if "f1" in k or "acc" in k:
                    if results["valid"][k] > self.__best_valid_result[k]:
                        flag = True
                        break
            if flag:
                for k, v in self.__best_valid_result.items():
                    self.__best_valid_result[k] = (v + results["valid"][k]) / 2
                self.__best_test_result = results["test"]

        with self.__train_summary_writer.as_default():
            for k, v in results["train"].items():
                tf.summary.scalar(k, v, step=epoch)
        with self.__valid_summary_writer.as_default():
            for k, v in results["valid"].items():
                tf.summary.scalar(k, v, step=epoch)
        with self.__test_summary_writer.as_default():
            for k, v in results["test"].items():
                tf.summary.scalar(k, v, step=epoch)

        return flag