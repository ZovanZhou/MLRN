import os

# Cancel the warning info from tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import pprint
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from model import MT4SFID
import tensorflow_addons as tfa
from loader4data import DataLoader
from tblogger import TensorboardLogger
from utils import train_one_step, evaluate_model

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--order", default=1, type=int)
parser.add_argument("--epoch", default=30, type=int)
parser.add_argument("--rm_num", action="store_true")
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--patience", default=20, type=int)
parser.add_argument("--gamma", default=0.5, type=float)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--max_seq_len", default=64, type=int)
parser.add_argument("--log_dir", default="./logs", type=str)
parser.add_argument("--graph_attn_heads", default=2, type=int)
parser.add_argument("--dropout_rate", default=0.5, type=float)
parser.add_argument("--weight_decay", default=1e-6, type=float)
parser.add_argument("--graph_output_dim", default=512, type=int)
parser.add_argument("--save_model", default=0, choices=[0, 1], type=int)
parser.add_argument("--weights", default="./weights/model.h5", type=str)
parser.add_argument("--undirected_graph", default=0, type=int, choices=[0, 1])
parser.add_argument("--opt", choices=["adam", "adamw"], default="adam", type=str)
parser.add_argument("--mode", default="train", type=str, choices=["train", "test"])
parser.add_argument("--dataset", choices=["atis", "snips"], default="atis", type=str)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

DATASET_PATH = f"./dataset/{args.dataset}"
BERT_PATH = "./wwm_uncased_L-24_H-1024_A-16"
dataLoader = DataLoader(
    dataset=DATASET_PATH,
    bert_path=BERT_PATH,
    batch_size=args.batch_size,
    max_seq_len=args.max_seq_len,
    rm_num=args.rm_num,
    undirected_graph=args.undirected_graph,
)
n_slots = len(dataLoader.SLOTS)
n_entity = len(dataLoader.ENTITY)
n_intents = len(dataLoader.INTENTS)

logger = TensorboardLogger(
    path=args.log_dir,
    desc=f"{args.dataset}"
    + f"_rm_num_{args.rm_num}"
    + f"_gamma_{args.gamma}"
    + f"_dropout_{args.dropout_rate}"
    + f"_gh_{args.graph_attn_heads}"
    + f"_go_{args.graph_output_dim}"
    + f"_order_{args.order}"
    + f"_seed_{args.seed}",
)

model = MT4SFID(
    dataset=DATASET_PATH,
    bert_path=BERT_PATH,
    n_bio_label=n_entity,
    n_entity_label=n_slots,
    n_intent_label=n_intents,
    max_seq_len=args.max_seq_len,
    order=args.order,
    graph_attn_heads=args.graph_attn_heads,
    graph_output_dim=args.graph_output_dim,
    dropout_rate=args.dropout_rate,
)

if args.opt == "adam":
    optimizer = tf.optimizers.Adam(learning_rate=args.lr)
elif args.opt == "adamw":
    optimizer = tfa.optimizers.AdamW(
        learning_rate=args.lr, weight_decay=args.weight_decay
    )


def train_model(model, dataLoader, optimizer, args, logger):
    n_patience = 0
    for e in range(args.epoch):
        losses = []
        for (
            _,
            mask,
            indice,
            segment,
            dep_graph,
            bio_label,
            entity_label,
            intent_label,
        ) in tqdm(dataLoader.Data("train"), ascii=True):
            loss = train_one_step(
                model,
                optimizer,
                mask,
                indice,
                segment,
                tf.sparse.to_dense(
                    tf.sparse.reshape(
                        dep_graph, (-1, args.max_seq_len, args.max_seq_len)
                    )
                ),
                tf.one_hot(bio_label, n_entity),
                tf.one_hot(entity_label, n_slots),
                tf.one_hot(intent_label, n_intents),
                args.gamma,
            )
            losses.append(loss.numpy())

        tb_plot_results = {"train": {"loss": np.mean(losses)}}
        result = evaluate_model(model, dataLoader, "valid")
        tb_plot_results["valid"] = result
        pprint.pprint({"Valid": result})
        result = evaluate_model(model, dataLoader, "test")
        tb_plot_results["test"] = result
        pprint.pprint({"Test": result})

        if logger.save_results(e, tb_plot_results):
            logger.plot_best_results()
            n_patience = 0
            if args.save_model:
                model.save_weights(args.weights)
        else:
            n_patience += 1
            if n_patience == args.patience:
                break
    logger.plot_best_results()


if __name__ == "__main__":
    if args.mode == "train":
        train_model(model, dataLoader, optimizer, args, logger)
    elif args.mode == "test":
        pass