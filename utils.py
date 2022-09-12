import numpy as np
from tqdm import tqdm
import tensorflow as tf
from conlleval import evaluate
from typing import List, Dict, Tuple


@tf.function
def train_one_step(
    model,
    optimizer,
    mask,
    indice,
    segment,
    dep_graph,
    bio_label,
    entity_label,
    intent_label,
    gamma,
):
    with tf.GradientTape() as tape:
        logits_bio, logits_entity, logits_intent = model(
            mask, indice, segment, dep_graph, training=True
        )
        loss_bio = tf.reduce_mean(
            tf.reduce_sum(
                tf.losses.categorical_crossentropy(bio_label, logits_bio), axis=-1
            )
        )
        loss_entity = tf.reduce_mean(
            tf.losses.categorical_crossentropy(entity_label, logits_entity)
        )
        loss_intent = tf.reduce_mean(
            tf.losses.categorical_crossentropy(intent_label, logits_intent)
        )
        loss = (1.0 - gamma) * (loss_bio + loss_entity) + gamma * loss_intent
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def extract_bio_label(bio_label: List) -> List:
    dict_bio = []
    tmp_bio = []
    for i, label in enumerate(bio_label):
        if label != "O":
            if len(tmp_bio):
                if label == "B-":
                    dict_bio.append(tmp_bio.copy())
                    tmp_bio.clear()
                    tmp_bio.extend([i, i + 1])
                else:
                    tmp_bio[-1] += 1
            else:
                tmp_bio.extend([i, i + 1])
        else:
            if len(tmp_bio):
                dict_bio.append(tmp_bio.copy())
                tmp_bio.clear()
    return dict_bio


def predict_one_sample(model, dataLoader, indice, segment, dep_graph) -> Tuple:
    tf_indice, tf_segment, tf_dep_graph = [
        tf.constant([ele]) for ele in [indice, segment, dep_graph]
    ]
    tf_mask = tf.constant([[0] * len(indice)])
    logits_bio, _, logits_intent = model(tf_mask, tf_indice, tf_segment, tf_dep_graph)
    intent_label = dataLoader.idx2intentlabel(
        tf.argmax(logits_intent, axis=-1).numpy().tolist()
    )
    pred_bio = dataLoader.idx2biolabel(tf.argmax(logits_bio, axis=-1).numpy().tolist())
    pred_bio, intent_label = [ele[0] for ele in [pred_bio, intent_label]]
    dict_bio = extract_bio_label(pred_bio)

    bio_entity_label = ["O"] * len(indice)
    if len(dict_bio):
        masks = []
        indices, segments, dep_graphs = [], [], []
        for ele in dict_bio:
            start_idx, end_idx = ele
            mask = [0] * len(indice)
            for i in range(start_idx, end_idx):
                mask[i] = 1
            masks.append(mask)
            indices.append(indice)
            segments.append(segment)
            dep_graphs.append(np.expand_dims(dep_graph, 0))

        tf_mask, tf_indice, tf_segment, tf_dep_graph = [
            tf.constant(np.vstack(ele))
            for ele in [masks, indices, segments, dep_graphs]
        ]
        logits_entity = model(tf_mask, tf_indice, tf_segment, tf_dep_graph)[1]
        # processing slot prediction
        pred_entity = dataLoader.idx2slotlabel(
            tf.argmax(logits_entity, axis=-1).numpy().tolist()
        )
        for i in range(len(masks)):
            bio_entity_label = _merge_bio_entity_label(
                masks[i], pred_bio, pred_entity[i], bio_entity_label
            )
    return (bio_entity_label, intent_label)


def _merge_bio_entity_label(
    mask: List, bio_label: List, entity_label: str, bio_entity_label: List
) -> List:
    for i in range(len(mask)):
        if mask[i]:
            if bio_label[i] == "X":
                bio_entity_label[i] = "X"
            else:
                bio_entity_label[i] = bio_label[i] + entity_label
    return bio_entity_label


def _parse_bio_entity_sequence(pred: List, truth: List) -> Tuple:
    res_pred, res_truth = [], []
    for p, t in zip(pred, truth):
        if t != "X":
            res_truth.append(t)
            if p == "X":
                res_pred.append("O")
            else:
                res_pred.append(p)
    return (res_pred, res_truth)


def _process_prediction_result(dict_result: Dict) -> Dict:
    slot_pred = []
    slot_truth = []
    intent_summary = []
    sentence_summary = []
    for v in dict_result.values():
        length = v["slot"]["length"]
        tmp_slot_pred, tmp_slot_truth = _parse_bio_entity_sequence(
            v["slot"]["pred"][1 : 1 + length], v["slot"]["truth"][1 : 1 + length]
        )
        slot_pred.extend(tmp_slot_pred)
        slot_truth.extend(tmp_slot_truth)
        intent_summary.append(int(v["intent"]["pred"] == v["intent"]["truth"]))
        sentence_summary.append(
            int(intent_summary[-1] and tmp_slot_pred == tmp_slot_truth)
        )
    slot_prec, slot_reca, slot_f1 = evaluate(slot_truth, slot_pred, verbose=False)
    return {
        "slot (f1)": slot_f1 / 100,
        "slot (recall)": slot_reca / 100,
        "slot (precision)": slot_prec / 100,
        "intent (acc)": np.mean(intent_summary),
        "sentence (acc)": np.mean(sentence_summary),
    }


def evaluate_model(model, dataLoader, dtype: str = "valid") -> Dict:
    dict_result = {}
    for (
        ids,
        mask,
        indice,
        segment,
        dep_graph,
        bio_label,
        entity_label,
        intent_label,
    ) in tqdm(dataLoader.Data(dtype), ascii=True):
        ids = ids.numpy().tolist()
        truth_bio = dataLoader.idx2biolabel(bio_label.numpy().tolist())
        truth_entity = dataLoader.idx2slotlabel(entity_label.numpy().tolist())
        truth_intent = dataLoader.idx2intentlabel(intent_label.numpy().tolist())

        dep_graph = tf.sparse.to_dense(
            tf.sparse.reshape(
                dep_graph, (-1, dataLoader.MAX_SEQ_LEN, dataLoader.MAX_SEQ_LEN)
            )
        )
        for i, _id in enumerate(ids):
            if _id not in dict_result.keys():
                dict_result[_id] = {"slot": {}, "intent": {}}
                (
                    dict_result[_id]["slot"]["pred"],
                    dict_result[_id]["intent"]["pred"],
                ) = predict_one_sample(
                    model,
                    dataLoader,
                    indice.numpy().tolist()[i],
                    segment.numpy().tolist()[i],
                    dep_graph.numpy().tolist()[i],
                )
                dict_result[_id]["slot"]["truth"] = ["O"] * len(truth_bio[i])
                dict_result[_id]["slot"]["length"] = (
                    indice.numpy().tolist()[i].index(0) - 2
                )
                dict_result[_id]["intent"]["truth"] = truth_intent[i]
            dict_result[_id]["slot"]["truth"] = _merge_bio_entity_label(
                mask.numpy()[i],
                truth_bio[i],
                truth_entity[i],
                dict_result[_id]["slot"]["truth"],
            )
    return _process_prediction_result(dict_result)