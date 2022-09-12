import tensorflow as tf
from keras import initializers
from attention import AttentionWeightedAverage
from graph_attention_layer import GraphAttention
from keras.layers import Dense, Lambda, RepeatVector
from keras_bert import load_trained_model_from_checkpoint
from embedding4label import SlotLabelEmbedding, IntentLabelEmbedding


class MT4SFID(tf.keras.models.Model):
    def __init__(
        self,
        dataset: str,
        bert_path: str,
        max_seq_len: int,
        n_bio_label: int,
        n_entity_label: int,
        n_intent_label: int,
        order: int = 1,
        graph_attn_heads: int = 2,
        graph_output_dim: int = 512,
        dropout_rate: float = 0.5,
    ):
        super(MT4SFID, self).__init__()

        self.order = order
        self.repeat_sent_layer = RepeatVector(max_seq_len)
        self.repeat_slot_layer = RepeatVector(n_entity_label)
        self.repeat_intent_layer = RepeatVector(n_intent_label)
        (
            self.slotlabel_embedding_fixed,
            self.intentlabel_embedding_fixed,
        ) = self.__load_label_embedding(dataset, bert_path)
        self.slotlabel_embedding_trainable = self.add_weight(
            shape=self.slotlabel_embedding_fixed.get_shape(),
            initializer=initializers.he_uniform,
            name="trainable_slotlabel_embedding",
            trainable=True,
        )
        self.intentlabel_embedding_trainable = self.add_weight(
            shape=self.intentlabel_embedding_fixed.get_shape(),
            initializer=initializers.he_uniform,
            name="trainable_intentlabel_embedding",
            trainable=True,
        )

        self.slotLabelAttention = AttentionWeightedAverage(
            return_attention=True, name="slot_label_attention"
        )
        self.intentLabelAttention = AttentionWeightedAverage(
            return_attention=True, name="intent_label_attention"
        )
        self.graphAttention = GraphAttention(
            graph_output_dim,
            attn_heads=graph_attn_heads,
            dropout_rate=dropout_rate,
        )

        self.cls_fc = Lambda(lambda x: x[:, 0])
        self.slot_gated_fc = Dense(1024, activation="sigmoid")
        self.bio_fc = Dense(n_bio_label, activation="softmax")
        self.entity_fc = Dense(n_entity_label, activation="softmax")
        self.intent_fc = Dense(n_intent_label, activation="softmax")
        self.__ckpt_path = f"{bert_path}/bert_model.ckpt"
        self.__config_path = f"{bert_path}/bert_config.json"
        self.bert_model = load_trained_model_from_checkpoint(
            self.__config_path, self.__ckpt_path, seq_len=None
        )
        for l in self.bert_model.layers:
            l.trainable = True

    def __load_label_embedding(self, dataset: str, bert_path: str):
        slotLabelEmbedding = SlotLabelEmbedding(dataset, bert_path)
        intentLabelEmbedding = IntentLabelEmbedding(dataset, bert_path)
        tf_slotlabel_weights = tf.constant(slotLabelEmbedding.Embedding)
        tf_intentlabel_weights = tf.constant(intentLabelEmbedding.Embedding)
        return (tf_slotlabel_weights, tf_intentlabel_weights)

    @tf.function
    def call(self, mask, ind, seg, dep_graph, training=False):
        slotlabel_embedding = tf.concat(
            [self.slotlabel_embedding_fixed, self.slotlabel_embedding_trainable],
            axis=-1,
        )
        intentlabel_embedding = tf.concat(
            [self.intentlabel_embedding_fixed, self.intentlabel_embedding_trainable],
            axis=-1,
        )
        sent_feature = self.bert_model([ind, seg])
        repeat_batch_layer = RepeatVector(sent_feature.get_shape().as_list()[0])

        pos_feature = sent_feature
        for _ in tf.range(self.order):
            pos_feature = self.graphAttention([pos_feature, dep_graph], training)

        # feature extractors
        span_feature = tf.reduce_sum(
            tf.expand_dims(tf.cast(mask, tf.float32), -1) * sent_feature, axis=1
        )
        cls_feature = self.cls_fc(sent_feature)

        # intent label attention
        intent_attention_feature, intent_attention_value = self.intentLabelAttention(
            tf.concat(
                [
                    self.repeat_intent_layer(cls_feature),
                    tf.transpose(repeat_batch_layer(intentlabel_embedding), [1, 0, 2]),
                ],
                axis=-1,
            )
        )

        # slot gated
        slot_gated = self.slot_gated_fc(
            tf.concat(
                [pos_feature, self.repeat_sent_layer(intent_attention_feature)],
                axis=-1,
            )
        )
        slot_gated_sent_feature = tf.concat(
            [slot_gated * sent_feature, pos_feature], axis=-1
        )

        # slot label attention
        slot_attention_feature, slot_attention_value = self.slotLabelAttention(
            tf.concat(
                [
                    self.repeat_slot_layer(span_feature),
                    tf.transpose(repeat_batch_layer(slotlabel_embedding), [1, 0, 2]),
                ],
                axis=-1,
            )
        )

        # output prediction logits
        logits_bio = self.bio_fc(slot_gated_sent_feature)
        logits_entity = self.entity_fc(slot_attention_feature)
        logits_intent = self.intent_fc(intent_attention_feature)

        return (logits_bio, logits_entity, logits_intent)
