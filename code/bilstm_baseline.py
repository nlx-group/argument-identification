import tensorflow as tf
import numpy as np
import logging
import tools.tf_data
import statistics as st
import itertools as it
from collections import namedtuple

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s.%(msecs)06d:%(levelname)s:%(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")


def main():
    epochs = 10
    batch_size = 64
    sequence_length = 30
    emb_trainable = False
    learning_rate = 0.01
    dropout = 0.8
    lstm_units = 48
    fully_connected_activation_fn = None
    fully_connected_trainable = True
    n_outputs = 2
    runs = 10

    dataset = "ukp_sentential_pt"
    embeddings = "random"
    topics = ["all"]
    data = tools.tf_data.Data()

    logging.info(f"loading embeddings")
    data.load_embeddings(embeddings)

    logging.info(f"loading dataset")
    data.load_dataset(dataset, batch_size, sequence_length)

    with tf.name_scope("init"):
        is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

    with tf.name_scope("input"):
        input_data = tf.placeholder(tf.int32, shape=[None, None])
        output_data = tf.placeholder(tf.int32, shape=[None])

        tf_emb_matrix = tf.get_variable(name="tf_emb_matrix",
                                        shape=[data.emb_vocab_size, data.emb_dim],
                                        initializer=tf.constant_initializer(np.array(data.emb_matrix)),
                                        trainable=emb_trainable)

        input_layer = tf.nn.embedding_lookup(tf_emb_matrix, input_data)

    with tf.name_scope("bilstm"):
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units)
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units)
        (output_fw, output_bw), last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                             cell_bw=lstm_bw_cell,
                                                                             inputs=input_layer,
                                                                             dtype=tf.float32)

    with tf.name_scope("loss"):
        keep_prob = tf.cond(is_training,
                            true_fn=lambda: tf.constant(dropout),
                            false_fn=lambda: tf.constant(1.0))

        output = tf.concat([output_fw, output_bw], axis=2)
        output = tf.nn.dropout(output, rate=1 - keep_prob)
        logits = tf.contrib.layers.fully_connected(inputs=output[:, - 1],
                                                   num_outputs=n_outputs,
                                                   activation_fn=fully_connected_activation_fn,
                                                   trainable=fully_connected_trainable)

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=output_data,
                                                                  logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        predictions = tf.argmax(logits, 1)
        confusion = tf.confusion_matrix(labels=output_data,
                                        predictions=predictions,
                                        num_classes=2)

    Metrics = namedtuple('metrics', ['topic', 'run', 'epoch', 'accuracy',
                                     'precision', 'recall', 'fmeasure',
                                     'partition'])
    scores = []

    for topic in topics:
        for run in range(runs):
            logging.info(f"training run {run}")
            with tf.Session() as sess:
                # reset of weights
                sess.run(tf.global_variables_initializer())
                data.create_data_partitions_in_topic(topic)

                for epoch in range(epochs):
                    for batch in range(data.total_batches):
                        input_train_batch, output_train_batch = data.next_batch()
                        sess.run(training, feed_dict={is_training: True,
                                                      input_data: input_train_batch,
                                                      output_data: output_train_batch})

                    partition_data = {}
                    partition_data["train"] = (data.input_train, data.output_train)
                    partition_data["val"] = (data.input_validation, data.output_validation)
                    partition_data["test"] = (data.input_test, data.output_test)

                    for partition in ["train", "val", "test"]:
                        conf_matrix = confusion.eval(feed_dict={is_training: False,
                                                                input_data: partition_data[partition][0],
                                                                output_data: partition_data[partition][1]})
                        acc_metric, pre_metric, rec_metric, fm_metric = data.metrics(conf_matrix)
                        metric = Metrics(topic=topic,
                                         run=run,
                                         epoch=epoch,
                                         accuracy=acc_metric,
                                         precision=pre_metric,
                                         recall=rec_metric,
                                         fmeasure=fm_metric,
                                         partition=partition)
                        scores.append(metric)
                    logging.info(f"epoch {epoch} train accuracy {scores[-3].accuracy:.4f} (validation {scores[-2].accuracy:.4f}a, {scores[-2].precision:.4f}p, {scores[-2].recall:.4f}r, {scores[-2].fmeasure:.4f}fm)")

        # unintentionally unclear sorry
        scores_val = [(s.fmeasure, s.epoch, s.run) for s in scores if s.topic == topic and s.partition == "val"]
        best_val_epoch_run = [max(x[1]) for x in it.groupby(scores_val, key=lambda k: k[2])]
        avg_test = []
        for _, e, r in best_val_epoch_run:
            avg_test.extend(*[(s.accuracy, s.precision, s.recall, s.fmeasure) for s in scores if s.topic == topic and s.partition == "test" and s.epoch == e and s.run == r])
        logging.info(f"averages {st.mean(avg_test[0::4]):.4f}a\t{st.mean(avg_test[1::4]):.4f}p\t{st.mean(avg_test[2::4]):.4f}r\t{st.mean(avg_test[3::4]):.4f}fm")


if __name__ == '__main__':
    main()
