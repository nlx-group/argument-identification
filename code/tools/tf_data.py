import numpy as np
import math
from collections import namedtuple
from nltk.corpus import stopwords


class Data:

    def __init__(self):
        self.data_path = "./data/"
        self.dataset = None

        self.emb_word_vocab = None
        self.emb_matrix = None
        self.emb_vocab_to_int = None
        self.emb_dim = None
        self.emb_vocab_size = None

        self.input_embedding = []
        self.batch_current = 0
        self.input_sequence_length = 60

        self.dt_instances = []
        self.stopWords = set(stopwords.words('portuguese'))

    def load_embeddings(self, embedding_path, dim=None, vocab=None, random_initialized=False):

        if embedding_path == "fasttextpt":
            embedding_path = self.data_path + "fasttextpt.vec"
            dim = 300

        if embedding_path == "random":
            embedding_path = self.data_path + "fasttextpt.vec"
            dim = 300
            random_initialized = True

        vocab = []
        with open(f"{self.data_path}restrict_vocab_ukp_sentential_pt.txt", "r") as fh:
            for line in fh:
                line = line.rstrip()
                vocab.append(line)

        with open(embedding_path, 'r') as file_obj:
            self.emb_word_vocab = []
            self.emb_matrix = []
            self.emb_word_vocab.extend(['PAD', 'UNK'])
            self.emb_matrix.append(np.random.uniform(-1.0, 1.0, (1, dim))[0])
            self.emb_matrix.append(np.random.uniform(-1.0, 1.0, (1, dim))[0])

            for line in file_obj:
                if vocab is None or line.split()[0] in vocab:
                    self.emb_word_vocab.append(line.split()[0])

                    if random_initialized:
                        vector = np.random.uniform(-1.0, 1.0, (1, dim))[0]
                    else:
                        vector = np.array([float(i) for i in line.split()[1:]])

                    # googlenews semantic space have some strange vectors
                    if len(vector) != dim:
                        vector = np.random.uniform(-1.0, 1.0, (1, dim))[0]

                    self.emb_matrix.append(vector)

        self.emb_vocab_to_int = {word: index for index, word in enumerate(self.emb_word_vocab)}
        self.emb_dim = dim
        self.emb_vocab_size = len(self.emb_word_vocab)

    def load_dataset(self, dataset_name, batch_size=1, sequence_length=None):
        self.dataset = dataset_name
        self.input_sequence_length = sequence_length
        self.dt_instances = []
        self.batch_size = batch_size

        Dt_instance = namedtuple('dt_instance', ['topic', 'sentence', 'annotation', 'partition', 'embedding', 'output'])

        if dataset_name == "ukp_sentential_pt":
            data_sets = ['train', 'test', 'dev']
            for dt in data_sets:
                with open(f"{self.data_path}{dt}_pt.tsv", "r") as fh:
                    for line in fh:

                        contents = line.rstrip().split("\t")
                        sentence = contents[1].lower()
                        annotation = contents[0]

                        if dt in ['train', 'test', 'dev']:
                            partition = dt
                        if partition == "dev":
                            partition = "val"

                        embedding = self.sentence_embedding(sentence)
                        instance = Dt_instance(topic="all",
                                               sentence=sentence,
                                               annotation=annotation,
                                               partition=partition,
                                               embedding=embedding,
                                               output=annotation)
                        self.dt_instances.append(instance)
        self.dt_num_instances = len(self.dt_instances)

    def sentence_embedding(self, sentence):

        sentence = sentence.replace('.', '')
        sentence = sentence.replace(',', '')
        sentence = sentence.replace('!', ' ! ')
        sentence = sentence.replace('?', ' ? ')
        sentence = sentence.replace('\'s', '')
        sentence = sentence.replace('\'re', '')
        sentence = sentence.replace('\'t', '')
        sentence = sentence.replace('"', '')
        sentence = sentence.split(" ")
        sentence = [s for s in sentence if s]

        if self.emb_matrix is None:
            return sentence

        sentence = [self.emb_vocab_to_int["UNK"] if word not in self.emb_vocab_to_int else self.emb_vocab_to_int[word] for word in sentence]

        if len(sentence) >= self.input_sequence_length:
            return sentence[:self.input_sequence_length]
        else:
            sentence.extend([self.emb_vocab_to_int["PAD"]] * (self.input_sequence_length - len(sentence)))
            return sentence

    def create_data_partitions_in_topic(self, topic):
        self.input_train = []
        self.input_test = []
        self.input_validation = []
        self.output_train = []
        self.output_test = []
        self.output_validation = []

        if topic == "all":
            data_train_batch = [instance for instance in self.dt_instances]
        else:
            data_train_batch = [instance for instance in self.dt_instances if instance.topic == topic]

        self.input_train = [instance.embedding for instance in data_train_batch if instance.partition == "train"]
        self.output_train = [instance.output for instance in data_train_batch if instance.partition == "train"]

        self.input_test = [instance.embedding for instance in data_train_batch if instance.partition == "test"]
        self.output_test = [instance.output for instance in data_train_batch if instance.partition == "test"]

        self.input_validation = [instance.embedding for instance in data_train_batch if instance.partition == "val"]
        self.output_validation = [instance.output for instance in data_train_batch if instance.partition == "val"]

        self.total_batches = math.ceil(len(self.input_train) / self.batch_size)

    def next_batch(self):
        input_data = self.input_train[self.batch_current:self.batch_current + self.batch_size]
        output_data = self.output_train[self.batch_current:self.batch_current + self.batch_size]

        self.batch_current += self.batch_size
        if self.batch_current >= len(self.input_train):
            self.batch_current = 0

        return (input_data, output_data)

    def metrics(self, confusion_matrix):
        TP = confusion_matrix[0][0]
        TN = confusion_matrix[1][1]
        FP = confusion_matrix[1][0]
        FN = confusion_matrix[0][1]

        accuracy = (TP + TN) / (TP + FP + FN + TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        fmeasure = (2 * precision * recall) / (precision + recall)

        return (accuracy, precision, recall, fmeasure)
