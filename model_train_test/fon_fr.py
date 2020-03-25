# This has been inspired from [1] and [2]
# Tensorflow Tutorial NMT with attention : https://www.tensorflow.org/tutorials/text/nmt_with_attention
# James Brownlee - Deep Learning for NLP : Section 9 - Machine Translation
# Everything has been added for the specify task
# Written by Bonaventure DOSSOU
from __future__ import absolute_import, division, print_function, unicode_literals

import io
import os
import re
import string
import time
import unicodedata

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
# from nltk import bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import corpus_gleu
from sklearn.model_selection import train_test_split

tf.enable_eager_execution()


# Added : load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read().split("\n")
    # close the file
    file.close()
    return text


# Added : save list to file
def save_list(lines, filename):
    # convert lines to a single blob of text
    data = '\n'.join(lines)
    # open file
    file = open(filename, 'w', encoding="utf-8")
    # write text
    file.write(data)
    # close file
    file.close()


current_languages = ['Fon', 'Fr']
src_lang, dest_lang = map(str, input("Available languages are : " + ', '.join(i.strip().capitalize() for i in
                                                                              current_languages) + "\n\nEnter the source"
                                                                                                   " language and "
                                                                                                   "destination "
                                                                                                   "separated by a "
                                                                                                   "space : ").split())

path_to_file = "dataset_fon_fr.txt"


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


# Modified to handle Fon diacritics
def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    w = re_punc.sub('', w)

    lines_str = w.replace("”", "")
    lines_str = lines_str.replace("“", "")
    lines_str = lines_str.replace("’", "'")
    lines_str = lines_str.replace("«", "")
    lines_str = lines_str.replace("»", "")
    lines_str = ' '.join([word for word in lines_str.split() if word.isalpha()])
    w = '<start> ' + lines_str + ' <end>'
    return w


def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples] if
                  len(l.split("\t")) == 2]  # to make sure the element has two pairs :
    # Fon sentence and its French translation
    return zip(*word_pairs)


# en for Fongbe, sp for French
fon, fr = create_dataset(path_to_file, None)


def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    if src_lang.lower().strip() == "fon":
        inp_lang, targ_lang = create_dataset(path, num_examples)
        # save_list(inp_lang, "training_fon_sentences.txt")
        # save_list(targ_lang, "training_french_sentences.txt")
    else:
        targ_lang, inp_lang = create_dataset(path, num_examples)
        # not handled yet : This part will create the model for French - Fon translation
        # save_list(inp_lang, "training_fon_sentences.txt")
        # save_list(targ_lang, "training_french_sentences.txt")

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


num_examples = int(0.9 * len(fon))

print("Total Dataset Size : {} - Training Size : {} - Testing Size (with BLEU) : {}".format(len(fon), num_examples,
                                                                                            len(fon) - num_examples))
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)

# Creating training and validation sets using an 90-10 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                target_tensor,
                                                                                                test_size=0.1)

# parameters chosen after many trials :)
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 100
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
units = 128
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1
embedding_dim = 512

print("Fon vocabulary size : {} - French vocabulary : {}".format(vocab_inp_size, vocab_tar_size))

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


attention_layer = BahdanauAttention(30)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

optimizer = tf.keras.optimizers.Adam(0.001)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


checkpoint_dir = './training_checkpoints_1'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


# DEFINES BATCH TRAINING PROCESS
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)
        # dec_input = tf.expand_dims([1]*BATCH_SIZE, 1)
        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


# MODEL TRAINING

EPOCHS = 50
array_epochs, array_losses = [], []
for epoch in range(EPOCHS):

    array_epochs.append(epoch)

    start = time.time()
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every epoch
    checkpoint.save(file_prefix=checkpoint_prefix)

    print('Epoch_{} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / steps_per_epoch))
    array_losses.append(total_loss / steps_per_epoch)
    print('Time taken for epoch_{} : {} sec\n'.format(epoch + 1, time.time() - start))

np.save("all_epoch_fr_fon_{}".format(EPOCHS), np.array(array_epochs))
np.save("all_losses_fr_fon_{}".format(EPOCHS), np.array(array_losses))


def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word_index[i] for i in sentence.split(' ') if i in inp_lang.word_docs]
    inputs_not = [(i, sentence.index(i)) for i in sentence.split(' ') if i not in inp_lang.word_docs]
    # print(inputs, inputs_not)
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot, inputs_not

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)
    # print(inputs_not)
    return result, sentence, attention_plot, inputs_not


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence, index):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # uncomment to save and and show plot of the attention weights
    # plt.savefig("plot_attention_{}.png".format(index))
    # plt.show()


# made to re-insert words which are not in the dictionary after the translation has been done
def insert_word_at_index(phrase, array_words):
    phrases = phrase.split()
    for i in range(len(array_words)):
        phrases.insert(array_words[i][1], array_words[i][0])

    result = ' '.join(i.strip() for i in phrases)
    return result


def translate(sentence, index, target):
    saved_sentence = sentence
    result, sentence, attention_plot, left_word = evaluate(sentence)
    if "<end>" not in result:
        result = result + "<end>"

    index_end = result.index("<end>")
    result_end = result[:index_end].strip()

    print('Input: %s' % saved_sentence)
    input_ = 'Input: %s' % saved_sentence + "\n"

    processed_target = preprocess_sentence(target)
    processed_target = processed_target.replace("<start>", "")
    processed_target = processed_target.replace("<end>", "")

    print("Target : %s" % target)
    targ = "Target : %s" % target + "\n"

    print('Predicted translation: {}'.format(insert_word_at_index(result_end.capitalize(), left_word)))
    print("=" * 40)

    prediction = 'Predicted translation: {}'.format(insert_word_at_index(result_end.capitalize(), left_word)) + "\n"
    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '), index)

    all_together = input_ + targ + prediction
    return result, processed_target.strip().capitalize(), all_together


# restoring the latest checkpoint in checkpoint_dir to test on test dataset
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

all_bleu_scores = []
all_records_prediction = []

path_to_test = "fon_french_test_dataset.txt"

test_dataset = load_doc(path_to_test)

# split pairs and make sure they're unique
fon_test, fr_test = [], []
for sentence in test_dataset:
    pairs = sentence.split("\t")
    if len(pairs) == 2:  # making sure each element has fon text and its french translation
        if pairs[0] not in fon_test:
            fon_test.append(pairs[0])
            fr_test.append(pairs[1])

list_of_references = []
hypothesis = []

for text_index in range(len(fon_test)):
    pred, processed_target, record_translation = translate(fon_test[text_index], text_index, fr_test[text_index])
    if pred is not None:
        actual = processed_target.lower().split()
        predicted = pred.lower().split()[:len(pred.lower().split()) - 1]
        hypothesis.append(predicted)
        list_of_references.append([actual])

        # current_bleu_score_1_gram = bleu([actual], predicted, weights=(1.0, 0, 0, 0))
        # current_bleu_score_2_gram = bleu([actual], predicted, weights=(0.5, 0.5, 0, 0))
        # current_bleu_score_3_gram = bleu([actual], predicted, weights=(0.3, 0.3, 0.3, 0))
        # current_bleu_score_4_gram = bleu([actual], predicted,
        #                              weights=(0.25, 0.25, 0.25, 0.25))  # this is the bleu score function by default
        # current_bleu_score = max(current_bleu_score_1_gram, current_bleu_score_2_gram, current_bleu_score_3_gram,
        #                          current_bleu_score_4_gram)
        # all_bleu_scores.append(current_bleu_score)
        # print("Current bleu score on sentence {} : {}".format(text_index + 1, current_bleu_score))
        end_line = "=" * 40
        record_translation = record_translation + end_line
        all_records_prediction.append(record_translation)
    # print("=" * 40)
    print("Done : ", text_index + 1)

bleu_score_4 = corpus_bleu(list_of_references, hypothesis)
bleu_score_1 = corpus_bleu(list_of_references, hypothesis, weights=(1.0, 0, 0, 0))
bleu_score_2 = corpus_bleu(list_of_references, hypothesis, weights=(0.5, 0.5, 0, 0))
bleu_score_3 = corpus_bleu(list_of_references, hypothesis, weights=(0.3, 0.3, 0.3, 0))

gleu_score = corpus_gleu(list_of_references, hypothesis)

bleu_score_final = "Overall BLEU Score on FFR v1.0 Test Dataset : {}".format(
    round(max(bleu_score_1, bleu_score_2, bleu_score_3, bleu_score_4) * 100, 2))

gleu_score_ = "Overall GLEU Score on FFR v1.0 Test Dataset : {}".format(round(gleu_score * 100), 2)

testing_scores = list()
testing_scores.append(bleu_score_final)
testing_scores.append(gleu_score_)
# np_all_results = np.array(all_bleu_scores)
# np_all_predictions = np.array(all_records_prediction)
# np.save("all_bleu_results_fr", np_all_results)
# np.save("all_records_prediction", np_all_predictions)
# save_list(all_records_prediction, "all_records_prediction_bleu_scores.txt")
save_list(testing_scores, "testing_bleu_gleu_scores.txt")
