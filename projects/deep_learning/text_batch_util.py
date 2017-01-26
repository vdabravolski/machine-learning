import string
import numpy as np
from nltk import tokenize

text = "The call to basic_rnn_seq2seq returns two arguments: outputs and states. Both of them are lists " \
       "of tensors of the same length as decoder_inputs. Naturally, outputs correspond to the outputs of " \
       "the decoder in each time-step, in the first picture above that would be W, X, Y, Z, EOS. " \
       "The returned states represent the internal state of the decoder at every time-step."


class BatchUtil(object):
    '''
    BatchUtil is used to separate incoming text into batches with predefined number of words in each batch.
    '''
    def __init__(self, batch_size=3, num_batches=10):
        self.text = [word.lower() for word in text.split(" ")]
        self.text = [i.replace(".", "") for i in self.text]
        self.text = [i.replace(":", "") for i in self.text]
        self.text = [i.replace(",", "") for i in self.text]
        self.batch_size = batch_size
        self.cursor = 0
        self.num_batches = num_batches
        self.text_size = len(self.text)

    def _produce_batch(self):
        next_batch = []
        for _ in range(self.batch_size):
            next_batch.append(self.text[self.cursor])
            self.cursor = (self.cursor + 1) % self.text_size
        return next_batch

    def batch_array(self):
        batch_array = []
        reverse_batch_array = []
        for _ in range(self.num_batches):
            batch_el = self._produce_batch()
            batch_array.append(batch_el)
            reverse_batch_array.append([el[::-1] for el in batch_el])

        return batch_array, reverse_batch_array

    def vocabulary_size(self):
        return len(set(self.text))


class EncodeDecodeUtil(object):
    '''
    EncodeDecodeUtil is used to process each sentences and provide 1-hot encoding and decoding.
    '''
    def __init__(self, vocabulary_size=30):
        self.vocabulary_size = vocabulary_size  # include 26 letters, space, "*" as GO and "-" as padding symbols.
        self.buckets = [10, 20, 40, 60, 100]
        self.first_letter = ord(string.ascii_lowercase[0])

    def __char2id(self, char):

        if char in string.ascii_lowercase:
            return ord(char) - self.first_letter + 1
        elif char == ' ':
            return 0
        elif char == '*':  # start of the sentence
            return 27
        elif char == '-':  # end of the sentence
            return 28
        elif char == '.':  # padding
            return 29
        else:
            print('Unexpected character: %s' % char)
            return 0

    def __id2char(self, id):
        if (id > 0) and (id <27):
            return chr(id + self.first_letter - 1)
        elif id == 0:
            return ' '
        elif id == 27:
            return ''
        elif id == 28:
            return ''
        elif id == 29:
            return '.'
        else:
            raise Exception("Incorrect symbol %d" %id)

    def __sentence_size(self, sentence):
        __sentence = sentence
        for i in self.buckets:
            print(float(len(__sentence)))
            print(i + 2)
            if (float(len(__sentence) + 1) // i) < 1:
                #__lenght = i + 2 + (i + 2) % len(__sentence)
                return i # TODO: need to account here for paddings.

    def encode(self, plain_sentence):
        plain_sentence = plain_sentence.lower()
        print("plain sent",plain_sentence)
        encoded_length = self.__sentence_size(plain_sentence)
        encoded_sentence = np.zeros(shape=(encoded_length, self.vocabulary_size))

        encoded_sentence[0, self.__char2id('*')] = 1  # encoding start of the sentence

        __iterator = 1
        for char in plain_sentence:  # encoding of sentence body
            encoded_sentence[__iterator, self.__char2id(char)] = 1
            __iterator += 1
        encoded_sentence[(len(plain_sentence) + 1):, self.__char2id('-')] = 1  # adding padding to fit into bucket

        return np.fliplr(encoded_sentence)

    def decode(self, encoded_sentence):
        decoded_text = ''
        encoded_text = np.fliplr(encoded_sentence)
        for i in range(np.shape(encoded_text)[0]):
             id = np.argmax(encoded_text[i, :])
             decoded_text += self.__id2char(id)
        return decoded_text

    def encode_text(self, text):
        tokens = tokenize.sent_tokenize(text.lower())
        encoded_text = np.empty(shape = (len(tokens)))
        for sentence in tokens:
            np.append(encoded_text, self.encode(sentence), axis = 0)
        return encoded_text

    def decode_text(self, enc_text):
        text = ""

        for i in range(np.shape(enc_text)[0]):
            text += self.decode(enc_text[i,:])
        return text

enc = EncodeDecodeUtil()

sentence = " abcde dfgdsfg sdfg."
text = "These things are usually done by dedicated sentence splitter tools library modules. " \
       "Trying to do with regexes alone is not going to produce good results. " \
       "The better splitters have been trained. "

assert enc.decode(enc.encode(sentence)) == sentence
assert enc.decode_text(enc.encode_text(text)) == text

#dec_txt = enc.encode(txt)
#print("encoded", dec_txt)
#print("decoded", enc.decode(dec_txt))
#print("orig", txt)


#b = BatchUtil(3, 50)
#print(b.batch_array())
