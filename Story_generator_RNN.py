import json
import re

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from torch import nn, LongTensor
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

# !pip install pytorch-nlp
import torch.word_to_vector

glove = GloVe('6B')

vec_king = glove['king']
vec_woman = glove['woman']
vec_man = glove['man']
vec_output = vec_king - vec_man + vec_woman

all_tokens = list(glove.stoi.keys())
distances_to_tokens = []
for token in all_tokens:
    dist = (glove[token] - vec_output) ** 2
    distances_to_tokens.append((dist, token))

# sort by closest and print top 10

#TODO forget gate
#TODO word2vec
#TODO attention

RNN_LAYERS = 1
HIDDEN_SIZE = 512
EMBEDDING_SIZE = 32
BATCH_SIZE = 128
LEARNING_RATE = 1e-2
NUM_EPOCHS = 50

MAX_SENTENCES = 10000
MAX_SENTENCE_LEN = 10

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using: {DEVICE}')


word_occurances = {}
sum_occurances = 0

def load_quotes():
    global sum_occurances
    # try:
    #     return pickle.load(open('./quotes_db.pkl', 'rb'))
    # except FileNotFoundError:
    #     print('pkl file does not exist yet')

    with open('./quotes_db.json', 'r') as fp:
        quotes_db = json.load(fp)
    word_to_idx = {}
    idx_to_word = []
    sentences = []
    sentences_max_len = 0
    for each in quotes_db:
        quote_str = each['Quote'].lower().strip()
        sentence_words = list(re.findall(r"([a-z]+\'?[a-z]*)", quote_str))

        if len(sentence_words) > MAX_SENTENCE_LEN:
            continue

        sentence = []
        for word in sentence_words:
            if word.endswith("'"):
                word = word[:-1]

            try:
                word_idx = word_to_idx[word]
            except:
                word_idx = len(idx_to_word)
                word_to_idx[word] = word_idx
                idx_to_word.append(word)

            if word not in word_occurances.keys():
                word_occurances[word] = 0
            word_occurances[word] += 1
            sum_occurances += 1

            sentence.append(word_idx)
        if sentence not in sentences and len(sentence) > 1:
            sentences.append(sentence)
        sentences_max_len = max(sentences_max_len, len(sentence))

        if len(sentences) > MAX_SENTENCES:
            break

    print(f'sentences: {len(sentences)} sentences_max_len: {sentences_max_len} word_dict: {len(idx_to_word)}')
    quotes_ = {
        'idx_to_word': idx_to_word,
        'sentences': sentences,
        'sentences_max_len': sentences_max_len,
    }
   # pickle.dump(quotes_, open('quotes_db.pkl', 'wb'))
    print('pickle saved')

    # ,
    csv_data = [
        ['word', 'occurrences']
    ]
    csv_data += list(zip(word_occurances.keys(), [str(it) for it in word_occurances.values()]))
    with open('./quotes_db.csv', 'w') as fp:
        fp.writelines([','.join(it) + '\n' for it in csv_data])

    return quotes_


quotes = load_quotes()

# generate dummy test set
# sentences = []
# sentences_max_len = 0
# idx_to_word =  {}
# for i in range(1000):
#     sentence = []
#     j_start = random.randint(0, 17)
#     for j in range(j_start, 20):
#         sentence.append(j)
#     if random.random() < 0.5:
#         sentence.reverse()
#     sentences.append(sentence)
#     sentences_max_len = max(sentences_max_len, len(sentence))
#
# for j in range(20):
#     idx_to_word[j] = str(j)
#
# quotes = {
#     'idx_to_word': idx_to_word,
#     'sentences': sentences,
#     'sentences_max_len': sentences_max_len,
# }

sentence_count = len(quotes['sentences'])
sentences_max_len = quotes['sentences_max_len']
corpus_size = len(quotes['idx_to_word']) + 1  # +1 for end of sentence
sentence_end = corpus_size - 1 #bug

sum_occurances += sentence_count

print(f'sentence_count: {sentence_count} corpus_size: {corpus_size}')


class SentenceDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.sentences = [self.prepare(s) for s in quotes['sentences']]

    @staticmethod
    def prepare(sentence_):
        x = LongTensor(sentence_)

        y = torch.roll(x, shifts=-1, dims=0)
        y[-1] = sentence_end

        x_len = len(sentence_)
        return x, y, x_len

    def __getitem__(self, index):
        return self.sentences[index]

    def __len__(self):
        return sentence_count

def collate_fn(batch_input):
    x_input, y_input, x_len = zip(*batch_input)

    x_len_max = int(np.max(x_len))

    # sort batch so that max x_len first (desc.)
    input_idxes_sorted = list(reversed(np.argsort(x_len).tolist()))

    x = torch.zeros((len(x_len), x_len_max), dtype=torch.long)
    y = torch.zeros_like(x)
    x_len_out = torch.LongTensor(x_len)

    for i in range(len(x_len)):
        i_sorted = input_idxes_sorted[i]
        x_len_out[i] = x_len[i_sorted]
        x[i, 0:x_len_out[i]] = x_input[i_sorted]
        y[i, 0:x_len_out[i]] = y_input[i_sorted]

    return (x, y, x_len_out)


dataset = SentenceDataset()

size_dataset = len(dataset)
size_train = int(size_dataset * 0.8)

torch.manual_seed(7)
dataset_train, dataset_test = torch.utils.data.random_split(dataset, [size_train, size_dataset-size_train])

data_loader_train = DataLoader(
    dataset_train,
    BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    collate_fn=collate_fn
)

data_loader_test = DataLoader(
    dataset_test,
    BATCH_SIZE,
    shuffle=False,
    drop_last=True,
    collate_fn=collate_fn
)


weights_classes = []

for idx, word in enumerate(quotes['idx_to_word']):
    weight_class = 1.0 - word_occurances[word] / sum_occurances
    weights_classes.append(weight_class)

weight_class = 1.0 - sentence_count / sum_occurances
weights_classes.append(weight_class)

np_weights_classes = np.array(weights_classes)
np_weights_classes /= np.sum(np_weights_classes)

t_weights_classes = torch.FloatTensor(weights_classes)
t_weights_classes = t_weights_classes.to(DEVICE)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding_layer = nn.Embedding(num_embeddings=corpus_size, embedding_dim=EMBEDDING_SIZE)
        self.rnn = torch.nn.LSTM(input_size=EMBEDDING_SIZE, num_layers=RNN_LAYERS, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.fc_h = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc = nn.Linear(HIDDEN_SIZE, EMBEDDING_SIZE)
        self.softmax = nn.Softmax(dim=1)


    def reset_hidden(self, batch_size, device=DEVICE):
        self.hidden = torch.zeros((RNN_LAYERS, batch_size, HIDDEN_SIZE)).to(device)
        self.cell = torch.zeros((RNN_LAYERS, batch_size, HIDDEN_SIZE)).to(device)

    def forward(self, x, x_len):
        batch_size = x.size(0)

        # x => (BATCH, SEQ)
        x_enc = []
        for idx, x_each in enumerate(x):
            x_enc.append(x_each[0:x_len[idx]])
        x_enc = torch.cat(x_enc, dim=0)

        x_emb = self.embedding_layer(x_enc)

        x_dec = []
        x_idx = 0
        for idx in range(batch_size):
            x_dec.append(x_emb[x_idx:x_idx+x_len[idx]])
            x_idx += x_len[idx]

        x_pack = torch.nn.utils.rnn.pack_sequence(x_dec)

        out_packed, (self.hidden, self.cell) = self.rnn.forward(x_pack, (self.hidden, self.cell))

        out_padded, len_out = torch.nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=x_len[0])

        # x => (BATCH, SEQ, FEATURES)
        out_flat = []
        for idx in range(batch_size):
            out_flat.append(out_padded[idx, 0:x_len[idx]])
        out_flat = torch.cat(out_flat, dim=0)

        out_flat = self.fc_h.forward(out_flat)
        out_flat = self.fc.forward(out_flat)

        out_flat = torch.matmul(out_flat, self.embedding_layer.weight.t())

        # TODO hierarchical softmax
        out_flat = self.softmax.forward(out_flat)


        x_dec = []
        x_idx = 0
        for idx in range(batch_size):
            x_dec.append(out_flat[x_idx:x_idx+x_len[idx]])
            x_idx += x_len[idx]
        out_seq = torch.nn.utils.rnn.pack_sequence(x_dec)

        #TODO use following function to replace for loop
        #out_seq = torch.nn.utils.rnn.PackedSequence(out_flat, batch_sizes=x_len.to('cpu'))

        out_seq_padded, len_out = torch.nn.utils.rnn.pad_packed_sequence(out_seq, batch_first=True, total_length=x_len[0])

        return out_seq_padded


loss_tot_train = []
loss_tot_test = []

model = Model().to(DEVICE)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

for epoch in range(1, NUM_EPOCHS+1):

    losses = []
    for data_loader in [data_loader_train, data_loader_test]:

        if data_loader == data_loader_train:
            model = model.train()
            torch.set_grad_enabled(True)
        else:
            model = model.eval()
            torch.set_grad_enabled(False)

        for batch_x, batch_y, len_x in data_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            len_x = len_x.to(DEVICE)

            model.reset_hidden(batch_x.size(0))
            y_prim = model.forward(batch_x, len_x)
            y_prim = y_prim.contiguous().view((y_prim.size(0) * y_prim.size(1), corpus_size))

            y_target = batch_y.contiguous().view((batch_y.size(0) * batch_y.size(1), 1))
            tmp = torch.arange(corpus_size).reshape(1, corpus_size).to(DEVICE)
            y_target = (y_target == tmp).float() # one hot encoded 0.0 or 1.0
            y_target = y_target.to(DEVICE)

            #loss = torch.mean(-y_target*torch.log(y_prim + 1e-20))
            #loss = (y_prim - y_target) ** 2

            loss = -y_target*torch.log(y_prim + 1e-20)
            loss = torch.mean(loss, dim=0) * t_weights_classes
            loss = torch.mean(loss)

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            losses.append(loss.item())

            #debug
            #break

        loss_tot = np.average(losses)
        stage = 'train'
        if data_loader == data_loader_test:
            loss_tot_test.append(loss_tot)
            stage = 'test'
        else:
            loss_tot_train.append(loss_tot)
        print(f'epoch: {epoch} stage:{stage} loss:{loss_tot}')

        if data_loader == data_loader_test:
            #batch_x, batch_y, len_x = next(iter(data_loader))
            model = model.to('cpu')
            batch_x = batch_x.to('cpu')
            len_x = len_x.to('cpu')

            for idx_sample, seq_x in enumerate(batch_x):
                seq_len = len_x[idx_sample].item()

                model.reset_hidden(batch_size=1, device='cpu')

                seq_simulated = []

                # feed beginning of sentence
                seq_len_feed = int(seq_len / 2)
                seq_x_feed = seq_x[:seq_len_feed]
                y_prim = model.forward(seq_x_feed.unsqueeze(0), torch.LongTensor([seq_len_feed]))

                y_prim_sample = y_prim[0] # remove batch
                x_last = torch.argmax(y_prim_sample[-1]) # last timestep

                seq_simulated_feed = seq_x_feed.data.numpy().tolist()

                if x_last != sentence_end: # BUG
                    seq_simulated.append(x_last.item())

                    # rollout
                    for idx_rollout in range(quotes['sentences_max_len']):

                        y_prim = model.forward(torch.LongTensor([[x_last]]), torch.LongTensor([1]))
                        y_prim_sample = y_prim[0] # remove batch
                        x_last = torch.argmax(y_prim_sample[-1]) # last timestep
                        if x_last == sentence_end: # BUG
                            break
                        seq_simulated.append(x_last.item())

                # word indexes / token => words
                #print(json.dumps(seq_simulated))
                print(f'idx_sample: {idx_sample}: ' + ' '.join([quotes['idx_to_word'][it] for it in seq_simulated_feed]) +
                      ' >> ' + ' '.join([quotes['idx_to_word'][it] for it in seq_simulated]))
                break

            model = model.to(DEVICE)

            plt.title('train')
            plt.plot(loss_tot_train)

            plt.title('test')
            plt.plot(loss_tot_test)

            plt.show()






