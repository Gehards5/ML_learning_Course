import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn
import torch.nn.init
import torch.nn.functional as F
import torch.utils
import torch.utils.data

TIMESTEPS_INPUT = 16
TIMESTEPS_EXT = 8
TIMESTEPS_PREDICT = 16

DATASET_WINDOW_OVERLAP = 1
DATASET_LEN = TIMESTEPS_INPUT * int(1e2)
DATASET_X_DELTA = 1e-1
DATASET_NOISE_PERCENT = 0

HIDDEN_SIZE = 32
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
DEVICE = 'cpu'
NUM_EPOCHS = 300
RNN_LAYERS = 1

y_data = []
for x in range(DATASET_LEN):
    y = np.sin(x * DATASET_X_DELTA)
    y_data.append(y)

y_data = np.array(y_data)
#plt.plot(np.arange(0, TIMESTEPS_INPUT), y_data[:TIMESTEPS_INPUT])
#plt.show()


class DatasetTimeseries(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        self.data_x = []
        self.data_y = []
        for idx in range(0, len(y_data) - 1 - TIMESTEPS_INPUT - TIMESTEPS_PREDICT, DATASET_WINDOW_OVERLAP):
            x_slice = torch.FloatTensor(y_data[idx:idx + TIMESTEPS_INPUT])
            x_slice = x_slice.unsqueeze(dim=1)

            x_noise = torch.zeros_like(x_slice).uniform_()
            x_slice = (1.0 - DATASET_NOISE_PERCENT) * x_slice + DATASET_NOISE_PERCENT * x_noise
            self.data_x.append(x_slice)

            y_slice = torch.FloatTensor(y_data[idx + 1:idx + 1 + TIMESTEPS_INPUT + TIMESTEPS_PREDICT])
            y_slice = y_slice.unsqueeze(dim=1)
            self.data_y.append(y_slice)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return len(self.data_x)


dataset = DatasetTimeseries()
print(f'datapoints: {len(dataset)}')
data_loader = torch.utils.data.DataLoader(
    dataset,
    BATCH_SIZE,
    shuffle=True,
    drop_last=True
)


# TODO 1. use Linear layers instead of W, b
# TODO 2. implement manually LSTM cell

class VanillaRNN(torch.nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, batch_first):
        super().__init__()

        self.input_size = input_size
        self.num_layers = num_layers # ignore LAYERS
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Glorot initialization.
        self.W_ih = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(hidden_size, input_size)))
        self.b_ih = torch.nn.Parameter(torch.zeros(hidden_size))

        #torch.nn.init.kaiming_uniform
        self.W_hh = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(hidden_size, hidden_size)))
        self.b_hh = torch.nn.Parameter(torch.zeros(hidden_size))


    def forward(self, input, hidden_and_cell): # (BATCH, SEQ, input_size)

        hidden = hidden_and_cell[0] # (RNN_LAYERS, batch_size, HIDDEN_SIZE)
        hidden_t = hidden[0] # (batch_size, HIDDEN_SIZE)

        output_seq = []
        input_seq = torch.transpose(input, 0, 1) # (SEQ, BATCH, input_size)
        for idx_seq in range(input_seq.size(0)): # iterate seq timesteps
            x_t = input_seq[idx_seq] # (BATCH, input_size)

            hidden_t = F.tanh(
                torch.matmul(x_t, self.W_ih.t()) + self.b_ih +
                torch.matmul(hidden_t, self.W_hh.t()) + self.b_hh
            )
            # hidden_t => (BATCH, hidden_size) => unsqueeze => (1, BATCH, hidden_size)
            output_seq.append(hidden_t.unsqueeze(dim=0))

        output = torch.cat(output_seq, dim=0) # => (SEQ, BATCH, hidden_size)
        output = torch.transpose(output, 0, 1)

        return output, None # (BATCH, SEQ, hidden_size)

# weight (32, 8) => (output, input)
# input (batch, input)

# output = weight * input
# out = torch.matmul(input, torch.transpose(weight, 0, 1))
# output = torch.transpose(torch.matmul(weight, torch.transpose(input, 0, 1)), 0, 1)

# * pairwise
# torch.matmul() matrix

# rnn = VanillaRNN(input_size=3, num_layers=1, hidden_size=6, batch_first=True)
#
# input = torch.nn.init.uniform_(torch.zeros((32, 16, 3))) # (BATCH, SEQ, input_size)
#
# hidden = torch.zeros((RNN_LAYERS, 32, 6))
# cell = torch.zeros((RNN_LAYERS, 32, 6))
#
# output = rnn.forward(input, (hidden, cell)) # (BATCH, SEQ, hidden_size)

class ModelPytorch(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = VanillaRNN(input_size=1, num_layers=RNN_LAYERS, hidden_size=HIDDEN_SIZE, batch_first=True)

        self.decoder = torch.nn.Sequential(
            #torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=1)
        )

        self.hidden = None
        self.reset_hidden(BATCH_SIZE)

    def reset_hidden(self, batch_size):
        self.hidden = torch.zeros((RNN_LAYERS, batch_size, HIDDEN_SIZE))
        self.cell = torch.zeros((RNN_LAYERS, batch_size, HIDDEN_SIZE))

    def forward(self, input):

        # many to many
        out, _ = self.rnn.forward(input, (self.hidden, self.cell))
        y = self.decoder.forward(out)

        if self.training:
            for _ in range(TIMESTEPS_PREDICT):
                out, _ = self.rnn.forward(y[:, -TIMESTEPS_EXT:], (self.hidden, self.cell))
                y_last = self.decoder.forward(out)
                y = torch.cat([y, y_last[:,-1:]], dim=1) #(batch, timesteps, features)

        return y


model = ModelPytorch().to(DEVICE)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

for epoch in range(1, NUM_EPOCHS+1):

    losses = []
    for batch_x, batch_y in data_loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        # np_x = batch_x.data.numpy()
        # np_y = batch_y.data.numpy()
        # np_y = np_y[:, TIMESTEPS_INPUT:]
        # for idx in range(batch_x.size(0)):
        #     plt.plot(np.arange(0, TIMESTEPS_INPUT), np_x[idx])
        #     plt.plot(np.arange(TIMESTEPS_INPUT, TIMESTEPS_INPUT + len(np_y[idx])), np_y[idx])
        #     plt.show()

        model.reset_hidden(batch_x.size(0))
        batch_y_prim = model.forward(batch_x)
        loss = torch.mean((batch_y_prim - batch_y)**2)
        #loss = torch.sum(torch.abs(batch_y_prim - batch_y))
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f'epoch: {epoch} loss: {np.average(losses)}')

    # Simulation

    offset = 0
    sim_x = y_data[offset:TIMESTEPS_INPUT + offset]
    sim_out = sim_x.tolist()
    sim_x = torch.FloatTensor(sim_x)
    sim_x = sim_x.unsqueeze(dim=1)

    model = model.eval()

    # inital memory
    model.reset_hidden(batch_size=1)

    sim_x = sim_x.unsqueeze(dim=0)
    sim_y = model.forward(sim_x) # Input is TIMESTEPS_INPUT

    for idx_rollout in range(TIMESTEPS_PREDICT*5):

        sim_y_prim = model.forward(sim_y[:,-TIMESTEPS_EXT:])
        sim_y = torch.cat([sim_y, sim_y_prim[:,-1:]], dim=1)

        sim_y_last_timestep = sim_y[:,-1] # last timestep
        sim_out.append(sim_y_last_timestep[0].item())

    plt.title(f'epoch: {epoch}')
    plt.plot(np.arange(0, TIMESTEPS_INPUT), sim_out[:TIMESTEPS_INPUT])
    plt.plot(np.arange(TIMESTEPS_INPUT, len(sim_out)), sim_out[TIMESTEPS_INPUT:])
    plt.show()

    model = model.train()

