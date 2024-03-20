class Model(FModule):
    def __init__(self, input_size, hidden_size, output_size, n_layer=1, bidirectional=True):
        super(Module, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_direction = 2 if bidirectional else 1
        self.n_layer = n_layer
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=n_layer, bidirectional=bidirectional)
        # gru Input Seq, batch, input Output Seq, batch, hidden*nDirection
        self.fc = nn.Linear(self.hidden_size * self.n_direction, output_size)

    def _init_hidden(self, batch_size):
        # 注意，每次调用该函数batch_size会改变
        # gru Hidden Input nLayer*nDirection,BatchSize, HiddenSize
        # Hidden Output nLayer*nDirection, BatchSize, HiddenSize
        hidden = torch.zeros(self.n_layer * self.n_direction, batch_size, self.hidden_size)
        return hidden

    def forward(self, seq, seq_len):
        # BatchSize * SeqLen -> SeqLen * BatchSize
        seq = seq.t()
        batch_size = seq_len.shape[0]
        hidden = self._init_hidden(batch_size)
        embedding = self.embedding(seq)
        gru_input = pack_padded_sequence(embedding, seq_len)
        output, hidden = self.gru(gru_input, hidden)
        # 值得注意的是, 我们使用最后一层的hidden作为输出
        # n_layer*n_direction, BatchSize, hidden -> Batch, hiddenSize*n_direction
        # print('GRU的hidden的shape', hidden.shape)
        if self.n_direction == 2:
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
            # 实际上这里有个问题，对于多层n_layer，如何拿到最后的Hidden_forward和Hidden_back
        else:
            hidden = hidden[-1]
        fc_output = self.fc(hidden)
        return fc_output

