from torch import nn


class MLPSpatialFilter(nn.Module):

    def __init__(self, num_sensor, num_hidden, activation):
        super(MLPSpatialFilter, self).__init__()
        self.fc11 = nn.Linear(num_sensor, num_sensor)
        self.fc12 = nn.Linear(num_sensor, num_sensor)
        self.fc21 = nn.Linear(num_sensor, num_hidden)
        self.fc22 = nn.Linear(num_hidden, num_hidden)
        self.fc23 = nn.Linear(num_sensor, num_hidden)
        self.value = nn.Linear(num_hidden, num_hidden)
        self.activation = nn.__dict__[activation]()

    def forward(self, x):
        out = dict()
        x = self.activation(self.fc12(self.activation(self.fc11(x))) + x)
        x = self.activation(self.fc22(self.activation(self.fc21(x))) + self.fc23(x))
        out['value'] = self.value(x)
        out['value_activation'] = self.activation(out['value'])
        return out


class TemporalFilter(nn.Module):

    def __init__(self, input_size, num_source, num_layer, activation):
        super(TemporalFilter, self).__init__()
        self.rnns = nn.ModuleList()
        self.rnns.append(nn.LSTM(input_size, num_source, batch_first=True, num_layers=num_layer))
        self.num_layer = num_layer
        self.input_size = input_size
        self.activation = nn.__dict__[activation]()

    def forward(self, x):
        out = dict()
        # c0/h0 : num_layer, T, num_out
        for l in self.rnns:
            l.flatten_parameters()
            x, _ = l(x)

        out['rnn'] = x  # seq_len, batch, num_directions * hidden_size
        return out


class TemporalInverseNet(nn.Module):

    def __init__(self, num_sensor=64, num_source=994, rnn_layer=1,
                 spatial_model=MLPSpatialFilter, temporal_model=TemporalFilter,
                 spatial_output='value_activation', temporal_output='rnn',
                 spatial_activation='ReLU', temporal_activation='ReLU', temporal_input_size=500):
        super(TemporalInverseNet, self).__init__()
        self.attribute_list = [num_sensor, num_source, rnn_layer,
                               spatial_model, temporal_model, spatial_output, temporal_output,
                               spatial_activation, temporal_activation, temporal_input_size]
        self.spatial_output = spatial_output
        self.temporal_output = temporal_output
        # Spatial filtering
        self.spatial = spatial_model(num_sensor, temporal_input_size, spatial_activation)
        # Temporal filtering
        self.temporal = temporal_model(temporal_input_size, num_source, rnn_layer, temporal_activation)

    def forward(self, x):
        out = dict()
        out['fc2'] = self.spatial(x)[self.spatial_output]
        x = out['fc2']
        out['last'] = self.temporal(x)[self.temporal_output]
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)