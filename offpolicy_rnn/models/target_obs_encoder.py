import torch
from torch import nn
from .rnn_base import RNNBase
from .mlp_base import MLPBase
from .RNNHidden import RNNHidden
from .torch_utility import cumsum_fun_selective_scan

class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, output_dim, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, output_dim)
        )
        self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images)
        latent = self.output_activation(images_compressed)
        return latent


class TargetObsEncoder(torch.nn.Module):
    def __init__(self, target_image_dim, target_vector_dim, common_obs_dim,
                 action_dim, embedding_dim, rnn_type, middle_ebd_dim=32, output_activation='elu',
                 learn_std=False):
        super(TargetObsEncoder, self).__init__()
        self.target_image_dim = target_image_dim
        self.target_vector_dim = target_vector_dim
        self.common_obs_dim = common_obs_dim
        self.action_dim = action_dim
        self.learn_std = learn_std
        self.embedding_dim = embedding_dim
        self.original_image_dim_row, self.original_image_dim_col = 58, 87
        self.image_encoder = DepthOnlyFCBackbone58x87(middle_ebd_dim, num_frames=1)
        self.output_activation = output_activation
        if self.learn_std:
            assert output_activation in ['linear', 'tanh']
            output_activation = 'linear'
        self.target_observation_encoder = RNNBase(target_vector_dim + common_obs_dim + middle_ebd_dim + action_dim,
                                                  embedding_dim * 2 if self.learn_std else embedding_dim,
                                                  [512, 256, 256, 128],
                                                  ['elu', 'elu', 'elu', 'elu', output_activation],
                                                  ['fc', 'fc', rnn_type, 'fc', 'fc']
                                                  )
        self.rnn_num = self.target_observation_encoder.rnn_num
        self.encoder = self.target_observation_encoder
        pass

    def make_rnd_init_state(self, batch_size, device):
        return self.target_observation_encoder.make_rnd_init_state(batch_size, device)

    def make_init_state(self, batch_size, device):
        return self.target_observation_encoder.make_init_state(batch_size, device)

    def forward(self, x, last_action, hidden):
        assert x.shape[-1] == self.target_image_dim + self.target_vector_dim + self.common_obs_dim, f'expect: {self.target_image_dim + self.target_vector_dim + self.common_obs_dim}, got: {x.shape[-1]}'
        target_image = x[..., -self.target_image_dim:]
        target_vector = x[...,
                            -(self.target_image_dim + self.target_vector_dim):(-self.target_image_dim)]
        common_obs = x[..., :-(self.target_image_dim + self.target_vector_dim)]

        image_shape = list(target_image.shape)
        image_target_shape = image_shape[:-1] + [1, self.original_image_dim_row, self.original_image_dim_col]
        image = target_image.reshape(image_target_shape)
        image = image.reshape([-1] + [1, self.original_image_dim_row, self.original_image_dim_col])
        image_features = self.image_encoder.forward(image)
        image_features = image_features.reshape(image_shape[:-1] + [image_features.shape[-1]])

        mixup = torch.cat((image_features, target_vector, common_obs, last_action), dim=-1)
        embedding, hidden, _ = self.target_observation_encoder.meta_forward(
            mixup, hidden
        )
        if self.learn_std:
            mean, logstd = embedding.chunk(2, dim=-1)
            if self.output_activation == 'tanh':
                mean = torch.tanh(mean)
            # if self.output_activation == 'tanh':
            #     LOGSTD_MAX, LOGSTD_MIN = 1.0, -10.0
            #     logstd = (logstd + 1) / 2 * (LOGSTD_MAX - LOGSTD_MIN) + LOGSTD_MIN
        else:
            mean, logstd = embedding, None
        return mean, logstd, hidden

    def logp(self, x, last_action, ground_truth, hidden):
        assert self.learn_std
        mean, logstd, hidden = self.forward(x, last_action, hidden)
        gaussian_distribution = torch.distributions.Normal(mean, logstd.exp())
        return gaussian_distribution.log_prob(ground_truth), hidden


class TargetObsEncoderNoImage(torch.nn.Module):
    def __init__(self, target_image_dim, target_vector_dim, common_obs_dim,
                 action_dim, embedding_dim, rnn_type, middle_ebd_dim=32, output_activation='elu',
                 learn_std=False):
        super(TargetObsEncoderNoImage, self).__init__()
        self.target_image_dim = target_image_dim
        self.target_vector_dim = target_vector_dim
        self.common_obs_dim = common_obs_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.learn_std = learn_std
        self.embedding_dim = embedding_dim
        self.output_activation = output_activation
        self.image_encoder = MLPBase(target_image_dim, middle_ebd_dim, [256, 256], ['elu', 'elu', 'elu'])
        if self.learn_std:
            assert output_activation in ['linear', 'tanh']
            output_activation = 'linear'
        self.target_observation_encoder = RNNBase(target_vector_dim + common_obs_dim + middle_ebd_dim + action_dim,
                                                  embedding_dim * 2 if self.learn_std else embedding_dim,
                                                  [512, 256, 256, 128],
                                                  ['elu', 'elu', 'elu', 'elu', output_activation],
                                                  ['fc', 'fc', rnn_type, 'fc', 'fc']
                                                  )
        self.encoder = self.target_observation_encoder
        self.rnn_num = self.target_observation_encoder.rnn_num

    def forward(self, x, last_action, hidden):
        assert x.shape[-1] == self.target_image_dim + self.target_vector_dim + self.common_obs_dim, f'expect: {self.target_image_dim + self.target_vector_dim + self.common_obs_dim}, got: {x.shape[-1]}'
        target_image = x[..., -self.target_image_dim:]
        target_vector = x[..., -(self.target_image_dim + self.target_vector_dim):(-self.target_image_dim)]
        common_obs = x[..., :-(self.target_image_dim + self.target_vector_dim)]
        image_features = self.image_encoder.forward(target_image)
        mixup = torch.cat((image_features, target_vector, common_obs, last_action), dim=-1)
        embedding, hidden, _ = self.target_observation_encoder.meta_forward(mixup, hidden)
        if self.learn_std:
            mean, logstd = embedding.chunk(2, dim=-1)
            if self.output_activation == 'tanh':
                mean = torch.tanh(mean)

                # LOGSTD_MAX, LOGSTD_MIN = 1.0, -10.0
                # logstd = (logstd + 1) / 2 * (LOGSTD_MAX - LOGSTD_MIN) + LOGSTD_MIN
        else:
            mean, logstd = embedding, None
        return mean, logstd, hidden


    def make_rnd_init_state(self, batch_size, device):
        return self.target_observation_encoder.make_rnd_init_state(batch_size, device)

    def make_init_state(self, batch_size, device):
        return self.target_observation_encoder.make_init_state(batch_size, device)

    def logp(self, x, last_action, ground_truth, hidden):
        assert self.learn_std
        mean, logstd, hidden = self.forward(x, last_action, hidden)
        gaussian_distribution = torch.distributions.Normal(mean, logstd.exp())
        return gaussian_distribution.log_prob(ground_truth), hidden


class TargetObsEncoderExtreme(torch.nn.Module):
    def __init__(self, target_image_dim, target_vector_dim, common_obs_dim,
                 action_dim, embedding_dim, rnn_type, middle_ebd_dim=32, output_activation='elu',
                 learn_std=False):
        super(TargetObsEncoderExtreme, self).__init__()
        self.target_image_dim = target_image_dim
        self.target_vector_dim = target_vector_dim
        self.common_obs_dim = common_obs_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.learn_std = learn_std
        self.embedding_dim = embedding_dim
        self.output_activation = output_activation
        # self.image_encoder = MLPBase(target_image_dim, middle_ebd_dim, [256, 256], ['elu', 'elu', 'elu'])
        self.target_observation_encoder = RNNBase(target_vector_dim + common_obs_dim  + action_dim,
                                                  embedding_dim * 2 if self.learn_std else embedding_dim,
                                                  [512, 256, 256, 128],
                                                  ['elu', 'elu', 'elu', 'elu', output_activation],
                                                  ['fc', 'fc', rnn_type, 'fc', 'fc']
                                                  )
        self.encoder = self.target_observation_encoder
        if self.learn_std:
            assert output_activation in ['linear', 'tanh']

        self.rnn_num = self.target_observation_encoder.rnn_num

    def forward(self, x, last_action, hidden):
        assert x.shape[-1] == self.target_image_dim + self.target_vector_dim + self.common_obs_dim, f'expect: {self.target_image_dim + self.target_vector_dim + self.common_obs_dim}, got: {x.shape[-1]}'
        # target_image = x[..., -self.target_image_dim:]
        target_vector = x[..., -(self.target_image_dim + self.target_vector_dim):(-self.target_image_dim)]
        common_obs = x[..., :-(self.target_image_dim + self.target_vector_dim)]
        # image_features = self.image_encoder.forward(target_image)
        mixup = torch.cat((target_vector, common_obs, last_action), dim=-1)
        embedding, hidden, _ = self.target_observation_encoder.meta_forward(mixup, hidden)
        if self.learn_std:
            mean, logstd = embedding.chunk(2, dim=-1)
            if self.output_activation == 'tanh':
                LOGSTD_MAX, LOGSTD_MIN = 1.0, -10.0
                logstd = (logstd + 1) / 2 * (LOGSTD_MAX - LOGSTD_MIN) + LOGSTD_MIN
        else:
            mean, logstd = embedding, None
        return mean, logstd, hidden


    def make_rnd_init_state(self, batch_size, device):
        return self.target_observation_encoder.make_rnd_init_state(batch_size, device)

    def make_init_state(self, batch_size, device):
        return self.target_observation_encoder.make_init_state(batch_size, device)

    def logp(self, x, last_action, ground_truth, hidden):
        assert self.learn_std
        mean, logstd, hidden = self.forward(x, last_action, hidden)
        gaussian_distribution = torch.distributions.Normal(mean, logstd.exp())
        return gaussian_distribution.log_prob(ground_truth), hidden

class MaximumEncoder:
    def __init__(self, image_dim, output_dim):
        self.image_dim = image_dim
        self.output_dim = output_dim

    def forward(self, x):
        shape = list(x.shape)
        shape[-1] = self.output_dim
        maximum = torch.max(x, dim=-1, keepdim=True).values
        output = torch.zeros(shape, device=x.device)
        output = output + maximum
        return output


class TargetObsEncoderNoImage2Head(torch.nn.Module):
    def __init__(self, target_image_dim, target_vector_dim, common_obs_dim,
                 action_dim, embedding_dim, rnn_type, middle_ebd_dim=32, output_activation='elu',
                 learn_std=False, layer_norm=False, maximum_image_encoder=False):
        super(TargetObsEncoderNoImage2Head, self).__init__()
        self.target_image_dim = target_image_dim
        self.target_vector_dim = target_vector_dim
        self.common_obs_dim = common_obs_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.learn_std = learn_std
        self.embedding_dim = embedding_dim
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.maximum_image_encoder = maximum_image_encoder
        if layer_norm:
            self.ln = torch.nn.LayerNorm(self.embedding_dim)
        if target_image_dim > 0:
            if self.maximum_image_encoder:
                self.image_encoder = MaximumEncoder(target_image_dim, middle_ebd_dim)
            else:
                self.image_encoder = MLPBase(target_image_dim, middle_ebd_dim, [512, 256], ['elu', 'elu', 'elu'])
        self.other_encoder = MLPBase(common_obs_dim + target_vector_dim + action_dim, middle_ebd_dim, [512, 256], ['elu', 'elu', 'elu'])
        assert self.learn_std
        # if self.learn_std:
        #     assert output_activation in ['linear', 'tanh']
        #     output_activation = 'linear'
        self.target_observation_encoder = RNNBase(middle_ebd_dim * 2 if target_image_dim else middle_ebd_dim,
                                                  128,
                                                  [1024, 512, 256, 256, 256],
                                                  ['elu', 'elu', 'elu', 'elu', 'elu', 'elu'],
                                                  ['fc', 'fc', 'fc', rnn_type, 'fc', 'fc']
                                                  )
        self.mean_head = RNNBase(
            128,
            embedding_dim ,
            [512, 256],
            ['elu', 'elu', output_activation],
            ['fc', 'fc', 'fc'],
        )

        self.logstd_head = RNNBase(
            128,
            embedding_dim,
            [512, 256],
            ['elu', 'elu', 'linear'],
            ['fc', 'fc', 'fc'],
        )

        self.encoder = self.target_observation_encoder
        self.rnn_num = self.target_observation_encoder.rnn_num

    def forward(self, x, last_action, hidden):
        assert x.shape[-1] == self.target_image_dim + self.target_vector_dim + self.common_obs_dim, f'expect: {self.target_image_dim + self.target_vector_dim + self.common_obs_dim}, got: {x.shape[-1]}'
        if self.target_image_dim > 0:
            target_image = x[..., -self.target_image_dim:]
        else:
            target_image = x[..., :0]

        target_vector = x[..., -(self.target_image_dim + self.target_vector_dim):(-self.target_image_dim)]
        if self.target_image_dim + self.target_vector_dim > 0:
            common_obs = x[..., :-(self.target_image_dim + self.target_vector_dim)]
        else:
            common_obs = x
        if self.target_image_dim > 0:
            image_features = self.image_encoder.forward(target_image)
            common_things = self.other_encoder.forward(torch.cat((target_vector, common_obs, last_action), dim=-1))

            mixup = torch.cat((image_features, common_things), dim=-1)
        else:
            common_things = self.other_encoder.forward(torch.cat((target_vector, common_obs, last_action), dim=-1))
            mixup = common_things
        embedding, hidden, _ = self.target_observation_encoder.meta_forward(mixup, hidden)
        mean, _, _ = self.mean_head.meta_forward(embedding, None)
        if self.layer_norm:
            mean = self.ln(mean)
        logstd, _, _ = self.logstd_head.meta_forward(embedding, None)

        return mean, logstd, hidden


    def make_rnd_init_state(self, batch_size, device):
        return self.target_observation_encoder.make_rnd_init_state(batch_size, device)

    def make_init_state(self, batch_size, device):
        return self.target_observation_encoder.make_init_state(batch_size, device)

    def logp(self, x, last_action, ground_truth, hidden):
        assert self.learn_std
        mean, logstd, hidden = self.forward(x, last_action, hidden)
        gaussian_distribution = torch.distributions.Normal(mean, logstd.exp())
        return gaussian_distribution.log_prob(ground_truth), hidden


class TargetObsEncoderNoImage2HeadStdDescend(torch.nn.Module):
    def __init__(self, target_image_dim, target_vector_dim, common_obs_dim,
                 action_dim, embedding_dim, rnn_type, middle_ebd_dim=32, output_activation='elu',
                 learn_std=False, layer_norm=False):
        super(TargetObsEncoderNoImage2HeadStdDescend, self).__init__()
        self.target_image_dim = target_image_dim
        self.target_vector_dim = target_vector_dim
        self.common_obs_dim = common_obs_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.learn_std = learn_std
        self.embedding_dim = embedding_dim
        self.output_activation = output_activation
        if target_image_dim > 0:
            self.image_encoder = MLPBase(target_image_dim, middle_ebd_dim, [512, 256], ['elu', 'elu', 'elu'])
        self.other_encoder = MLPBase(common_obs_dim + target_vector_dim + action_dim, middle_ebd_dim, [512, 256], ['elu', 'elu', 'elu'])
        assert self.learn_std
        self.target_observation_encoder = RNNBase(middle_ebd_dim * 2 if target_image_dim else middle_ebd_dim,
                                                  128,
                                                  [1024, 512, 256, 256, 256],
                                                  ['elu', 'elu', 'elu', 'elu', 'elu', 'elu'],
                                                  ['fc', 'fc', 'fc', rnn_type, 'fc', 'fc']
                                                  )
        self.mean_head = RNNBase(
            128,
            embedding_dim ,
            [512, 256],
            ['elu', 'elu', output_activation],
            ['fc', 'fc', 'fc'],
        )

        self.logstd_head = RNNBase(
            128,
            embedding_dim,
            [512, 256],
            ['elu', 'elu', 'tanh'],
            ['fc', 'fc', 'fc'],
        )

        self.encoder = self.target_observation_encoder
        self.rnn_num = self.target_observation_encoder.rnn_num + 1

    @staticmethod
    def cumsum(x, start, lst_cumsum):
        cumsum = cumsum_fun_selective_scan(x, start, lst_cumsum)
        return cumsum

    def forward(self, x, last_action, hidden):
        hidden_base_logstd = hidden[:1]
        hidden = hidden[1:]
        assert x.shape[-1] == self.target_image_dim + self.target_vector_dim + self.common_obs_dim, f'expect: {self.target_image_dim + self.target_vector_dim + self.common_obs_dim}, got: {x.shape[-1]}'
        if self.target_image_dim > 0:
            target_image = x[..., -self.target_image_dim:]
        else:
            target_image = x[..., :0]

        target_vector = x[..., -(self.target_image_dim + self.target_vector_dim):(-self.target_image_dim)]
        if self.target_image_dim + self.target_vector_dim > 0:
            common_obs = x[..., :-(self.target_image_dim + self.target_vector_dim)]
        else:
            common_obs = x
        if self.target_image_dim > 0:
            image_features = self.image_encoder.forward(target_image)
            common_things = self.other_encoder.forward(torch.cat((target_vector, common_obs, last_action), dim=-1))

            mixup = torch.cat((image_features, common_things), dim=-1)
        else:
            common_things = self.other_encoder.forward(torch.cat((target_vector, common_obs, last_action), dim=-1))
            mixup = common_things
        embedding, hidden, _ = self.target_observation_encoder.meta_forward(mixup, hidden)
        mean, _, _ = self.mean_head.meta_forward(embedding, None)
        delta_logstd, _, _ = self.logstd_head.meta_forward(embedding, None)
        delta_logstd = -delta_logstd

        logstd_old = hidden_base_logstd[0].transpose(0, 1) # (B, 1, C)
        if hidden.rnn_start is None:
            logstd = torch.cumsum(delta_logstd, dim=-2) + logstd_old
        else:
            logstd = self.cumsum(delta_logstd, hidden.rnn_start, logstd_old)
        hidden_base_logstd[0] = logstd[..., -1:, :].transpose(0, 1)    # (B, 1, C) -> (1, B, C)
        hidden = hidden_base_logstd + hidden
        return mean, logstd, hidden

    def make_rnd_init_state(self, batch_size, device):
        init_states = RNNHidden(1, ['gru'], device)
        init_states.append(init_states.init_random_hidden_by_type('gru', batch_size, self.embedding_dim, device))
        return init_states + self.target_observation_encoder.make_rnd_init_state(batch_size, device)

    def make_init_state(self, batch_size, device):
        init_states = RNNHidden(1, ['gru'], device)
        init_states.append(init_states.init_hidden_by_type('gru', batch_size, self.embedding_dim, device))
        return init_states + self.target_observation_encoder.make_init_state(batch_size, device)


class TargetObsEncoderNoScanDot2Head(torch.nn.Module):
    def __init__(self, target_image_dim, target_vector_dim, common_obs_dim,
                 action_dim, embedding_dim, rnn_type, middle_ebd_dim=32, output_activation='elu',
                 learn_std=False):
        super(TargetObsEncoderNoScanDot2Head, self).__init__()
        self.target_image_dim = target_image_dim
        self.target_vector_dim = target_vector_dim
        self.common_obs_dim = common_obs_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.learn_std = learn_std
        self.embedding_dim = embedding_dim
        self.output_activation = output_activation
        # self.image_encoder = MLPBase(target_image_dim, middle_ebd_dim, [512, 256], ['elu', 'elu', 'elu'])
        self.other_encoder = MLPBase(common_obs_dim + target_vector_dim + action_dim, middle_ebd_dim, [512, 256], ['elu', 'elu', 'elu'])
        assert self.learn_std
        # if self.learn_std:
        #     assert output_activation in ['linear', 'tanh']
        #     output_activation = 'linear'
        self.target_observation_encoder = RNNBase(middle_ebd_dim,
                                                  128,
                                                  [1024, 512, 256, 256, 256],
                                                  ['elu', 'elu', 'elu', 'elu', 'elu', 'elu'],
                                                  ['fc', 'fc', 'fc', rnn_type, 'fc', 'fc']
                                                  )
        self.mean_head = RNNBase(
            128,
            embedding_dim ,
            [512, 256],
            ['elu', 'elu', output_activation],
            ['fc', 'fc', 'fc'],
        )

        self.logstd_head = RNNBase(
            128,
            embedding_dim,
            [512, 256],
            ['elu', 'elu', 'linear'],
            ['fc', 'fc', 'fc'],
        )

        self.encoder = self.target_observation_encoder
        self.rnn_num = self.target_observation_encoder.rnn_num

    def forward(self, x, last_action, hidden):
        assert x.shape[-1] == self.target_image_dim + self.target_vector_dim + self.common_obs_dim, f'expect: {self.target_image_dim + self.target_vector_dim + self.common_obs_dim}, got: {x.shape[-1]}'
        # target_image = x[..., -self.target_image_dim:]
        target_vector = x[..., -(self.target_image_dim + self.target_vector_dim):(-self.target_image_dim)]
        common_obs = x[..., :-(self.target_image_dim + self.target_vector_dim)]
        # image_features = self.image_encoder.forward(target_image)
        common_things = self.other_encoder.forward(torch.cat((target_vector, common_obs, last_action), dim=-1))
        # mixup = torch.cat((image_features, common_things), dim=-1)
        mixup = common_things
        embedding, hidden, _ = self.target_observation_encoder.meta_forward(mixup, hidden)
        mean, _, _ = self.mean_head.meta_forward(embedding, None)
        logstd, _, _ = self.logstd_head.meta_forward(embedding, None)

        return mean, logstd, hidden


    def make_rnd_init_state(self, batch_size, device):
        return self.target_observation_encoder.make_rnd_init_state(batch_size, device)

    def make_init_state(self, batch_size, device):
        return self.target_observation_encoder.make_init_state(batch_size, device)

    def logp(self, x, last_action, ground_truth, hidden):
        assert self.learn_std
        mean, logstd, hidden = self.forward(x, last_action, hidden)
        gaussian_distribution = torch.distributions.Normal(mean, logstd.exp())
        return gaussian_distribution.log_prob(ground_truth), hidden


class TargetObsEncoder2Head(torch.nn.Module):
    def __init__(self, target_image_dim, target_vector_dim, common_obs_dim,
                 action_dim, embedding_dim, rnn_type, middle_ebd_dim=32, output_activation='elu',
                 learn_std=False):
        super(TargetObsEncoder2Head, self).__init__()
        self.target_image_dim = target_image_dim
        self.target_vector_dim = target_vector_dim
        self.common_obs_dim = common_obs_dim
        self.action_dim = action_dim
        self.learn_std = learn_std
        self.embedding_dim = embedding_dim
        self.original_image_dim_row, self.original_image_dim_col = 58, 87
        self.image_encoder = DepthOnlyFCBackbone58x87(middle_ebd_dim, num_frames=1)
        self.output_activation = output_activation
        assert self.learn_std
        self.other_encoder = MLPBase(common_obs_dim + target_vector_dim + action_dim, middle_ebd_dim, [512, 256],
                                     ['elu', 'elu', 'elu'])
        self.target_observation_encoder = RNNBase(middle_ebd_dim * 2,
                                                  128,
                                                  [1024, 512, 256, 256, 256],
                                                  ['elu', 'elu', 'elu', 'elu', 'elu', 'elu'],
                                                  ['fc', 'fc', 'fc', rnn_type, 'fc', 'fc']
                                                  )
        self.mean_head = RNNBase(
            128,
            embedding_dim,
            [512, 256],
            ['elu', 'elu', output_activation],
            ['fc', 'fc', 'fc'],
        )

        self.logstd_head = RNNBase(
            128,
            embedding_dim,
            [512, 256],
            ['elu', 'elu', 'linear'],
            ['fc', 'fc', 'fc'],
        )

        self.rnn_num = self.target_observation_encoder.rnn_num
        self.encoder = self.target_observation_encoder
        pass

    def make_rnd_init_state(self, batch_size, device):
        return self.target_observation_encoder.make_rnd_init_state(batch_size, device)

    def make_init_state(self, batch_size, device):
        return self.target_observation_encoder.make_init_state(batch_size, device)

    def forward(self, x, last_action, hidden):
        assert x.shape[-1] == self.target_image_dim + self.target_vector_dim + self.common_obs_dim, f'expect: {self.target_image_dim + self.target_vector_dim + self.common_obs_dim}, got: {x.shape[-1]}'
        target_image = x[..., -self.target_image_dim:]
        target_vector = x[...,
                            -(self.target_image_dim + self.target_vector_dim):(-self.target_image_dim)]
        common_obs = x[..., :-(self.target_image_dim + self.target_vector_dim)]

        image_shape = list(target_image.shape)
        image_target_shape = image_shape[:-1] + [1, self.original_image_dim_row, self.original_image_dim_col]
        image = target_image.reshape(image_target_shape)
        image = image.reshape([-1] + [1, self.original_image_dim_row, self.original_image_dim_col])
        image_features = self.image_encoder.forward(image)
        image_features = image_features.reshape(image_shape[:-1] + [image_features.shape[-1]])
        common_things = self.other_encoder.forward(torch.cat((target_vector, common_obs, last_action), dim=-1))
        mixup = torch.cat((image_features, common_things), dim=-1)
        middle_embedding, hidden, _ = self.target_observation_encoder.meta_forward(
            mixup, hidden
        )
        mean, _, _ = self.mean_head.meta_forward(middle_embedding, None)
        logstd, _, _ = self.logstd_head.meta_forward(middle_embedding, None)
        return mean, logstd, hidden

    def logp(self, x, last_action, ground_truth, hidden):
        assert self.learn_std
        mean, logstd, hidden = self.forward(x, last_action, hidden)
        gaussian_distribution = torch.distributions.Normal(mean, logstd.exp())
        return gaussian_distribution.log_prob(ground_truth), hidden


class TargetObsEncoder2HeadDescend(torch.nn.Module):
    def __init__(self, target_image_dim, target_vector_dim, common_obs_dim,
                 action_dim, embedding_dim, rnn_type, middle_ebd_dim=32, output_activation='elu',
                 learn_std=False):
        super(TargetObsEncoder2HeadDescend, self).__init__()
        self.target_image_dim = target_image_dim
        self.target_vector_dim = target_vector_dim
        self.common_obs_dim = common_obs_dim
        self.action_dim = action_dim
        self.learn_std = learn_std
        self.embedding_dim = embedding_dim
        self.original_image_dim_row, self.original_image_dim_col = 58, 87
        self.image_encoder = DepthOnlyFCBackbone58x87(middle_ebd_dim, num_frames=1)
        self.output_activation = output_activation
        assert self.learn_std
        self.other_encoder = MLPBase(common_obs_dim + target_vector_dim + action_dim, middle_ebd_dim, [512, 256],
                                     ['elu', 'elu', 'elu'])
        self.target_observation_encoder = RNNBase(middle_ebd_dim * 2 + action_dim,
                                                  128,
                                                  [1024, 512, 256, 256, 256],
                                                  ['elu', 'elu', 'elu', 'elu', 'elu', 'elu'],
                                                  ['fc', 'fc', 'fc', rnn_type, 'fc', 'fc']
                                                  )
        self.mean_head = RNNBase(
            128,
            embedding_dim,
            [512, 256],
            ['elu', 'elu', output_activation],
            ['fc', 'fc', 'fc'],
        )

        self.logstd_head = RNNBase(
            128,
            embedding_dim,
            [512, 256],
            ['elu', 'elu', 'sigmoid'],
            ['fc', 'fc', 'fc'],
        )

        self.rnn_num = self.target_observation_encoder.rnn_num + 1
        self.encoder = self.target_observation_encoder
        pass

    def make_rnd_init_state(self, batch_size, device):
        init_states = RNNHidden(1, ['gru'], device)
        init_states.append(init_states.init_random_hidden_by_type('gru', batch_size, self.action_dim, device))
        return init_states + self.target_observation_encoder.make_rnd_init_state(batch_size, device)

    def make_init_state(self, batch_size, device):
        init_states = RNNHidden(1, ['gru'], device)
        init_states.append(init_states.init_hidden_by_type('gru', batch_size, self.action_dim, device))
        return init_states + self.target_observation_encoder.make_init_state(batch_size, device)

    def forward(self, x, last_action, hidden):
        assert x.shape[-1] == self.target_image_dim + self.target_vector_dim + self.common_obs_dim, f'expect: {self.target_image_dim + self.target_vector_dim + self.common_obs_dim}, got: {x.shape[-1]}'
        hidden_base_logstd = hidden[:1]
        hidden = hidden[1:]
        target_image = x[..., -self.target_image_dim:]
        target_vector = x[...,
                            -(self.target_image_dim + self.target_vector_dim):(-self.target_image_dim)]
        common_obs = x[..., :-(self.target_image_dim + self.target_vector_dim)]

        image_shape = list(target_image.shape)
        image_target_shape = image_shape[:-1] + [1, self.original_image_dim_row, self.original_image_dim_col]
        image = target_image.reshape(image_target_shape)
        image = image.reshape([-1] + [1, self.original_image_dim_row, self.original_image_dim_col])
        image_features = self.image_encoder.forward(image)
        image_features = image_features.reshape(image_shape[:-1] + [image_features.shape[-1]])
        common_things = self.other_encoder.forward(torch.cat((target_vector, common_obs, last_action), dim=-1))
        mixup = torch.cat((image_features, common_things), dim=-1)
        middle_embedding, hidden, _ = self.target_observation_encoder.meta_forward(
            mixup, hidden
        )
        mean, _, _ = self.mean_head.meta_forward(middle_embedding, None)
        logstd, _, _ = self.logstd_head.meta_forward(middle_embedding, None)
        hidden = hidden_base_logstd + hidden
        return mean, logstd, hidden

    def logp(self, x, last_action, ground_truth, hidden):
        assert self.learn_std
        mean, logstd, hidden = self.forward(x, last_action, hidden)
        gaussian_distribution = torch.distributions.Normal(mean, logstd.exp())
        return gaussian_distribution.log_prob(ground_truth), hidden
