import torch
from .mlp_base import MLPBase


class PrivilegedObsEncoderSmall(torch.nn.Module):
    def __init__(self, privileged_image_dim, privileged_vector_dim, common_obs_dim, embedding_dim, middle_ebd_dim=32, output_activation='elu'):
        super(PrivilegedObsEncoderSmall, self).__init__()
        middle_ebd_dim = middle_ebd_dim // 2
        self.privileged_image_dim = privileged_image_dim
        self.privileged_vector_dim = privileged_vector_dim
        self.common_obs_dim = common_obs_dim
        self.embedding_dim = embedding_dim
        self.image_encoder = MLPBase(privileged_image_dim, middle_ebd_dim, [256, 128], ['elu', 'elu', 'elu'])
        self.pre_encoder = MLPBase(privileged_vector_dim, middle_ebd_dim, [128,], ['elu', 'elu'])
        self.common_encoder = MLPBase(common_obs_dim, middle_ebd_dim, [128,], ['elu', 'elu'])
        self.privileged_encoder = MLPBase(middle_ebd_dim + middle_ebd_dim + middle_ebd_dim, embedding_dim, [256, 128], ['elu', 'elu', output_activation])
        self.rnn_num = 0

        pass

    def forward(self, x, hidden):
        assert x.shape[-1] == self.privileged_image_dim + self.privileged_vector_dim + self.common_obs_dim, f'expect: {self.privileged_image_dim + self.privileged_vector_dim + self.common_obs_dim}, got: {x.shape[-1]}'
        privileged_image = x[..., -self.privileged_image_dim:]
        privileged_vector = x[..., -(self.privileged_image_dim + self.privileged_vector_dim):(-self.privileged_image_dim)]
        common_obs = x[..., :-(self.privileged_image_dim + self.privileged_vector_dim)]
        image_features = self.image_encoder.forward(privileged_image)
        # return torch.cat((image_features, privileged_vector), dim=-1), hidden
        mixup = torch.cat((image_features, self.pre_encoder(privileged_vector), self.common_encoder(common_obs)), dim=-1)
        embedding = self.privileged_encoder.forward(mixup)
        return embedding, hidden

class PrivilegedObsEncoder(torch.nn.Module):
    def __init__(self, privileged_image_dim, privileged_vector_dim, common_obs_dim, embedding_dim, middle_ebd_dim=32, output_activation='elu'):
        super(PrivilegedObsEncoder, self).__init__()
        self.privileged_image_dim = privileged_image_dim
        self.privileged_vector_dim = privileged_vector_dim
        self.common_obs_dim = common_obs_dim
        self.embedding_dim = embedding_dim
        self.image_encoder = MLPBase(privileged_image_dim, middle_ebd_dim, [256, 256], ['elu', 'elu', 'elu'])
        self.pre_encoder = MLPBase(privileged_vector_dim, middle_ebd_dim, [256, 128], ['elu', 'elu', 'elu'])
        self.common_encoder = MLPBase(common_obs_dim, middle_ebd_dim, [256, 128], ['elu', 'elu', 'elu'])
        self.privileged_encoder = MLPBase(middle_ebd_dim + middle_ebd_dim + middle_ebd_dim, embedding_dim, [256, 128], ['elu', 'elu', output_activation])
        self.rnn_num = 0

        pass

    def forward(self, x, hidden):
        assert x.shape[-1] == self.privileged_image_dim + self.privileged_vector_dim + self.common_obs_dim, f'expect: {self.privileged_image_dim + self.privileged_vector_dim + self.common_obs_dim}, got: {x.shape[-1]}'
        privileged_image = x[..., -self.privileged_image_dim:]
        privileged_vector = x[..., -(self.privileged_image_dim + self.privileged_vector_dim):(-self.privileged_image_dim)]
        common_obs = x[..., :-(self.privileged_image_dim + self.privileged_vector_dim)]
        image_features = self.image_encoder.forward(privileged_image)
        # return torch.cat((image_features, privileged_vector), dim=-1), hidden
        mixup = torch.cat((image_features, self.pre_encoder(privileged_vector), self.common_encoder(common_obs)), dim=-1)
        embedding = self.privileged_encoder.forward(mixup)
        return embedding, hidden


class PrivilegedObsEncoderNoCommon(torch.nn.Module):
    def __init__(self, privileged_image_dim, privileged_vector_dim, common_obs_dim, embedding_dim, middle_ebd_dim=32, output_activation='elu'):
        super(PrivilegedObsEncoderNoCommon, self).__init__()
        self.privileged_image_dim = privileged_image_dim
        self.privileged_vector_dim = privileged_vector_dim
        self.common_obs_dim = common_obs_dim
        self.embedding_dim = embedding_dim
        self.image_encoder = MLPBase(privileged_image_dim, middle_ebd_dim, [256, 256], ['elu', 'elu', 'elu'])
        self.pre_encoder = MLPBase(privileged_vector_dim, middle_ebd_dim, [256, 128], ['elu', 'elu', 'elu'])
        # self.common_encoder = MLPBase(common_obs_dim, middle_ebd_dim, [256, 128], ['elu', 'elu', 'elu'])
        self.privileged_encoder = MLPBase(middle_ebd_dim + middle_ebd_dim, embedding_dim, [256, 128], ['elu', 'elu', output_activation])
        self.rnn_num = 0

        pass

    def forward(self, x, hidden):
        assert x.shape[-1] == self.privileged_image_dim + self.privileged_vector_dim + self.common_obs_dim, f'expect: {self.privileged_image_dim + self.privileged_vector_dim + self.common_obs_dim}, got: {x.shape[-1]}'
        privileged_image = x[..., -self.privileged_image_dim:]
        privileged_vector = x[..., -(self.privileged_image_dim + self.privileged_vector_dim):(-self.privileged_image_dim)]
        # common_obs = x[..., :-(self.privileged_image_dim + self.privileged_vector_dim)]
        image_features = self.image_encoder.forward(privileged_image)
        # return torch.cat((image_features, privileged_vector), dim=-1), hidden
        mixup = torch.cat((image_features, self.pre_encoder(privileged_vector)), dim=-1)
        embedding = self.privileged_encoder.forward(mixup)
        return embedding, hidden

class PrivilegedObsEncoderExpand(torch.nn.Module):
    def __init__(self, privileged_image_dim, privileged_vector_dim, common_obs_dim, embedding_dim, middle_ebd_dim=32, output_activation='elu'):
        super(PrivilegedObsEncoderExpand, self).__init__()
        self.privileged_image_dim = privileged_image_dim
        self.privileged_vector_dim = privileged_vector_dim
        self.common_obs_dim = common_obs_dim
        self.embedding_dim = embedding_dim
        self.image_encoder = MLPBase(privileged_image_dim, middle_ebd_dim, [512, 256], ['elu', 'elu', 'elu'])
        self.pre_encoder = MLPBase(privileged_vector_dim, middle_ebd_dim, [512, 256], ['elu', 'elu', 'elu'])
        self.common_encoder = MLPBase(common_obs_dim, middle_ebd_dim, [512, 256], ['elu', 'elu', 'elu'])
        self.privileged_encoder = MLPBase(middle_ebd_dim + middle_ebd_dim + middle_ebd_dim, embedding_dim, [768, 512, 256], ['elu', 'elu', 'elu', output_activation])
        self.rnn_num = 0

        pass

    def forward(self, x, hidden):
        assert x.shape[-1] == self.privileged_image_dim + self.privileged_vector_dim + self.common_obs_dim, f'expect: {self.privileged_image_dim + self.privileged_vector_dim + self.common_obs_dim}, got: {x.shape[-1]}'
        privileged_image = x[..., -self.privileged_image_dim:]
        privileged_vector = x[..., -(self.privileged_image_dim + self.privileged_vector_dim):(-self.privileged_image_dim)]
        common_obs = x[..., :-(self.privileged_image_dim + self.privileged_vector_dim)]
        image_features = self.image_encoder.forward(privileged_image)
        # return torch.cat((image_features, privileged_vector), dim=-1), hidden
        mixup = torch.cat((image_features, self.pre_encoder(privileged_vector), self.common_encoder(common_obs)), dim=-1)
        embedding = self.privileged_encoder.forward(mixup)
        return embedding, hidden
