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
    def __init__(self, privileged_image_dim, privileged_vector_dim, common_obs_dim,
                 embedding_dim, middle_ebd_dim=32, output_activation='elu', no_common=False, layer_norm=False):
        super(PrivilegedObsEncoder, self).__init__()
        self.privileged_image_dim = privileged_image_dim
        self.privileged_vector_dim = privileged_vector_dim
        self.common_obs_dim = common_obs_dim
        self.embedding_dim = embedding_dim
        self.no_common = no_common
        if privileged_image_dim > 0:
            self.image_encoder = MLPBase(privileged_image_dim, middle_ebd_dim, [256, 256], ['elu', 'elu', 'elu'])
        self.pre_encoder = MLPBase(privileged_vector_dim, middle_ebd_dim, [256, 128], ['elu', 'elu', 'elu'])
        if not no_common:
            self.common_encoder = MLPBase(common_obs_dim, middle_ebd_dim, [256, 128], ['elu', 'elu', 'elu'])
        encoding_dim = middle_ebd_dim
        if privileged_image_dim > 0:
            encoding_dim += middle_ebd_dim
        if not no_common:
            encoding_dim += middle_ebd_dim
        self.privileged_encoder = MLPBase(encoding_dim, embedding_dim, [256, 128], ['elu', 'elu', output_activation])
        self.layer_norm = layer_norm
        if layer_norm:
            self.ln = torch.nn.LayerNorm(embedding_dim)

        self.rnn_num = 0

        pass

    def forward(self, x, hidden):
        assert x.shape[-1] == self.privileged_image_dim + self.privileged_vector_dim + self.common_obs_dim, f'expect: {self.privileged_image_dim + self.privileged_vector_dim + self.common_obs_dim}, got: {x.shape[-1]}'
        if self.privileged_image_dim > 0:
            privileged_image = x[..., -self.privileged_image_dim:]
            privileged_vector = x[..., -(self.privileged_image_dim + self.privileged_vector_dim):(-self.privileged_image_dim)]
            common_obs = x[..., :-(self.privileged_image_dim + self.privileged_vector_dim)]
        else:
            assert self.privileged_vector_dim > 0
            privileged_image = x[..., :0]
            privileged_vector = x[...,
                                -(self.privileged_image_dim + self.privileged_vector_dim):]
            common_obs = x[..., :-(self.privileged_image_dim + self.privileged_vector_dim)]
        # if self.privileged_image_dim > 0:
        #     image_features = self.image_encoder.forward(privileged_image)
        #     if not self.no_common:
        #         mixup = torch.cat((image_features, self.pre_encoder(privileged_vector), self.common_encoder(common_obs)), dim=-1)
        #     else:
        #         mixup = torch.cat((image_features, self.pre_encoder(privileged_vector)), dim=-1)
        # else:
        #     if not self.no_common:
        #         mixup = torch.cat((self.pre_encoder(privileged_vector), self.common_encoder(common_obs)), dim=-1)
        #     else:
        #         mixup = self.pre_encoder(privileged_vector)
        # embedding = self.privileged_encoder.forward(mixup)
        # if self.layer_norm:
        #     embedding = self.ln(embedding)
        # TODO 0314 modified
        embedding = privileged_vector
        return embedding, hidden


class PrivilegedObsEncoderDirect(torch.nn.Module):
    def __init__(self, privileged_image_dim, privileged_vector_dim, common_obs_dim, embedding_dim, middle_ebd_dim=32, output_activation='elu'):
        super(PrivilegedObsEncoderDirect, self).__init__()
        self.privileged_image_dim = privileged_image_dim
        self.privileged_vector_dim = privileged_vector_dim
        self.common_obs_dim = common_obs_dim
        self.embedding_dim = embedding_dim
        if privileged_image_dim > 0:
            self.image_encoder = MLPBase(privileged_image_dim, middle_ebd_dim, [256, 256], ['elu', 'elu', 'elu'])
        self.common_encoder = MLPBase(common_obs_dim, middle_ebd_dim, [256, 128], ['elu', 'elu', 'elu'])
        encoding_dim = middle_ebd_dim + middle_ebd_dim
        if privileged_image_dim > 0:
            encoding_dim += privileged_image_dim
        self.privileged_encoder = MLPBase(encoding_dim, embedding_dim, [256, 128], ['elu', 'elu', output_activation])
        self.rnn_num = 0

        pass

    def forward(self, x, hidden):
        assert x.shape[-1] == self.privileged_image_dim + self.privileged_vector_dim + self.common_obs_dim, f'expect: {self.privileged_image_dim + self.privileged_vector_dim + self.common_obs_dim}, got: {x.shape[-1]}'
        if self.privileged_image_dim > 0:
            privileged_image = x[..., -self.privileged_image_dim:]
            privileged_vector = x[..., -(self.privileged_image_dim + self.privileged_vector_dim):(-self.privileged_image_dim)]
            common_obs = x[..., :-(self.privileged_image_dim + self.privileged_vector_dim)]
        else:
            assert self.privileged_vector_dim > 0
            privileged_image = x[..., :0]
            privileged_vector = x[...,
                                -(self.privileged_image_dim + self.privileged_vector_dim):]
            common_obs = x[..., :-(self.privileged_image_dim + self.privileged_vector_dim)]
        if self.privileged_image_dim > 0:
            image_features = self.image_encoder.forward(privileged_image)
            mixup = torch.cat((image_features, privileged_vector, self.common_encoder(common_obs)), dim=-1)
        else:
            mixup = torch.cat((self.pre_encoder(privileged_vector), self.common_encoder(common_obs)), dim=-1)

        # return torch.cat((image_features, privileged_vector), dim=-1), hidden
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
