from .contextual_ppo_value_mlp_encoder import ContextualPPOValue
from .contextual_ppo_policy_mlp_encoder import ContextualPPOPolicy

def make_policy_model(policy_args, base_alg_name, discrete):
    policy = ContextualPPOPolicy(**policy_args)
    return policy

def make_value_model(value_args, base_alg_name, discrete):
    value = ContextualPPOValue(**value_args)
    return value
