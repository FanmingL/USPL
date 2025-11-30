from ..parameter.Parameter import Parameter
from ..algorithm.ppo import PPO
from ..algorithm.ppo_seperate_optim import PPO_SEPERATE_OPTIM
from ..algorithm.ppo_seperate_optim_multi_gpu import PPO_SEPERATE_OPTIM_MULTI_GPU


def alg_init(parameter: Parameter, env=None):
    if parameter.alg_name == 'ppo':
        sac = PPO(parameter, env)
    elif parameter.alg_name == 'ppo_seperate_optim':
        sac = PPO_SEPERATE_OPTIM(parameter, env)
    elif parameter.alg_name == 'ppo_seperate_optim_multi_gpu':
        sac = PPO_SEPERATE_OPTIM_MULTI_GPU(parameter, env)
    else:
        raise NotImplementedError(f'Algorithm {parameter.alg_name} has not been implemented!')
    return sac
