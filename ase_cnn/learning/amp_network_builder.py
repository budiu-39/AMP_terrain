# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np

DISC_LOGIT_INIT_SCALE = 1.0

class AMPBuilder(network_builder.A2CBuilder):   # 这里不只是加入disc，而是把全部网络重写了。
    def __init__(self, **kwargs):   # 应该把相关网络参数作为参数输到这里，具体应该就是run.py里面的调用代码（build_factory）这种
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):   #


            super().__init__(params, **kwargs)

            self.env_sensor_shape = kwargs.get('env_sensor_shape')

            if self.env_sensor_shape is not None:
                self._build_env_cnn(self.env_sensor_shape)  #好像输入必须要是3维的,不知道这样子可不可以(应该可以

            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)

            amp_input_shape = kwargs.get('amp_input_shape')
            self._build_disc(amp_input_shape)   #地形如果也学习了会导致范化性降低吧

            return

        def load(self, params):
            super().load(params)

            self._disc_units = params['disc']['units']
            self._disc_activation = params['disc']['activation']
            self._disc_initializer = params['disc']['initializer']
            self.cnn = params['env_cnn']
            # self._env_cnn_units = params['env_cnn']['units']   #这个不确定params里面有没有
            # self._env_cnn_activation = params['env_cnn']['activation']
            # self._env_cnn_initializer = params['env_cnn']['initializer']
            return
        def forward(self, obs_dict):   # obs_dict 里面存着state？
            # if self.env_sensor_shape:
            #     obs = obs_dict['obs']  # 这里的是所有env的，所以info也要传入所有（已经传了hhh）
            #     env = obs_dict['env_obs']
            #     states = obs_dict.get('rnn_states', None)
            #     env = self.eval_env(env)  # 这里已经是输出1维的tensor了
            #     obs_env = torch.cat((obs, env), 1)
            #     actor_outputs = self.eval_actor(obs_env)  # 这里可以简单合并吗？类型是tensor 用个cat也可以
            #     value = self.eval_critic(obs_env)
            # else:
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            actor_outputs = self.eval_actor(obs)  # 这里可以简单合并吗？类型是tensor
            value = self.eval_critic(obs)

            output = actor_outputs + (value, states)  #输出去了哪里？ 这个是最先的步骤，前向传播后得到s用来更新critic

            return output   #为什么output是32*2*28呢，应为一个是sigma，一个是mu hhhh

        def eval_env(self, obs):
            e_out = self.env_cnn(obs)
            e_out = e_out.contiguous().view(e_out.size(0), -1)  # 这里是变形
            e_out = self.env_mlp(e_out)
            return e_out

        def eval_actor(self, obs):   # 计算actor  也就是决策网络  （这里的obs是什么，环境输入+人物状态

            a_out = self.actor_cnn(obs)    #这个真的有吗？如果cnn不存在的话，首先要决定cnn的参数，然后把obs区分来方，cnn层只输入obs_env, mlp层输入 a_out 和 obs_state
            # 这个可以保留，但是通过env_cnn 在actor_mlp 里面加入 不过mlp是不是..初始化的时候大小已经确定了。那这样的话得完全重写了。
            a_out = a_out.contiguous().view(a_out.size(0), -1)  #这里是变形
            a_out = self.actor_mlp(a_out)  #这里输出是512
                     
            if self.is_discrete:  #这里是做一个log，求对数  discrete multi_discrete continuous 有空也可以研究一下

                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits

            if self.is_continuous:     # 是连续的
                mu = self.mu_act(self.mu(a_out))    #mu是一个全连接层，为什么要专门拿出来，可能是为了探索（比方说，神经元是否激活）
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma   #输出是28，28个自由度？好像真的是
            return

        def eval_critic(self, obs):   #这里是计算critic，输入是obs（在哪里被调用的呢？  突然想到不是self.obs
            c_out = self.critic_cnn(obs)    #目前输入是
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)
            value = self.value_act(self.value(c_out))    #这里是给出动作的价值
            return value

        def eval_disc(self, amp_obs): #这里输入的是什么，参考运动要不要输入到这里？ 等一下... 环境是要跟运动挂钩的，如果运动变成了style的话，
            disc_mlp_out = self._disc_mlp(amp_obs)
            disc_logits = self._disc_logits(disc_mlp_out)
            return disc_logits

        def get_disc_logit_weights(self):
            return torch.flatten(self._disc_logits.weight)

        def get_disc_weights(self):
            weights = []
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            weights.append(torch.flatten(self._disc_logits.weight))
            return weights   #这个是返回权重 应该在记录pth文件的时候被调用

        def _build_env_cnn(self, input_shape):
            self.env_cnn = nn.Sequential()  # 空的序列
            input_shape = torch_ext.shape_whc_to_cwh(input_shape)
            cnn_args = {
                'ctype': self.cnn['type'],
                'input_shape': input_shape,
                'convs': self.cnn['convs'],
                'activation': self.cnn['activation'],
                'norm_func_name': None,
            }
            self.env_cnn = self._build_conv(**cnn_args)
            mlp_input_shape = self._calc_input_size(input_shape, self.env_cnn)
            self.env_mlp = torch.nn.Linear(mlp_input_shape, 64)

            mlp_init = self.init_factory.create(**self.initializer)
            cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.env_cnn.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            return

        def _build_disc(self, input_shape):
            self._disc_mlp = nn.Sequential()  #空的序列

            mlp_args = {
                'input_size' : input_shape[0], 
                'units' : self._disc_units,   #先传到self里面了
                'activation' : self._disc_activation, 
                'dense_func' : torch.nn.Linear
            }
            self._disc_mlp = self._build_mlp(**mlp_args)  #把字典中的值作为关键字上传
            
            mlp_out_size = self._disc_units[-1]
            self._disc_logits = torch.nn.Linear(mlp_out_size, 1)   #这是一个线性层

            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias) 

            torch.nn.init.uniform_(self._disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits.bias) 

            return

    def build(self, name, **kwargs):   #self的 params
        net = AMPBuilder.Network(self.params, **kwargs)
        return net