import torch
import torch.nn as nn
from models.basic_layers.ResBlock import ResBlock, conv1x1, conv3x3
from typing import Type, Any, Callable, Union, List, Optional


class GiWModel(nn.Module):
    def __init__(self, cur_image_shape, fur_image_shape, n_fur_states_training=35):
        super(GiWModel, self).__init__()
        self.cur_shape = cur_image_shape
        self.fur_shape = fur_image_shape
        self.n_fur_states_training = n_fur_states_training
        self.cur_state_encoder = nn.Sequential(
            nn.Conv2d(self.cur_shape[0], 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64, 128, stride=2, downsample=nn.Sequential(
                conv1x1(64, 128, 2),
                nn.BatchNorm2d(128),
            )),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResBlock(128, 128),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResBlock(128, 128)
        )
        self.fur_state_encoder = nn.Sequential(
            nn.Conv2d(self.fur_shape[0], 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64, 128, stride=1, downsample=nn.Sequential(
                conv1x1(64, 128, 1),
                nn.BatchNorm2d(128),
            )),
            ResBlock(128, 128),
            ResBlock(128, 128)
        )
        self.action_estimator = nn.Sequential(
            ResBlock(256, 128, stride=1, downsample=nn.Sequential(
                conv1x1(256, 128, 1),
                nn.BatchNorm2d(128),
            )),
            ResBlock(128, 128),
            ResBlock(128, 64, stride=1, downsample=nn.Sequential(
                conv1x1(128, 64),
                nn.BatchNorm2d(64),
            )),
            nn.Conv2d(64, 1, 1, stride=1, padding=0)
        )

    def forward(self, cur_input, fur_inputs):
        assert list(fur_inputs.size())[2:] == self.fur_shape
        cf, hf, wf = self.fur_shape

        cur_states = self.cur_state_encoder(cur_input)
        fur_n_states = self.fur_state_encoder(fur_inputs.view(-1, cf, hf, wf))

        _, codes_c, codes_h, codes_w = fur_n_states.size()
        cur_states = cur_states.unsqueeze(1).repeat_interleave(self.n_fur_states_training, dim=1)
        fur_n_states = fur_n_states.view(-1, self.n_fur_states_training, codes_c, codes_h, codes_w)
        cat_states = torch.cat([cur_states, fur_n_states], dim=2)

        fur_n_rewards = self.action_estimator(cat_states.view(-1, 2 * codes_c, codes_h, codes_w))
        fur_n_rewards = fur_n_rewards.view((-1, self.n_fur_states_training) + fur_n_rewards.size()[-2:])

        return fur_n_rewards

    def load_w(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['state_dict'])


if __name__ == "__main__":
    # batch size=1
    cur_state_input = torch.randn([1, 7, 360, 640], dtype=torch.float32)
    fur_state_inputs = torch.randn([1, 35, 7, 45, 80], dtype=torch.float32)
    model = GiWModel(cur_image_shape=[7, 360, 640], fur_image_shape=[7, 45, 80], n_fur_states_training=35)

    rewards = model(cur_state_input, fur_state_inputs)
    print("rewards: ", rewards.size())
