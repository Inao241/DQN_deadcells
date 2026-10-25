import torch
import torch.nn as nn


class DQN_action(nn.Module):
    def __init__(self, action_dim, dueling=True):
        super(DQN_action, self).__init__()
        self.dueling = dueling

        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(9218, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128)

        )

        if self.dueling:
            # Dueling head: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
            self.a = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )
            self.v = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        else:
            # Plain Q head with the same depth/width, for the ablation
            self.q = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )

    def forward(self, x, boss, player):
        x = self.net(x)
        x = torch.cat((x, boss, player), dim=1)
        x = self.fc(x)
        return self._head(x)

    def _head(self, x):
        if not self.dueling:
            return self.q(x)
        a = self.a(x)
        v = self.v(x)
        return v + (a - a.mean(dim=-1, keepdim=True))


class DQN_move(nn.Module):
    def __init__(self, action_dim, dueling=True):
        super(DQN_move, self).__init__()
        self.dueling = dueling

        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),

            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(9221, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128)

        )

        if self.dueling:
            # Dueling head: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
            self.a = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )
            self.v = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        else:
            # Plain Q head with the same depth/width, for the ablation
            self.q = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )

    def forward(self, x, action):
        x = self.net(x)
        x = torch.cat((x, action), 1)
        x = self.fc(x)
        return self._head(x)

    def _head(self, x):
        if not self.dueling:
            return self.q(x)
        a = self.a(x)
        v = self.v(x)
        return v + (a - a.mean(dim=-1, keepdim=True))


if __name__ == '__main__':
    a = torch.rand((1, 4, 128, 128))
    net = DQN_move(3)

    action = torch.zeros(5)
    action[0] = 1
    net(a, action.unsqueeze(0))
