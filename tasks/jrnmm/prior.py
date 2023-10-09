import torch
from sbi.utils import BoxUniform


class prior_JRNMM(BoxUniform):
    def __init__(self, parameters):
        self.parameters = parameters
        low = []
        high = []
        for i in range(len(parameters)):
            low.append(parameters[i][1])
            high.append(parameters[i][2])
        super().__init__(
            low=torch.tensor(low, dtype=torch.float32),
            high=torch.tensor(high, dtype=torch.float32),
        )

    def condition(self, param_name):
        """
        This functions returns the prior distribution for [C, mu, sigma]
        parameter. It is written like this for compatibility purposes with
        the Pyro framework
        """

        low = []
        high = []
        for i in range(len(self.parameters)):
            if self.parameters[i][0] == param_name:
                pass
            else:
                low.append(self.parameters[i][1])
                high.append(self.parameters[i][2])
        low = torch.tensor(low, dtype=torch.float32)
        high = torch.tensor(high, dtype=torch.float32)
        return BoxUniform(low=low, high=high)
