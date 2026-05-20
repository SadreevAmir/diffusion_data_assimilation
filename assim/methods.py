import numpy as np


class IdentityMethod:
    name = "identity"

    @staticmethod
    def init_params(model_config, full_config=None):
        return {}

    @staticmethod
    def preprocess_tensors(tensors, config):
        return tensors

    def fit(self, tensors, steps=None, meta=None):
        self.background = np.asarray(tensors["background"])
        self.analysis = self.background.copy()
        self.observations = tensors.get("observations")

    def get_x(self):
        return self.background, self.analysis

    def get_obs(self):
        return self.observations


methods = {
    IdentityMethod.name: IdentityMethod,
}
