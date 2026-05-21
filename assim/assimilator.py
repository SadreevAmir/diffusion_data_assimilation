from .logger import AssimilationLogger


class AssimilationHandler:
    def __init__(self, config, dataloader, method_class, model_config, data_config):
        self.config = config
        self.dataloader = dataloader
        self.method_class = method_class
        self.model_config = model_config
        self.data_config = data_config
        self.logger = AssimilationLogger(config, model_config, data_config)

    def run_assimilation(self):
        params = self.method_class.init_params(self.model_config, self.config)
        method = self.method_class(**params)

        for tensors, targets, meta in self.dataloader:
            tensors = method.preprocess_tensors(tensors, self.config)
            method.fit(tensors, self.config.get("steps"), meta)
            background, analysis = method.get_x()
            obs = method.get_obs()
            self.logger.log(meta, background, analysis, targets, obs)

        self.logger.close()
