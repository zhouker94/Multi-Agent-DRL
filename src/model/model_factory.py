import model.ddpg
import model.dqn
import model.base_model


class ModelFactory(object):
    def __init__(self, uid, config, ckpt_path):
        self._uid = uid
        self._config = config
        self._ckpt_path = ckpt_path

    def get_model(self, model_type: str) -> model.base_model.BaseModel:
        if model_type == "DQN":
            return model.dqn.DQNModel(
                self._uid,
                self._config,
                self._ckpt_path
            )
        elif model_type == "DDPG":
            return model.ddpg.DDPGModel(
                self._uid,
                self._config,
                self._ckpt_path
            )
        else:
            raise ValueError("Not supported model type")
