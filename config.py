import json

""" configuration json """
class Config(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)

config = Config({

        # optimization
        "batch_size": 6, #48 20 2
        "learning_rate": 1e-3, #3
        "weight_decay": 1e-2, #3
        "n_epoch": 70,
        "patience":60,
        "val_freq": 1,
        "num_avg_val": 1, 
        "num_workers": 1,
        
        # data
        "dataset":"VALID",

        # model
        "type": "ECAvalidMobSwin",
        "svPath": "results",

        # load & save checkpoint
        "model_name": "ECAvalidMobSwin",
        "type_name": "MobSwin",
        "ckpt_path": "./output/models/",               # directory for saving checkpoint
        "log_path": "./output/log/",
        "log_file": ".log",
        "tensorboard_path": "./output/tensorboard/"

})