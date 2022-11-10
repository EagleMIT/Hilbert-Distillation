import six
import yaml
import codecs
from ast import literal_eval


def load_cfg(config):
    with codecs.open(config, 'r', 'utf-8') as file :
        loaded_cfg = yaml.load(file, Loader=yaml.FullLoader)
    return Config(loaded_cfg)


class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    def __setattr__(self, key, value, create_if_not_exist=True):
        t = self
        keylist = key.split(".")
        for k in keylist[:-1]:
            t = t.__getattr__(k, create_if_not_exist)

        t.__getattr__(keylist[-1], create_if_not_exist)
        t[keylist[-1]] = value

    def __getattr__(self, key, create_if_not_exist=True):
        if not key in self:
            if not create_if_not_exist:
                raise KeyError
            self[key] = Config()
        return self[key]

    def __setitem__(self, key, value):
        if isinstance(value, six.string_types):
            try:
                value = literal_eval(value)
            except ValueError:
                pass
            except SyntaxError:
                pass
        super(Config, self).__setitem__(key, value)
