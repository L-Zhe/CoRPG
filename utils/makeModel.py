from    Transformer.Model import transformer
import  importlib
from    utils import constants

def make_model(config):
    return transformer(config)
