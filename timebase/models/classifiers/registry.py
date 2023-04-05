from timebase.models import utils
from timebase.utils.tensorboard import Summary

_MODELS = dict()


def register(name):
    def add_to_dict(fn):
        global _MODELS
        _MODELS[name] = fn
        return fn

    return add_to_dict


def get_model(args, summary: Summary = None):
    if args.model not in _MODELS:
        raise KeyError(f"model {args.model} not found")
    model = _MODELS[args.model](args)
    if summary is not None:
        summary.scalar(
            f"model/trainable_parameters", utils.count_trainable_params(model)
        )
        model_summary = utils.model_summary(args, model)
        if args.verbose == 2:
            print(model_summary)
    return model
