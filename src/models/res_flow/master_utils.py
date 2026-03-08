from res_flow.res_flow_back_prob import update_lipschitz
from master.config import DEVICE
from master.training.utils import initialize_model


def set_resnet_eval(model):
    update_lipschitz(model)
    model.ema.swap()


def unset_resnet_eval(model):
    model.ema.swap()
    ...


def load_res_flow_model(model, checkpoint, args):
    model.to(DEVICE)
    initialize_model(model, args)

    sd = {k: v for k, v in checkpoint['model'].items() if 'last_n_samples' not in k}
    state = model.state_dict()
    state.update(sd)
    model.load_state_dict(state, strict=True)

    # load ema
    model.ema.set(checkpoint['ema'])
    model.ema.swap()

    update_lipschitz(model)
