import torch
import res_flow.layers.base as base_layers


def update_lipschitz(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
                m.compute_weight(update=True)
            if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                m.compute_weight(update=True)


def res_flow_back_prob(train_model):
    args = train_model.args
    model = train_model.model
    optimizer = train_model.optimizer

    ### is set to 1 per default
    #if global_itr % args.update_freq == args.update_freq - 1:

    if args.update_freq > 1:
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= args.update_freq

    grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.)
    ## default is False
    #if args.learn_p: compute_p_grads(model)

    optimizer.step()
    optimizer.zero_grad()
    update_lipschitz(model)
    model.ema.apply()
