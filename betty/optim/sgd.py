import torch


def fsgd(
    params,
    param_mapping,
    param_groups,
    states
):
    for group_idx, group_mapping in enumerate(param_mapping):
        group = param_groups[group_idx]

        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for param_idx in group_mapping:
            p = params[param_idx]

            if p.gradient is None:
                continue
            grad = p.gradient
            #print('sgd grad:', grad)
            if weight_decay != 0:
                grad = grad + weight_decay * p

            param_state = states[param_idx]
            if 'momentum_buffer' not in param_state or param_state['momentum_buffer'] is None:
                buf = param_state['momentum_buffer'] = grad
            else:
                buf = param_state['momentum_buffer']
                buf = momentum * buf + (1 - dampening) * grad
                param_state['momentum_buffer'] = buf
            if nesterov:
                grad = grad + momentum * buf
            else:
                grad = buf

            p.update = group['lr'] * grad
            #params[param_idx] = p - group['lr'] * grad
    #return tuple(params)
    return tuple(p - p.update for p in params)
