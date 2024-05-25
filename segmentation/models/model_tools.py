import torch


def load_model(model, model_path, use_ema=False, optimizer=None, resume=False, start_lr=None):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    start_epoch = checkpoint['epoch']
    print('loaded {}, epoch {}'.format(model_path, start_epoch))
    if use_ema:
        state_dict_ = checkpoint['ema_state_dict']
    else:
        state_dict_ = checkpoint['model_state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            if start_lr is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = start_lr
                print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')

    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, ema=None, best=None, optimizer=None, scheduler=None):
    parallel = isinstance(model, torch.nn.DataParallel)
    model_state_dict = model.module.state_dict() if parallel else model.state_dict()
    data = {'epoch': epoch, 'model_state_dict': model_state_dict}

    if best is not None:
        data['best'] = best
    if ema is not None:
        ema_model = ema.model
        parallel = isinstance(ema_model, torch.nn.DataParallel)
        ema_state_dict = ema_model.module.state_dict() if parallel else ema_model.state_dict()
        data['ema_state_dict'] = ema_state_dict
    if optimizer is not None:
        data['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        data['scheduler'] = scheduler.state_dict()

    torch.save(data, path)