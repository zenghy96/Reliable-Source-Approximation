import torch
import numpy as np


def add_to_logging_dict(logging_dict, header, values):
    for index, a in enumerate(values):
        a_mean = torch.mean(a).detach().cpu().numpy()
        a_variance = torch.var(a).detach().cpu().numpy()
        # print(names[index], " " , a_mean, a_variance)
        logging_dict[header[index] + "_mean"] = a_mean
        # logging_dict[header[index]+"_variance"] = a_variance
    return logging_dict


def NIG_NLL(y, mu, v, alpha, beta):
    epsilon = 1e-16
    twoBlambda = 2 * beta * (1 + v)

    a1 = 0.5723649429247001 - 0.5 * torch.log(v + epsilon)
    # a1 = 0.5 * torch.log(np.pi / torch.max(v,epsilon))
    a2a = - alpha * torch.log(2 * beta + epsilon)
    a2b = - alpha * torch.log(1 + v)
    a3 = (alpha + 0.5) * torch.log(v * (y - mu) ** 2 + twoBlambda + epsilon)
    a4 = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

    a2 = a2a + a2b

    nll = a1 + a2 + a3 + a4

    # nll = torch.exp(nll)
    likelihood = (np.pi / v) ** (0.5) / (twoBlambda ** alpha) * ((v * (y - mu) ** 2 + twoBlambda) ** (alpha + 0.5))
    # nll = 1 * (y - mu)**2
    likelihood *= torch.exp(a4)
    # nll = likelihood

    mse = (mu - y) ** 2
    mse += 1e-15
    mse = torch.log(mse)

    header = ['y', 'mu', 'v', 'alpha', 'beta', 'nll', 'mse', 'a1', 'a2a', 'a2b', 'a2', 'a3', 'a4', 'likelihood',
              'twoblambda']
    values = [y, mu, v, alpha, beta, nll, mse, a1, a2a, a2b, a2, a3, a4, likelihood, twoBlambda]

    logging_dict = {}
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    is_nan = torch.stack([torch.isnan(x) * 1 for x in values])

    return nll, logging_dict


def NIG_Reg(y, gamma, v, alpha, beta):
    error = torch.abs(y - gamma)

    # alternatively can do
    # error = (y-gamma)**2

    evi = v + alpha + 1 / (beta + 1e-15)
    reg = error * evi

    return reg
    # return torch.mean(reg)


def calculate_evidential_loss(y, mu, v, alpha, beta, lambda_coef=1.0):
    nig_nll, logging_dict = NIG_NLL(y, mu, v, alpha, beta)
    nig_reg = NIG_Reg(y, mu, v, alpha, beta)

    ev_sum = nig_nll + lambda_coef * nig_reg
    evidential_loss = torch.mean(ev_sum)

    header = ['nig_nll', 'nig_reg', 'nig_loss', 'mse']
    values = [nig_nll, nig_reg, ev_sum, (y - mu) ** 2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    return evidential_loss


def calculate_evidential_loss_constraints(it, y, mu, v, alpha, beta, lambda_coef=1.0):
    nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha, beta)
    nig_reg = NIG_Reg(y, mu, v, alpha, beta)

    ev_sum = nig_nll + lambda_coef * nig_reg
    evidential_loss = torch.mean(ev_sum)

    header = ['nig_nll', 'nig_reg', 'nig_loss', 'mse']
    values = [nig_nll, nig_reg, ev_sum, (y - mu) ** 2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    return evidential_loss, logging_dict


def calc_ev_krnl_reg(target_x, context_x, v, lambda_ker=0):
    diff_mat = target_x[:, :, None, :] - context_x[:, None, :, :]
    sq_mat = diff_mat ** 2

    dist_mat = torch.einsum('bijk->bij', sq_mat)
    dist_mat = torch.sqrt(dist_mat)

    min_dist = torch.min(dist_mat, dim=-1)[0]

    kernel_reg_val = lambda_ker * min_dist[:, :, None] * v

    kernel_reg_val = torch.mean(kernel_reg_val)

    return kernel_reg_val
