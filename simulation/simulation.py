from torch import nn

from simulation.env import Env, Mem
from simulation.model_builder import ModelBuilder
from simulation.status import Status

INPUT = 50
OUTPUT = 5
CHECK_FREQ = 10

def one_observation(env, model):
    X = env.observation()
    pred = model(X)
    X_new, Y = env.step(pred)
    loss_fn = nn.MSELoss()
    loss = loss_fn(pred, Y)
    return loss.item()

def simulation(mb: ModelBuilder, env, opt, bs, lr, eph):
    mb.set_input(INPUT)
    mb.set_classes(OUTPUT)
    model = mb.finalize()
    if env.get_required_sizes() is not None:
        inp, out = env.get_required_sizes()
        env.set_input_size(inp)
        env.set_output_size(out)
    else:
        env.set_input_size(INPUT)
        env.set_output_size(OUTPUT)
    env.reset()

    res = [one_observation(env, model)]
    for i in range(eph):
        Status.get_status().tic_epoch(f"epoch: {i}")
        r = bs_train_loop(env, model, opt(params=model.parameters(), lr=lr), bs)
        res.append(r)
    return res

def bs_train_loop(env: Env, model, optimizer, batch):
    X = env.observation()
    loss_fn = nn.MSELoss()
    check = 0
    sum_loss = 0.0
    min = 1000
    optimizer.zero_grad()
    for i in range(batch):
        pred = model(X)

        X_new, Y = env.step(pred)
        loss = loss_fn(pred, Y)

        X = X_new

        loss.backward()
        optimizer.step()

        loss_l = loss.item()
        sum_loss += float(loss_l)
        check += 1
        if loss < min:
            min = loss_l

        if i % CHECK_FREQ == 0:
            Status.get_status().print(f"loss: {loss_l:>7f}")

    return min

def loss_criterium(loss):
    epsilon = 1e-6
    return loss[0] - loss[-1] < epsilon

def avg(loss):
    return sum(loss)/len(loss)
def simulation_mem(mb: ModelBuilder, env_configuration, opt, lr, bs):
    EPOCH = 20
    mb.set_input(INPUT)
    mb.set_classes(OUTPUT)
    model = mb.finalize()

    env = Mem(1)
    env.set_input_size(INPUT)
    env.set_output_size(OUTPUT)
    env.reset()

    last_size = env_configuration[0]

    res = []


    for size in env_configuration:
        env.enlarge(size - last_size)
        last_size = size - last_size
        big_epoch = 0

        losses = [n for n in range(EPOCH)]

        while loss_criterium(losses):
            big_epoch += 1
            if big_epoch > 1:
                Status.get_status().add_epoch(EPOCH)

            for eph in range(EPOCH):
                Status.get_status().tic_epoch(f"epoch: {(big_epoch-1)*EPOCH + eph}")
                r = bs_train_loop(env, model, opt(params=model.parameters(), lr=lr), bs)
                losses[eph] = r
        res.append(avg(losses))
    return res