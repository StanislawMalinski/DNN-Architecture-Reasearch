from torch import nn

from simulation.env import Mem, Env, SqApr, LinApr
from simulation.model_builder import ModelBuilder

INPUT = 50
OUTPUT = 5
CHECK_FREQ = 10

def simulation(opt, mb: ModelBuilder, bs, lr, eph):
    res = []
    mb.set_input(INPUT)
    mb.set_classes(OUTPUT)
    model = mb.finalize()
    env = Mem()
    env.set_input_size(INPUT)
    env.set_output_size(OUTPUT)
    env.reset()
    for i in range(eph):
        print(f"epoch: {i}")
        r = train_loop(env, model, opt(params=model.parameters(), lr=lr), bs)
        res.append(r)
    return res


def train_loop(env: Env, model, optimizer, batch):
    X = env.observation()
    loss_fn = nn.MSELoss()
    loss = 1
    check = 0.0
    sum_loss = 0.0
    for i in range(batch):
        pred = model(X)
        X_new, Y = env.step(pred)
        loss = loss_fn(pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        X = X_new

        if i % CHECK_FREQ == 0:
            loss = loss.item()
            sum_loss += float(loss)
            check += 1
            current = (i + 1) * CHECK_FREQ
            print(f"loss: {loss:>7f}  [{current:>5d}]")

    return sum_loss/check
