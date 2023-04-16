from torch.nn import CrossEntropyLoss

from simulation.env import Mem, Env
from simulation.model_builder import ModelBuilder

INPUT = 20
OUTPUT = 2
CHECK_FREQ = 10

def simulation(opt, mb: ModelBuilder, bs, lr, eph):
    res = []
    mb.set_input(INPUT)
    mb.set_classes(OUTPUT)
    model = mb.finalize()
    env = Mem()
    env.set_input_size(INPUT)
    env.set_output_size(OUTPUT)
    for i in range(eph):
        print(f"epoch: {i}")
        r = train_loop(env, model, CrossEntropyLoss(), opt(params=model.parameters(), lr=lr), bs)
        res.append(r)
    return res


def train_loop(env: Env, model, loss_fn, optimizer, batch): #Code copied from https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    X = env.reset()
    loss = 100
    for i in range(batch):
        pred = model(X)
        X, Y = env.step(pred)
        loss = loss_fn(pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % CHECK_FREQ == 0:
            loss = loss.item()
            current = (i + 1) * CHECK_FREQ
            print(f"loss: {loss:>7f}  [{current:>5d}]")
    return float(loss)




