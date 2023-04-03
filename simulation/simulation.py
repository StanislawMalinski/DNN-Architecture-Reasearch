from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from environment import RB, Env
from model_builder import ModelBuilder

def simulation(opt, mb, bs, lr, eph):
    pass

def train_loop(env: Env, model, loss_fn, optimizer): #Code copied from https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    for i in range(batch):
        X = env.reset()
        y = env.expected(X)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            loss, current = loss.item(), (i + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}]")


if __name__ == "__main__":
    env = RB()
    env.set_input(2)
    env.set_output_size(2)
    env.reset()

    mb = ModelBuilder()
    mb.new_model()
    mb.set_input(env.get_input_size())
    mb.set_classes(env.get_output_size())
    model = mb.finalize()

    objective = CrossEntropyLoss()
    optim = SGD(params=model.parameters(), lr=0.1)

    train_loop(env, model, objective, optim)
