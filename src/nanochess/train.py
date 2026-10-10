from nanochess.dataset import get_loader
from pprint import pprint


# WIP

def train_model():
    train_loader = get_loader("train", 2)
    val_loader = get_loader("val", 2)


    x_train, y_train = next(iter(train_loader))
    pprint(x_train)
    pprint(y_train)
    print(f"{x_train.shape = }")
    print(f"{y_train.shape = }")

