import pandas as pd
from utils.all_utils import prepare_data
from utils.model import perceptron


def main(data,eta,epochs):

    df = pd.DataFrame(data)

    X, y = prepare_data(df)


    model = perceptron(eta=eta,epochs=epochs)
    model.fit(X,y)

    _ = model.totalloss() #dummy variable

if __name__ == '__main__': #entry point

    AND = {
        'x1': [0, 0, 1, 1],
        'x2': [0, 1, 0, 1],
        'y': [0, 0, 0, 1]
    }

    ETA = 0.3
    EPOCHS = 10
    main(data=AND,eta=ETA,epochs=EPOCHS)