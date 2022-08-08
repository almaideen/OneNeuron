import pandas as pd
from utils.all_utils import prepare_data
from utils.model import perceptron

OR = {
    'x1':[0,0,1,1],
    'x2':[0,1,0,1],
    'y':[0,1,1,1]
}

df = pd.DataFrame(OR)
X, y = prepare_data(df)
ETA = 0.3
EPOCHS = 10

model = perceptron(eta=ETA,epochs=EPOCHS)
model.fit(X,y)

_ = model.totalloss()