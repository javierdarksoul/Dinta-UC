import bentoml
from bentoml.io import NumpyNdarray
import numpy as np
vit16_runner = bentoml.pytorch.get("vit16:latest").to_runner()
example= np.zeros((1,3,224,224))
svc = bentoml.Service("vit16", runners=[vit16_runner])
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = vit16_runner.run(example)
    return result.argmax()