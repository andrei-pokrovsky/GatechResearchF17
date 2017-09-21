import torch
import torch.nn as nn


NUM_ANCHORS = 9

class AnchorSet(nn.ParameterList):
    def __init__(self):
        super().__init__()

        standard = torch.Tensor([0.8, 0.8, 0.8])
        scales = [1.0, 1.5, 2.0]
        aspects = [1.0, 2.0/1.0, 1.0/2.0]
        print(aspects)

        for scale in scales:
            for aspect in aspects:
                new_box = scale * standard
                new_box[0] *= aspect
                new_box[1] /= aspect
                self.append(nn.Parameter(new_box))


if __name__ == "__main__":
    anchors = AnchorSet()

    for anchor in anchors:
        print(anchor.data)

