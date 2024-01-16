import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM


class RespondCAM(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            RespondCAM,
            self).__init__(
            model,
            target_layers,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.sum(activations * grads, axis=(2, 3)) / np.sum(activations + 1e-10, axis=(2, 3))
        # in case of a 3D tensor, the return statement would be:
        # return np.sum(activations * grads, axis=(2, 3, 4)) / np.sum(activations + 1e-10, axis=(2, 3, 4))
