import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM


class SaliencyTubes(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            SaliencyTubes,
            self).__init__(
            model,
            target_layers,
            reshape_transform)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        preds, output_layer = self.model(input_tensor).cpu().numpy()
        output_layer = output_layer.cpu().numpy().transpose(1, 2, 3, 0)
        pred_weights = self.model.fc.weight[target_category].cpu().numpy()

        cam = np.zeros(dtype=np.float32, shape=output_layer.shape[0:3])

        for i, w in enumerate(pred_weights):
            cam += w * output_layer[:, :, :, i]
        return cam
