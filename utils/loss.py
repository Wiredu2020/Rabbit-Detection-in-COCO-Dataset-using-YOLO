import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.config as config


class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction='sum')
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model predictions (batch_size, grid_size, grid_size, 6), 6 = [x, y, w, h, conf, class]
            targets: Target values (batch_size, grid_size, grid_size, 6)
        """
        batch_size = predictions.size(0)
        
        # Create masks for cells with and without objects
        obj_mask = targets[..., 4] > 0  # mask for cells containing objects
        noobj_mask = ~obj_mask  # mask for cells without objects

        # Extract predictions and targets for object cells
        obj_pred = predictions[obj_mask]
        obj_target = targets[obj_mask]
        
        # Compute losses for object presence
        loss_xy = self.mse(obj_pred[:, :2], obj_target[:, :2])
        loss_wh = self.mse(
            torch.sqrt(torch.clamp(obj_pred[:, 2:4], min=config.EPS)),
            torch.sqrt(torch.clamp(obj_target[:, 2:4], min=config.EPS)),
        )
        loss_obj = self.mse(obj_pred[:, 4], obj_target[:, 4])
        loss_class = self.mse(obj_pred[:, 5], obj_target[:, 5])

        # Compute loss for no-object confidence
        loss_noobj = self.mse(predictions[noobj_mask][:, 4], targets[noobj_mask][:, 4])

        # Compute total loss
        total_loss = (self.lambda_coord * (loss_xy + loss_wh) +
                      loss_obj +
                      self.lambda_noobj * loss_noobj +
                      loss_class)
        
        if torch.isnan(total_loss).any():
            print("NaN detected in loss!")
            print("loss_xy:", loss_xy)
            print("loss_wh:", loss_wh)
            print("loss_obj:", loss_obj)
            print("loss_noobj:", loss_noobj)
            print("loss_class:", loss_class)
        
        return total_loss / batch_size
