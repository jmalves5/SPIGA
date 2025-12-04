import torch
import torch.nn as nn
from typing import Dict, Tuple

# ======================== Loss Functions ========================

class AWingLoss(nn.Module):
    """
    Adaptive Wing Loss for heatmap regression
    Reference: "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"
               Wang et al. (ICCV 2019)
    """
    def __init__(self, alpha: float = 2.1, omega: float = 14, epsilon: float = 1, theta: float = 0.5):
        super(AWingLoss, self).__init__()
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted heatmap [B, N, H, W]
            target: Ground truth heatmap [B, N, H, W]
        Returns:
            loss: Scalar loss value
        """
        # Clamp predictions to prevent numerical instability
        pred = torch.clamp(pred, min=0.0, max=1.0)
        target = torch.clamp(target, min=0.0, max=1.0)
        
        delta = (target - pred).abs()
        
        # Add small epsilon for numerical stability
        eps = 1e-7
        
        A = (
            self.omega
            * (1 / (1 + torch.pow(self.theta / (self.epsilon + eps), self.alpha - target) + eps))
            * (self.alpha - target)
            * torch.pow(self.theta / (self.epsilon + eps), self.alpha - target - 1)
            * (1 / (self.epsilon + eps))
        )
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / (self.epsilon + eps), self.alpha - target) + eps
        )

        losses = torch.where(
            delta < self.theta,
            self.omega
            * torch.log(1 + torch.pow(delta / (self.epsilon + eps), self.alpha - target) + eps),
            A * delta - C,
        )
        
        # Check for NaN/Inf before returning
        if torch.isnan(losses).any() or torch.isinf(losses).any():
            print(f"WARNING: AWingLoss produced NaN/Inf values")
            print(f"  pred stats: min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}")
            print(f"  target stats: min={target.min():.4f}, max={target.max():.4f}, mean={target.mean():.4f}")
            print(f"  delta stats: min={delta.min():.4f}, max={delta.max():.4f}")
            # Return NaN to signal the training loop to skip this batch
            return torch.tensor(float('nan'), device=pred.device, dtype=pred.dtype)

        return losses.mean()
        
class SmoothL1Loss(nn.Module):
    """Smooth L1 loss for coordinate regression"""
    def __init__(self, beta: float = 1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.beta, 
            0.5 * diff**2 / self.beta, 
            diff - 0.5 * self.beta
        )
        return loss.mean()


class LandmarkLoss(nn.Module):
    """Combined loss for landmark detection (Equation 1 from paper)"""
    def __init__(self, num_stages: int = 4, lambda_coord: float = 4.0, lambda_att: float = 50.0):
        super(LandmarkLoss, self).__init__()
        self.num_stages = num_stages
        self.lambda_coord = lambda_coord
        self.lambda_att = lambda_att
        self.coord_loss = SmoothL1Loss()
        self.awing_loss = AWingLoss()
        
        # For Stage 1: Add simple regression heads to predict landmarks from VisualField
        # This allows supervised training of the CNN backbone
        # Use smaller initialization and normalization for stability
        self.stage1_heads = nn.ModuleList()
        for _ in range(num_stages):
            head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, 98 * 2),  # 98 landmarks * 2 coords
            )
            # Initialize with small weights to prevent explosion
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            self.stage1_heads.append(head)

    def forward(self, predictions: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute landmark loss with doubling weights across HG stages (paper Eq. 1)
        Llnd = Σ(h=1 to M) 2^(h-1) * (λc * Lh_coord + λatt * (Lh_points + Lh_edges))
        
        For Stage 1 (backbone_forward): Uses VisualField features with simple regression heads
        For Stage 2/3 (full forward): Uses Landmarks and Heatmaps with proper losses
        """
        losses_detail = {}
        
        # Get device
        if predictions:
            device = list(predictions.values())[0][0].device if isinstance(list(predictions.values())[0], list) else list(predictions.values())[0].device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # Aggregate losses from all HG stages with doubling weights
        total_loss = torch.tensor(0.0, requires_grad=True, device=device)
        
        # Stage 1: backbone_forward returns only VisualField (no Landmarks/Heatmaps)
        # Use simple regression heads to predict landmarks from features
        if "VisualField" in predictions and "Landmarks" not in predictions and "landmarks" in targets:
            visual_fields = predictions["VisualField"]
            landmarks_target = targets["landmarks"]  # [B, 98, 2]
            B = landmarks_target.shape[0]
            landmarks_target_flat = landmarks_target.view(B, -1)  # [B, 196]
            
            # Move regression heads to correct device if needed
            if self.stage1_heads[0][0].training != visual_fields[0].requires_grad:
                for head in self.stage1_heads:
                    head.to(device)
            
            for stage_idx, visual_field in enumerate(visual_fields):
                # Check for NaN in input features
                if torch.isnan(visual_field).any() or torch.isinf(visual_field).any():
                    print(f"WARNING: NaN/Inf in visual_field at stage {stage_idx}")
                    continue
                
                # Predict landmarks from visual features
                landmarks_pred_flat = self.stage1_heads[stage_idx](visual_field)  # [B, 196]
                landmarks_pred = landmarks_pred_flat.view(B, 98, 2)  # [B, 98, 2]
                
                # Check predictions before loss computation
                if torch.isnan(landmarks_pred).any() or torch.isinf(landmarks_pred).any():
                    print(f"WARNING: NaN/Inf in landmarks_pred at stage {stage_idx}")
                    print(f"  visual_field stats: mean={visual_field.mean():.6f}, std={visual_field.std():.6f}")
                    print(f"  visual_field range: [{visual_field.min():.6f}, {visual_field.max():.6f}]")
                    # Skip this stage
                    continue
        
                # Compute coordinate loss (both in pixel space)
                coord_loss = self.coord_loss(landmarks_pred, landmarks_target)
                
                # Check for NaN in coordinate loss
                if torch.isnan(coord_loss) or torch.isinf(coord_loss):
                    print(f"WARNING: NaN/Inf in coord_loss at stage {stage_idx}")
                    print(f"  landmarks_pred range: [{landmarks_pred.min():.6f}, {landmarks_pred.max():.6f}]")
                    print(f"  landmarks_target range: [{landmarks_target.min():.6f}, {landmarks_target.max():.6f}]")
                    continue
                
                stage_loss = self.lambda_coord * coord_loss
                
                weight = 2 ** stage_idx
                total_loss = total_loss + weight * stage_loss
                losses_detail[f"stage_{stage_idx}_coord"] = coord_loss.item()
                losses_detail[f"stage_{stage_idx}_total"] = stage_loss.item()
        
        # Full model: Use coordinate and heatmap losses
        elif "Landmarks" in predictions and "landmarks" in targets:
            # predictions["Landmarks"] is a list of predictions from each HG stage
            landmarks_predictions = predictions["Landmarks"]  # List of [B, L, 2] tensors
            landmarks_target = targets["landmarks"]
            
            # Check for NaN/Inf in targets
            if torch.isnan(landmarks_target).any() or torch.isinf(landmarks_target).any():
                print(f"WARNING: NaN/Inf in landmarks_target before loss computation")
                return torch.tensor(float('nan'), device=landmarks_target.device, dtype=landmarks_target.dtype), {"landmarks": float('nan'), "loss_landmark": float('nan')}
            
            # Aggregate losses across all stages with doubling weights: 1x, 2x, 4x, 8x, ...
            for stage_idx, landmarks_pred in enumerate(landmarks_predictions):
                # Check for NaN/Inf in predictions
                if torch.isnan(landmarks_pred).any() or torch.isinf(landmarks_pred).any():
                    print(f"WARNING: NaN/Inf in landmarks_pred at stage {stage_idx}")
                    return torch.tensor(float('nan'), device=landmarks_pred.device, dtype=landmarks_pred.dtype), {"landmarks": float('nan'), "loss_landmark": float('nan')}
                
                # Compute coordinate loss for this stage (λc * Lh_coord)
                coord_loss = self.coord_loss(landmarks_pred, landmarks_target)
                
                # Check if loss is valid
                if torch.isnan(coord_loss).any() or torch.isinf(coord_loss).any():
                    print(f"WARNING: NaN/Inf in coordinate loss at stage {stage_idx}")
                    print(f"  landmarks_pred stats: min={landmarks_pred.min():.4f}, max={landmarks_pred.max():.4f}")
                    print(f"  landmarks_target stats: min={landmarks_target.min():.4f}, max={landmarks_target.max():.4f}")
                    # Return NaN loss to signal the training loop to skip this batch
                    return torch.tensor(float('nan'), device=landmarks_pred.device, dtype=landmarks_pred.dtype), {"landmarks": float('nan'), "loss_landmark": float('nan')}
                
                stage_loss = self.lambda_coord * coord_loss
                losses_detail[f"stage_{stage_idx}_coord"] = coord_loss.item()
                
                # Add heatmap losses if available (λatt * (Lh_points + Lh_edges))
                heatmap_loss = torch.tensor(0.0, device=landmarks_pred.device)
                if "Heatmaps" in predictions and len(predictions["Heatmaps"]) > stage_idx:
                    # Points heatmaps
                    if "heatmaps_points" in targets:
                        points_pred = predictions["Heatmaps"][stage_idx]
                        points_target = targets["heatmaps_points"]
                        points_loss = self.awing_loss(points_pred, points_target)
                        heatmap_loss = heatmap_loss + points_loss
                        losses_detail[f"stage_{stage_idx}_points"] = points_loss.item()
                    
                    # Edges heatmaps (if available)
                    if "heatmaps_edges" in targets and "HeatmapsEdges" in predictions:
                        edges_pred = predictions["HeatmapsEdges"][stage_idx]
                        edges_target = targets["heatmaps_edges"]
                        edges_loss = self.awing_loss(edges_pred, edges_target)
                        heatmap_loss = heatmap_loss + edges_loss
                        losses_detail[f"stage_{stage_idx}_edges"] = edges_loss.item()
                    
                    stage_loss = stage_loss + self.lambda_att * heatmap_loss
                
                # Apply doubling weight: 2^stage_idx (equivalent to 2^(h-1) where h=stage_idx+1)
                weight = 2 ** stage_idx
                total_loss = total_loss + weight * stage_loss
                losses_detail[f"stage_{stage_idx}_total"] = stage_loss.item()
        
        losses_detail["loss_landmark"] = total_loss.item() if torch.isfinite(total_loss).all() else 0.0
        return total_loss, losses_detail


class CombinedLoss(nn.Module):
    """Combined loss for landmark detection and pose estimation"""
    def __init__(self, num_stages: int = 4, lambda_coord: float = 4.0, 
                 lambda_att: float = 50.0, lambda_p: float = 1.0):
        super(CombinedLoss, self).__init__()
        self.num_stages = num_stages
        self.lambda_p = lambda_p
        self.landmark_loss = LandmarkLoss(num_stages, lambda_coord, lambda_att)
        self.pose_loss = nn.MSELoss()
        self.coord_loss = SmoothL1Loss()

    def forward(self, predictions: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        """Compute combined landmark and pose loss"""
        # Landmark loss
        lnd_loss, lnd_details = self.landmark_loss(predictions, targets)
        losses_detail = lnd_details.copy()

        # Pose loss (if predictions include pose)
        pose_loss_total = torch.tensor(0.0, device=lnd_loss.device, dtype=lnd_loss.dtype)
        
        if 'Pose' in predictions and 'pose' in targets:
            pose_pred = predictions['Pose']
            pose_loss_total = self.lambda_p * self.pose_loss(pose_pred, targets['pose'])
            losses_detail['pose'] = pose_loss_total.item()

        total_loss = lnd_loss + pose_loss_total
        losses_detail['loss_pose'] = pose_loss_total.item()
        losses_detail['loss_total'] = total_loss.item()

        return total_loss, losses_detail