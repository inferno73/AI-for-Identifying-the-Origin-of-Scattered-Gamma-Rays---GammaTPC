import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# mock if spconv not available (basely for local debugging)
try:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor, SparseConv3d, SubMConv3d, SparseInverseConv3d
    import functools
    MOCK_MODE = False
    print(">> GPU Environment detected. Using Real spconv.")
except ImportError:
    MOCK_MODE = True
    print(">> Spconv not found. Using Dense Fallback (GPU Enabled if available).")

    # Define minimal Mock classes to mimic spconv API for local debugging
    class SparseConvTensor:
        def __init__(self, features, indices, spatial_shape, batch_size):
            self.features = features
            self.indices = indices
            self.spatial_shape = spatial_shape
            self.batch_size = batch_size
        
        @property
        def spatial_size(self):
            return self.spatial_shape

    class MockLayer(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, indice_key=None):
            super().__init__()
            if isinstance(kernel_size, int): kernel_size = (kernel_size,)*3
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            self.stride = stride
            self.indice_key = indice_key

        def forward(self, input_tensor):
            # 1. "Dense-ify": Scatter sparse features to a small dense grid
            # For local mock, we assume small grid size to fit in memory
            # We just take the features and reshape them essentially, strictly for logic flow testing
            
            B = input_tensor.batch_size
            Spatial = input_tensor.spatial_shape
            C_in = input_tensor.features.shape[1]
            
            # Create dense map
            dense_map = torch.zeros((B, C_in, Spatial[0], Spatial[1], Spatial[2]), device=input_tensor.features.device)
            
            # Fill (naive loop for mock simple debugging, or scatter)
            # only for debugging logic as its slow
            idx = input_tensor.indices.long()
            feat = input_tensor.features
            
            # Filter out of bounds indices (if any, due to simplistic mock logic)
            mask = (idx[:,1] < Spatial[0]) & (idx[:,2] < Spatial[1]) & (idx[:,3] < Spatial[2])
            idx = idx[mask]
            feat = feat[mask]
            
            dense_map[idx[:,0], :, idx[:,1], idx[:,2], idx[:,3]] = feat
            
            # 2. Run Dense Conv
            out_dense = self.conv(dense_map)
            
            # 3. "Sparsify": Gather back to sparse tensor
            # For U-Net logic to hold, we need "meaningful" indices. 
            
            # Let's assume output indices are where output is non-zero (or just > small thresh)
            # This mimics "active" output
            b_dim, c_dim, x_dim, y_dim, z_dim = out_dense.shape
            
            
            indices = torch.nonzero(torch.sum(torch.abs(out_dense), dim=1) > -1) # All pixels "active" for now to test shape flow
            
            
            new_features = out_dense.permute(0, 2, 3, 4, 1).reshape(-1, c_dim)
            if indices.shape[0] != new_features.shape[0]:
                 pass
        
            out_features = out_dense[indices[:,0], :, indices[:,1], indices[:,2], indices[:,3]]
            
            new_shape = [x_dim, y_dim, z_dim]
            
            return SparseConvTensor(out_features, indices, new_shape, B)

    # Aliases
    class SparseConv3d(MockLayer): pass
    class SubMConv3d(MockLayer): pass
    class SparseInverseConv3d(MockLayer): 
        def __init__(self, in_channels, out_channels, kernel_size, indice_key=None, bias=True):
            # Inverse conv usually upsamples. 
            # In mock (dense), we user ConvTranspose3d
            super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=bias, indice_key=indice_key) # parameters dummy
            if isinstance(kernel_size, int): kernel_size = (kernel_size,)*3
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=2, padding=0, output_padding=0, bias=bias)
            self.stride = 2

# DATASET
class GammaDataset(Dataset):
    def __init__(self, npz_file, spatial_size=(64,64,64)):
        data = np.load(npz_file, allow_pickle=True)
        self.coords = data['coords']
        self.features = data['features']
        self.labels_origin = data['labels_origin']
        self.labels_direction = data['labels_direction']
        if 'filenames' in data:
            self.filenames = data['filenames']
        else:
            self.filenames = None
            
        self.spatial_size = spatial_size
        
        # Determine center offset to shift coords to [0, spatial_size]
        # Box is 64mm. Center is (0,0,0). So range is -32 to +32.
        # Shift by +32 to get 0..64.
        self.offset = np.array(spatial_size) / 2.0

    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        # 1. Quantize Coords
        # raw coords are centered at 0. Add offset to make them positive indices
        raw_coords = self.coords[idx] + self.offset
        quant_coords = np.floor(raw_coords).astype(np.int32)
        
        # 2. Filter out of bounds
        mask = (quant_coords[:,0] >= 0) & (quant_coords[:,0] < self.spatial_size[0]) & \
               (quant_coords[:,1] >= 0) & (quant_coords[:,1] < self.spatial_size[1]) & \
               (quant_coords[:,2] >= 0) & (quant_coords[:,2] < self.spatial_size[2])
        
        quant_coords = quant_coords[mask]
        feats = self.features[idx][mask]
        
        feats = self.features[idx][mask]
        
        # Normalize Features (N, 4)
        # 0: Charge (Log1p)
        # 1: dQ (ArcSinh - handles negative diffs)
        # 2: Linearity (0-1, keep as is)
        # 3: Density (Log1p)
        
        # We need to handle if feats is just (N, 1) (legacy dataset) or (N, 4)
        if feats.shape[1] == 4:
            feats[:, 0] = np.log1p(feats[:, 0])
            feats[:, 1] = np.arcsinh(feats[:, 1])
            # feats[:, 2] is 0-1
            feats[:, 3] = np.log1p(feats[:, 3])
        else:
             # Fallback for old dataset (N, 1)
             feats = np.log1p(feats)
        
        if len(quant_coords) == 0:
            # Fallback for empty after crop (should happen rarely with correct crop)
            quant_coords = np.zeros((1,3), dtype=np.int32)
            feats = np.zeros((1,1), dtype=np.float32)

        fname = ""
        if self.filenames is not None:
            fname = str(self.filenames[idx])
            
        return quant_coords, feats, self.labels_origin[idx], self.labels_direction[idx], fname

def sparse_collate_fn(batch):
    # Check if batch has 5 items (with filename) or 4
    if len(batch[0]) == 5:
        coords, feats, labels_org, labels_dir, fnames = zip(*batch)
        has_fnames = True
    else:
        coords, feats, labels_org, labels_dir = zip(*batch)
        has_fnames = False
        fnames = []
    
    # Create batch index column for sparse tensor
    batch_coords = []
    for i, c in enumerate(coords):
        # concat [batch_idx, x, y, z]
        b_idx = np.full((len(c), 1), i, dtype=np.int32)
        batch_coords.append(np.hstack([b_idx, c]))
        
    batch_coords = np.vstack(batch_coords)
    batch_feats = np.vstack(feats)
    batch_labels_org = np.vstack(labels_org)
    batch_labels_dir = np.vstack(labels_dir)
    
    if has_fnames:
        return torch.tensor(batch_coords), torch.tensor(batch_feats).float(), \
               torch.tensor(batch_labels_org).float(), torch.tensor(batch_labels_dir).float(), list(fnames)
    else:
        return torch.tensor(batch_coords), torch.tensor(batch_feats).float(), \
               torch.tensor(batch_labels_org).float(), torch.tensor(batch_labels_dir).float()


# Model Architecture

class UnifiedSparseUNet(nn.Module):
    def __init__(self, in_channels, base_filters=16, spatial_size=[64,64,64]):
        super().__init__()
        self.spatial_size = spatial_size
        
        # Encoder
        # Block 1: Input -> 16
        self.enc1 = SubMConv3d(in_channels, base_filters, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        # Down 1: 16 -> 32 (Stride 2)
        self.down1 = SparseConv3d(base_filters, base_filters*2, kernel_size=2, stride=2, bias=False, indice_key='down1')
        
        # Block 2: 32 -> 32
        self.enc2 = SubMConv3d(base_filters*2, base_filters*2, kernel_size=3, padding=1, bias=False, indice_key='subm2')
        # Down 2: 32 -> 64 (Stride 2)
        self.down2 = SparseConv3d(base_filters*2, base_filters*4, kernel_size=2, stride=2, bias=False, indice_key='down2')
        
        # Bottleneck: 64
        self.bot = SubMConv3d(base_filters*4, base_filters*4, kernel_size=3, padding=1, bias=False, indice_key='subm3')
        
        # Decoder
        # Up 1: 64 -> 32
        self.up2 = SparseInverseConv3d(base_filters*4, base_filters*2, kernel_size=2, indice_key='down2', bias=False)
        self.dec2 = SubMConv3d(base_filters*2, base_filters*2, kernel_size=3, padding=1, bias=False, indice_key='subm2')
        
        # Up 2: 32 -> 16
        self.up1 = SparseInverseConv3d(base_filters*2, base_filters, kernel_size=2, indice_key='down1', bias=False)
        self.dec1 = SubMConv3d(base_filters, base_filters, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        
        # Heads
        
        # Head A: Origin (Heatmap / Offset)
        # 16 -> 1 (per voxel scalar)
        self.head_origin = SubMConv3d(base_filters, 1, kernel_size=1, bias=True, indice_key='origin')
        
        # Head B: Direction (Vector)
        # 16 -> 3 (per voxel vector)
        self.head_direction = SubMConv3d(base_filters, 3, kernel_size=1, bias=True, indice_key='direction')
        
        # Head C: Reconstruction (Inverse Diffusion)
        # 16 -> 1 (Binary Classification: Is this voxel part of the sharp track?)
        self.head_recon = SubMConv3d(base_filters, 1, kernel_size=1, bias=True, indice_key='recon')
        
        # Transformer Pooling (for Direction)
        self.pool_proj = nn.Linear(3, 16)
        encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.final_dir = nn.Linear(16, 3)

    def forward(self, input_tensor):
        # Encoder
        x = self.enc1(input_tensor)
        x = self.down1(x) 
        x = self.enc2(x)
        x = self.down2(x)
        x = self.bot(x)
        
        # Decoder (simplified, missing skip concats for brevity in V1, assumed SubM handles dense-state logic via indice_key in real spconv)
        # Note: True U-Net concatenates encoder features. 
        # In spconv, creating tensor with concat features is verbose. 
        # For this version (ResNet-like U-Net), we often just Add (Residual) or skip concat if simple.
        
        x = self.up2(x)
        x = self.dec2(x)
        x = self.up1(x)
        x = self.dec1(x)
        
        # Heads
        # 1. Origin Offset Prediction (Per Voxel)
        out_origin = self.head_origin(x) # (N, 1)
        
        # 2. Direction Prediction
        out_dir = self.head_direction(x) # (N, 3)
        
        # Global Pooling Logic
        # We need to aggregate per-voxel predictions into global predictions per batch.
        
        # Get dense features and indices to do batch-wise pooling
        
        feats_org = out_origin.features
        feats_dir = out_dir.features
        indices = out_origin.indices # (N, 4) -> b, x, y, z
        
        bs = input_tensor.batch_size
        
        pred_global_origins = []
        pred_global_dirs = []
        
        for b in range(bs):
            mask = (indices[:, 0] == b)
            if not mask.any(): 
                # Handle empty track
                pred_global_origins.append(torch.zeros(3, device=x.features.device))
                pred_global_dirs.append(torch.tensor([1,0,0], dtype=torch.float, device=x.features.device))
                continue
                
            # Origin: Weighted Avg (SoftArgMax style)
            # Here feats_org is logits. Sigmoid -> Prob.
            probs = torch.sigmoid(feats_org[mask])
            probs = torch.clamp(probs, min=1e-6, max=1.0) # Safety
            
            # Coords of these voxels
            voxel_coords = indices[mask, 1:].float()
            # Predicted Global Offset = Sum(p * coord) / Sum(p)
            sum_p = torch.sum(probs) + 1e-8 # higher epsilon
            
            # Weighted Centroid (in local quantized coords)
            centroid = torch.sum(probs * voxel_coords, dim=0) / sum_p
            
            # Check for NaNs (Differentiable)
            # Create safety tensor (needs to be on same device)
            # We use torch.where to keep graph intact
            safe_centroid = torch.tensor([32.0, 32.0, 32.0], device=x.features.device)
            is_nan_centroid = torch.isnan(centroid).any()
            # If we don't handle this carefully, backprop might fail on the nan branch anyway.
            # But the error 'requires grad' usually means we replaced a variable with a non-grad tensor.
            # Fix: avoid breaking the chain.
            if is_nan_centroid:
                 # If NaN, we just detach/stop-grad on this sample effectively by using constant
                 centroid = safe_centroid
            
            # Note: The error "element 0 of tensors does not require grad" suggests 'loss' has no grad_fn.
            # This happens if 'pred_org' or 'pred_dir' are replaced by constants that don't track history.
            # To fix, we ensure even fallback has dummy grad or we just accept 'safe_centroid' breaks that specific sample's grad (which is fine).
            # Wait, if we replace 'centroid' with 'safe_centroid' (created with torch.tensor), it has no Grad!
            # So if ALL samples are NaN, loss has no Grad. -> Crash.
            
            # Better fix: Use inputs to generate dummy zero with grad, then add safe value.
            # zero_grad = torch.sum(x.features) * 0.0
            # centroid = centroid + zero_grad 
            
            # Let's try the torch.where approach which is cleaner
            centroid = torch.where(torch.isnan(centroid), safe_centroid, centroid)

            # Since label is "Offset from Center (32,32,32)", and centroid is in (0..64)
            # We want (Centroid - 32).
            # If centroid is at 33, it means true origin is at +1mm relative to noisy center.
            global_off = centroid - (torch.tensor(self.spatial_size, device=x.features.device) / 2.0)
            
            # Scale back if voxel scale != 1mm (assuming 1mm for now)
            pred_global_origins.append(global_off)
            
            # Direction: Transformer Pool
            # Valid voxels
            voxel_dirs = feats_dir[mask] # (M, 3)
            
            if MOCK_MODE:
                # Mock Transformer input
                # Transformers act on (Batch, Seq, Dim). Here Batch=1 (pooling per track).
                toks = self.pool_proj(voxel_dirs).unsqueeze(0) # (1, M, 16)
                # If too many voxels, subsample? Transformer is O(N^2).
                if toks.shape[1] > 500: toks = toks[:, :500, :]
                
                trans_feats = self.transformer(toks) # (1, M, 16)
                # Mean over sequence
                global_feat = torch.mean(trans_feats, dim=1).squeeze(0) # (16)
                final_vec = self.final_dir(global_feat) # (3)
            else:
                 # Real mode logic is same, PyTorch handles it
                toks = self.pool_proj(voxel_dirs).unsqueeze(0)
                if toks.shape[1] > 500: toks = toks[:, :500, :]
                trans_feats = self.transformer(toks)
                global_feat = torch.mean(trans_feats, dim=1).squeeze(0)
                final_vec = self.final_dir(global_feat)

            # Normalize safely
            norm_val = torch.norm(final_vec)
            # Safe norm to avoid div by zero in graph
            safe_norm = torch.clamp(norm_val, min=1e-8)
            final_vec_norm = final_vec / safe_norm
            
            # Handle NaN or small norm (fallback)
            # Again use torch.where to maintain graph flow (if possible) or just constant (breaking flow for that sample)
            safe_vec = torch.tensor([1.0, 0.0, 0.0], device=x.features.device)
            # If norm was too small or nan, use safe_vec
            condition = (norm_val < 1e-8) | torch.isnan(norm_val)
            final_vec = torch.where(condition, safe_vec, final_vec_norm)
                
            
            pred_global_dirs.append(final_vec)
            
        # 3. Reconstruction Output (Sparse)
        out_recon = self.head_recon(x)
            
        return torch.stack(pred_global_origins), torch.stack(pred_global_dirs), out_recon

# Training Loop

def train(args):
    dataset_path = os.path.join(args.dataset_dir, "dataset_train.npz")
    dataset = GammaDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=sparse_collate_fn, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    print(f"Training on {device}")
    
    # Updated to 4 channels for PCA features
    model = UnifiedSparseUNet(in_channels=4, spatial_size=[64,64,64]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    criterion_mse = nn.MSELoss()
    criterion_cos = nn.CosineSimilarity()
    
    min_loss = float('inf')
    min_loss_org = float('inf')
    min_loss_dir = float('inf')
    
    model.train()
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        total_loss_org = 0
        total_loss_dir = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            # Handle variable return length
            if len(batch) == 5:
                coords, feats, lbl_org, lbl_dir, _ = batch
            else:
                coords, feats, lbl_org, lbl_dir = batch
                
            coords = coords.to(device)
            feats = feats.to(device)
            lbl_org = lbl_org.to(device)
            lbl_dir = lbl_dir.to(device)
            
            # Input Safety Checks
            if torch.isnan(feats).any() or torch.isinf(feats).any():
                print(f"WARNING: Batch has NaN/Inf features! Skipping.")
                continue
            if torch.isnan(lbl_dir).any():
                print(f"WARNING: Batch has NaN labels! Skipping.")
                continue
                
            bs = lbl_org.shape[0]
            
            # Create SparseTensor
            input_st = SparseConvTensor(feats, coords, spatial_shape=[64,64,64], batch_size=bs)
            
            # Forward
            pred_org, pred_dir, out_recon = model(input_st)
            
            # --- Losses ---
            # 1. Regression Losses
            loss_org = criterion_mse(pred_org, lbl_org)
            loss_dir = torch.mean(1 - criterion_cos(pred_dir, lbl_dir))
            
            # 2. Reconstruction Loss (Inverse Diffusion)
            # Generate Target Mask on the fly: pixels close to the true line are 1, else 0
            
            # Get Voxel Coordinates of the active sites
            # indices: (N_active, 4) -> [batch, x, y, z]
            # features: (N_active, 1) -> logits
            recon_indices = out_recon.indices
            recon_logits = out_recon.features.squeeze(1)
            
            # Prepare Batch Data for vectorized distance calc
            batch_idx = recon_indices[:, 0].long()
            voxel_coords = recon_indices[:, 1:].float() # (x, y, z)
            
            # Get True Origin and Direction for each voxel based on its batch index
            # lbl_org is offset from center (32,32,32). 
            # So True Line Origin = (32,32,32) + lbl_org[b]
            center_offset = torch.tensor([32.0, 32.0, 32.0], device=device)
            true_origins = center_offset + lbl_org[batch_idx]
            true_dirs = lbl_dir[batch_idx]
            
            # Point-to-Line Distance
            # Vector from LineOrigin to Voxel
            vec_PV = voxel_coords - true_origins
            
            # Cross Product of PV and Dir
            # Distance = |PV x Dir| / |Dir| (Dir is unit, so just norm of cross)
            cross_prod = torch.cross(vec_PV, true_dirs, dim=1)
            dists = torch.norm(cross_prod, dim=1)
            
            # Target: 1 if dist < Radius (e.g. 1.5mm - sharp core), 0 otherwise
            # This teaches the model to suppress the "fuzz" (diffusion) and highlight the core
            # Target: 1 if dist < Radius (e.g. 1.5mm - sharp core), 0 otherwise
            recon_target = (dists < 1.5).float()
            
            # --- STABILITY FIX ---
            if torch.isnan(recon_logits).any():
                print("WARNING: recon_logits contain NaNs BEFORE clamp!")
            
            # Clamp logits to prevent explosion in exp() inside BCE
            recon_logits = torch.clamp(recon_logits, min=-10.0, max=10.0)
            
            criterion_bce = nn.BCEWithLogitsLoss()
            loss_recon = criterion_bce(recon_logits, recon_target)
            
            # Clamp regression losses individually for safety
            loss_org = torch.clamp(loss_org, max=100.0)
            
            # Total Loss (Reduced weight)
            loss = loss_org + loss_dir + 1.0 * loss_recon # Reduced from 5.0

            if torch.isnan(loss):
                print(f"WARNING: NaN Loss detected! Org:{loss_org.item():.4f}, Dir:{loss_dir.item():.4f}, Rec:{loss_recon.item():.4f}")
                # Check for Model Death
                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"CRITICAL: Model parameter {name} has NaNs! Training is broken.")
                        sys.exit(1)
                
                optimizer.zero_grad()
                continue
            
            optimizer.zero_grad()
            loss.backward()
            # Gradient Clipping (Prevent explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_loss_org += loss_org.item()
            total_loss_dir += loss_dir.item()
            pbar.set_postfix({'loss': loss.item(), 'org': loss_org.item(), 'dir': loss_dir.item()})
       
        # Checkpointing
        epoch_avg_loss = total_loss / len(dataloader)
        epoch_avg_loss_org = total_loss_org / len(dataloader)
        epoch_avg_loss_dir = total_loss_dir / len(dataloader)
        
        print(f"Epoch {epoch+1} Loss: {epoch_avg_loss:.4f} (Avg Org: {epoch_avg_loss_org:.4f}, Avg Dir: {epoch_avg_loss_dir:.4f})")
        
        # 1. Best Overall
        if epoch_avg_loss < min_loss:
            min_loss = epoch_avg_loss
            
            # Backup
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_name = f"model_best_{timestamp}_loss_{min_loss:.4f}.pth"
            backup_path = os.path.join(args.dataset_dir, backup_name)
            torch.save(model.state_dict(), backup_path)
            
            # Main Link
            save_path = os.path.join(args.dataset_dir, "model_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New Best Overall Logic! Loss: {min_loss:.4f}")

        # 2. Best Origin
        if epoch_avg_loss_org < min_loss_org:
            min_loss_org = epoch_avg_loss_org
            save_path = os.path.join(args.dataset_dir, "model_best_origin.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New Best Origin Model! Loss: {min_loss_org:.4f}")

        # 3. Best Direction
        if epoch_avg_loss_dir < min_loss_dir:
            min_loss_dir = epoch_avg_loss_dir
            save_path = os.path.join(args.dataset_dir, "model_best_direction.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New Best Direction Model! Loss: {min_loss_dir:.4f}")
            min_loss = epoch_avg_loss
            
            # 1. Save Backup
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_name = f"model_best_{timestamp}_loss_{min_loss:.4f}.pth"
            backup_path = os.path.join(args.dataset_dir, backup_name)
            torch.save(model.state_dict(), backup_path)
            
            # 2. Update Main Link
            save_path = os.path.join(args.dataset_dir, "model_best.pth")
            torch.save(model.state_dict(), save_path)
            
            print(f"New Best Model Saved! Loss: {min_loss:.4f}")
            print(f"  -> Backup: {os.path.abspath(backup_path)}")
            print(f"  -> Main:   {os.path.abspath(save_path)}")
            
    print("Training finished.")
    final_path = os.path.join(args.dataset_dir, "model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Saved Final Model at {os.path.abspath(final_path)}")
    
    # DEBUG: List files to prove existence to user
    print("-" * 30)
    print(f"Contents of {args.dataset_dir}:")
    try:
        # Cross-platform list
        files = os.listdir(args.dataset_dir)
        for f in files:
            full = os.path.join(args.dataset_dir, f)
            stat = os.stat(full)
            size = stat.st_size / 1024 # KB
            mtime = time.ctime(stat.st_mtime)
            print(f"  {f:<30} {size:>8.1f} KB  Last Mod: {mtime}")
    except Exception as e:
        print(f"Could not list directory: {e}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='AI_model_method/dataset')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5) # Reduced from 1e-4
    parser.add_argument('--mock', action='store_true', help="Force mock mode")
    args = parser.parse_args()
    
    train(args)
