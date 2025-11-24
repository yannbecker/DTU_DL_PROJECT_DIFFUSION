import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import anndata as ad
import scanpy as sc
import numpy as np

class ScimilarityVAE(nn.Module):
    """
    VAE adapted from scimilarity for bulk RNA-seq data
    Based on scDiffusion's implementation
    """
    def __init__(self, num_genes, latent_dim=128, hidden_dims=[512, 256]):
        super(ScimilarityVAE, self).__init__()
        
        # Encoder
        encoder_layers = []
        input_dim = num_genes
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        input_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], num_genes))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def load_scimilarity_weights(vae_model, scimilarity_checkpoint_path):
    """
    Load pretrained scimilarity weights into VAE
    You need to download annotation_model_v1.pt from 
    https://zenodo.org/records/8286452
    """
    checkpoint = torch.load(scimilarity_checkpoint_path, map_location='cpu')
    
    # scimilarity checkpoint structure may vary
    # You'll need to inspect the checkpoint and map weights appropriately
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Map compatible weights from scimilarity to your VAE
    # This is a simplified example - adjust based on actual checkpoint structure
    vae_state = vae_model.state_dict()
    pretrained_dict = {}
    
    for k, v in state_dict.items():
        # Map encoder weights
        if 'encoder' in k and k in vae_state:
            if v.shape == vae_state[k].shape:
                pretrained_dict[k] = v
    
    # Update model with pretrained weights
    vae_state.update(pretrained_dict)
    vae_model.load_state_dict(vae_state, strict=False)
    
    print(f"Loaded {len(pretrained_dict)} pretrained weight tensors")
    return vae_model

def train_vae(adata, num_genes, max_steps=200000, batch_size=256, 
              lr=1e-4, device='cuda', scimilarity_ckpt=None):
    """
    Train VAE on bulk data with optional scimilarity initialization
    Following scDiffusion's VAE_train.py approach
    """
    # Prepare data
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()
    
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=4, pin_memory=True)
    
    # Initialize model
    vae = ScimilarityVAE(num_genes=num_genes).to(device)
    
    # Load scimilarity weights if provided
    if scimilarity_ckpt is not None:
        vae = load_scimilarity_weights(vae, scimilarity_ckpt)
    
    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    
    # Training loop
    vae.train()
    step = 0
    
    while step < max_steps:
        for batch_idx, (batch_x,) in enumerate(dataloader):
            batch_x = batch_x.to(device)
            
            # Forward pass
            recon_x, mu, logvar = vae(batch_x)
            
            # Loss computation
            # Reconstruction loss (MSE for log-normalized data)
            recon_loss = nn.MSELoss(reduction='sum')(recon_x, batch_x) / batch_x.size(0)
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss /= batch_x.size(0)
            
            # Total loss with KL annealing
            beta = min(1.0, step / 50000)  # KL annealing
            loss = recon_loss + beta * kl_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            
            # Logging
            if step % 1000 == 0:
                print(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | "
                      f"Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f}")
            
            # Save checkpoint
            if step % 10000 == 0 and step > 0:
                torch.save({
                    'step': step,
                    'model_state_dict': vae.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'vae_checkpoint_step_{step}.pt')
            
            step += 1
            if step >= max_steps:
                break
    
    return vae

# Usage
adata_bulk_genes = ad.read_h5ad('/work3/s193518/scIsoPred/data/bulk_processed_genes.h5ad')

# Preprocess
adata_bulk_genes.layers['counts'] = adata_bulk_genes.X.copy()
sc.pp.normalize_total(adata_bulk_genes, target_sum=1e4)
sc.pp.log1p(adata_bulk_genes)

# Train VAE (with optional scimilarity initialization)
num_genes = adata_bulk_genes.n_vars  # 45263

vae = train_vae(
    adata_bulk_genes,
    num_genes=num_genes,
    max_steps=200000,
    batch_size=256,
    device='cuda',
    scimilarity_ckpt='annotation_model_v1.pt'  # Optional: set to None to train from scratch
)

# Get embeddings
vae.eval()
with torch.no_grad():
    X = torch.FloatTensor(adata_bulk_genes.X.toarray() if hasattr(adata_bulk_genes.X, 'toarray') else adata_bulk_genes.X)
    X = X.to('cuda')
    mu, logvar = vae.encode(X)
    embeddings = mu.cpu().numpy()

adata_bulk_genes.obsm['X_vae'] = embeddings
adata_bulk_genes.write_h5ad('bulk_genes_with_embeddings.h5ad')
