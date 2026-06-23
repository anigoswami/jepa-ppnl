"""Minimal Text-LeWorldModel: end-to-end next-embedding MSE + SIGReg for text sequences."""
import math, random, os
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def param_groups(modules, wd):
    np_ = [(n, p) for m in modules for n, p in m.named_parameters() if p.requires_grad]
    nd = [p for n, p in np_ if p.ndim < 2 or n.endswith("bias")]
    d = [p for n, p in np_ if p.ndim >= 2 and not n.endswith("bias")]
    return [{"params": d, "weight_decay": wd}, {"params": nd, "weight_decay": 0.0}]


def sincos_1d(n, dim):
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.) / dim))
    pe = torch.zeros(n, dim); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


class SIGReg(nn.Module):
    """Single-GPU SIGReg / Epps-Pulley statistic with Gaussian-windowed quadrature."""
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        w = torch.full((knots,), 2 * dt, dtype=torch.float32); w[[0, -1]] = dt   # trapezoid weights
        phi = torch.exp(-t.square() / 2.0)                                       # N(0,1) char fn at knots
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", w * phi)

    def forward(self, proj):                                                     # proj: (T, B, D)
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)        # fresh projections per step
        A = A.div_(A.norm(p=2, dim=0))                                           # unit-norm
        x_t = (proj @ A).unsqueeze(-1) * self.t                                  # (T, B, P, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()                     # mean(-3) averages over batch B


class Projector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        shape = x.shape
        y = self.linear(x.reshape(-1, shape[-1]))
        if self.training and y.size(0) == 1:
            y = F.batch_norm(y, self.bn.running_mean, self.bn.running_var,
                             self.bn.weight, self.bn.bias, training=False)
        else:
            y = self.bn(y)
        return y.view(shape)


class CondBlock(nn.Module):
    def __init__(self, dim, heads, mlp=4.0):
        super().__init__()
        self.heads = heads; self.dh = dim // heads
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Linear(int(dim * mlp), dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.ada[-1].weight, 0); nn.init.constant_(self.ada[-1].bias, 0)  # AdaLN-zero init

    def _attn(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).reshape(B, T, D))

    def forward(self, x, c):
        sa, ka, ga, sm, km, gm = self.ada(c).chunk(6, dim=-1)           # AdaLN-zero modulators from context c
        x = x + ga * self._attn(self.n1(x) * (1 + ka) + sa)
        x = x + gm * self.mlp(self.n2(x) * (1 + km) + sm)
        return x


class TextEncoder(nn.Module):
    """Maps frozen 384-dim sentence vectors down to our internal world model dimensions."""
    def __init__(self, in_dim=384, dim=128):
        super().__init__()
        self.dim = dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.projector = Projector(dim)

    def forward(self, text_vectors):
        B, T, D = text_vectors.shape
        x = self.fc(text_vectors.reshape(-1, D))
        return self.projector(self.norm(x).view(B, T, self.dim))


class ARPredictor(nn.Module):
    def __init__(self, num_frames, dim=128, depth=4, heads=4, action_dim=2):
        super().__init__()
        self.act_proj = nn.Sequential(nn.Linear(action_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.register_buffer("time_pe", sincos_1d(num_frames, dim))
        self.blocks = nn.ModuleList([CondBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.projector = Projector(dim)

    def forward(self, z, a):                                          # z: (B, T, D); a: (B, T, action_dim)
        x = z + self.time_pe[None, :z.size(1)]
        c = self.act_proj(a)
        for blk in self.blocks: x = blk(x, c)
        return self.projector(self.norm(x))


class TextSequenceDataset(Dataset):
    """Scans an entire directory of text files, vectorizes them independently, 
    and pools them together to create a multi-domain replay buffer."""
    
    def __init__(self, dirpath, n_frames=3):
        super().__init__()
        self.n_frames = n_frames
        
        print(f"--> Scanning directory: {dirpath} for training data...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get all text files in the target directory
        txt_files = [os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.endswith(".txt")]
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in your data folder: {dirpath}")
            
        all_sequences = []
        all_actions = []
        
        for filepath in txt_files:
            print(f"    Processing: {os.path.basename(filepath)}")
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                
            if len(lines) < n_frames:
                continue  # Skip files that don't have enough lines to form a sequence
                
            # Vectorize this specific file's lines offline
            embeddings = embedder.encode(lines, convert_to_tensor=True, show_progress_bar=False).cpu()
            
            # Build clean chronological windows ONLY within this file's boundaries
            for i in range(len(embeddings) - n_frames + 1):
                all_sequences.append(embeddings[i : i + n_frames])
                
                # Action Vector: Alternating binary indicator columns for turn shifts
                act = torch.zeros(n_frames, 2)
                for t in range(n_frames):
                    act[t] = torch.tensor([1.0, 0.0]) if (i + t) % 2 == 0 else torch.tensor([0.0, 1.0])
                all_actions.append(act)
        #Ensure data was actually discovered before stacking ---
        if not all_sequences:
            raise ValueError(f"Dataset Build Failed: No valid text trajectories could be formed from files in '{dirpath}'. Check your text file formatting.")        
        
        # Stack everything into master tensors
        self.sequences = torch.stack(all_sequences)
        self.actions = torch.stack(all_actions)
        print(f"--> Total training trajectories assembled: {len(self.sequences)}")

    def __len__(self): return len(self.sequences)
    def __getitem__(self, i): return self.sequences[i], self.actions[i]

def train(epochs=10, batch_size=64, lr=3e-4, wd=0.01, sigreg_w=0.2, device=None):
    device = device or pick_device(); print(f"device: {device}")
    
    data_dir = "./data/usr_data"
    #make directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)

    ds = TextSequenceDataset(dirpath=data_dir, n_frames=3)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    
    encoder = TextEncoder(in_dim=384, dim=128).to(device)
    predictor = ARPredictor(num_frames=ds.n_frames, dim=encoder.dim, action_dim=2).to(device)
    sigreg = SIGReg().to(device)
    
    checkpoint_path = "./data/usr_data/leworld_text.pt"
    if os.path.exists(checkpoint_path):
        print(f"--> Found existing brain at {checkpoint_path}. Loading weights for fine-tuning...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        predictor.load_state_dict(checkpoint['predictor_state_dict'])
    else:
        print("--> No existing brain found. Initializing brand new random synapses...")

    opt = torch.optim.AdamW(param_groups([encoder, predictor], wd), lr=lr)

    losses = {"pred": [], "sigreg": [], "total": [], "zero_act": []}
    step = 0
    
    print("\nTraining LeWorldModel on Text Semantics...")
    for epoch in range(epochs):
        for text_vecs, actions in loader:
            text_vecs = text_vecs.to(device); actions = actions.to(device)
            
            emb = encoder(text_vecs)                                           # z (B, T, D)
            ctx_z, ctx_a = emb[:, :-1], actions[:, :-1]                       # Current states & speaker actions
            pred = predictor(ctx_z, ctx_a)                                    # Predict next-turn embeddings
            
            tgt = emb[:, 1:]                                                  # Un-detached target vectors
            pred_loss = (pred - tgt).pow(2).mean()                            # L_pred
            sr = sigreg(emb.transpose(0, 1))                                  # L_sigreg
            loss = pred_loss + sigreg_w * sr
            
            opt.zero_grad(); loss.backward(); opt.step()
            
            with torch.no_grad():                                             # Check baseline error without speaker profile
                pz = predictor(ctx_z, torch.zeros_like(ctx_a))
                zero_act = (pz - tgt).pow(2).mean().item()
            losses["pred"].append(pred_loss.item())
            losses["sigreg"].append(sr.item())
            losses["total"].append(loss.item())
            losses["zero_act"].append(zero_act)     
            if step % 5 == 0:
                gap = zero_act - pred_loss.item()
                print(f"ep={epoch} step={step:4d} pred={pred_loss.item():.4f} "
                      f"sigreg={sr.item():.4f} neutral_ctx={zero_act:.4f} gap={gap:+.4f}")
            step += 1
        print(f"--> Retaining memory: Saving 'leworld_text.pt' at epoch {epoch}...")
        torch.save({
            'epoch': epoch,
            'step': step,
            'encoder_state_dict': encoder.state_dict(),
            'predictor_state_dict': predictor.state_dict(),
        }, checkpoint_path)

    # This is the final return statement at the very end of the function
    return {"encoder": encoder, "predictor": predictor, "sigreg": sigreg,
            "losses": losses, "loader": loader, "device": device}


if __name__ == "__main__":
    train()