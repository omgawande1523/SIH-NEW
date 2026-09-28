"""Ablation: retrain the diffusion model with the physics loss switched off (~30-50 min on
CPU), save it as models/<source>/stage2_diffusion_nophys.pt, then re-run the Stage 2
evaluation so both variants are scored on identical noise. run_demo.py picks the variant
up automatically afterwards."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sih import config as C  # noqa: E402
from sih.stage2 import diffusion as D  # noqa: E402
from sih.stage2 import pipeline as S2  # noqa: E402

if __name__ == "__main__":
    torch.set_num_threads(4)
    data = D.FineData(C.CACHE / "fine_train.nc", C.CACHE / "fine_static.nc")
    m = D.train_diffusion(data, steps=3000, lam_cons=0.0, lam_conv=0.0)
    torch.save(m.state_dict(), C.MODELS / "stage2_diffusion_nophys.pt")
    S2.evaluate(extra_diffusion={"diffusion_no_physics": m.eval()})
