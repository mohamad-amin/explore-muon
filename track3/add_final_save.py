"""Append an after-training save of the final weights (and PD input statistics) to a Track 3 script.
Analysis only: the save runs after the last validation, so training is unchanged.
Usage: python add_final_save.py SRC DST"""
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_text()
old = "\ndist.destroy_process_group()\n"
assert text.count(old) == 1 and "_final.pt" not in text
save = '''
if dist.get_rank() == 0:  # analysis only: final weights (and PD input statistics) for offline probes
    pd_stats = {name: {"cov": m.pd_cov.cpu(), "weight": m.pd_weight.cpu()}
                for name, m in model.named_modules() if hasattr(m, "pd_cov")}
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "pd_stats": pd_stats},
               logfile[:-4] + "_final.pt")
'''
dst.write_text(text.replace(old, save + old))
print(f"{dst}: +{len(save.splitlines())} lines")
