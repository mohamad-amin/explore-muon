import json
from pathlib import Path
import sys,time
p=Path(__file__).resolve().parent/"PROCESS_EXIT.json"
with p.open("x") as stream:
 json.dump({"exit_code":int(sys.argv[1]),"finished_unix":time.time(),"gpu_requested":False},stream,indent=2)
 stream.write("\n")
