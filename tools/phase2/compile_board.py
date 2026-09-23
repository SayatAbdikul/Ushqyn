#!/usr/bin/env python3
"""Lower a checked software-v2 VM image into the supported hardware ABI."""
import argparse,hashlib,json,os,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'compiler'))
from program_image import load_image
from hardware_v2 import lower
p=argparse.ArgumentParser();p.add_argument('software_image',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
source=a.software_image.read_bytes();blob,metadata=lower(load_image(source))
metadata.update(image_sha256=hashlib.sha256(blob).hexdigest(),software_image_sha256=hashlib.sha256(source).hexdigest())
a.output.parent.mkdir(parents=True,exist_ok=True)
fd,tmp=tempfile.mkstemp(dir=a.output.parent,prefix=a.output.name+'.')
try:
    with os.fdopen(fd,'wb') as f:f.write(blob)
    os.replace(tmp,a.output)
finally:
    if os.path.exists(tmp):os.unlink(tmp)
a.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(f'{a.output}: {metadata["used_bytes"]} / {len(blob)} SRAM bytes, {metadata["target"]}')
