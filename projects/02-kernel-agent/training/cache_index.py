"""Rebuild a missing shard index privately; never write into the HF cache."""
import argparse
import json
from pathlib import Path
import struct


def build(snapshot,output):
    snapshot=Path(snapshot).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    index={};total=0
    for p in snapshot.iterdir():
        if not p.is_file():continue
        (output/p.name).symlink_to(p.resolve())
        if p.suffix!='.safetensors':continue
        with p.open('rb') as f:
            size=struct.unpack('<Q',f.read(8))[0]
            if size>100000000:raise ValueError('Unexpected tensor header size')
            header=json.loads(f.read(size))
        for name,tensor in header.items():
            if name=='__metadata__':continue
            if name in index:raise ValueError('Duplicate tensor key')
            index[name]=p.name;total+=tensor['data_offsets'][1]-tensor['data_offsets'][0]
    if not index:raise ValueError('No cached weights found')
    destination=output/'model.safetensors.index.json'
    if destination.is_symlink():destination.unlink() # Only our newly created private link.
    destination.write_text(json.dumps({'metadata':{'total_size':total},'weight_map':index},indent=2))
    return {'tensor_count':len(index),'total_bytes':total,'snapshot':str(snapshot),'shared_cache_modified':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(build(a.snapshot,a.output))
