import hashlib,json,tarfile,time
from pathlib import Path
root=Path('/home/campus/zhangweihao/FLASH1')
manifest=json.loads((root/'artifacts/registry-manifest.json').read_text())
expected={item['digest'].split(':')[1]:item['size'] for item in manifest['layers']}
config=manifest['config']['digest'].split(':')[1]
expected[config]=manifest['config']['size']
verified=[]
with tarfile.open('/home/test/greyhound-ae-linux-amd64.tar','r|') as archive:
    for member in archive:
        name=member.name.split('/')[-1].removesuffix('.tar.gz').removeprefix('sha256:')
        if name not in expected: continue
        assert member.size==expected[name],member.name
        digest=hashlib.sha256()
        f=archive.extractfile(member)
        while chunk:=f.read(8*1024*1024): digest.update(chunk)
        assert digest.hexdigest()==name,member.name
        verified.append(name)
        print(f'verified {len(verified)}/65 {name}',flush=True)
assert set(expected)==set(verified)
(root/'results/image-verification.json').write_text(json.dumps({'all_blobs_verified':True,'count':len(verified),'manifest_digest':hashlib.sha256((root/'artifacts/registry-manifest.json').read_bytes()).hexdigest()},indent=2))
