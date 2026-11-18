"""Shared, fixed protocol for laboratory 3. No test loading during training."""
import csv, hashlib, json, os, platform, subprocess
from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('TORCH_HOME', str(ROOT / 'cache'))
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
MEAN, STD = (.485,.456,.406), (.229,.224,.225)
RUNS = [('scratch',1)] + [(r,s) for r in ['frozen','unfreeze'] for s in [1,2,3]]
MODULES = ['conv1','bn1','layer1','layer2','layer3','layer4','fc']
EXPECTED = {
 'train': [(200,1996761),(200,2032625),(200,2183923),(200,1950345),(200,1987613),(200,1841817)],
 'validation': [(218,4837323),(237,5276962),(276,6135652),(262,5789535),(255,5655068),(250,5541845)],
 'test': [(219,4837559),(237,5291326),(277,6157250),(263,5849765),(255,5668011),(251,5547533)]}

def dump(path, obj):
 Path(path).parent.mkdir(parents=True, exist_ok=True)
 Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dirs(data):
 data=Path(data)
 found={}
 for name in ['seg_train','seg_test']:
  candidates=[data/name/name, data/name, data]
  found[name]=next((p for p in candidates if (p/'buildings').is_dir()), None)
  if found[name] is None: raise FileNotFoundError(f'{name} missing under {data}')
 return found

def make_split(data):
 folders=dirs(data); rows=[]
 for source in ['seg_train','seg_test']:
  rng=np.random.default_rng(0)
  for label in CLASSES:
   files=sorted(p.name for p in (folders[source]/label).glob('*.jpg'))
   idx=np.arange(len(files)); rng.shuffle(idx)
   groups=[('train',idx[:200])] if source=='seg_train' else [('validation',idx[:len(idx)//2]),('test',idx[len(idx)//2:])]
   for part,indices in groups:
    rows += [dict(file=files[i],label=label,part=part) for i in indices]
 with (ROOT/'split.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['file','label','part']);w.writeheader();w.writerows(rows)
 return check_split()

def read_split():
 with (ROOT/'split.csv').open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))

def check_split():
 rows=read_split(); assert len({r['file'] for r in rows})==len(rows)==4200
 table=[]
 for part in EXPECTED:
  for k,label in enumerate(CLASSES):
   subset=[r for r in rows if r['part']==part and r['label']==label]
   actual=(len(subset),sum(int(Path(r['file']).stem) for r in subset))
   assert actual==EXPECTED[part][k],(part,label,actual,EXPECTED[part][k])
   table.append(dict(part=part,label=label,n=actual[0],sum=actual[1],matches=True))
 return table

class Images(Dataset):
 def __init__(self,data,part):
  self.rows=[r for r in read_split() if r['part']==part]
  self.folder=dirs(data)['seg_train' if part=='train' else 'seg_test']
  ops=[T.RandomResizedCrop(160,scale=(.7,1.)),T.RandomHorizontalFlip()] if part=='train' else [T.Resize((160,160))]
  self.transform=T.Compose(ops+[T.ToTensor(),T.Normalize(MEAN,STD)])
 def __len__(self): return len(self.rows)
 def path(self,i):
  r=self.rows[i]; return self.folder/r['label']/r['file']
 def __getitem__(self,i):
  with Image.open(self.path(i)) as im: x=self.transform(im.convert('RGB'))
  return x,CLASSES.index(self.rows[i]['label'])

def loader(data,part):
 return DataLoader(Images(data,part),batch_size=32 if part=='train' else 64,shuffle=part=='train',drop_last=False,num_workers=0)

def build(regime,seed,device='cpu'):
 torch.manual_seed(seed); np.random.seed(seed)
 model=resnet18(weights=None if regime=='scratch' else ResNet18_Weights.IMAGENET1K_V1)
 model.fc=torch.nn.Linear(512,6)
 for name,p in model.named_parameters():
  p.requires_grad_(regime=='scratch' or name.startswith('fc.') or (regime=='unfreeze' and name.startswith('layer4.')))
 model=model.to(device)
 if regime=='scratch': groups=[dict(params=list(model.parameters()),lr=.003)]
 else:
  groups=[dict(params=list(model.fc.parameters()),lr=.003)]
  if regime=='unfreeze': groups.append(dict(params=list(model.layer4.parameters()),lr=.0003))
 opt=torch.optim.AdamW(groups,weight_decay=.0001)
 assert all(p.requires_grad for g in opt.param_groups for p in g['params'])
 assert sum(p.numel() for p in model.parameters())==11179590
 assert sum(p.numel() for p in model.parameters() if p.requires_grad)==dict(scratch=11179590,frozen=3078,unfreeze=8396806)[regime]
 return model,opt

def set_train_mode(model,regime):
 model.train()
 frozen=[] if regime=='scratch' else ['conv1','bn1','layer1','layer2','layer3']+(['layer4'] if regime=='frozen' else [])
 for name in frozen: getattr(model,name).eval()

@torch.no_grad()
def logits(model,dl,device='cpu'):
 model.eval()
 return torch.cat([model(x.to(device)).cpu() for x,y in dl]).numpy()

def labels(dl): return np.array([CLASSES.index(r['label']) for r in dl.dataset.rows])

def provenance():
 def git(*args):
  try:
   p=subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}',*args],cwd=ROOT,capture_output=True,text=True)
  except OSError:
   return None
  return p.stdout.strip() if p.returncode==0 else None
 commit=git('rev-parse','HEAD'); status=git('status','--porcelain')
 return dict(commit=commit,dirty=None if status is None else bool(status),
   git_note=None if commit else 'Source folder is not a Git repository; source SHA256 recorded instead.',
   source_sha256={p.name:sha(p) for p in sorted(ROOT.glob('*.py'))},
   python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,
   platform=platform.platform(),processor=platform.processor(),threads=torch.get_num_threads())

def settings(model,opt,regime):
 return dict(img_size=160,classes=CLASSES,dtype='float32',weights='None' if regime=='scratch' else 'IMAGENET1K_V1',
  train_transform=dict(crop_size=160,scale=[.7,1.],ratio=[.75,4/3],interpolation='BILINEAR',antialias=True,horizontal_flip_p=.5,mean=MEAN,std=STD),
  eval_transform=dict(resize=[160,160],interpolation='BILINEAR',max_size=None,antialias=True,mean=MEAN,std=STD),
  loader=dict(train_batch=32,eval_batch=64,shuffle_train=True,shuffle_eval=False,drop_last=False,num_workers=0,generator=None,sampler=None,pin_memory=False,persistent_workers=False),
  epochs=8,steps_per_epoch=38,steps=304,optimizer='AdamW',optimizer_groups=[dict(numel=sum(p.numel() for p in g['params']),**{k:v for k,v in g.items() if k!='params'}) for g in opt.param_groups],
  loss=dict(name='CrossEntropyLoss',weight=None,ignore_index=-100,reduction='mean',label_smoothing=0.),bn=dict(momentum=.1,eps=1e-5,affine=True,track_running_stats=True),
  total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
  scheduler=None,gradient_clipping=None,early_stopping=False,checkpoint_selection='epoch8',mixed_precision=False,
  fc_initialization='Linear default U(-1/sqrt(512),1/sqrt(512)); weights and bias',
  frozen_eval_modules=[] if regime=='scratch' else MODULES[:-1 if regime=='frozen' else -2])

def select(runs):
 means={r:float(np.mean([x['epochs'][-1]['val_macro_f1'] for x in runs if x['regime']==r])) for r in ['scratch','frozen','unfreeze']}
 regime=max(means,key=means.get)
 # Stable tie order: scratch, frozen, unfreeze; then smallest seed.
 winner=max((x for x in runs if x['regime']==regime),key=lambda x:(x['epochs'][-1]['val_macro_f1'],-x['seed']))
 return winner,means


def device_name(device):
 kind = torch.device(device).type
 if kind == 'cuda': return torch.cuda.get_device_name(torch.device(device))
 if kind == 'mps': return 'Apple Metal (MPS)'
 return platform.processor() or platform.machine() or 'CPU'

def validate_device(device):
 kind = torch.device(device).type
 if kind not in ('cpu', 'cuda', 'mps'):
  raise ValueError('Choose cpu, cuda, or mps.')
 if kind == 'cuda' and not torch.cuda.is_available():
  raise RuntimeError('CUDA is unavailable. Use --device cpu.')
 if kind == 'mps' and not torch.backends.mps.is_available():
  raise RuntimeError('MPS is unavailable on this Mac/Python environment. Use --device cpu.')
 return device_name(device)
