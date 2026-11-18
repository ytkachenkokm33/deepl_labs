"""uv run python train.py --data PATH [--device cpu]"""
import argparse, copy, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from common import *

def checks(data,device):
 result=dict(split=check_split(),initial_losses={},parameter_groups={},changes=[],bn_prediction='With plain model.train(), backbone parameters stay fixed but BN buffers change; training loss alone does not reliably detect this error.')
 loss_fn=torch.nn.CrossEntropyLoss()
 for regime in ['scratch','frozen','unfreeze']:
  model,opt=build(regime,1,device)
  result['parameter_groups'][regime]=settings(model,opt,regime)
  if regime in ['scratch','frozen']:
   set_train_mode(model,regime); x,y=next(iter(loader(data,'train')))
   with torch.no_grad(): result['initial_losses'][regime]=loss_fn(model(x.to(device)),y.to(device)).item()
   if regime=='scratch':
    fig,axes=plt.subplots(4,4,figsize=(8,8))
    pixels=(x[:16]*torch.tensor(STD)[None,:,None,None]+torch.tensor(MEAN)[None,:,None,None]).clamp(0,1)
    for i,ax in enumerate(axes.flat): ax.imshow(pixels[i].permute(1,2,0));ax.set_title(CLASSES[y[i]],fontsize=10);ax.axis('off')
    fig.tight_layout();fig.savefig(ROOT/'figures/batch.png',dpi=180);plt.close(fig)
 for regime,bug in [('frozen',False),('unfreeze',False),('frozen',True)]:
  model,opt=build(regime,1,device);before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  if bug: model.train()
  else: set_train_mode(model,regime)
  dl=iter(loader(data,'train')); losses=[]
  for step in range(5):
   x,y=next(dl);opt.zero_grad();loss=loss_fn(model(x.to(device)),y.to(device));loss.backward();opt.step();losses.append(loss.item())
  after={k:v.detach().cpu() for k,v in model.state_dict().items()}
  pn=set(dict(model.named_parameters()))
  for name in MODULES:
   keys=[k for k in before if k.startswith(name+'.')]
   params=any(not torch.equal(before[k],after[k]) for k in keys if k in pn)
   bkeys=[k for k in keys if 'running_mean' in k or 'running_var' in k]
   stats=any(not torch.equal(before[k],after[k]) for k in bkeys) if bkeys else None
   counters=[k for k in keys if 'num_batches_tracked' in k]
   counter=any(not torch.equal(before[k],after[k]) for k in counters) if counters else None
   expected_params=name=='fc' or (regime=='unfreeze' and name=='layer4')
   expected_stats=(bug or (regime=='unfreeze' and name=='layer4')) if bkeys else None
   ok=params==expected_params and stats==expected_stats and counter==expected_stats
   assert ok,(regime,bug,name,params,stats,counter)
   result['changes'].append(dict(regime=regime,bug=bug,module=name,parameters_changed=params,bn_changed=stats,counter_changed=counter,matches=ok))
  if bug:
   result['bn_bug']={s:float((after['layer1.0.bn1.'+s]-before['layer1.0.bn1.'+s]).abs().max()) for s in ['running_mean','running_var']}
   result['bn_bug']['five_losses']=losses
 dump(ROOT/'checks.json',result)
 print('All split, parameter and BN checks passed',flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--device',default='cpu');ap.add_argument('--threads',type=int,default=6);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 validate_device(args.device)
 torch.set_num_threads(args.threads)
 (ROOT/'figures').mkdir(exist_ok=True);(ROOT/'runs').mkdir(exist_ok=True)
 make_split(args.data)
 if not args.resume or not (ROOT/'checks.json').exists(): checks(args.data,args.device)
 val=loader(args.data,'validation');yv=labels(val); runs=[]
 for regime,seed in RUNS:
  out=ROOT/'runs'/f'{regime}_seed{seed}';out.mkdir(exist_ok=True)
  if args.resume and (out/'run.json').exists() and (out/'model.pt').exists():
   old=json.loads((out/'run.json').read_text());assert old['split_sha256']==sha(ROOT/'split.csv');runs.append(old);continue
  model,opt=build(regime,seed,args.device);dl=loader(args.data,'train');loss_fn=torch.nn.CrossEntropyLoss()
  run=dict(regime=regime,seed=seed,settings=settings(model,opt,regime),split_sha256=sha(ROOT/'split.csv'),provenance=provenance(),device=args.device,device_name=device_name(args.device),epochs=[])
  t0=time.perf_counter()
  for epoch in range(1,9):
   set_train_mode(model,regime);total=0.;steps=0
   for x,y in dl:
    opt.zero_grad();loss=loss_fn(model(x.to(args.device)),y.to(args.device));loss.backward();opt.step();total+=loss.item()*len(y);steps+=1
   assert steps==38 and len(y)==16
   z=logits(model,val,args.device);f=float(f1_score(yv,z.argmax(1),average='macro'))
   run['epochs'].append(dict(epoch=epoch,train_loss=total/1200,val_macro_f1=f))
   print(f'{regime} seed={seed} epoch={epoch}/8 loss={total/1200:.6f} val_F1={f:.6f} elapsed={time.perf_counter()-t0:.1f}s',flush=True)
  run['training_seconds']=time.perf_counter()-t0
  np.save(out/'val_logits.npy',z)
  torch.save({k:v.cpu() for k,v in model.state_dict().items()},out/'model.pt')
  dump(out/'run.json',run);runs.append(run)
 winner,means=select(runs);key=f"{winner['regime']}_seed{winner['seed']}"
 torch.save(dict(state_dict=torch.load(ROOT/'runs'/key/'model.pt',map_location='cpu',weights_only=True),regime=winner['regime'],seed=winner['seed'],classes=CLASSES,img_size=160),ROOT/'lab3_model.pt')
 dump(ROOT/'selection.json',dict(regime=winner['regime'],seed=winner['seed'],means=means,model_sha256=sha(ROOT/'lab3_model.pt')))
 print('Selected '+key+'; test has not been evaluated.',flush=True)

if __name__=='__main__': main()
