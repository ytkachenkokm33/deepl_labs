"""Validate stored runs and selection, then evaluate the chosen model once on CPU."""
import argparse, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from common import *

def inspect_runs():
 runs=[]; expected_hash=sha(ROOT/'split.csv'); y=np.array([CLASSES.index(r['label']) for r in read_split() if r['part']=='validation'])
 for regime,seed in RUNS:
  out=ROOT/'runs'/f'{regime}_seed{seed}'
  r=json.loads((out/'run.json').read_text(encoding='utf-8'))
  assert (r['regime'],r['seed'])==(regime,seed)
  assert r['split_sha256']==expected_hash,'Split SHA256 mismatch; retrain all seven runs.'
  assert len(r['epochs'])==8 and r['settings']['steps']==304
  z=np.load(out/'val_logits.npy');assert z.shape==(1498,6) and np.isfinite(z).all()
  assert abs(f1_score(y,z.argmax(1),average='macro')-r['epochs'][-1]['val_macro_f1'])<1e-12
  runs.append(r)
 return runs

def effects(runs):
 winner,means=select(runs)
 ranges={r:float(np.ptp([x['epochs'][-1]['val_macro_f1'] for x in runs if x['regime']==r])) for r in ['frozen','unfreeze']}
 R=max(ranges.values());pre=means['frozen']-means['scratch'];unf=means['unfreeze']-means['frozen']
 return dict(means=means,ranges=ranges,R=R,delta_pre=pre,delta_unf=unf,total=pre+unf,
  pre_exceeds_noise=abs(pre)>R,unf_exceeds_noise=abs(unf)>R)

def curves(runs):
 fig,axes=plt.subplots(1,2,figsize=(11,4.5));colors=dict(scratch='gray',frozen='#2166ac',unfreeze='#b2182b')
 for r in runs:
  label=f"{r['regime']} s{r['seed']}";style=['-','--',':'][r['seed']-1]
  for ax,key in zip(axes,['train_loss','val_macro_f1']):
   ax.plot(range(1,9),[e[key] for e in r['epochs']],style,color=colors[r['regime']],label=label,marker='o',markersize=3)
 for ax,ylabel in zip(axes,['Train cross-entropy','Validation macro-F1']):
  ax.set(xlabel='Epoch',ylabel=ylabel,xticks=range(1,9));ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(ROOT/'figures/curves.png',dpi=200);plt.close(fig)

def choose_errors(cm,rows,y,pred,probs):
 pairs=sorted([(i,j,int(cm[i,j])) for i in range(6) for j in range(6) if i!=j and cm[i,j]],key=lambda x:(-x[2],x[0],x[1]))
 chosen=[];rank=0
 while len(chosen)<min(5,int((y!=pred).sum())):
  for i,j,n in pairs:
   indices=sorted(np.flatnonzero((y==i)&(pred==j)),key=lambda k:rows[k]['file'])
   if rank<len(indices):
    k=int(indices[rank]);chosen.append(dict(index=k,file=rows[k]['file'],true=CLASSES[i],predicted=CLASSES[j],probability=float(probs[k,j]),pair_count=n))
    if len(chosen)==5: break
  rank+=1
 return chosen

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--threads',type=int,default=6);args=ap.parse_args()
 torch.set_num_threads(args.threads);check_split();runs=inspect_runs();winner,means=select(runs)
 checkpoint=torch.load(ROOT/'lab3_model.pt',map_location='cpu',weights_only=True)
 assert (checkpoint['regime'],checkpoint['seed'])==(winner['regime'],winner['seed'])
 assert checkpoint['classes']==CLASSES and checkpoint['img_size']==160
 selection=json.loads((ROOT/'selection.json').read_text())
 assert selection['model_sha256']==sha(ROOT/'lab3_model.pt')
 # Verify actual selected weights through saved validation logits before accessing test.
 model=resnet18(weights=None);model.fc=torch.nn.Linear(512,6);model.load_state_dict(checkpoint['state_dict']);model.eval()
 zv=logits(model,loader(args.data,'validation'))
 stored=np.load(ROOT/'runs'/f"{winner['regime']}_seed{winner['seed']}"/'val_logits.npy')
 # CPU kernels differ across Windows/x86 and macOS/ARM. Checkpoint SHA256 above remains exact.
 max_diff=float(np.max(np.abs(zv-stored)))
 assert np.allclose(zv,stored,rtol=1e-3,atol=1e-3),f'Validation logits differ substantially (max abs={max_diff:.6g}).'
 print(f'Validation logits verified; max absolute difference: {max_diff:.6g}',flush=True)
 curves(runs)
 dl=loader(args.data,'test');y=labels(dl);z=logits(model,dl);pred=z.argmax(1)
 probs=torch.from_numpy(z).softmax(1).numpy();cm=confusion_matrix(y,pred,labels=range(6))
 accuracy=float(accuracy_score(y,pred));f=float(f1_score(y,pred,average='macro'))
 selected_errors=choose_errors(cm,dl.dataset.rows,y,pred,probs)
 result=dict(effects=effects(runs),selected=dict(regime=winner['regime'],seed=winner['seed'],val_macro_f1=winner['epochs'][-1]['val_macro_f1']),test_macro_f1=f,test_accuracy=accuracy,accuracy_se=math.sqrt(accuracy*(1-accuracy)/len(y)),confusion_matrix=cm.tolist(),errors=selected_errors,split_sha256=sha(ROOT/'split.csv'),model_sha256=sha(ROOT/'lab3_model.pt'),analysis_provenance=provenance())
 np.save(ROOT/'test_logits.npy',z);dump(ROOT/'analysis.json',result)
 fig,ax=plt.subplots(figsize=(6.8,5.5));im=ax.imshow(cm,cmap='Blues')
 ax.set(xticks=range(6),yticks=range(6),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted class',ylabel='True class')
 plt.setp(ax.get_xticklabels(),rotation=35,ha='right')
 for i in range(6):
  for j in range(6):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
 fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(ROOT/'figures/confusion.png',dpi=200);plt.close(fig)
 fig,axes=plt.subplots(1,5,figsize=(12,3.2))
 for ax,e in zip(axes,selected_errors):
  with Image.open(dl.dataset.path(e['index'])) as im:ax.imshow(im.convert('RGB'))
  ax.set_title(f"{e['file']}\n{e['true']} → {e['predicted']}\np={e['probability']:.3f}",fontsize=9);ax.axis('off')
 fig.tight_layout();fig.savefig(ROOT/'figures/errors.png',dpi=220);plt.close(fig)
 print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
