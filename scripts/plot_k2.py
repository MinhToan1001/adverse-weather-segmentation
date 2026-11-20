"""Export training diagrams and measured diagnostics as PNG + SVG artifacts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype']='none'  # Editable report labels, no thousands of glyph paths.
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import yaml

from scripts.k2_results import ResultFiles, _summary
from src.data.labels import CLASS_NAMES


def _save(fig, out, name):
    for suffix in ('png','svg'):
        fig.savefig(out/(name+'.'+suffix), dpi=150, bbox_inches='tight')
        if suffix=='svg':
            path=out/(name+'.svg')
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


def pipeline(out, arm='k2', recipe='legacy_v1', epochs=20):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(16,6.7));ax.set_xlim(0,16);ax.set_ylim(0,7);ax.axis('off')
    ax.text(.3,6.65,'Train on Cityscapes; evaluate a frozen checkpoint on adverse weather',fontsize=17,weight='bold')
    ax.text(.3,6.15,f'ARM: {arm}  |  recipe: {recipe}  |  planned epochs: {epochs}',fontsize=11,color='#475569')
    has_weather=arm not in ('baseline','film_clean');has_film=arm not in ('baseline','aug')
    xs=[.3,3.45,6.6,9.75,12.9];w=2.7;h=1.15
    upper=['Cityscapes train\n2,975 labeled images',
           ('K1 on native RGB + IDs\n' if has_weather else 'Clean RGB\n')+'Aligned scale/crop/flip',
           'MiT-B0 + decoder\n'+('Stage-wise FiLM' if has_film else 'Plain B0 control'),
           'Segmentation CE\n'+('+ 0.1 condition CE' if arm=='k2' else 'No condition loss'),
           'AdamW + AMP\nBackpropagation']
    lower=['Cityscapes val\n500 images; select best', 'Native-label validation\nFP32 class IoU / mIoU',
           'Best source mIoU\nSave best + last + RNG', 'Freeze checkpoint\nVerify weights/config hash',
           'ACDC val: 406 images\nRead-only evaluation']
    def box(x,y,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08,rounding_size=.12',
            facecolor=color,edgecolor='#94a3b8',linewidth=1.2))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10.3,linespacing=1.6)
    for x,t,b in zip(xs,upper,lower):
        box(x,4.25,t,'#e0f2fe');box(x,1.65,b,'#dcfce7')
    def arrow(a,b,**kw):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color='#475569',linewidth=1.4,**kw))
    for row in (4.25,1.65):
        for x in xs[:-1]:arrow((x+w+.08,row+h/2),(x+3.15-.1,row+h/2))
    ax.plot([14.25,14.25,7.95],[5.48,5.8,5.8],color='#64748b',linestyle='--',linewidth=1.2)
    arrow((7.95,5.8),(7.95,5.48))
    ax.text(11.1,5.88,'Repeat train / validate for the fixed horizon',ha='center',fontsize=9,color='#475569')
    arrow((xs[2]+w/2,4.15),(xs[1]+w/2,2.95),connectionstyle='arc3,rad=.15')
    ax.text(7.8,3.2,'Same model; no validation gradients',ha='center',fontsize=9,color='#475569')
    ax.text(.3,.9,'ACDC images/labels never enter the training loss or checkpoint selection. K3/K4 are not enabled.',fontsize=11)
    ax.text(.3,.45,'Public validation scores are not hidden-test results. Target feedback used for later changes makes that analysis exploratory.',fontsize=10,color='#475569')
    _save(fig,out,'training_pipeline')


def architecture(out, arm):
    fig,ax=plt.subplots(figsize=(14,5.5));ax.set_xlim(0,15);ax.set_ylim(0,6);ax.axis('off')
    has_film=arm not in ('baseline','aug')
    ax.text(.2,5.65,'B0 architecture and the K2 additions',fontsize=17,weight='bold')
    ax.text(.2,5.15,'FiLM uses channel mean/variance; its output feeds the next stage and the decoder.',fontsize=10.5,color='#475569')
    xs=[.25,3.15,6.05,8.95,11.85];w=2.35
    texts=['Normalized RGB\nH x W x 3']+[f'MiT stage {i+1}\nC={c}, stride={s}'+('\nFiLM: scale + shift' if has_film else '\nPlain B0 features')
              for i,(c,s) in enumerate(zip([32,64,160,256],[4,8,16,32]))]
    def box(x,y,text,color,width=w):
        ax.add_patch(FancyBboxPatch((x,y),width,1.05,boxstyle='round,pad=.06',facecolor=color,edgecolor='#94a3b8'))
        ax.text(x+width/2,y+.525,text,ha='center',va='center',fontsize=10,linespacing=1.5)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color='#64748b',linewidth=1.2))
    for x,text in zip(xs,texts):box(x,3.55,text,'#e0f2fe')
    for x in xs[:-1]:arrow((x+w+.08,4.07),(x+2.9-.1,4.07))
    box(8.4,1.3,'Four-scale projection\nUpsample / concat / fuse','#dcfce7',width=3.1)
    box(12,1.3,'19-class pixel logits\nSegmentation CE','#dcfce7',width=2.6)
    for x in xs[1:]:arrow((x+w/2,3.45),(9.95,2.45))
    arrow((11.6,1.825),(11.9,1.825))
    if has_film:
        box(.6,1.3,'Stage4 global pooling\nCondition head: 5 logits','#fef3c7',width=3.7)
        arrow((13,3.45),(4.4,2.45))
        ax.text(.65,.65,'Condition CE x 0.1 is active.' if arm=='k2' else 'Auxiliary head present, loss disabled for this control.',fontsize=10)
    ax.text(8.4,.65,'FiLM starts as identity; no EMA teacher or test-time adaptation.',fontsize=10,color='#475569')
    _save(fig,Path(out),'k2_architecture')


def plot_result(result, out, evaluation=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with ResultFiles(result) as files:
        summary=_summary(files)
        cfg=yaml.safe_load(files.read('train.yaml'))
        rows=[json.loads(s) for s in files.read('run/history.jsonl').decode().splitlines() if s.strip()]
    pipeline(out,summary['arm'],summary['recipe'],summary['planned_epochs'])
    architecture(out,summary['arm'])
    x=[r['epoch'] for r in rows]
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    axes[0,0].plot(x,[r['train_loss'] for r in rows],color='#0284c7')
    axes[0,0].set_title('Training objective (CE + auxiliary loss when enabled)')
    axes[0,1].plot(x,[100*r['val']['miou'] for r in rows],color='#16a34a')
    axes[0,1].axvline(summary['best_epoch'],color='#64748b',linestyle='--',label='Selected best')
    axes[0,1].legend();axes[0,1].set_title('Cityscapes val500 selection mIoU (%)')
    ce=[r['val'].get('cross_entropy',np.nan) for r in rows]
    axes[1,0].plot(x,ce,color='#d97706');axes[1,0].set_title('Validation unweighted CE (separate objective)')
    axes[1,1].plot(x,[r.get('learning_rate',np.nan) for r in rows],color='#7c3aed')
    axes[1,1].set_title('Backbone learning rate')
    for ax in axes.flat:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    fig.suptitle(f"{summary['arm']} / {summary['recipe']} / seed {summary['seed']} — measured source diagnostics")
    fig.tight_layout();_save(fig,out,'learning_curves')
    values=np.array([[np.nan if v is None else v for v in r['val']['per_class_iou']] for r in rows])
    fig,ax=plt.subplots(figsize=(11,7));heat=ax.imshow(values.T*100,aspect='auto',vmin=0,vmax=100,cmap='viridis')
    ax.set_yticks(range(19),CLASS_NAMES);ticks=np.linspace(0,len(x)-1,min(10,len(x)),dtype=int)
    ax.set_xticks(ticks,[x[i] for i in ticks]);ax.set(xlabel='Epoch',title='Cityscapes per-class IoU across epochs')
    fig.colorbar(heat,ax=ax,label='IoU (%)');fig.tight_layout();_save(fig,out,'class_learning')
    fig,ax=plt.subplots(figsize=(11,6));best=100*np.array([np.nan if v is None else v for v in summary['best_per_class_iou']])
    ax.barh(CLASS_NAMES,best,color='#0284c7');ax.set_xlim(0,100)
    ax.set(xlabel='IoU (%)',title=f"Best source checkpoint: epoch {summary['best_epoch']}")
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2);fig.tight_layout();_save(fig,out,'best_class_iou')
    if any('condition_train' in r for r in rows):
        fig,axes=plt.subplots(1,2,figsize=(10,3.7))
        axes[0].plot(x,[r.get('condition_train',{}).get('cross_entropy',np.nan) for r in rows])
        axes[0].set_title('Auxiliary condition CE')
        axes[1].plot(x,[100*r.get('condition_train',{}).get('accuracy',np.nan) for r in rows])
        axes[1].set_title('Synthetic-transform recognition accuracy (%)')
        for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
        fig.suptitle('K1 labels: not segmentation accuracy or real-weather classification');fig.tight_layout()
        _save(fig,out,'condition_learning')
    if evaluation:
        evaluation=Path(evaluation)
        metrics={}
        for scope in ('cityscapes_val','acdc_val'):
            path=evaluation/scope/'metrics.json'
            state=evaluation/scope/'run-state.json'
            if path.is_file() and state.is_file() and json.loads(state.read_text())['status']=='complete':
                metrics[scope]=json.loads(path.read_text())
        if metrics:
            labels=[];scores=[]
            for scope,report in metrics.items():
                for domain,part in report['by_domain'].items():labels.append(domain);scores.append(100*part['miou'])
            fig,ax=plt.subplots(figsize=(8,4.7));ax.bar(labels,scores,color=['#16a34a' if d=='cityscapes' else '#0284c7' for d in labels])
            for index,value in enumerate(scores):ax.text(index,min(value+1.2,98),f'{value:.2f}%',ha='center',fontsize=10)
            ax.set_ylim(0,100);ax.set(ylabel='mIoU (%)',title='Frozen checkpoint: source / real adverse-weather validation')
            ax.grid(axis='y',alpha=.2);fig.tight_layout();_save(fig,out,'weather_miou')
            for scope in metrics:
                cache=sorted((evaluation/scope/'confusions').glob('*.npy'))
                if not cache:continue
                confusion=sum(np.load(p,allow_pickle=False) for p in cache)
                normalized=confusion/np.maximum(confusion.sum(1,keepdims=True),1)
                fig,ax=plt.subplots(figsize=(10,8));heat=ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues')
                ax.set_xticks(range(19),CLASS_NAMES,rotation=70,ha='right');ax.set_yticks(range(19),CLASS_NAMES)
                ax.set(xlabel='Predicted class',ylabel='Ground truth class',title=scope+' row-normalized confusion matrix')
                fig.colorbar(heat,ax=ax);fig.tight_layout();_save(fig,out,scope+'_confusion')
    details=dict(summary=summary,config=cfg,figures=sorted(p.name for p in out.glob('*.png')),
        interpretation='Measured diagnostics only; optimization gains require matched recipe/seed controls. No ACDC hidden-test claim.')
    (out/'figure_manifest.json').write_text(json.dumps(details,indent=2)+'\n',encoding='utf-8')
    return details


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--evaluation');a=p.parse_args();plot_result(a.input,a.out,a.evaluation)
