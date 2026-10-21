"""Render aggregate historical validation/parameter counts, never dataset pixels."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from steel_defect.constants import PROJECT_ROOT


def main():
    report=json.loads((PROJECT_ROOT/'docs/CHECKPOINT_AUDIT.json').read_text())
    order=['cnn_sgd','cnn_adam','gap_adam','resnet_frozen','resnet_finetuned']
    by_id={row['model_id']:row for row in report['checkpoints'] if row['included']}
    rows=[by_id[key] for key in order]
    names=['CNN + SGD','CNN + Adam','GAP + Adam','Frozen ResNet18','Fine-tuned ResNet18']
    colors=['#64748b','#2563eb','#0891b2','#7c3aed','#16a34a']
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    y=range(len(rows))
    axes[0].barh(list(y),[r['historical_val_accuracy_percent'] for r in rows],color=colors)
    axes[0].set_yticks(list(y),names);axes[0].invert_yaxis();axes[0].set_xlim(0,110)
    axes[0].set_xlabel('Historical validation accuracy (%)')
    for i,row in enumerate(rows):axes[0].text(row['historical_val_accuracy_percent']+1,i,f"{row['historical_val_accuracy_percent']:.2f}%",va='center')
    axes[1].barh(list(y),[r['parameters'] for r in rows],color=colors)
    axes[1].set_yticks(list(y),names);axes[1].invert_yaxis();axes[1].set_xscale('log');axes[1].set_xlim(10000,4e7)
    axes[1].set_xlabel('Verified total parameters (log scale)')
    fig.suptitle('Recorded model comparison — validation records, not a new evaluation',fontsize=13)
    fig.text(.5,.01,'No held-out evaluation or latency benchmark rerun; historical runs do not establish universal causal effects.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94))
    out=PROJECT_ROOT/'docs/figures/model_comparison.svg';out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,metadata={'Date':None});plt.close(fig)


if __name__=='__main__':main()
