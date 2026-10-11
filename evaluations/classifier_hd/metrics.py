"""Metrics for independently labelled source images, never pseudo-labels.

Binary Brier uses the Mangrove probability; multiclass Brier is the sum of
squared errors across classes (range 0..2). ECE uses ten equal-width bins of
maximum-class confidence. Undefined metrics are None and serialize as null.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, log_loss, roc_curve, precision_recall_curve


def score(y, probabilities, names):
    y=np.asarray(y,dtype=int);p=np.asarray(probabilities,dtype=float);k=len(names)
    if not len(y):raise ValueError('Independent ground-truth labels are required.')
    if p.shape!=(len(y),k) or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError('Invalid probability matrix.')
    if not np.allclose(p.sum(1),1,atol=1e-6) or (y<0).any() or (y>=k).any():raise ValueError('Invalid labels or probability normalization.')
    pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=list(range(k)));rows=[];curves={}
    div=lambda a,b:float(a/b) if b else None
    for i,name in enumerate(names):
        tp=int(cm[i,i]);fn=int(cm[i,:].sum()-tp);fp=int(cm[:,i].sum()-tp);tn=len(y)-tp-fn-fp
        binary=(y==i).astype(int);valid=bool(binary.any() and (1-binary).any())
        rows.append(dict(label=name,support=int(binary.sum()),tp=tp,fp=fp,fn=fn,tn=tn,
                         precision=div(tp,tp+fp),recall=div(tp,tp+fn),f1=div(2*tp,2*tp+fp+fn),
                         specificity=div(tn,tn+fp),false_positive_rate=div(fp,fp+tn),
                         roc_auc=float(roc_auc_score(binary,p[:,i])) if valid else None,
                         average_precision=float(average_precision_score(binary,p[:,i])) if valid else None,
                         prevalence=float(binary.mean())))
        if valid:
            fpr,tpr,_=roc_curve(binary,p[:,i]);pr,re,_=precision_recall_curve(binary,p[:,i])
            curves[name]=dict(fpr=fpr.tolist(),tpr=tpr.tolist(),precision=pr.tolist(),recall=re.tolist())
    conf=p.max(1);correct=(pred==y);bins=[];ece=0.
    for i in range(10):
        mask=(conf>=i/10)&((conf<(i+1)/10) if i<9 else (conf<=1))
        n=int(mask.sum());acc=float(correct[mask].mean()) if n else None;confidence=float(conf[mask].mean()) if n else None
        if n:ece+=n/len(y)*abs(acc-confidence)
        bins.append(dict(lower=i/10,upper=(i+1)/10,n=n,accuracy=acc,confidence=confidence))
    mean=lambda key:float(np.mean([r[key] for r in rows])) if all(r[key] is not None for r in rows) else None
    truth=np.eye(k)[y];brier=np.mean((p[:,1]-truth[:,1])**2) if k==2 else np.mean(np.sum((p-truth)**2,axis=1))
    return dict(n=len(y),accuracy=float(correct.mean()),macro_f1=mean('f1'),balanced_accuracy=mean('recall'),
                weighted_f1=float(sum(r['support']*(r['f1'] or 0) for r in rows)/len(y)),
                log_loss=float(log_loss(y,p,labels=list(range(k)))),brier=float(brier),ece=float(ece),
                confusion_matrix=cm.tolist(),classes=rows,reliability=bins,curves=curves,
                pr_auc_definition='Average precision (step integral), not trapezoidal PR area.',
                macro_definition='Macro F1 is null if any class F1 is undefined; balanced accuracy is null if a class has no ground truth.')
