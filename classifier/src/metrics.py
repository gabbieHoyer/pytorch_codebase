import numpy as np
from sklearn.metrics import confusion_matrix
import pdb

#Note this of course doesn't work for multi-class classification hence will throw error :D
def update_confusionMatrix(labels,yhat,current_cm):
    try:
        print(labels.view(1))
        print('break')
        print(yhat.argmax(-1))
        print(yhat.argmax(-1).view(-1))
        tn, fp, fn, tp = confusion_matrix(labels.view(-1).cpu(), yhat.argmax(-1).view(-1).cpu()).ravel()

#        tn, fp, fn, tp = confusion_matrix(labels.view(-1).cpu(), yhat.argmax(-1).view(-1).cpu()).ravel()

#, labels=['plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    except:
        if (labels.view(-1).cpu()==1).all() and (yhat.argmax(-1).view(-1).cpu()==1).all():
            tn, fp, fn, tp = 0., 0., 0., len(labels.view(-1).cpu())

        elif (labels.view(-1).cpu()==0).all() and (yhat.argmax(-1).view(-1).cpu()==0).all():
            tn, fp, fn, tp = len(labels.view(-1).cpu()), 0., 0., 0.

        else:
            print('problem with the confusion matrix')
            pdb.set_trace()

    running_tn, running_fp, running_fn, running_tp = current_cm#[0],current_cm[1],current_cm[2],current_cm[3]

#    running_tn, running_fp, running_fn, running_tp = #current_cm[0],current_cm[1],current_cm[2],current_cm[3]

    running_tn+=tn
    running_fp+=fp
    running_fn+=fn
    running_tp+=tp
    return running_tn, running_fp, running_fn, running_tp

def scores_from_confusionMatrix(tn, fp, fn, tp):
    sn = tp.astype(np.float)/(tp.astype(np.float)+fn.astype(np.float))
    sp = tn.astype(np.float)/(tn.astype(np.float)+fp.astype(np.float))
    ppv= tp.astype(np.float)/(tp.astype(np.float)+fp.astype(np.float))
    f1score = 2*(ppv*sn)/(ppv+sn)
    return sn, sp, ppv, f1score
