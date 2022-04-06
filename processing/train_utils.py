import os

import torch
from tqdm import tqdm

from processing.utils import acc_class as balanced_acc
from processing.utils import AverageMeter

import wandb


def supervised_training(dataloaders, model, device, train_tools, settings, num_epochs, split):
    split = split  
    print('Using {}'.format(device))
    model.cuda()

    loss_fn   = train_tools[0].cuda()
    optimizer = train_tools[1]
    scheduler = train_tools[2]

    train_loss = AverageMeter()
    val_loss   = AverageMeter()
    best_metric = 0
    patience    = 0
    least_loss  = 0
    # See how to store these together, try using a dictionary!

    for epoch in range(num_epochs):
        predicted_labels,gt_labels, val_predicted_labels,val_gt_labels  = [], [], [],[]

        print('\n{}\nTraining epoch {} \n{}'.format('-'*40,epoch,'-'*40))

        for x_step, label in tqdm(dataloaders['train']):
            
            if torch.cuda.is_available():
                x_step = x_step.cuda(non_blocking=True)
                label  = label.cuda(non_blocking=True)
            bsz = label.shape[0]
            output = model(x_step)

            loss = loss_fn(output, label) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), bsz)

            # Store for metric computation 
            pred = output.data.max(1, keepdim=True)[1].cpu()
            predicted_labels.extend(pred.numpy())
            gt_labels.extend(label.cpu().numpy())
        
        del x_step, label

        scheduler.step()

        model.eval()
        # Validation
        for x_step, label in tqdm(dataloaders['val']):
            if torch.cuda.is_available():
                x_step = x_step.cuda(non_blocking=True)
                label  = label.cuda(non_blocking=True)
            bsz = label.shape[0]

            output = model(x_step)

            loss = loss_fn(output, label)
            val_loss.update(loss.item(), bsz)
        
            # Store for metric computation 
            pred = pred = output.data.max(1, keepdim=True)[1].cpu()
            val_predicted_labels.extend(pred.numpy())
            val_gt_labels.extend(label.cpu().numpy())

        train_metric = balanced_acc(gt_labels, predicted_labels)
        val_metric   = balanced_acc(val_gt_labels, val_predicted_labels)

        print('='*41)
        print('Train acc : {} ---------- Val acc : {} \nTrain loss {} ---------- Val loss {}'.format(
            round(train_metric,3), round(val_metric, 3), round(train_loss.avg,4), round(val_loss.avg, 4)))
        print('='*41)
       #wandb.log({'Train acc' : train_metric, 'Val acc' : val_metric,
       #                'Train loss' : train_loss.avg, 'Val loss' : val_loss.avg})

        train_loss.reset()
        val_loss.reset()
        if val_metric > best_metric:
            best_metric = val_metric
            update_best_model(val_metric, model, split, train_metric)
            patience = 0 

        if val_loss.avg >= least_loss:
            patience +=1
        else:
            least_loss = val_loss.avg
        
        if patience>=50:
            return 0 

def update_best_model(acc_val, model, split, acc_train, classifier=None):
    if not os.path.exists('saved_models/'):
        os.mkdir('saved_models/')
        
    save_dir = 'saved_models/' + split 
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print('model saved in {}'.format(save_dir))
    save_path =  save_dir + '/model_' + str(round(acc_val, 4)) + '_' + str(round(acc_train, 4)) + '.pt'
    if classifier!=None:
        models_dict = {'model' : model.state_dict(), 'classifier' : classifier.state_dict()}
        torch.save(models_dict,save_path)
    else:
        torch.save(model.state_dict(),save_path)
