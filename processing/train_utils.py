import os

import torch
from tqdm import tqdm

from processing.utils import acc_class as balanced_acc
from processing.utils import AverageMeter

import wandb
def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()  # Set model to training mode
    train_loss = AverageMeter()
    predicted_labels = []
    gt_labels = []

    for x_step, label in tqdm(dataloader):
        x_step, label = x_step.to(device), label.to(device)

        output = model(x_step)
        loss = loss_fn(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), label.size(0))

        # Store for metric computation
        pred = output.data.max(1, keepdim=True)[1].cpu()
        predicted_labels.extend(pred.numpy())
        gt_labels.extend(label.cpu().numpy())

    return train_loss.avg, predicted_labels, gt_labels

def validation_epoch(model, dataloader, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    val_loss = AverageMeter()
    predicted_labels = []
    gt_labels = []

    with torch.no_grad():
        for x_step, label in tqdm(dataloader):
            x_step, label = x_step.to(device), label.to(device)

            output = model(x_step)
            loss = loss_fn(output, label)

            val_loss.update(loss.item(), label.size(0))

            # Store for metric computation
            pred = output.data.max(1, keepdim=True)[1].cpu()
            predicted_labels.extend(pred.numpy())
            gt_labels.extend(label.cpu().numpy())

    return val_loss.avg, predicted_labels, gt_labels

def supervised_training(dataloaders, model, device, train_tools, args, num_epochs, split):
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

    # Create a config dict which is the args plus the split
    config_dict = args
    config_dict.split = split[-1]
    # Comment out the following line to use wandb
    # TODO comment this before sending back the reviews
    wandb.init(name = f'{args.model_name.upper()} split {split[-1]}', project="BCN 20000 paper", config=config_dict, reinit=True)

    for epoch in range(num_epochs):
        print(f'\n{"-"*40}\nTraining epoch {epoch}\n{"-"*40}')

        train_loss, train_preds, train_gts = train_epoch(model, dataloaders['train'], loss_fn, optimizer, device)
        val_loss, val_preds, val_gts = validation_epoch(model, dataloaders['val'], loss_fn, device)

        train_metric = balanced_acc(train_gts, train_preds)
        val_metric = balanced_acc(val_gts, val_preds)

        wandb.log({
            'Train Accuracy': train_metric,
            'Validation Accuracy': val_metric,
            'Train Loss': train_loss,
            'Validation Loss': val_loss,
            'Epoch': epoch,
            'Learning Rate': optimizer.param_groups[0]['lr']
        })

        if val_metric > best_metric:
            best_metric = val_metric
            wandb.log({"Best Validation Accuracy": best_metric})
            update_best_model(val_metric, model, split, train_metric)
            patience = 0

            # Perform test evaluation
            _, test_preds, test_gts = validation_epoch(model, dataloaders['test'], loss_fn, device)
            test_metric = balanced_acc(test_gts, test_preds)
            print(f'Test accuracy: {test_metric}')
            wandb.log({'Test Accuracy': test_metric})

            if val_loss < least_loss:
                least_loss = val_loss
            else:
                patience += 1

            if patience >= 50:
                print("Early stopping triggered")
                break

            scheduler.step()

        print('='*41)
        print('Train acc : {} ---------- Val acc : {} \nTrain loss {} ---------- Val loss {}'.format(
            round(train_metric, 3), round(val_metric, 3), round(train_loss, 4), round(val_loss, 4)))
        print('='*41)
        
    # Optionally finish the wandb run
    wandb.finish()

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
