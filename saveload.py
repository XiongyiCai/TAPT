import os
import torch


def save_checkpoint(checkpoint_dir, model, optimizer, scheduler,epoch, loss, max_checkpoints = 3):
    """
    Save model checkpoint
    """
    if len(os.listdir(checkpoint_dir)) >= max_checkpoints:
        checkpoint_list = os.listdir(checkpoint_dir)
        checkpoint_list.sort(key=lambda x: float(x.split("_")[-1].split(".")[0]))
        remove_path = os.path.join(checkpoint_dir, checkpoint_list[-1])
        os.remove(remove_path)
        
    
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_out_path = os.path.join(checkpoint_dir, "model_epoch_{}_loss_{:.3f}.pth".format(epoch,loss))
    state = {"epoch": epoch ,"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}


    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))
    

def load_checkpoint(checkpoint_dir, model, optimizer, scheduler, best_loss = True):
    """
    Load model checkpoint
    """
    print("Loading checkpoint: {} ...".format(checkpoint_dir))
    
    if len(os.listdir(checkpoint_dir)) == 0:
        print('NOT ANY CHECKPOINTS IN THIS FOLDER, please CHECK')
        return 0
    
    if best_loss == True:
        checkpoint_list = os.listdir(checkpoint_dir)
        checkpoint_list.sort(key=lambda x: float(x.split("_")[-1].split(".")[0]))
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_list[0])
    else:
        checkpoint_list = os.listdir(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_list[-1])
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = checkpoint["epoch"]
    print("Checkpoint loaded. Resume training from epoch {} and checkpoint is in {}".format(epoch, checkpoint_path))

    return epoch
