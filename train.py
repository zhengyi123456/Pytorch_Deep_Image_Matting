from dataloader import *
from utils import *
#from model import *
from segnet import *
from demo import *
import time
import tensorboardX as tb
import sys
import os
from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

learning_rate = 1e-4
current_learning_rate = 1e-4
best_val_loss = 100

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def Tran_batch(batch_x, batch_y):
    # [16, 320, 320, 4]
    batch_x = batch_x.squeeze(0)
    # [16, 320, 320, 2] alpha mask
    batch_y = batch_y.squeeze(0)
    batch_x = batch_x.cuda()
    batch_y = batch_y.cuda()
    batch_x = batch_x.transpose(1,2)
    batch_x = batch_x.transpose(1,3)
    # batch_y = batch_y.transpose(1,2)
    # batch_y = batch_y.transpose(1,3)
    #print('Read the data')
    return batch_x, batch_y

def load_model(model):
    pretrained_dict = torch.load('./logs/model-best.pth')
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
        # load params
    model.load_state_dict(new_state_dict)
    return model

# [batch_size, channel, width, height]
if __name__ == '__main__':
    tb_summary_writer = tb.SummaryWriter('./logs')

    model = EncoderDecoder()
    model = model.cuda()


    train_loader = train_gen()
    valid_loader = valid_gen()

    optimizer = torch.optim.Adam(model.parameters(), current_learning_rate, (0.9, 0.999), 1e-8, weight_decay=0)

    iteration = 0
    epoch = -1
    start = time.time()
    model.train()

    #model = load_model(model)

    model = torch.nn.DataParallel(model)

    while True:
        epoch += 1
        if epoch >= 300:
            break
        # learning rate decay
        #if epoch > 10:
        if epoch > 5:
            #frac = (epoch - 10) / 5
            frac = (epoch - 5) / 3
            decay_factor = learning_rate ** frac
            current_learning_rate = learning_rate * decay_factor
        set_lr(optimizer, current_learning_rate)

        for iteration, (batch_x, batch_y) in enumerate(train_loader):
            sys.stdout.flush()

            batch_x, batch_y = Tran_batch(batch_x, batch_y)

            optimizer.zero_grad()

            output = model(batch_x)
            loss = overall_loss(batch_y, output)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            train_loss = loss.item()

            optimizer.step()

            end = time.time()
            if iteration % 10 == 0:
                print('iter {} (epoch {}), train_loss = {:.3f}, time = {:.3f}'.format(iteration, epoch, train_loss, (end-start)))
                add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
                add_summary_value(tb_summary_writer, 'learning_rate', current_learning_rate, iteration)
                start = time.time()

        # Calculate validation loss and save best model
        val_loss = 0
        model.eval()
        start = time.time()
        for iteration, (batch_x, batch_y) in enumerate(valid_loader):
            batch_x, batch_y = Tran_batch(batch_x, batch_y)
            output = model(batch_x)
            loss = overall_loss(batch_y, output)
            val_loss += loss.item()
        end = time.time()
        val_loss = val_loss / (iteration+1)
        print('epoch {} validation loss {:.3f}'.format(epoch, val_loss))
        epoch_model_path = './logs/model_{}_{:.4f}.pth'.format(epoch, val_loss)
        torch.save(model.state_dict(), epoch_model_path)
        print('Save epoch model!')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = './logs/model-best.pth'
            torch.save(model.state_dict(), best_model_path)
            print('Save best model!')
            compute_test(model)
        model.train()
