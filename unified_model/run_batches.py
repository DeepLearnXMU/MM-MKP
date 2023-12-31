import torch.nn as nn
import time
import math
import torch
import os
from predict import predict_seq2seq, predict_classifier
from torch.nn import MSELoss

EPS = 1e-8
LR_GAP_EPS = 1e-5


class LossStatistics:
    """
    Accumulator for loss staistics. Modified from OpenNMT
    """

    def __init__(self, loss=0.0, seq2seq_loss=0.0, classifier_loss=0.0,
                 n_tokens=0, n_batch=0, forward_time=0.0, loss_compute_time=0.0, backward_time=0.0,kl_loss=0.0,use_kl=False):
        assert type(loss) is float or type(loss) is int
        assert type(n_tokens) is int
        self.loss = loss
        self.seq2seq_loss = seq2seq_loss
        self.classifier_loss = classifier_loss
        self.use_kl=use_kl
        self.kl_loss=kl_loss

        if math.isnan(loss):
            raise ValueError("Loss is NaN")
        self.n_tokens = n_tokens
        self.n_batch = n_batch
        self.forward_time = forward_time
        self.loss_compute_time = loss_compute_time
        self.backward_time = backward_time

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `LossStatistics` object

        Args:
            stat: another statistic object
        """
        self.loss += stat.loss
        self.seq2seq_loss += stat.seq2seq_loss
        self.classifier_loss += stat.classifier_loss
        self.kl_loss+= stat.kl_loss
        if math.isnan(stat.loss):
            raise ValueError("Loss is NaN")
        self.n_tokens += stat.n_tokens
        self.n_batch += stat.n_batch
        self.forward_time += stat.forward_time
        self.loss_compute_time += stat.loss_compute_time
        self.backward_time += stat.backward_time
        

    def xent(self):
        """ compute normalized cross entropy """
        assert self.n_tokens > 0, "n_tokens must be larger than 0"
        return self.seq2seq_loss / self.n_tokens

    def ppl(self):
        """ compute normalized perplexity """
        assert self.n_tokens > 0, "n_tokens must be larger than 0"
        return math.exp(min(self.seq2seq_loss / self.n_tokens, 100))

    def mean_classifier_loss(self):
        assert self.n_batch > 0
        return self.classifier_loss / self.n_batch

    def mean_kl_loss(self):
        assert self.n_batch > 0
        return self.kl_loss / self.n_batch

    def total_time(self):
        return self.forward_time, self.loss_compute_time, self.backward_time

    def clear(self):
        self.loss = 0.0
        self.n_tokens = 0
        self.n_batch = 0
        self.forward_time = 0.0
        self.loss_compute_time = 0.0
        self.backward_time = 0.0
        self.seq2seq_loss = 0
        self.classifier_loss = 0
        self.kl_loss = 0

def convert_time2str(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dh-%02dm" % (h, m)


def masked_cross_entropy(class_dist, target, trg_mask):
    num_classes = class_dist.size(2)
    class_dist_flat = class_dist.view(-1, num_classes)  # [batch_size*trg_seq_len, num_classes]
    log_dist_flat = torch.log(class_dist_flat + EPS)
    target_flat = target.view(-1, 1)  # [batch*trg_seq_len, 1]
    losses_flat = -torch.gather(log_dist_flat, dim=1, index=target_flat)  # [batch * trg_seq_len, 1]
    losses = losses_flat.view(*target.size())  # [batch, trg_seq_len]

    if trg_mask is not None:
        losses = losses * trg_mask

    loss = losses.sum(dim=1)  # [batch_size]
    loss = loss.sum()

    return loss


def train_valid_mixture(model, optimizer, train_data_loader, valid_data_loader, start_epoch, opt):
    print('\n==============================Training=================================')
    report_train_loss_statistics = LossStatistics()
    report_train_loss = []
    report_valid_loss = []
    best_valid_loss = float('inf')
    num_stop_dropping = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    total_train_step = len(train_data_loader)
    total_valid_step = len(valid_data_loader)
    t0 = time.time()
    print("Entering the training and validating for %d epochs" % opt.num_epochs)
    for epoch in range(start_epoch, opt.num_epochs):
        print('\nBegin training for epoch %d' % (epoch + 1))

        #print("total_train_step",total_train_step)
        for i, batch in enumerate(train_data_loader):
            #print("batch",i)
            batch_loss_stat = train_one_generation_batch(batch, model, optimizer, criterion, opt)
            report_train_loss_statistics.update(batch_loss_stat)
            # Print log info
            if (i + 1) % max(1, total_train_step // 10) == 0:
                seq2seq_loss = batch_loss_stat.xent()
                classifier_loss = batch_loss_stat.mean_classifier_loss()
                total_loass = seq2seq_loss + classifier_loss
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}, Seq2seq: {:.6f}, Classifier: {:.6f}, Time: {:.2f}'
                      .format(epoch + 1, opt.num_epochs, i + 1, total_train_step,
                              total_loass, seq2seq_loss, classifier_loss, time.time() - t0))
        current_seq2seq_train_loss = report_train_loss_statistics.xent()
        current_classifier_train_loss = report_train_loss_statistics.mean_classifier_loss()
        current_train_loss = current_seq2seq_train_loss + current_classifier_train_loss
        print('Train Epoch {}, Avg Loss: {:.4f}, Seq2seq: {:.4f}, Classifier: {:.4f}, Time: {:.2f}'
              .format(epoch + 1, current_train_loss, current_seq2seq_train_loss, current_classifier_train_loss,
                      time.time() - t0))
        report_train_loss_statistics.clear()
        # Start validating
        # print('\nBegin validating for epoch %d with %d batches' % (epoch + 1, total_valid_step))
        valid_loss_stat = valid_one_generation_epoch(valid_data_loader, model, criterion, opt)
        current_seq2seq_valid_loss = valid_loss_stat.xent()
        current_classifier_valid_loss = valid_loss_stat.mean_classifier_loss()
        current_valid_loss = current_seq2seq_valid_loss + current_classifier_valid_loss
        print('\nValid Epoch {}, Avg Loss: {:.4f}, Seq2seq: {:.4f}, Classifier: {:.4f}, Time: {:.2f}'
               .format(epoch + 1, current_valid_loss, current_seq2seq_valid_loss, current_classifier_valid_loss,
                       time.time() - t0))

        # Save the model checkpoints
        if epoch + 1 >= opt.epochs_to_save:
            if not os.path.exists(opt.model_dir):
                os.makedirs(opt.model_dir)
            if not os.path.exists(opt.model_save_dir):
                os.makedirs(opt.model_save_dir)
            save_path = os.path.join(opt.model_dir, 'e{}_TL{:.2f}_VL{:.2f}_{}.ckpt'
                                     .format(epoch + 1, current_train_loss, current_valid_loss,
                                             convert_time2str(time.time() - t0)))
            torch.save(model.state_dict(), save_path)
            if not opt.fix_classifier:
                torch.save(model.state_dict(), opt.model_save_dir + opt.tag+'kl'+opt.use_kl+'itm'+opt.itm+'last.ckpt')
            print('\nSaving checkpoint into %s' % save_path)
            if opt.continue_to_predict:
                if epoch + 1 == opt.epochs_to_save:
                    with open(opt.res_fn, 'a') as f:
                        f.write('\n')
                # cmd_str = 'python predict_seq2seq.py -model_path %s -res_fn %s' % (save_path, opt.res_fn)
                # print('\n==================================Predict===================================')
                # print('Command: %s' % cmd_str)
                # os.system(cmd_str)
                opt.model_path = save_path
                seq2seq_results = predict_seq2seq(opt, model)
                print('=================>Seq2seq F1@1: %.5f' % seq2seq_results[0])
                if not opt.combine_pred or not opt.fix_classifier:
                    classifier_results = predict_classifier(opt, model)
                    print('=================>Classifier F1@1: %.5f' % classifier_results[0])

        # early stopping via monitoring the validation loss
        if current_valid_loss < best_valid_loss:  # or epoch + 1 == opt.num_epochs
            print("\nValid loss drops")
            best_valid_loss = current_valid_loss
            num_stop_dropping = 0

        else:
            num_stop_dropping += 1
            print("\nValid loss does not drop for %d epochs" % num_stop_dropping)
            for i, param_group in enumerate(optimizer.param_groups):
                old_lr = float(param_group['lr'])
                new_lr = old_lr * opt.learning_rate_decay
                if old_lr - new_lr > LR_GAP_EPS:
                    param_group['lr'] = new_lr
                    print("The new learning rate is decayed to %.6f" % new_lr)
                else:
                    print('The new learning rate (%.8f) is too small, stop the program!!' % new_lr)
                    return

        report_train_loss.append(current_train_loss)
        report_valid_loss.append(current_valid_loss)


        if num_stop_dropping > opt.early_stop_tolerance:
            print('Have not increased for %d checkpoints, early stop training' % num_stop_dropping)
            return



def train_valid_itm_mixture(model, optimizer, train_data_loader, valid_data_loader, itm_train_loader,itm_test_loader,start_epoch, opt):
    print('\n==============================Training=================================')
    report_train_loss_statistics = LossStatistics()
    report_train_loss = []
    report_valid_loss = []
    best_valid_loss = float('inf')
    num_stop_dropping = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    total_train_step = len(train_data_loader)
    total_valid_step = len(valid_data_loader)
    t0 = time.time()
    
    print("Entering the training and validating for %d epochs" % opt.num_epochs)
    #jv tprint("opt.itm",opt.itm)
    '''if opt.itm and not opt.fix_classifier:
        total_trainitm_step=len(itm_train_loader)
        total_validitm_step=len(itm_test_loader)
        for epoch in range(start_epoch, opt.num_epochs):
            print('\nBegin training Itm for epoch %d' % (epoch + 1))
        
            current_itm_loss=0.0
            num_batch =  0
            for i,batch in enumerate(itm_train_loader):
                itm_loss,bsz=train_one_itm_batch(batch, model, optimizer, criterion, opt)
                current_itm_loss+=itm_loss.item()
                num_batch += bsz
                print('Epoch [{}/{}], Step [{}/{}], ItmLoss: {:.4f},  Time: {:.2f}'
                      .format(epoch + 1, opt.num_epochs, i + 1, total_trainitm_step,itm_loss,time.time() - t0))
            mean_batch_itm_loss=current_itm_loss/num_batch
            print('Train Epoch {}, Itm Loss: {:.4f},  Time: {:.2f}'
              .format(epoch + 1, mean_batch_itm_loss,time.time() - t0))
        
            valid_itm_loss,n_batch=valid_one_itm_epoch(itm_test_loader, model, criterion, opt)
            current_itm_valid_loss=valid_itm_loss/n_batch
            print('Valid Epoch {}, Itm Loss: {:.4f},  Time: {:.2f}'
              .format(epoch + 1, current_itm_valid_loss,time.time() - t0))
    '''
    for epoch in range(start_epoch, opt.num_epochs):
        
        if opt.itm and not opt.fix_classifier:
            print('\nBegin training Itm for epoch %d' % (epoch + 1))
            current_itm_loss=0.0
            num_batch =  0
            for i,batch in enumerate(itm_train_loader):
                itm_loss,bsz=train_one_itm_batch(batch, model, optimizer, criterion, opt)
                current_itm_loss+=itm_loss.item()
                num_batch += bsz
                
            mean_batch_itm_loss=current_itm_loss/num_batch
            print('Train Epoch {}, Itm Loss: {:.4f},  Time: {:.2f}'
              .format(epoch + 1, mean_batch_itm_loss,time.time() - t0))

        print('\nBegin training for epoch %d' % (epoch + 1))
        #itm_task
            # print("epoch",epoch)
        for i, batch in enumerate(train_data_loader):
            # print("batch",i)
            batch_loss_stat = train_one_generation_batch(batch, model, optimizer, criterion, opt,epoch)
            report_train_loss_statistics.update(batch_loss_stat)
            # Print log info
            if (i + 1) % max(1, total_train_step // 10) == 0:
                seq2seq_loss = batch_loss_stat.xent()
                classifier_loss = batch_loss_stat.mean_classifier_loss()
                total_loass = seq2seq_loss + classifier_loss
                if opt.use_kl:
                    kl_loss=batch_loss_stat.mean_kl_loss()
                    total_loass=total_loass+kl_loss
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}, Seq2seq: {:.6f}, Classifier: {:.6f}, KL: {:.6f},Time: {:.2f}'
                      .format(epoch + 1, opt.num_epochs, i + 1, total_train_step,
                              total_loass, seq2seq_loss, classifier_loss,kl_loss, time.time() - t0))
                else:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}, Seq2seq: {:.6f}, Classifier: {:.6f}, Time: {:.2f}'
                      .format(epoch + 1, opt.num_epochs, i + 1, total_train_step,
                              total_loass, seq2seq_loss, classifier_loss, time.time() - t0))
        current_seq2seq_train_loss = report_train_loss_statistics.xent()
        current_classifier_train_loss = report_train_loss_statistics.mean_classifier_loss()

        current_train_loss = current_seq2seq_train_loss + current_classifier_train_loss
        if opt.use_kl:
            current_kl_train_loss=report_train_loss_statistics.mean_kl_loss()
            current_train_loss=current_train_loss+current_kl_train_loss
            print('Train Epoch {}, Avg Loss: {:.4f}, Seq2seq: {:.4f}, Classifier: {:.4f}, KL: {:.6f}, Time: {:.2f}'
              .format(epoch + 1, current_train_loss, current_seq2seq_train_loss, current_classifier_train_loss,current_kl_train_loss,
                      time.time() - t0))
        else:
            print('Train Epoch {}, Avg Loss: {:.4f}, Seq2seq: {:.4f}, Classifier: {:.4f}, Time: {:.2f}'
              .format(epoch + 1, current_train_loss, current_seq2seq_train_loss, current_classifier_train_loss,
                      time.time() - t0))
        report_train_loss_statistics.clear()
        # Start validating
        # print('\nBegin validating for epoch %d with %d batches' % (epoch + 1, total_valid_step))
        
        
        valid_loss_stat = valid_one_generation_epoch(valid_data_loader, model, criterion, opt)
        current_seq2seq_valid_loss = valid_loss_stat.xent()
        current_classifier_valid_loss = valid_loss_stat.mean_classifier_loss()
        current_valid_loss = current_seq2seq_valid_loss + current_classifier_valid_loss
        #if opt.use_kl and epoch>=4:
        if opt.use_kl:
            current_kl_valid_loss=valid_loss_stat.mean_kl_loss()
            current_valid_loss=current_valid_loss+current_kl_valid_loss
            print('\nValid Epoch {}, Avg Loss: {:.4f}, Seq2seq: {:.4f}, Classifier: {:.4f}, KL: {:.6f},  Time: {:.2f}'
               .format(epoch + 1, current_valid_loss, current_seq2seq_valid_loss, current_classifier_valid_loss,current_kl_valid_loss,
                       time.time() - t0))
        else:
            print('\nValid Epoch {}, Avg Loss: {:.4f}, Seq2seq: {:.4f}, Classifier: {:.4f}, Time: {:.2f}'
               .format(epoch + 1, current_valid_loss, current_seq2seq_valid_loss, current_classifier_valid_loss,
                       time.time() - t0))

        # Save the model checkpoints
        if epoch + 1 >= opt.epochs_to_save:
            if not os.path.exists(opt.model_dir):
                os.makedirs(opt.model_dir)
            if not os.path.exists(opt.model_save_dir):
                os.makedirs(opt.model_save_dir)

            save_path = os.path.join(opt.model_dir, 'e{}_itm{}_kl{}_TL{:.2f}_VL{:.2f}_{}.ckpt'
                                     .format(epoch + 1,opt.itm,opt.use_kl,current_train_loss, current_valid_loss,
                                             convert_time2str(time.time() - t0)))
            torch.save(model.state_dict(), save_path)
            print('\nSaving checkpoint into %s' % save_path)
            if not opt.fix_classifier:
                torch.save(model.state_dict(), opt.model_save_dir+str(opt.tag)+ 'kl'+str(opt.use_kl)+'itm'+str(opt.itm) + 'last.ckpt')
            if opt.continue_to_predict:
                if epoch + 1 == opt.epochs_to_save:
                    with open(opt.res_fn, 'a') as f:
                        f.write('\n')
                # cmd_str = 'python predict_seq2seq.py -model_path %s -res_fn %s' % (save_path, opt.res_fn)
                # print('\n==================================Predict===================================')
                # print('Command: %s' % cmd_str)
                # os.system(cmd_str)
                opt.model_path = save_path
                seq2seq_results = predict_seq2seq(opt, model)
                print('=================>Seq2seq F1@1: %.5f' % seq2seq_results[0])
                if not opt.combine_pred or not opt.fix_classifier:
                    classifier_results = predict_classifier(opt, model)
                    print('=================>Classifier F1@1: %.5f' % classifier_results[0])

        # early stopping via monitoring the validation loss
        if current_valid_loss < best_valid_loss:  # or epoch + 1 == opt.num_epochs
            print("=====================  VALID DROP ！！! ==============================")
            print("\nValid loss drops!")
            best_valid_loss = current_valid_loss
            if not opt.fix_classifier:
                torch.save(model.state_dict(), opt.model_save_dir +str(opt.tag)+'kl'+str(opt.use_kl)+'itm'+str(opt.itm)+'best1.ckpt')
            num_stop_dropping = 0

        else:
            num_stop_dropping += 1
            print("\nValid loss does not drop for %d epochs" % num_stop_dropping)
            for i, param_group in enumerate(optimizer.param_groups):
                old_lr = float(param_group['lr'])
                new_lr = old_lr * opt.learning_rate_decay
                if old_lr - new_lr > LR_GAP_EPS:
                    param_group['lr'] = new_lr
                    print("The new learning rate is decayed to %.6f" % new_lr)
                else:
                    print('The new learning rate (%.8f) is too small, stop the program!!' % new_lr)
                    return

        report_train_loss.append(current_train_loss)
        report_valid_loss.append(current_valid_loss)


        if num_stop_dropping > opt.early_stop_tolerance:
            print('Have not increased for %d checkpoints, early stop training' % num_stop_dropping)
            return



def train_one_generation_batch(batch, model, optimizer, criterion, opt,epoch=0):
    # Set mini-batch dataset
    if opt.use_type:
        src, src_lens, src_mask, src_oov, oov_lists, trg, trg_class, trg_lens, trg_mask, trg_oov, img, attr,type = batch
    else:
        src, src_lens, src_mask, src_oov, oov_lists, trg, trg_class, trg_lens, trg_mask, trg_oov, img, attr = batch
    
    batch_size = src.size(0)

    max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

    src = src.to(opt.device)
    src_mask = src_mask.to(opt.device)
    src_oov = src_oov.to(opt.device)
    
    if opt.use_type:
        type=type.to(opt.device)

    trg = trg.to(opt.device)
    trg_class = trg_class.to(opt.device)
    trg_mask = trg_mask.to(opt.device)
    trg_oov = trg_oov.to(opt.device)

    if opt.use_img:
        img = img.to(opt.device)
    if opt.use_attr:
        attr = attr.to(opt.device)

    model.train()
    optimizer.zero_grad()

    t0 = time.time()
    

    forward_time = time.time() - t0
    t0 = time.time()
    if opt.use_kl:
        if opt.use_type:
            # decoder_dist, h_t, attention_dist, encoder_final_state, classifier_output,trg2img_attn, text2img_attn \
            #     = model(src, src_lens, src_mask, src_oov, trg, max_num_oov, img, attr,type)
            decoder_dist, h_t, attention_dist, encoder_final_state, classifier_output,trg2img_attn, text2img_attn \
            = model(src, src_lens, src_mask, src_oov, trg, trg_lens, trg_mask, max_num_oov, img, attr,type)
        else:
            # decoder_dist, h_t, attention_dist, encoder_final_state, classifier_output,trg2img_attn, text2img_attn \
            #     =model(src, src_lens, src_mask, src_oov, trg, max_num_oov, img, attr)
            decoder_dist, h_t, attention_dist, encoder_final_state, classifier_output,trg2img_attn, text2img_attn \
            = model(src, src_lens, src_mask, src_oov, trg, trg_lens, trg_mask, max_num_oov, img, attr)
        #todo
        #ranking_loss=nn.KLDivLoss(reduction='sum')
        #kl_loss=ranking_loss(nn.functional.log_softmax(text2img_attn.squeeze(1),dim=-1),nn.functional.softmax(trg2img_attn.squeeze(1),dim=-1))
        
        # kl_loss=ranking_loss(text2img_attn.squeeze(1),trg2img_attn.squeeze(1))
        # print(f'text2img_attn:{text2img_attn.size()},trg2img_attn:{trg2img_attn.size()},kl:{kl_loss.size()},kl:{kl_loss}')
        Mse_loss = MSELoss(reduction='sum')
        text2img_attn = torch.where(text2img_attn <= -1e2, torch.zeros_like(text2img_attn),text2img_attn)
        trg2img_attn = torch.where(trg2img_attn <= -1e2, torch.zeros_like(trg2img_attn),trg2img_attn)
        #if opt.debug:
        #    print("text2img_attn",text2img_attn)
        #    print("trg2img_attn",trg2img_attn)
        kl_loss = Mse_loss(text2img_attn.squeeze(1),trg2img_attn.squeeze(1))
        # print(f'train kl :{kl_loss}')
        # print(f'loss_mse:{loss_mse.size()},loss_mse:{loss_mse}')   
    else:
        if opt.use_type:
            # decoder_dist, h_t, attention_dist, encoder_final_state, classifier_output \
            # = model(src, src_lens, src_mask, src_oov, trg, max_num_oov, img, attr,type)
            decoder_dist, h_t, attention_dist, encoder_final_state, classifier_output \
            = model(src, src_lens, src_mask, src_oov, trg, trg_lens, trg_mask, max_num_oov, img, attr,type)
        else:
            # decoder_dist, h_t, attention_dist, encoder_final_state, classifier_output \
            # = model(src, src_lens, src_mask, src_oov, trg,  max_num_oov, img, attr)
            decoder_dist, h_t, attention_dist, encoder_final_state, classifier_output \
                = model(src, src_lens, src_mask, src_oov, trg, trg_lens, trg_mask, max_num_oov, img, attr)
  


    if opt.copy_attn:  # Compute the loss using target with oov words
        seq2seq_loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask)
    else:  # Compute the loss using target without oov words
        seq2seq_loss = masked_cross_entropy(decoder_dist, trg, trg_mask)

    
    loss_compute_time = time.time() - t0
    t0 = time.time()

    total_trg_tokens = sum(trg_lens)

    # seq2seq_loss.div(total_trg_tokens)  # yue

    classifier_loss = criterion(classifier_output, trg_class)

    if opt.combine_pred: # yue
        loss = seq2seq_loss
    else:
        loss = seq2seq_loss + classifier_loss

    if opt.use_kl:
        loss=loss+kl_loss
    loss.backward()
    backward_time = time.time() - t0

    if opt.max_grad_norm > 0:
        nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)

    optimizer.step()

    # construct a statistic object for the loss
    if opt.use_kl:
        stat = LossStatistics(loss.item(), seq2seq_loss.item(), classifier_loss.item(),
                              total_trg_tokens, n_batch=batch_size,
                              forward_time=forward_time, loss_compute_time=loss_compute_time, backward_time=backward_time,kl_loss=kl_loss.item(),use_kl=True)
     
    else:
        stat = LossStatistics(loss.item(), seq2seq_loss.item(), classifier_loss.item(),
                              total_trg_tokens, n_batch=batch_size,
                              forward_time=forward_time, loss_compute_time=loss_compute_time, backward_time=backward_time)
    
    return stat



def train_one_itm_batch(batch, model, optimizer, criterion, opt):
    # Set mini-batch dataset
    src, src_lens, src_mask, src_oov, oov_lists, label, img, attr = batch
    batch_size = src.size(0)


    max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

    src = src.to(opt.device)
    src_mask = src_mask.to(opt.device)
    src_oov = src_oov.to(opt.device)
    label=torch.tensor(label)
    label=label.to(opt.device).long()
   

    if opt.use_img:
        img = img.to(opt.device)
    if opt.use_attr:
        attr = attr.to(opt.device)

    model.train()
    optimizer.zero_grad()

    t0 = time.time()
    type=None
    trg=None
    trg_lens = None
    trg_mask = None
    relation_pred = model(src, src_lens, src_mask, src_oov,trg, trg_lens, trg_mask, max_num_oov, img, attr,type,False,True)
    # relation_pred = model(src, src_lens, src_mask, src_oov,trg, max_num_oov, img, attr,type,False,True)
    
    forward_time = time.time() - t0
    t0 = time.time()
    
    #print("relation_pred",relation_pred)
    #print("label",label)

    itm_loss=nn.functional.cross_entropy(relation_pred,label.long())
    
    '''if opt.copy_attn:  # Compute the loss using target with oov words
        seq2seq_loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask)
    else:  # Compute the loss using target without oov words
        seq2seq_loss = masked_cross_entropy(decoder_dist, trg, trg_mask)'''

    loss_compute_time = time.time() - t0
    t0 = time.time()

    #total_trg_tokens = sum(trg_lens)

    # seq2seq_loss.div(total_trg_tokens)  # yue

    #classifier_loss = criterion(classifier_output, trg_class)

    #if opt.combine_pred: # yue
    #    loss = seq2seq_loss
    #else:
    #    loss = seq2seq_loss + classifier_loss

    itm_loss.backward()
    backward_time = time.time() - t0

    if opt.max_grad_norm > 0:
        nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)

    optimizer.step()

    # construct a statistic object for the loss
    '''stat = LossStatistics(loss.item(), seq2seq_loss.item(), classifier_loss.item(),
                          total_trg_tokens, n_batch=batch_size,
                          forward_time=forward_time, loss_compute_time=loss_compute_time, backward_time=backward_time)'''

    return itm_loss,batch_size


def valid_one_generation_epoch(data_loader, model, criterion, opt):
    model.eval()
    evaluation_seq2seq_loss_sum = 0.0
    evaluation_classifier_loss_sum = 0.0
    evaluation_loss_sum = 0.0
    total_trg_tokens = 0
    n_batch = 0
    loss_compute_time_total = 0.0
    forward_time_total = 0.0
    evaluation_kl_loss_sum=0.0
    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            

            if opt.use_type:
                src, src_lens, src_mask, src_oov, oov_lists, trg, trg_class, trg_lens, trg_mask, trg_oov, img, attr,type = batch
            else:
                src, src_lens, src_mask, src_oov, oov_lists, trg, trg_class, trg_lens, trg_mask, trg_oov, img, attr = batch
             
            max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

            batch_size = src.size(0)
            n_batch += batch_size

            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            if opt.use_type:
                type=type.to(opt.device)
            else:
                type=None
            trg = trg.to(opt.device)
            trg_class = trg_class.to(opt.device)
            trg_mask = trg_mask.to(opt.device)
            trg_oov = trg_oov.to(opt.device)

            if opt.use_img:
                img = img.to(opt.device)
            if opt.use_attr:
                attr = attr.to(opt.device)

            t0 = time.time()
            if opt.use_kl:
                # decoder_dist, h_t, attention_dist, encoder_final_state, classifier_output,trg2img_attn, text2img_attn \
                #     = model(src, src_lens, src_mask, src_oov, trg, max_num_oov, img, attr,type)
                decoder_dist, h_t, attention_dist, encoder_final_state, classifier_output,trg2img_attn, text2img_attn \
                    = model(src, src_lens, src_mask, src_oov, trg, trg_lens, trg_mask, max_num_oov, img, attr,type)
                 #todo
                # ranking_loss=nn.KLDivLoss(reduction='sum')
                # kl_loss=ranking_loss(nn.functional.log_softmax(text2img_attn.squeeze(1),dim=-1),nn.functional.softmax(trg2img_attn.squeeze(1),dim=-1))
                # kl_loss=ranking_loss(torch.log(text2img_attn.squeeze(1)),trg2img_attn.squeeze(1))
                Mse_loss = MSELoss(reduction='sum')
                text2img_attn = torch.where(text2img_attn <= -1e2, torch.zeros_like(text2img_attn),text2img_attn)
                trg2img_attn = torch.where(trg2img_attn <= -1e2, torch.zeros_like(trg2img_attn),trg2img_attn)
                kl_loss = Mse_loss(text2img_attn.squeeze(1),trg2img_attn.squeeze(1))
                #print("text2img_attn",text2img_attn)
                #print("trg2img_attn",trg2img_attn)
                #print("kl_loss",kl_loss)
            else:
                # decoder_dist, h_t, attention_dist, encoder_final_state, classifier_output \
                #     = model(src, src_lens, src_mask, src_oov, trg, max_num_oov, img, attr,type)
                decoder_dist, h_t, attention_dist, encoder_final_state, classifier_output \
                    = model(src, src_lens, src_mask, src_oov, trg, trg_lens, trg_mask, max_num_oov, img, attr,type)

            forward_time = time.time() - t0
            forward_time_total += forward_time
            t0 = time.time()

            if opt.copy_attn:  # Compute the loss using target with oov words
                seq2seq_loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask)
            else:  # Compute the loss using target without oov words
                seq2seq_loss = masked_cross_entropy(decoder_dist, trg, trg_mask)

            classifier_loss = criterion(classifier_output, trg_class)

            loss = classifier_loss + seq2seq_loss

            if opt.use_kl:
                loss=loss+kl_loss
            # print(f'valid kl:{kl_loss}')
            loss_compute_time = time.time() - t0
            loss_compute_time_total += loss_compute_time

            evaluation_seq2seq_loss_sum += seq2seq_loss.item()
            evaluation_classifier_loss_sum += classifier_loss.item()
            if opt.use_kl:
                evaluation_kl_loss_sum+=kl_loss.item()
            evaluation_loss_sum += loss.item()
            total_trg_tokens += sum(trg_lens)
    # print(evaluation_classifier_loss_sum)
    if opt.use_kl:
        eval_loss_stat = LossStatistics(evaluation_loss_sum, evaluation_seq2seq_loss_sum, evaluation_classifier_loss_sum,
                                    total_trg_tokens, n_batch,
                                    forward_time_total,
                                    loss_compute_time_total, 0.0,evaluation_kl_loss_sum,True)
    else:
        eval_loss_stat = LossStatistics(evaluation_loss_sum, evaluation_seq2seq_loss_sum, evaluation_classifier_loss_sum,
                                    total_trg_tokens, n_batch,
                                    forward_time=forward_time_total,
                                    loss_compute_time=loss_compute_time_total)
    return eval_loss_stat


def valid_one_itm_epoch(data_loader, model, criterion, opt):
    model.eval()
    evaluation_itm_loss_sum = 0.0
    n_batch = 0
    loss_compute_time_total = 0.0
    forward_time_total = 0.0
    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            src, src_lens, src_mask, src_oov, oov_lists, label, img, attr = batch


            max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

            batch_size = src.size(0)
            n_batch += batch_size

            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            #type=type.to(opt.device)

            if opt.use_img:
                img = img.to(opt.device)
            if opt.use_attr:
                attr = attr.to(opt.device)

            t0 = time.time()
            trg=None
            trg_lens = None
            trg_mask = None
            type=None

            label=torch.tensor(label)
            label=label.to(opt.device).long()

            relation_pred = model(src, src_lens, src_mask, src_oov, trg, trg_lens, trg_mask, max_num_oov, img, attr,type,False,True)
            # relation_pred = model(src, src_lens, src_mask, src_oov, trg, max_num_oov, img, attr,type,False,True)

            forward_time = time.time() - t0
            forward_time_total += forward_time
            t0 = time.time()
            itm_loss=nn.functional.cross_entropy(relation_pred,label.long())
            

            loss_compute_time = time.time() - t0
            loss_compute_time_total += loss_compute_time

            evaluation_itm_loss_sum += itm_loss.item()

        return evaluation_itm_loss_sum,n_batch
