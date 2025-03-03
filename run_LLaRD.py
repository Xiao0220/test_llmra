import os, pdb, sys
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import argparse
from shutil import copyfile
from time import time
from tqdm import tqdm
sys.path.append('./')
sys.path.append('../')
from set import *
from evaluate import *
from log import Logger
from LLaRD import LLaRD
from load_dataset import Dataset
import pdb

torch.cuda.empty_cache()
def parse_args():
    parser = argparse.ArgumentParser(description='LLM4Denoise Parameters')
    ### general parameters ###
    parser.add_argument('--dataset', type=str, default='amazon_book', help='?')
    parser.add_argument('--model', type=str, default='LGN', help='LGN or GMF')
    parser.add_argument('--runid', type=str, default='0', help='current log id')
    parser.add_argument('--device_id', type=str, default='0', help='?')
    parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--topk', type=int, default=20, help='Topk value for evaluation')   # NDCG@20 as convergency metric
    parser.add_argument('--early_stops', type=int, default=10, help='model convergent when NDCG@20 not increase for x epochs')
    parser.add_argument('--num_neg', type=int, default=1, help='number of negetiva samples for each [u,i] pair')

    ### model parameters ###
    parser.add_argument('--gcn_layer', type=int, default=3, help='?')
    parser.add_argument('--num_user', type=int, default=13024, help='max uid')
    parser.add_argument('--num_item', type=int, default=22347, help='max iid')
    parser.add_argument('--latent_dim', type=int, default=64, help='latent embedding dimension')
    parser.add_argument('--init_type', type=str, default='norm', help='?')
    parser.add_argument('--l2_reg', type=float, default=1e-4, help='?')
    parser.add_argument('--beta', type=float, default=5.0, help='?')
    parser.add_argument('--alpha', type=float, default=0.1, help='?')
    parser.add_argument('--sigma', type=float, default=0.25, help='?')
    parser.add_argument('--keep_rate', type=float, default=0.8, help='?')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='?')
    parser.add_argument('--prf_weight', type=float, default=0.01, help='?')
    parser.add_argument('--str_weight', type=float, default=1.0, help='?')
    parser.add_argument('--kd_temperature', type=float, default=0.2, help='?')
    parser.add_argument('--edge_bias', type=float, default=0.5, help='observation bias of social relations')
    return parser.parse_args()


def makir_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_file(save_path):
    copyfile('./LLaRD.py', save_path + 'LLaRD.py')
    copyfile('./run_LLaRD.py', save_path + 'run_LLaRD.py')
    copyfile('./load_dataset.py', save_path + 'load_dataset.py')

def eval_test(model):
    model.eval()
    with torch.no_grad():
        masked_adj_matrix = model.graph_learner(model.user_embeddings, model.item_embeddings)
        user_emb, item_emb = model.forward(masked_adj_matrix, g_type='cf')
    return user_emb.cpu().detach().numpy(), item_emb.cpu().detach().numpy()


if __name__ == '__main__':
    seed_everything(2023)
    args = parse_args()
    if args.dataset == 'yelp':
        args.num_user = 11091
        args.num_item = 11010
    elif args.dataset == 'steam':
        args.num_user = 23310
        args.num_item = 5237
    elif args.dataset == 'amazon_book':
        args.num_user = 11000
        args.num_item = 9332

    args.data_path = './data/' + args.dataset + '/'
    record_path = './saved/' + args.dataset + '/LLaRD/' + args.runid + '/'
    model_save_path = record_path + 'models/'
    makir_dir(model_save_path)
    # save_file(record_path)
    log = Logger(record_path)
    for arg in vars(args):
        log.write(arg + '=' + str(getattr(args, arg)) + '\n')

    rec_data = Dataset(args)
    rec_model = LLaRD(args, rec_data)
    device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
    rec_model.to(device)
    optimizer = torch.optim.Adam(rec_model.parameters(), lr=args.lr)

    for name, param in rec_model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    max_hr, max_recall, max_ndcg, early_stop = 0, 0, 0, 0
    topk = args.topk
    best_epoch = 0

    model_files = []
    max_to_keep = 5

    for epoch in range(args.epochs):
        t1 = time()
        sum_auc, sum_llm_auc, all_rank_loss, all_reg_loss, all_prf_loss, all_str_loss, all_ib_loss, all_total_loss, batch_num = 0, 0, 0, 0, 0, 0, 0, 0, 0
        rec_model.train()
        #  batch数据
        loader = rec_data._batch_sampling(num_negative=args.num_neg)
        for batch_data in loader:
            # auc, rank_loss, reg_loss, ib_loss, total_loss = rec_model.calculate_all_loss(u, i, j)
            auc, llm_auc, cf_loss, llm_loss, ib_loss, total_loss = rec_model.calculate_LLaRD_loss(batch_data)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            sum_auc += auc.item()
            sum_llm_auc += llm_auc.item()
            all_rank_loss += cf_loss[0].item()
            all_reg_loss += cf_loss[1].item()
            all_prf_loss += llm_loss[0].item()
            all_str_loss += llm_loss[1].item()
            all_ib_loss += ib_loss.item()
            all_total_loss += total_loss.item()
            batch_num += 1
        mean_auc = sum_auc / batch_num
        mean_llm_auc = sum_llm_auc / batch_num
        mean_rank_loss = all_rank_loss / batch_num
        mean_reg_loss = all_reg_loss / batch_num
        mean_prf_loss = all_prf_loss / batch_num
        mean_str_loss = all_str_loss / batch_num
        mean_ib_loss = all_ib_loss / batch_num
        mean_total_loss = all_total_loss / batch_num
        log.write(('Epoch:{:d}, Train_AUC:{:.4f}, Loss_rank:{:.4f}, Loss_reg:{:.4f}, Loss_llm:{:.4f}, Loss_ib:{:.4f}, Loss_sum:{:.4f}\n'
                            .format(epoch, mean_auc, mean_rank_loss, mean_reg_loss, mean_prf_loss+mean_str_loss, mean_ib_loss, mean_total_loss)))
        t2 = time()


        # ***************************  evaluation on Top-20  *****************************#
        if epoch % 1 == 0:
            early_stop += 1
            user_emb, item_emb = eval_test(rec_model)
            hr, recall, ndcg = num_faiss_evaluate(rec_data.testdata, rec_data.traindata, [20], user_emb, item_emb,
                                        rec_data.testdata.keys())
            if ndcg[20] >= max_ndcg or ndcg[20] == max_ndcg and recall[20] >= max_recall:
                best_epoch = epoch
                max_hr = hr[20]
                max_recall = recall[20]
                max_ndcg = ndcg[20]
            log.write((
                'Current Evaluation: Epoch:{:d},  topk:{:d}, recall:{:.4f}, ndcg:{:.4f}\n'.format(epoch, topk,
                                                                                                  recall[20], ndcg[20])))
            log.write((
                'Best Evaluation: Epoch:{:d},  topk:{:d}, recall:{:.4f}, ndcg:{:.4f}\n'.format(best_epoch, topk,
                                                                                               max_recall, max_ndcg)))

            if ndcg[20] == max_ndcg:
                early_stop = 0
                best_ckpt = 'epoch_' + str(epoch) + '_ndcg_' + str(ndcg[20]) + '.ckpt'
                filepath = model_save_path + best_ckpt
                torch.save(rec_model.state_dict(), filepath)
                print(f"Saved model to {filepath}")
                model_files.append(filepath)
                if len(model_files) > max_to_keep:
                    oldest_file = model_files.pop(0)
                    os.remove(oldest_file)
                    print(f"Removed old model file: {oldest_file}")

            t3 = time()
            log.write('traintime:{:.4f}, valtime:{:.4f}\n\n'.format(t2 - t1, t3 - t2))
            if epoch > 50 and early_stop > args.early_stops:
                log.write('early stop: ' + str(epoch) + '\n')
                log.write(set_color('max_recall@20=:{:.4f}, max_ndcg@20=:{:.4f}\n'.format(max_recall, max_ndcg), 'green'))
                break
        # break

    # ***********************************  start evaluate testdata   ********************************#
    rec_model.load_state_dict(torch.load(model_save_path + best_ckpt))
    user_emb, item_emb = eval_test(rec_model)
    hr, recall, ndcg = num_faiss_evaluate(rec_data.testdata, rec_data.traindata,
                                          [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], user_emb, item_emb,
                                          rec_data.testdata.keys())
    for key in ndcg.keys():
        log.write(set_color(
            'Topk:{:3d}, HR:{:.4f}, Recall:{:.4f}, NDCG:{:.4f}\n'.format(key, hr[key], recall[key], ndcg[key]), 'cyan'))
    log.close()
    print('END')
