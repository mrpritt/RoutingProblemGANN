import argparse
import logging
import numpy as np
import torch
import os
import time
import torch.optim as optim
from VRP.VRP_Actor import Model
from VRP.creat_vrp import creat_data,reward1
from VRP.quantum_layers import decoder_config_from_env, encoder_attn_config_from_env
from torch.optim.lr_scheduler import LambdaLR
from VRP.rolloutBaseline1 import RolloutBaseline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_nodes',        type=int,   default=21)
    parser.add_argument('--n_epochs',       type=int,   default=100)
    parser.add_argument('--batch_size',     type=int,   default=512)
    parser.add_argument('--data_size',      type=int,   default=768000)
    parser.add_argument('--val_size',       type=int,   default=10000)
    parser.add_argument('--lr',             type=float, default=1e-3)
    parser.add_argument('--hidden_node_dim',type=int,   default=128)
    parser.add_argument('--hidden_edge_dim',type=int,   default=16)
    parser.add_argument('--conv_layers',    type=int,   default=4)
    parser.add_argument('--output_dir',     type=str,   default='results')
    parser.add_argument('--run_name',       type=str,   default=None)
    return parser.parse_args()
def rollout(model, dataset,batch_size, n_nodes):

    model.eval()
    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(bat,n_nodes*2,True)

            cost = reward1(bat.x,cost.detach(), n_nodes)
        return cost.cpu()
    totall_cost = torch.cat([eval_model_bat(bat.to(device))for bat in dataset], 0)
    return totall_cost

max_grad_norm = 2

rewardss = []
def adv_normalize(adv):
    std = adv.std()
    assert std != 0. and not torch.isnan(std), 'Need nonzero std'
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs

def train():
    args = get_args()
    n_nodes = args.n_nodes
    steps = n_nodes

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info('Starting PPO training for VRP')

    run_name = args.run_name or 'vrp{}_classical_{}'.format(n_nodes, time.strftime('%Y%m%dT%H%M%S'))
    save_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    decoder_backend, decoder_qnn_config = decoder_config_from_env()
    encoder_attn_backend, encoder_attn_qnn_config, encoder_attn_qnn_layers = encoder_attn_config_from_env(args.conv_layers)

    print('n_nodes:', n_nodes, 'lr:', args.lr, 'batch_size:', args.batch_size,
          'data_size:', args.data_size, 'val_size:', args.val_size,
          'hidden_node_dim:', args.hidden_node_dim, 'hidden_edge_dim:', args.hidden_edge_dim,
          'conv_layers:', args.conv_layers)

    data_loder = creat_data(n_nodes, args.data_size, batch_size=args.batch_size)
    valid_loder = creat_data(n_nodes, args.val_size,  batch_size=args.batch_size)
    logging.info('DATA CREATED/Problem size: %s' % n_nodes)

    actor = Model(
        3,
        args.hidden_node_dim,
        1,
        args.hidden_edge_dim,
        conv_laysers=args.conv_layers,
        decoder_backend=decoder_backend,
        decoder_qnn_config=decoder_qnn_config,
        encoder_attn_backend=encoder_attn_backend,
        encoder_attn_qnn_config=encoder_attn_qnn_config,
        encoder_attn_qnn_layers=encoder_attn_qnn_layers,
    ).to(device)
    rol_baseline = RolloutBaseline(actor, valid_loder, n_nodes=steps)
    actor_optim = optim.Adam(actor.parameters(), lr=args.lr)

    costs = []
    for epoch in range(args.n_epochs):
        print("epoch:", epoch, "------------------------------------------------")
        actor.train()

        times, losses, rewards = [], [], []
        epoch_start = time.time()
        start = epoch_start

        scheduler = LambdaLR(actor_optim, lr_lambda=lambda f: 0.96 ** epoch)
        for batch_idx, batch in enumerate(data_loder):
            batch = batch.to(device)
            tour_indices, tour_logp = actor(batch, steps * 2)

            rewar = reward1(batch.x, tour_indices.detach(), n_nodes)
            base_reward = rol_baseline.eval(batch, steps)

            advantage = (rewar - base_reward)
            if not advantage.ne(0).any():
                print("advantage==0.")
            advantage = adv_normalize(advantage)
            actor_loss = torch.mean(advantage.detach() * tour_logp)

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()
            scheduler.step()
            rewards.append(torch.mean(rewar.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            step = 200
            if (batch_idx + 1) % step == 0:
                end = time.time()
                times.append(end - start)
                start = end
                print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(data_loder), np.mean(rewards[-step:]),
                       np.mean(losses[-step:]), times[-1]))

        rol_baseline.epoch_callback(actor, epoch)

        epoch_dir = os.path.join(save_dir, 'epoch_%d' % epoch)
        os.makedirs(epoch_dir, exist_ok=True)
        torch.save(actor.state_dict(), os.path.join(epoch_dir, 'actor.pt'))

        cost = rollout(actor, valid_loder, args.batch_size, steps).mean()
        costs.append(cost.item())

        epoch_time = time.time() - epoch_start
        print('Finished epoch %d, took %s' % (epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_time))))
        print('Problem: VRP%d / Average distance: %.6f' % (n_nodes, cost.item()))
        print(costs)

    logging.info('Ending PPO training for VRP')

train()
