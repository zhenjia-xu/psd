import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import PSD
from data import MotionDataset
from utils import mkdir, flow2im, imwrite
import dominate
from dominate.tags import *


def main(args):
    # set device (cpu / gpu)
    if args.gpu == '-1':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))

    data, loaders = {}, {}
    data['demo'] = MotionDataset(
        data_path = args.data_path,
        split = 'demo',
        size = args.size,
        scale = args.scale
    )
    loaders['demo'] = DataLoader(
        dataset = data['demo'],
        batch_size = args.batch,
        shuffle = True,
        num_workers = args.workers
    )

    print('==> dataset loaded')
    print('[size] = {0}'.format(len(data['demo'])))

    model = PSD(dimensions=args.dimensions, size=args.size)
    model.load_state_dict(torch.load(args.resume, map_location='cpu'))
    model.to(device)

    structure = torch.sigmoid(model.structural_descriptor.structure).data.cpu().numpy()
    
    visualization_path = args.visualization_path
    mkdir(visualization_path, clean=True)
    figure_path = os.path.join(visualization_path, 'figures')
    mkdir(figure_path, clean=True)
    
    with torch.no_grad():
        means, log_vars = [], []
        for batch in tqdm(loaders['demo'], desc = 'demo'):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_size = batch['image_inputs'].size(0)
            total_size = len(data['demo'])
            outputs= model.forward(
                image_inputs=batch['image_inputs'],
                flow_inputs=batch['flow_inputs'],
                returns = ['mean', 'log_var']
            )
            means.extend(outputs['mean'].data.cpu().numpy())
            log_vars.extend(outputs['log_var'].data.cpu().numpy())
        means = np.asarray(means)
        log_vars = np.asarray(log_vars)

        x, ym, yv = [], [], []
        for k in range(means.shape[1]):
            x.extend([k, k])
            ym.extend([np.min(means[:, k]), np.max(means[:, k])])
            yv.extend([np.min(log_vars[:, k]), np.max(log_vars[:, k])])
  
        plt.switch_backend('agg')
        plt.figure()
        plt.bar(x, ym, .5, color = 'b')
        plt.xlabel('dimension')
        plt.ylabel('mean')
        plt.savefig(os.path.join(figure_path, 'means.png'), bbox_inches = 'tight')

        plt.figure()
        plt.bar(x, yv, .5, color = 'b')
        plt.xlabel('dimension')
        plt.ylabel('log(var)')
        plt.savefig(os.path.join(figure_path, 'vars.png'), bbox_inches = 'tight')

        batch = iter(loaders['demo']).next()
        batch = {k: v[:args.visualization_num].to(device) for k, v in batch.items()}
        batch_size = batch['image_inputs'].size(0)
        outputs = outputs= model.forward(
            image_inputs=batch['image_inputs'],
            flow_inputs=batch['flow_inputs']
        )

        samples = []
        for k in range(4):
            indices = np.random.choice(len(data['demo']), batch_size)
            sample = model.forward(
                image_inputs=batch['image_inputs'],
                mean = torch.tensor(means[indices], device=device),
                log_var = torch.tensor(log_vars[indices], device=device),
            )

            samples.append(sample['image_outputs'].cpu().numpy())

        vis_image_inputs = batch['image_inputs'].cpu().numpy()
        vis_image_targets = batch['image_targets'].cpu().numpy()
        vis_image_outputs = outputs['image_outputs'].cpu().numpy()
        vis_motions = outputs['motion_outputs'].cpu().numpy()

        for i in range(args.visualization_num):
            imwrite(os.path.join(figure_path, '{}_image_input.png'.format(i)), vis_image_inputs[i])
            imwrite(os.path.join(figure_path, '{}_image_target.png'.format(i)), vis_image_targets[i])
            imwrite(os.path.join(figure_path, '{}_image_output.png'.format(i)), np.clip(vis_image_outputs[i], 0, 1))

            for dim in args.dimensions:
                imwrite(os.path.join(figure_path, '{}_motion_{}.png'.format(i, dim)), flow2im(vis_motions[i, :, dim, ...]))
            
            for k in range(4):
                imwrite(os.path.join(figure_path, '{}_sample_{}.gif'.format(i, k)), [vis_image_inputs[i], np.clip(samples[k][i], 0, 1)])
            
    
    with dominate.document(title='PSD') as web:
        h1('PSD Demo Results')
        h3('Statistics')
        img(src=os.path.join('figures', 'means.png'))
        img(src=os.path.join('figures', 'vars.png'))

        h3('Structure')
        with table(border=1, style='table-layout: fixed;'):
            with tr():
                with td(style='word-wrap: break-word;', halign='center', align='center',):
                    p('')
                for y in args.dimensions:
                    with td(style='word-wrap: break-word;', halign='center', align='center',):
                        p('dimension-{}'.format(y))
            for x in args.dimensions:
                with tr():
                    with td(style='word-wrap: break-word;', halign='center', align='center',):
                        p('dimension-{}'.format(x))
                    for y in args.dimensions:
                        value = structure[x][y]
                        bgcolor = 'Orange' if value > 0.5 else 'LightGray'
                        with td(style='word-wrap: break-word;', halign='center', align='center', bgcolor=bgcolor):
                            p('%.5f' % value)

        h3('Visualizations')
        cols = ['image_input', 'image_target', 'image_output']
        with table(border=1, style='table-layout: fixed;'):
            with tr():
                for col in cols:
                    with td(style='word-wrap: break-word;', halign='center', align='center',):
                        p(col)

                for dim in args.dimensions:
                    with td(style='word-wrap: break-word;', halign='center', align='center',):
                        p('motion-{}'.format(dim))
                
                for k in range(4):
                    with td(style='word-wrap: break-word;', halign='center', align='center',):
                        p('sample-{}'.format(k))

            for id in range(args.visualization_num):
                with tr():
                    for col in cols:
                        with td(style='word-wrap: break-word;', halign='center', align='top'):
                            img(style='width:128px', src=os.path.join('figures', '{}_{}.png'.format(id, col)))
    
                    for dim in args.dimensions:
                        with td(style='word-wrap: break-word;', halign='center', align='center',):
                            img(style='width:128px', src=os.path.join('figures', '{}_motion_{}.png'.format(id, dim)))
    
                    for k in range(4):
                        with td(style='word-wrap: break-word;', halign='center', align='center',):
                            img(style='width:128px', src=os.path.join('figures', '{}_sample_{}.gif'.format(id, k)))

    with open(os.path.join(visualization_path, 'index.html'), 'w') as fp:
        fp.write(web.render())
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--resume', default = 'models/snapshot.pth')
    parser.add_argument('--gpu', default = '0')

    parser.add_argument('--data_path', default = 'data/shape3-demo')
    parser.add_argument('--size', default = 128, type = float)
    parser.add_argument('--scale', default = 100, type = float)
    parser.add_argument('--workers', default = 4, type = int)
    parser.add_argument('--batch', default = 32, type = int)
    parser.add_argument('--visualization_num', default = 8, type = int)
    parser.add_argument('--visualization_path', default = 'demo')


    parser.add_argument('--dimensions', default = '18,9,0')

    args = parser.parse_args()

    if args.dimensions is not None:
        args.dimensions = [int(x) for x in args.dimensions.split(',')]

    print('==> arguments parsed')
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))
    
    main(args)
