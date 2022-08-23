import argparse
import os
import time
from pathlib import Path

from config import cfg
from config import update_config
from data import ConcatDataset
from data import JRLDataLoader
from data import MyWeightedSampler
from model import Architecture
from model import MultiViewEmbeddingModel
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from utils import AverageMeter
from utils import check_weight_norm
from utils import jrl_lr_scheduler
from utils import ProgressLogger as Logger
from utils import results


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--auto_kill', action='store_true')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--search', action='store_true')
    parser.add_argument('--search_epoch', type=int, default=20)
    parser.add_argument('--train_random', action='store_true')
    parser.add_argument('--train_distill', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train_epoch', type=int, default=50)
    parser.add_argument('--budget', type=str, default='')

    args = parser.parse_args()

    return args


def set_train_mode(cfg, mode):
    cfg.defrost()
    cfg.MODEL.NAS.DERIVATION = mode
    cfg.freeze()


def main():
    args = parse_args()

    update_config(cfg, args)
    cfg.defrost()
    if args.budget != '':
        values = [float(i) for i in args.budget.split('+')]
        assert len(values) == 2, f"not enough values: {values}"
        cfg.MODEL.NAS.BUDGET.MIN, cfg.MODEL.NAS.BUDGET.MAX = min(values), max(
            values)

    cfg.freeze()
    print(cfg)
    ##

    ##
    final_output_dir = Path(cfg.OUTPUT_DIR)
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    gpu = args.gpu
    if gpu is not None:
        gpu = int(gpu)

    result = {}
    dataset_train = dataset_valid = None
    original_mode = cfg.MODEL.NAS.DERIVATION
    distill_setting = cfg.TRAIN.DISTILL.ENABLED

    if args.search:
        cfg.defrost()
        cfg.TRAIN.DISTILL.ENABLED = False
        cfg.TRAIN.END_EPOCH = args.search_epoch
        cfg.freeze()
        _, dataset_train, dataset_valid = train(gpu, args, cfg.OUTPUT_DIR,
                                                args.search)
        print("SEARCHING COMPLETE. TRAINING COMMENCE")
    if args.train:
        cfg.defrost()
        cfg.TRAIN.DISTILL.ENABLED = distill_setting
        cfg.TRAIN.END_EPOCH = args.train_epoch
        cfg.freeze()
        #set_train_mode(cfg, 'transference')
        result['train'], dataset_train, dataset_valid = train(
            gpu, args, cfg.OUTPUT_DIR, False, dataset_train, dataset_valid)
    if args.train_random:
        set_train_mode(cfg, 'random')
        result['random'], dataset_train, dataset_valid = train(
            gpu, args, cfg.OUTPUT_DIR, False, dataset_train, dataset_valid)
    if args.train_distill:
        cfg.defrost()
        cfg.TRAIN.DISTILL.ENABLED = True
        cfg.MODEL.NAS.DERIVATION = original_mode
        assert os.path.isfile(cfg.TRAIN.DISTILL.PATH)
        cfg.TRAIN.MASK.PATH = os.path.join(cfg.OUTPUT_DIR,
                                           'best_architecture.pth.tar')

        #cfg.MODEL.NAS.DERIVATION = 'transference'
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, 'distill')
        if not os.path.isdir(cfg.OUTPUT_DIR):
            os.mkdir(cfg.OUTPUT_DIR)
        cfg.freeze()
        result['distill'], dataset_train, dataset_valid = train(
            gpu, args, cfg.OUTPUT_DIR, False, dataset_train, dataset_valid)

    for k, v in result.items():
        print(f"{k} accuracy:")
        for metric, acc in v.items():
            print(f'\t{metric}: {acc}')


def train(gpu,
          args,
          final_output_dir,
          is_search,
          dataset_train=None,
          dataset_valid=None):
    if dataset_train is None:
        dataset_train = JRLDataLoader(cfg, 'train')
    if dataset_valid is None:
        dataset_valid = JRLDataLoader(cfg, 'test')
    search = is_search
    if not search:
        dataloader = DataLoader(
            dataset_train,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            sampler=MyWeightedSampler(
                weights=dataset_train.data_weights,
                num_samples=len(dataset_train),
            ),
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )
        mode = 'train'
    else:
        concat_dataset = ConcatDataset([dataset_train, dataset_valid])
        dataloader = DataLoader(
            concat_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            sampler=MyWeightedSampler(
                weights=dataset_train.data_weights,
                num_samples=len(dataset_train),
            ),
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )
        mode = 'search'
    dataloader_val = DataLoader(dataset_valid,
                                batch_size=cfg.TEST.BATCH_SIZE,
                                shuffle=False)
    cur_best = None
    model = MultiViewEmbeddingModel(cfg, dataset_train.get_sizes(),
                                    dataset_train.get_distributions())
    architecture = Architecture(model, cfg)
    if not search:
        if not os.path.isfile(cfg.TRAIN.MASK.PATH):
            cfg.defrost()
            best_p = os.path.join(final_output_dir,
                                  "best_architecture.pth.tar")
            p = os.path.join(final_output_dir, 'architecture.pth.tar')
            cfg.TRAIN.MASK.PATH = best_p if os.path.isfile(best_p) else p
            cfg.freeze()
        architecture.set_model_mask()

    if cfg.MODEL.PRETRAINED != '':
        _w = torch.load(cfg.MODEL.PRETRAINED)
        w = {}
        for key in _w.keys():
            if 'arch' in key: continue
            w[key] = _w[key]
        model.load_state_dict(w, strict=False)

    if cfg.TRAIN.DISTILL.ENABLED:
        model.enable_distillation(cfg.TRAIN.DISTILL.PATH)

    if gpu is not None:
        model = model.to(device=gpu)
    loss_meter = AverageMeter()
    optimizer = torch.optim.SGD(model.net_parameters(),
                                cfg.TRAIN.LR,
                                momentum=0.9)

    logger = Logger(cfg=cfg, is_search=search, save_dir=final_output_dir)
    if mode == 'search':
        osp_best = os.path.join(final_output_dir, f'model_best_{mode}.pth.tar')
        osp_ckp = os.path.join(final_output_dir, f'check_point_{mode}.pth.tar')
    else:
        derivation = cfg.MODEL.NAS.DERIVATION
        osp_ckp = os.path.join(final_output_dir,
                               f'check_point_{mode}_{derivation}.pth.tar')
        osp_best = os.path.join(final_output_dir,
                                f'model_best_{mode}_{derivation}.pth.tar')
    mode = 'search' if search else 'train'
    model.mode = 'train'
    start_epoch = 0
    if cfg.RESUME:
        if os.path.isfile(osp_ckp):
            ckp = torch.load(osp_ckp)
            model.load_state_dict(ckp['model'])
            optimizer.load_state_dict(ckp['optimizer'])
            model.tau = ckp['tau']
            start_epoch = ckp['epoch']
            if architecture:
                architecture.optimizer.load_state_dict(ckp['architecture'])

    auto_kill_countdown = 5
    for current_epoch in range(start_epoch, cfg.TRAIN.END_EPOCH):
        start_time = time.time()
        get_batch_time = time.time()
        step_time = time.time()
        if current_epoch >= cfg.TRAIN.BEGIN_EPOCH:
            model.mode = mode
        for i, batch in enumerate(dataloader):
            if gpu is not None:
                for bi in range(len(batch)):
                    batch[bi] = batch[bi].to(device=gpu, non_blocking=True)
            get_batch_time = time.time() - get_batch_time
            step_time = time.time()
            loss, _ = model(batch[0:2])
            loss_meter.update(loss.detach().clone().item())
            step_time = time.time() - step_time
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.net_parameters(),
                                           cfg.TRAIN.CLIP_NORM)
            optimizer.step()

            if search and current_epoch >= cfg.TRAIN.BEGIN_EPOCH:
                architecture(model, batch[2::])
                jrl_lr_scheduler(architecture.optimizer, cfg.MODEL.NAS.LR.MAX,
                                 0.0001, i, len(dataloader))

            model_lr = jrl_lr_scheduler(optimizer, cfg.TRAIN.LR, 0.0001, i,
                                        len(dataloader))

            if i % cfg.PRINT_FREQ == 0:
                msg = "Epoch %d Words %d/%d: lr = %5.3f loss = %6.3f words/sec = %5.2f prepare_time %.2f step_time %.2f" % (
                    current_epoch, i, len(dataloader), model_lr,
                    loss_meter.avg, cfg.TRAIN.BATCH_SIZE /
                    (time.time() - start_time), get_batch_time, step_time)

                status = architecture.get_status()
                for name in status:
                    if 'lr' in name:
                        msg = msg + f" {name} = {status[name]:.5f} "
                    if 'combine' in name:
                        msg = msg + f" {name} = {status[name]}"
                    else:
                        msg = msg + f" {name} = {status[name]:.3f} "

                print(msg)

                if cfg.DEBUG.ENABLED:
                    if cfg.DEBUG.CHECK_NORM:
                        check_weight_norm(model)

            start_time = time.time()
            get_batch_time = time.time()
        if current_epoch >= cfg.TRAIN.BEGIN_EPOCH:
            architecture.decay()

        chk_point = {
            "model": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'architecture': architecture.state_dict(),
            'epoch': current_epoch,
            'tau': model.tau
        }

        if mode == 'search':
            if current_epoch >= cfg.TRAIN.BEGIN_EPOCH:
                architecture.save_model_state()
                result = validate(cfg, model, dataset_valid, dataloader_val,
                                  final_output_dir)
                architecture.resume_model_state()
            else:
                result = validate(cfg, model, dataset_valid, dataloader_val,
                                  final_output_dir)
        else:
            result = validate(cfg, model, dataset_valid, dataloader_val,
                              final_output_dir)
        logger.log_results(result).save()
        print(f"saving checkpoint at {osp_ckp})")
        torch.save(chk_point, osp_ckp)
        if cur_best is None:
            if search:
                if current_epoch >= cfg.TRAIN.BEGIN_EPOCH:
                    cur_best = result
            else:
                cur_best = result
        # debugging print
        if search:
            if current_epoch >= cfg.TRAIN.BEGIN_EPOCH:
                if cur_best['recalls'] <= result['recalls']:
                    print(f"saving best architecture at {osp_best})")
                    torch.save(
                        {
                            'arch_state_dict': model.arch_state_dict(),
                            'features': model.feature_state_dict()
                        },
                        os.path.join(final_output_dir,
                                     'best_architecture.pth.tar'))
                    cur_best = result  #['recalls']
        else:
            if cur_best['recalls'] <= result['recalls']:
                print(f"saving best model at {osp_best})")
                torch.save(model.state_dict(), osp_best)
                cur_best = result
                auto_kill_countdown = 15
            else:
                auto_kill_countdown -= 1
        if mode == 'search':
            torch.save(
                {
                    'arch_state_dict': model.arch_state_dict(),
                    'features': model.feature_state_dict()
                }, os.path.join(final_output_dir, 'architecture.pth.tar'))
        if args.auto_kill and auto_kill_countdown == 0:
            break
    print("saving final model")
    logger.save()
    if mode == 'search':
        osp_final = os.path.join(final_output_dir,
                                 f"final_model_{mode}.pth.tar")
    else:
        osp_final = os.path.join(
            final_output_dir,
            f"final_model_{mode}_{cfg.MODEL.NAS.DERIVATION}.pth.tar")
    torch.save(model.state_dict(), osp_final)
    print("best final result")
    return cur_best, dataset_train, dataset_valid


def validate(cfg, model, dataset, dataloader, final_output_dir):
    model.eval()
    user_ranklist_map = {}
    user_ranklist_score_map = {}
    with torch.no_grad():
        for current_i, user_idxs in enumerate(dataloader):
            user_product_scores = model([user_idxs, None])

            user_product_scores = user_product_scores.cpu().numpy().tolist()
            user_idxs = user_idxs.cpu().numpy().tolist()
            for i in range(len(user_idxs)):
                u_idx = user_idxs[i]
                sorted_product_idxs = sorted(
                    range(len(user_product_scores[i])),
                    key=lambda k: user_product_scores[i][k],
                    reverse=True)
                user_ranklist_map[u_idx], user_ranklist_score_map[
                    u_idx] = dataset.compute_test_product_ranklist(
                        u_idx, user_product_scores[i], sorted_product_idxs,
                        cfg.MODEL.RANK_CUTOFF)  #(product name, rank)
            if current_i % cfg.PRINT_FREQ == 0:
                print("Finish test review %d/%d\r" %
                      (current_i, len(dataloader)))

    dataset.output_ranklist(user_ranklist_map, user_ranklist_score_map,
                            final_output_dir, cfg.MODEL.SIMILARITY_FUNC)
    best_result = results(
        os.path.join(final_output_dir, "test.product.ranklist"),
        os.path.join(cfg.DATA.PATH, 'test.qrels'), [10])[-1]
    model.train()
    return best_result


if __name__ == "__main__":
    main()
