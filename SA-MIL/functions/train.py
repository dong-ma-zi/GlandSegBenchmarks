import os
import torch
import torch.nn as nn
from .gm import gm
from sklearn import metrics


def train(model, dataloader_train, args, logger, valid_fn=None, dataloader_valid=None):
    if args.pretrain:
        model.pretrain()
        model.to(torch.device(args.device))
        print("The model has been pretrained.")
    best_result = 0
    r = args.r
    lr = args.lr
    wd = args.weight_decay
    loss_fn = nn.BCELoss()
    print('Learning Rate: ', args.lr)
    
    params1 = list(map(id, model.decoder1.parameters()))
    params2 = list(map(id, model.decoder2.parameters()))
    params3 = list(map(id, model.decoder3.parameters()))
    base_params = filter(lambda p: id(p) not in params1 + params2 + params3, model.parameters())
    params = [{'params': base_params},
              {'params': model.decoder1.parameters(), 'lr': lr / 100, 'weight_decay': wd},
              {'params': model.decoder2.parameters(), 'lr': lr / 100, 'weight_decay': wd},
              {'params': model.decoder3.parameters(), 'lr': lr / 100, 'weight_decay': wd}
              ]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    device_ids = list(map(int, args.device_ids))
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    device = torch.device(args.device)
    
    logger.info("{:*^50}".format("training start"))

    for epoch in range(args.epochs):
        logger.info(f"--------------------------------- epoch {epoch + 1} -------------------------------------------")
        model.train()
        epoch_loss = 0
        step = 0

        pred_list = []
        true_list = []
        for index, batch in enumerate(dataloader_train):
            image, label = batch
            image = image.to(device)
            label = label.float().to(device)

            loss = None
            preds = model(image)
            for pred in preds:
                # x = gm(pred, r)
                loss = loss + loss_fn(gm(pred, r), label) if loss else loss_fn(gm(pred, r), label)

            img_pred = (gm(preds[0], args.r) >= 0.5 + 0).int().to('cpu').numpy()
            pred_list += list(img_pred[:, 0])
            true_list += list(label.int().to("cpu").numpy()[:, 0])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            step += 1
        epochs = epoch + 1
        average_loss = epoch_loss / step
        average_f = metrics.f1_score(pred_list, true_list, pos_label=1)

        logger.info('epoch %d loss:%.4f f1:%.4f' % (epochs, average_loss, average_f))

        if valid_fn is not None:
            model.eval()
            result = valid_fn(model, dataloader_valid, args)
            logger.info('epoch %d valid_iou:%.4f' % (epochs, result))

            if result > best_result:
                best_result = result
                torch.save(model.module.state_dict(), os.path.join(args.work_path, f'model_best.pth'))

        if epochs % args.save_every == 0:
            torch.save(model.module.state_dict(), os.path.join(args.work_path, f'ep_{epochs}.pth'))

    print('best result: %.3f' % best_result)

