import torch
from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy

@torch.no_grad()
def compute_macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    """
    y_true, y_pred: shape [N], int64
    macro-F1 = (1/C) * Σ_c F1_c
    F1_c = 2*TP / (2*TP + FP + FN)
    """
    f1_sum = 0.0
    eps = 1e-12

    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().item()
        fp = ((y_pred == c) & (y_true != c)).sum().item()
        fn = ((y_pred != c) & (y_true == c)).sum().item()

        denom = 2 * tp + fp + fn
        f1_c = (2 * tp) / (denom + eps) if denom > 0 else 0.0
        f1_sum += f1_c

    return f1_sum / max(num_classes, 1)

def test_epoch(data_loader, model, criterion, opt):
    print("# -------------------------------------------------- #")
    print("Test model")
    model.eval()

    losses = AverageMeter() #배치마다 나온 loss, acc를 전체 test 평균으로 모으기 위해서 필요함
    accuracies = AverageMeter()

    all_preds = []
    all_targets = []
    all_video_ids = []

    with torch.no_grad():
        for i, data_item in enumerate(data_loader):
            # VAANet saliency utils 기준
            visual, saliency_map, target, audio, visualization_item, batch_size = process_data_item(opt, data_item)

            output, loss = run_model(
                opt,
                [visual, target, audio, saliency_map],
                model,
                criterion,
                i,
                print_attention=False
            )

            acc = calculate_accuracy(output, target)

            losses.update(loss.item(), batch_size)
            accuracies.update(acc, batch_size)

            preds = output.argmax(dim=1)
            all_preds.append(preds.detach().cpu())
            all_targets.append(target.detach().cpu())
            all_video_ids.extend(visualization_item)
    
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    num_classes = getattr(opt, "n_classes", output.size(1))
    macro_f1 = compute_macro_f1(y_true, y_pred, num_classes)

    print("Test loss    : {:.4f}".format(losses.avg))
    print("Test acc     : {:.4f}".format(accuracies.avg))
    print("Test macro F1: {:.4f}".format(macro_f1))

    return accuracies.avg, macro_f1, y_pred.tolist(), y_true.tolist(), all_video_ids