import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from models import ATSSModel
from datasets import split_dataset_custom
import argparse
import numpy as np
from torch.utils.data import DataLoader

# Initialize parameters
parser = argparse.ArgumentParser()
parser.add_argument('--test_neg_path', default="./AIGVD_1/dataset_wh/BLIP2_visual_textual/GenVideo/GenVideo-Val/Real")
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_frames', type=int, default=8)
parser.add_argument('--log_file', type=str, default="./outputs/log.txt")

args = parser.parse_args()

test_pos_dirs = [
    "./AIGVD_1/dataset_wh/BLIP2_visual_textual/EvalCrafter_wh/floor33",
    "./AIGVD_1/dataset_wh/BLIP2_visual_textual/EvalCrafter_wh/gen2",
    "./AIGVD_1/dataset_wh/BLIP2_visual_textual/EvalCrafter_wh/gen2_december",
    "./AIGVD_1/dataset_wh/BLIP2_visual_textual/EvalCrafter_wh/hotshot",
    "./AIGVD_1/dataset_wh/BLIP2_visual_textual/EvalCrafter_wh/lavie-base",
    "./AIGVD_1/dataset_wh/BLIP2_visual_textual/EvalCrafter_wh/lavie-interpolation",
    "./AIGVD_1/dataset_wh/BLIP2_visual_textual/EvalCrafter_wh/mix-sr",
    "./AIGVD_1/dataset_wh/BLIP2_visual_textual/EvalCrafter_wh/modelscope",
    "./AIGVD_1/dataset_wh/BLIP2_visual_textual/EvalCrafter_wh/MoonValley",
    "./AIGVD_1/dataset_wh/BLIP2_visual_textual/EvalCrafter_wh/pika",
    "./AIGVD_1/dataset_wh/BLIP2_visual_textual/EvalCrafter_wh/pika_v1_december_1",
    "./AIGVD_1/dataset_wh/BLIP2_visual_textual/EvalCrafter_wh/show_1",
    "./AIGVD_1/dataset_wh/BLIP2_visual_textual/EvalCrafter_wh/videocrafter-v1.0",
    "./AIGVD_1/dataset_wh/BLIP2_visual_textual/EvalCrafter_wh/zeroscope",
]

best_model_path = "./checkpoints/best.pt"

test_datasets = split_dataset_custom(test_pos_dirs, test_neg_path=args.test_neg_path, num_frames=args.num_frames)
test_loaders = [DataLoader(ds, batch_size=args.batch_size) for ds in test_datasets]

# Initialize model
model = ATSSModel(num_frames=args.num_frames).cuda()
model.load_state_dict(torch.load(best_model_path))

criterion = torch.nn.CrossEntropyLoss()


def calculate_metrics(loader, model, criterion):
    model.eval()
    all_preds, all_labels, losses = [], [], []
    with torch.no_grad():
        for sim_img, sim_txt, sim_cross, y in loader:
            sim_img = sim_img.cuda()
            sim_txt = sim_txt.cuda()
            sim_cross = sim_cross.cuda()
            y = y.cuda()

            logits = model(sim_img, sim_txt, sim_cross)
            y = y.squeeze()
            loss = criterion(logits, y)
            losses.append(loss.item())

            probs = torch.softmax(logits, dim=1)[:, 1]
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    preds_bin = [1 if p > 0.5 else 0 for p in all_preds]
    metrics = {
        'Loss': np.mean(losses) if losses else 0,
        'ACC': accuracy_score(all_labels, preds_bin),
        'AP': average_precision_score(all_labels, all_preds),
        'AUC': roc_auc_score(all_labels, all_preds),
        'Precision': precision_score(all_labels, preds_bin),
        'Recall': recall_score(all_labels, preds_bin),
        'F1': f1_score(all_labels, preds_bin)
    }
    return metrics


# Testing
with open(args.log_file, "a") as f:
    test_metrics_list = []
    for idx, test_loader in enumerate(test_loaders):
        test_metrics = calculate_metrics(test_loader, model, criterion)
        test_metrics_list.append(test_metrics)
        subset_name = os.path.basename(test_pos_dirs[idx])

        test_log_str = f"Test {subset_name} | Loss: {test_metrics['Loss']:.4f}, ACC: {test_metrics['ACC']:.4f}, AP: {test_metrics['AP']:.4f}, AUC: {test_metrics['AUC']:.4f}"
        print(test_log_str)
        f.write(test_log_str + "\n")

    # Calculate mean test metrics
    mean_test_metrics = {k: 0 for k in test_metrics_list[0]}
    for metrics in test_metrics_list:
        for k in mean_test_metrics:
            mean_test_metrics[k] += metrics[k]
    for k in mean_test_metrics:
        mean_test_metrics[k] /= len(test_metrics_list)

    mean_test_log_str = f"Mean Test | Loss: {mean_test_metrics['Loss']:.4f}, ACC: {mean_test_metrics['ACC']:.4f}, AP: {mean_test_metrics['AP']:.4f}, AUC: {mean_test_metrics['AUC']:.4f}"
    print(mean_test_log_str)
    f.write(mean_test_log_str + "\n")
