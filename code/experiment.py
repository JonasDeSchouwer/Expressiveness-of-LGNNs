import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import EdgeConvGNN, NodeConvGNN
from dataset import Dataset, Graph



class Configs:
    """
    ## Configurations
    Mostly the same hyperparameters were taken as in the LGLP experiments, to be found on https://github.com/LeiCaiwsu/LGLP
    """
    input_dim = 130 # 128 features from the Twitch dataset + 2 encoding the distance to the target nodes
    hidden_dim = 20      # chosen to limit expressivity
    epochs = 80            # enough based on experiments
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    optimizer = 'Adam'      # like in LGLP
    learning_rate = 5e-3    # like in LGLP
    batch_size = 50         # like in LGLP


def evaluate(outputs: torch.Tensor, targets: torch.Tensor, threshold: float=0.5):
    """
    Computes the accuracy with the given threshold
    """
    predictions = torch.where(outputs > threshold, 1, 0)
    tp = (predictions * targets).sum().item()
    tn = ((1 - predictions) * (1 - targets)).sum().item()
    fp = (predictions * (1 - targets)).sum().item()
    fn = ((1 - predictions) * targets).sum().item()
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "acc": (tp + tn) / len(outputs),
    }


def evaluate(model: nn.Module, set: List[Graph], loss_func=None, eval_auc=False):
    """
    Returns a dict with keys "loss" and "acc" for the given model and dataset.
    """

    model.eval()

    loss = 0
    outputs = torch.zeros(len(set), dtype=torch.float, device=model.device)
    targets = torch.zeros(len(set), dtype=torch.float, device=model.device)
    for i, graph in enumerate(set):
        output = model(graph.node_feat, graph.edge_index, graph.line_edge_index, graph.index01)
        outputs[i] = output
        targets[i] = graph.targets.to(torch.float)
        if loss_func is not None:
            loss += loss_func(output, graph.targets.to(torch.float)).item()

    auc = None
    tps = []
    fps = []
    n_samples = 50
    if eval_auc:
        for threshold in torch.linspace(0, 1, n_samples):
            eval = evaluate(outputs, targets, threshold=threshold)
            tps.append(eval["tp"])
            fps.append(eval["fp"])
        auc = torch.trapz(torch.tensor(tps), torch.tensor(fps))

    return {
        "loss": loss,
        "acc": evaluate(outputs, targets)["acc"],
        "auc": auc,
    }

def train(model: nn.Module, dataset: Dataset, conf: Configs, run_name=None):
    # define optimizer
    if conf.optimizer == 'Adam':
        optim = torch.optim.Adam(
            model.parameters(),
            lr=conf.learning_rate,
        )
    else:
        raise Exception(f"optimizer {conf.optimizer} unknown")
    
    # SummaryWriter for logging metrics
    writer = SummaryWriter(f"study/runs/{run_name}") if run_name is not None else SummaryWriter()
    
    # define loss function
    loss_func = nn.BCELoss()


    for epoch in tqdm(range(conf.epochs)):
        model.train()

        for batch in dataset.iter_batches(batch_size=conf.batch_size):
            optim.zero_grad()

            loss = torch.tensor(0).float().to(conf.device)
            for graph in batch:
                output = model(graph.node_feat, graph.edge_index, graph.line_edge_index, graph.index01)
                loss += loss_func(output, graph.targets.to(torch.float))
            loss /= len(batch)
            loss.backward()
            optim.step()

        # Log metrics
        model.eval()
        train_eval = evaluate(model, dataset.train_graphs, loss_func=loss_func, eval_auc=True)
        test_eval = evaluate(model, dataset.test_graphs, loss_func=loss_func, eval_auc=True)
        writer.add_scalar('Loss/train', train_eval["loss"], epoch)
        writer.add_scalar('Loss/test', test_eval["loss"], epoch)
        writer.add_scalar('Accuracy/train', train_eval["acc"], epoch)
        writer.add_scalar('Accuracy/test', test_eval["acc"], epoch)
        writer.add_scalar('AUC/train', train_eval["auc"], epoch)
        writer.add_scalar('AUC/test', test_eval["auc"], epoch)
        
        if epoch % math.ceil(conf.epochs/100) == 0:
            print (f"epoch {epoch}: {train_eval['loss']}")

    writer.close()


def main():
    # Create configurations
    conf = Configs()

    # Create models
    node_conv_model = NodeConvGNN(conf.input_dim).to(conf.device)
    edge_conv_model = EdgeConvGNN(conf.input_dim).to(conf.device)

    # Load dataset
    dataset = Dataset("code/data/TwitchENDataset")
    dataset_name = "TwitchEN"
    dataset.to(conf.device)

    try:
        train(node_conv_model, dataset, conf, run_name=f"{dataset_name}-node_conv")
    except Exception as e:
        print(e)
    try:
        train(edge_conv_model, dataset, conf, run_name=f"{dataset_name}-edge_conv")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()