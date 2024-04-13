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
    input_dim = 52 # 50 features from the PPI dataset + 2 encoding the distance to the target nodes
    hidden_dim = None      # chosen to limit expressivity
    epochs = 150            # enough based on experiments
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    optimizer = 'Adam'      # like in LGLP
    learning_rate = 5e-3    # like in LGLP
    batch_size = 50         # like in LGLP
    eval_auc = False


def evaluate_breakdown(outputs: torch.Tensor, targets: torch.Tensor, threshold: float=0.5):
    """
    Breaks down the results (tp,fp,tn,fn) for a given threshold.
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
        "tpr": tp / (tp + fn) if tp + fn > 0 else 0,    # true positive rate
        "fpr": fp / (fp + tn) if fp + tn > 0 else 0,    # false positive rate
        "acc": (tp + tn) / len(outputs),
    }


def evaluate(model: nn.Module, set: List[Graph], conf: Configs, loss_func=None, eval_auc=False):
    """
    Returns a dict with keys "loss" and "acc" for the given model and dataset.
    """

    model.eval()

    loss = 0
    outputs = torch.zeros(len(set), dtype=torch.float, device=conf.device)
    targets = torch.zeros(len(set), dtype=torch.float, device=conf.device)
    for i, graph in enumerate(set):
        output = model(graph.node_feat, graph.edge_index, graph.line_edge_index, graph.index01)
        outputs[i] = output
        targets[i] = graph.targets.to(torch.float)
        if loss_func is not None:
            loss += loss_func(output, graph.targets.to(torch.float)).item()

    auc = None
    tprs = []
    fprs = []
    n_samples = 50
    if eval_auc:
        for threshold in torch.linspace(0, 1, n_samples):
            eval = evaluate_breakdown(outputs, targets, threshold=threshold)
            tprs.append(eval["tpr"])
            fprs.append(eval["fpr"])
        auc = torch.trapz(torch.tensor(tprs), torch.tensor(fprs))

    return {
        "loss": loss,
        "acc": evaluate_breakdown(outputs, targets)["acc"],
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
        train_eval = evaluate(model, dataset.train_graphs, conf=conf, loss_func=loss_func, eval_auc=conf.eval_auc)
        test_eval = evaluate(model, dataset.test_graphs, conf=conf, loss_func=loss_func, eval_auc=conf.eval_auc)
        writer.add_scalar('Loss/train', train_eval["loss"], epoch)
        writer.add_scalar('Loss/test', test_eval["loss"], epoch)
        writer.add_scalar('Accuracy/train', train_eval["acc"], epoch)
        writer.add_scalar('Accuracy/test', test_eval["acc"], epoch)
        if conf.eval_auc:
            writer.add_scalar('AUC/train', train_eval["auc"], epoch)
            writer.add_scalar('AUC/test', test_eval["auc"], epoch)
        
        if epoch % math.ceil(conf.epochs/100) == 0:
            print (f"epoch {epoch}: {train_eval['loss']}")

    writer.close()


def main():
    # Create configurations
    conf = Configs()
    train_node_conv = False
    train_edge_conv = True

    # Load dataset
    dataset = Dataset("code/data/PPIDataset")
    dataset_name = "PPI"

    # Remove graphs with too large line graphs
    dataset.train_graphs = [graph for graph in dataset.train_graphs if graph.line_edge_index.shape[1] < 250000]
    dataset.test_graphs = [graph for graph in dataset.test_graphs if graph.line_edge_index.shape[1] < 250000]
    print("After removing large line graphs")
    print("Number of train graphs:", len(dataset.train_graphs))
    print("Number of test graphs:", len(dataset.test_graphs))
    dataset.to(conf.device)

    if train_node_conv:
        node_conv_model = NodeConvGNN(conf.input_dim, hidden_dim=conf.hidden_dim).to(conf.device)
        try:
            train(node_conv_model, dataset, conf, run_name=f"{dataset_name}-node_conv")
        except Exception as e:
            raise e
        finally:
            torch.save(node_conv_model.state_dict(), f"study/runs/{dataset_name}-node_conv.pth")
            node_conv_model = None  # free memory

    if train_edge_conv:
        edge_conv_model = EdgeConvGNN(conf.input_dim, hidden_dim=conf.hidden_dim).to(conf.device)
        try:
            train(edge_conv_model, dataset, conf, run_name=f"{dataset_name}-edge_conv")
        except Exception as e:
            raise e
        finally:
            torch.save(edge_conv_model.state_dict(), f"study/runs/{dataset_name}-edge_conv.pth")
            edge_conv_model = None


if __name__ == "__main__":
    main()