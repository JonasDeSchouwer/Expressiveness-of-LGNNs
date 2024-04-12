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
    epochs = 500
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    optimizer = 'Adam'      # like in LGLP
    learning_rate = 5e-3    # like in LGLP
    batch_size = 50         # like in LGLP


def evaluate(model: nn.Module, set: List[Graph], loss_func=None):
    """
    Returns a dict with keys "loss" and "acc" for the given model and dataset.
    """

    model.eval()

    loss = 0
    n_accurate = 0
    for graph in set:

        output = model(graph.node_feat, graph.edge_index, graph.line_edge_index, graph.index01)
        if loss_func is not None:
            loss += loss_func(output, graph.targets.to(torch.float)).item()

        prediction = torch.where(output > 0.5, 1, 0)
        n_accurate += 1 if (prediction == graph.targets).item() else 0

    return {
        "loss": loss,
        "acc": n_accurate/len(set)
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
        train_eval = evaluate(model, dataset.train_graphs, loss_func=loss_func)
        test_eval = evaluate(model, dataset.test_graphs, loss_func=loss_func)
        writer.add_scalar('Loss/train', train_eval["loss"], epoch)
        writer.add_scalar('Loss/test', test_eval["loss"], epoch)
        writer.add_scalar('Accuracy/train', train_eval["acc"], epoch)
        writer.add_scalar('Accuracy/test', test_eval["acc"], epoch)
        
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
    dataset.to(conf.device)

    train(node_conv_model, dataset, conf, run_name="node_conv")
    train(edge_conv_model, dataset, conf, run_name="edge_conv")


if __name__ == "__main__":
    main()