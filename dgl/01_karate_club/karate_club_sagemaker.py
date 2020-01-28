import dgl
import numpy as np
import pickle, os, argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

def build_karate_club_graph(edges):
    g = dgl.DGLGraph()
    g.add_nodes(node_count)
    src, dst = tuple(zip(*edges))
    g.add_edges(src, dst)
    # edges are directional in DGL; make them bidirectional
    g.add_edges(dst, src)
    return g

# Define the message and reduce function
# NOTE: We ignore the GCN's normalization constant c_ij for this tutorial.
def gcn_message(edges):
    # The argument is a batch of edges.
    # This computes a (batch of) message called 'msg' using the source node's feature 'h'.
    return {'msg' : edges.src['h']}

def gcn_reduce(nodes):
    # The argument is a batch of nodes.
    # This computes the new 'h' features by summing received 'msg' in each node's mailbox.
    return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}

# Define the GCNLayer module
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        # g is the graph and the inputs is the input node features
        # first set the node features
        g.ndata['h'] = inputs
        # trigger message passing on all edges
        g.send(g.edges(), gcn_message)
        # trigger aggregation at all nodes
        g.recv(g.nodes(), gcn_reduce)
        # get the result node features
        h = g.ndata.pop('h')
        # perform linear transformation
        return self.linear(h)
    
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        h = self.softmax(h)
        return h

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--node_count', type=int)
    args, _    = parser.parse_known_args()
    epochs     = args.epochs
    node_count = args.node_count
    
    training_dir = os.environ['SM_CHANNEL_TRAINING']
    model_dir    = os.environ['SM_MODEL_DIR']

    # Load edges from pickle file
    with open(os.path.join(training_dir, 'edge_list.pickle'), 'rb') as f:
        edge_list = pickle.load(f)
    print(edge_list)
    
    G = build_karate_club_graph(edge_list)
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())

    # The first layer transforms input features of size of 34 to a hidden size of 5.
    # The second layer transforms the hidden layer and produces output features of
    # size 2, corresponding to the two groups of the karate club.
    net = GCN(node_count, 5, 2)

    inputs = torch.eye(node_count)
    labeled_nodes = torch.tensor([0, node_count-1])  # only the instructor and the president nodes are labeled
    labels = torch.tensor([0,1])  # their labels are different

    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    all_preds = []

    for epoch in range(epochs):
        preds = net(G, inputs)
        all_preds.append(preds)
        # we only compute loss for labeled nodes
        loss = F.cross_entropy(preds[labeled_nodes], labels)
        optimizer.zero_grad() # PyTorch accumulates gradients by default
        loss.backward() 
        optimizer.step()
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

    last_epoch = all_preds[epochs-1].detach().numpy()
    predicted_class = np.argmax(last_epoch, axis=-1)

    print(predicted_class)

    torch.save(net.state_dict(), os.path.join(model_dir, 'karate_club.pt'))

