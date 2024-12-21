import random
import torch
from torch.utils.data import Dataset


########################
######### Data #########
########################
class GraphData:
    def __init__(self):
        """
        Initializes an empty graph data object.
        """
        self.x = None             # Node features
        self.node_y = None        # Node labels
        self.device_params = None # Device parameters (None when pretrining!!!)
        self.edge_index = None    # Edge indices
        self.edge_attr = None     # Edge attributes (optional)
        self.edge_y = None        # Edge labels (optional)
        self.graph_attrs = {}     # Graph-level attributes (e.g., labels)

    def set_node_attributes(self, x):
        """
        Sets node features.

        Parameters:
        - x (Tensor): Node features tensor of shape (num_nodes, num_node_features)
        """
        self.x = x

    def set_node_labels(self, ny):
        """
        Sets node labels.

        Parameters:
        - y (Tensor): Node label tensor of shape (num_nodes, )
        """
        self.node_y = ny

    def set_device_params(self, device_params):
        """
        Sets device parameters.

        Parameters:
        - device_params (Tensor): Device parameters tensor of shape (num_nodes, )
        """
        self.device_params = device_params

    def set_edge_attributes(self, edge_index, edge_attr=None):
        """
        Sets edge indices and edge attributes.

        Parameters:
        - edge_index (Tensor): Edge indices tensor of shape (2, num_edges)
        - edge_attr (Tensor, optional): Edge attributes tensor of shape (num_edges, num_edge_features)
        """
        self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr

    def set_edge_labels(self, ey):
        """
        Sets edge labels.

        Parameters:
        - ey (Tensor): Edge label tensor of shape (num_edges, )
        """
        self.edge_y = ey

    def set_graph_attributes(self, **kwargs):
        """
        Sets graph-level attributes.

        Parameters:
        - kwargs: Key-value pairs of graph attributes (e.g., y=label)
        """
        for key, value in kwargs.items():
            self.graph_attrs[key] = value

    def __repr__(self):
        return f"GraphData(x={self.x}, node_y={self.node_y}, device_params={self.device_params}, "\
               f"edge_index={self.edge_index}, edge_attr={self.edge_attr}, edge_y={self.edge_y}, "\
               f"graph_attrs={self.graph_attrs})"


########################
####### Dataset ########
########################
class GraphDataset(Dataset):
    def __init__(self, graph_list):
        """
        Initializes the dataset with a list of GraphData objects.

        Parameters:
        - graph_list (List[GraphData]): A list of graph data objects.
        """
        self.graph_list = graph_list

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        return self.graph_list[idx]


########################
###### Dataloader ######
########################
class GraphDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        """
        Initializes the data loader.

        Parameters:
        - dataset (GraphDataset): The dataset to load data from.
        - batch_size (int): Number of graphs per batch.
        - shuffle (bool): Whether to shuffle the data at the start of each epoch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.dataset)))

    def __iter__(self):
        """
        Returns an iterator object.
        """
        if self.shuffle:
            random.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self):
        """
        Returns the next batch of data.
        """
        if self.current_idx >= len(self.indices):
            raise StopIteration

        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size

        batch_graphs = [self.dataset[idx] for idx in batch_indices]
        return self.collate_fn(batch_graphs)

    def collate_fn(self, batch_graphs):
        """
        Collates a list of GraphData objects into a single batch.

        Parameters:
        - batch_graphs (List[GraphData]): List of graphs to batch.

        Returns:
        - batch_data (dict): A dictionary containing batched data.
        """
        batch_data = {}
        node_offset = 0  # Keeps track of the cumulative number of nodes

        # Initialize lists to hold batched data
        x_list = []
        ny_list = []
        dp_list = []
        edge_index_list = []
        edge_attr_list = []
        ey_list = []
        batch_vector = []

        # Collect graph-level attributes
        graph_level_attrs = {}

        for i, graph in enumerate(batch_graphs):
            x = graph.x                    # Node features
            ny = graph.node_y              # Node labels
            dp = graph.device_params       # Device parameters
            edge_index = graph.edge_index  # Edge indices
            edge_attr = graph.edge_attr    # Edge attributes (optional)
            ey = graph.edge_y              # Edge labels (optional)
            graph_attrs = graph.graph_attrs
            num_nodes = x.size(0)

            # Adjust edge indices
            adjusted_edge_index = edge_index + node_offset

            # Append node features
            x_list.append(x)

            # Append node labels
            ny_list.append(ny)

            # Append device parameters
            if dp is not None: # None in Pretraining!!!
                dp_list.append(dp)

            # Append adjusted edge indices
            edge_index_list.append(adjusted_edge_index)

            # Append edge attributes if they exist
            if edge_attr is not None:
                edge_attr_list.append(edge_attr)

            # Append edge labels
            ey_list.append(ey)

            # Create batch vector indicating graph membership
            batch_vector.append(torch.full((num_nodes,), i, dtype=torch.long))

            # Collect graph-level attributes
            for key, value in graph_attrs.items():
                if key not in graph_level_attrs:
                    graph_level_attrs[key] = []
                graph_level_attrs[key].append(torch.full((1,), value, dtype=torch.float32))

            # Update node offset
            node_offset += num_nodes

        # Concatenate all the lists
        batch_data['x'] = torch.cat(x_list, dim=0)
        batch_data['node_y'] = torch.cat(ny_list, dim=0)
        if dp_list != []:
            batch_data['device_params'] = torch.cat(dp_list, dim=0)
        batch_data['edge_index'] = torch.cat(edge_index_list, dim=1)
        if edge_attr_list:
            batch_data['edge_attr'] = torch.cat(edge_attr_list, dim=0)
        batch_data['edge_y'] = torch.cat(ey_list, dim=0)
        batch_data['batch'] = torch.cat(batch_vector, dim=0)

        # Stack graph-level attributes
        for key, value_list in graph_level_attrs.items():
            batch_data[key] = torch.concat(value_list, dim=0)

        return batch_data


if __name__ == "__main__":
    # Example usage

    # Create some sample graphs
    graph_list = []

    for i in range(5):
        num_nodes = random.randint(3, 6)
        num_edges = random.randint(4, 8)

        x = torch.randn(num_nodes, 3)  # Random node features
        edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Random edge indices
        edge_attr = torch.randn(num_edges, 2)  # Random edge attributes

        graph = GraphData()
        graph.set_node_attributes(x)
        graph.set_edge_attributes(edge_index, edge_attr)
        graph.set_graph_attributes(y=i)  # Some label

        graph_list.append(graph)

    dataset = GraphDataset(graph_list)
    dataloader = GraphDataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataloader:
        print()
        print(batch)
        print()
        break