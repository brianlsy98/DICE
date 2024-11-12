
import random
import torch
from torch.utils.data import Dataset


########################
######### Data #########
########################
class HeteroGraphData:
    def __init__(self):
        """
        Initializes an empty heterogeneous graph data object.
        """
        # Dictionary to store node features for each node type
        # Format: { node_type: {'x': tensor, 'attr1': tensor, ...} }
        self.node_dict = {}
        
        # Dictionary to store edge indices and attributes for each edge type
        # Edge type is a tuple: (source_node_type, relation_type, target_node_type)
        # Format: { edge_type: {'edge_index': tensor, 'edge_attr': tensor, ...} }
        self.edge_dict = {}
        
        # Optional graph-level attributes (e.g., labels)
        self.graph_attrs = {}
    
    def add_node(self, node_type, **kwargs):
        """
        Adds node features and attributes for a given node type.

        Parameters:
        - node_type (str): The type of the nodes.
        - kwargs: Key-value pairs of node attributes (e.g., x=node_features).
        """
        if node_type not in self.node_dict:
            self.node_dict[node_type] = {}
        for key, value in kwargs.items():
            self.node_dict[node_type][key] = value
    
    def add_edge(self, edge_type, **kwargs):
        """
        Adds edge indices and attributes for a given edge type.

        Parameters:
        - edge_type (tuple): A tuple (src_node_type, relation_type, dst_node_type).
        - kwargs: Key-value pairs of edge attributes (e.g., edge_index, edge_attr).
        """
        if edge_type not in self.edge_dict:
            self.edge_dict[edge_type] = {}
        for key, value in kwargs.items():
            self.edge_dict[edge_type][key] = value
    
    def set_graph_attributes(self, **kwargs):
        """
        Sets graph-level attributes.

        Parameters:
        - kwargs: Key-value pairs of edge attributes (e.g., y=label).
        """
        for key, value in kwargs.items():
            self.graph_attrs[key] = value



########################
####### Dataset ########
########################
class HeteroGraphDataset(Dataset):
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
class HeteroGraphDataLoader:
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
        Collates a list of HeteroGraphData objects into a single batch.

        Parameters:
        - batch_graphs (List[HeteroGraphData]): List of graphs to batch.

        Returns:
        - batch_data (dict): A dictionary containing batched data.
        """
        batch_data = {'edge_types': [], 'node_types': []}
        node_offset = {}  # Keeps track of the cumulative number of nodes per node type

        # Initialize structures to hold batched data
        # For each node type
        node_types = set()
        for graph in batch_graphs:
            node_types.update(graph.node_dict.keys())
        for ntype in node_types:
            batch_data['node_types'].append(ntype)
            batch_data[ntype] = {'dc_voltages': [], 'batch': []}
            node_offset[ntype] = 0

        # For each edge type
        edge_types = set()
        for graph in batch_graphs:
            edge_types.update(graph.edge_dict.keys())
        for etype in edge_types:
            batch_data['edge_types'].append(etype)
            batch_data[etype] = {'edge_index': []}



        # Process each graph in the batch
        for i, graph in enumerate(batch_graphs):
            # print()
            # print(graph.node_dict)
            # print(graph.edge_dict)
            # print(graph.graph_attrs)

            # Process node features
            for ntype, node_attrs in graph.node_dict.items():
                dc_v = node_attrs['dc_voltages']  # Node features
                num_nodes = dc_v.size(0)

                # Append node features
                batch_data[ntype]['dc_voltages'].append(dc_v)

                # Create batch vector indicating graph membership
                batch_vector = torch.full((num_nodes,), i, dtype=torch.long)
                batch_data[ntype]['batch'].append(batch_vector)

                # Update node offset for this node type
                node_offset[ntype] += num_nodes

            # Process edge indices
            for etype, edge_attrs in graph.edge_dict.items():
                edge_index = edge_attrs['edge_index']  # Shape: (2, num_edges)

                # Get source and target node types
                src_type, rel_type, dst_type = etype

                # Adjust edge indices
                src_offset = node_offset[src_type] - graph.node_dict[src_type]['dc_voltages'].size(0)
                dst_offset = node_offset[dst_type] - graph.node_dict[dst_type]['dc_voltages'].size(0)

                adjusted_edge_index = edge_index.clone()
                adjusted_edge_index[0] += src_offset
                adjusted_edge_index[1] += dst_offset

                batch_data[etype]['edge_index'].append(adjusted_edge_index)

                # Handle edge attributes if they exist
                if 'edge_attr' in edge_attrs:
                    if 'edge_attr' not in batch_data[etype]:
                        batch_data[etype]['edge_attr'] = []
                    if rel_type == 'nmos' or rel_type == 'pmos':
                        
                        adjusted_edge_attrs = edge_attrs['edge_attr'].clone()

                        vg_input_mask = adjusted_edge_attrs[:,0] == 0 # True if Vgate type is "input"
                        vb_input_mask = adjusted_edge_attrs[:,2] == 0 # True if Vbulk type is "input"

                        adjusted_edge_attrs[vg_input_mask, 1] += node_offset['input'] - graph.node_dict['input']['dc_voltages'].size(0)
                        adjusted_edge_attrs[~vg_input_mask, 1] += node_offset['output'] - graph.node_dict['output']['dc_voltages'].size(0)
                        adjusted_edge_attrs[vb_input_mask, 3] += node_offset['input'] - graph.node_dict['input']['dc_voltages'].size(0)
                        adjusted_edge_attrs[~vb_input_mask, 3] += node_offset['output'] - graph.node_dict['output']['dc_voltages'].size(0)
                        
                        batch_data[etype]['edge_attr'].append(adjusted_edge_attrs)
                    else: batch_data[etype]['edge_attr'].append(edge_attrs['edge_attr'])

            # Collect graph-level attributes
            for key, value in graph.graph_attrs.items():
                if key not in batch_data:
                    batch_data[key] = []
                batch_data[key].append(torch.tensor(value, dtype=torch.float))

        # Concatenate node features and batch vectors
        for ntype in node_types:
            batch_data[ntype]['dc_voltages'] = torch.cat(batch_data[ntype]['dc_voltages'], dim=0)
            batch_data[ntype]['batch'] = torch.cat(batch_data[ntype]['batch'], dim=0)

        # Concatenate edge indices and attributes
        for etype in edge_types:
            batch_data[etype]['edge_index'] = torch.cat(batch_data[etype]['edge_index'], dim=1)
            if 'edge_attr' in batch_data[etype]:
                batch_data[etype]['edge_attr'] = torch.cat(batch_data[etype]['edge_attr'], dim=0)
        

        # Stack labels
        for key, value in batch_data.items():
            if key not in node_types and key not in edge_types and\
                key != 'edge_types' and key != 'node_types':
                    if isinstance(value, list):
                        batch_data[key] = torch.stack(value, dim=0)

        return batch_data



if __name__ == "__main__":

    dataset = torch.load('./dc.pt')
    train_data = dataset['train_data']
    dataloader = HeteroGraphDataLoader(train_data, batch_size=2, shuffle=True)

    for batch in dataloader:
        print()
        print(batch)
        print()
        break
