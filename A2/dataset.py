import torch
import pickle


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = []
        self._preprocess_data(data_path)

    def _preprocess_data(self, data_path):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # TODO YOUR CODE HERE
        return {
            "state": torch.tensor(item[0], dtype=torch.float32),
            "action": torch.tensor(item[1], dtype=torch.long),
        }


if __name__ == "__main__":
    data_path = "CartPole-v1_dataset.pkl"
    dataset = Dataset(data_path)
    print(len(dataset))
    print(dataset[0])
    # The dimensionality of state and action
    print("State dimensionality:", dataset[0]["state"].shape)
    print("Action dimensionality:", dataset[0]["action"].shape)
    # The range of state and action
    all_states = torch.stack([dataset[i]["state"] for i in range(len(dataset))])
    all_actions = torch.stack([dataset[i]["action"] for i in range(len(dataset))])
    print(
        "State range:",
        (all_states.min(dim=0).values.tolist(), all_states.max(dim=0).values.tolist()),
    )
    print("Action range:", (all_actions.min().item(), all_actions.max().item()))
