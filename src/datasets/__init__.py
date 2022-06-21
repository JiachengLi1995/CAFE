from .interaction import ItemDataset

DATASETS = {
    ItemDataset.code(): ItemDataset,
}

def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
