from src.models import model_factory
from src.dataloaders import dataloader_factory
from src.datasets import dataset_factory
from src.trainers import trainer_factory
from src.utils.options import parser
from src.utils.utils import *

if __name__ == '__main__':

    # hyper-parameter config
    args = parser.parse_args()
    ckpt_root = setup_train(args)

    # dataset and data loader
    dataset = dataset_factory(args)
    train_loader, val_loader, test_loader, dataset = dataloader_factory(args, dataset)

    # pretrained products vectors
    pretrained_item_vectors = None
    if args.use_pretrained_vectors:
        pretrained_item_vectors = dataset.meta['p2v']
    
    # model setup
    model = model_factory(args, pretrained_item_vectors)
    if args.load_pretrained_weights is not None:
        print("weights loading from %s ..." % args.load_pretrained_weights)
        model = load_pretrained_weights(model, args.load_pretrained_weights)
    print("Model size:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # trainer setup
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, ckpt_root, dataset.data)
    
    # model training
    trainer.train()

    # model testing and saving
    trainer.test()
    trainer.logger_service.complete({'state_dict': (trainer._create_state_dict())})
    trainer.writer.close()
