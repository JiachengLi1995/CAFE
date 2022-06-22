from .cafe_model import CFRecLocalJointModel

MODELS = {
    "cafe": CFRecLocalJointModel
}

def model_factory(args, pretrained_item_vectors=None):
    model = MODELS[args.model_code]
    return model(args, pretrained_item_vectors)
