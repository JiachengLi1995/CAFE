from .sasrec_model import SASRecModel
from .ssept_model import SSEPTModel
from .fdsa_model import FDSAModel
from .pop_model import POPModel
from .cfrec_local_model import CFRecLocalModel
from .cfrec_model import CFRecModel
from .nova_model import NovaModel
from .cfrec_joint_model import CFRecJointModel
from .cfrec_local_joint_model import CFRecLocalJointModel

MODELS = {
    "sasrec": SASRecModel,
    "ssept": SSEPTModel,
    "fdsa": FDSAModel,
    "pop": POPModel,
    "cfrec-local": CFRecLocalModel,
    "cfrec": CFRecModel,
    "nova": NovaModel,
    "cfrec-joint": CFRecJointModel,
    "cfrec-local-joint": CFRecLocalJointModel
}

def model_factory(args, pretrained_item_vectors=None):
    model = MODELS[args.model_code]
    return model(args, pretrained_item_vectors)
