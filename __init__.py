from .conditioning_rebalance import (
    RebalanceGuider,
    StepRebalance,
    RebalanceCFG,
)
from .krea2 import (
    ConditioningKrea2Rebalance,
    Krea2EditRebalance,
    Krea2EncodeRebalance,
)
from .ideogram4 import (
    ConditioningIdeogram4Rebalance,
    #Ideogram4EditRebalance,
    Ideogram4EncodeRebalance,
)

NODE_CLASS_MAPPINGS = {
    "RebalanceGuider": RebalanceGuider,
    "StepRebalance": StepRebalance,
    "RebalanceCFG": RebalanceCFG,
    "ConditioningKrea2Rebalance": ConditioningKrea2Rebalance,
    "Krea2EditRebalance": Krea2EditRebalance,
    "Krea2EncodeRebalance": Krea2EncodeRebalance,
    "ConditioningIdeogram4Rebalance": ConditioningIdeogram4Rebalance,
    #"Ideogram4EditRebalance": Ideogram4EditRebalance,
    "Ideogram4EncodeRebalance": Ideogram4EncodeRebalance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RebalanceGuider": "Rebalance Guider",
    "StepRebalance": "Step Rebalance",
    "RebalanceCFG": "Rebalance CFG Custom",
    "ConditioningKrea2Rebalance": "Conditioning Krea2 Rebalance",
    "Krea2EditRebalance": "Krea 2 Image Edit Rebalance",
    "Krea2EncodeRebalance": "Krea 2 Encode Rebalance",
    "ConditioningIdeogram4Rebalance": "Conditioning Ideogram4 Rebalance",
    #"Ideogram4EditRebalance": "Ideogram 4 Image Edit Rebalance",
    "Ideogram4EncodeRebalance": "Ideogram 4 Encode Rebalance",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
