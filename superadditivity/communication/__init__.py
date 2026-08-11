"""Communication primitives for decentralized federated learning.

Re-exports
----------
GossipMixer
    W-weighted gossip mixing of client model parameters.
"""

from superadditivity.communication.gossip_mixer import GossipMixer

__all__ = ["GossipMixer"]
