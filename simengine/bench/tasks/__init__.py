"""Built-in bench tasks; importing this package registers them."""
from .bank import BankMarketing
from .gss import GSSWave
from .open_ecommerce import OpenEcommerce
from .opinionqa import OpinionQAGroups
from .simbench import SimBench
from .twin import Twin2K500
from .upworthy import Upworthy

__all__ = ["BankMarketing", "GSSWave", "OpenEcommerce", "OpinionQAGroups", "SimBench", "Twin2K500", "Upworthy"]
