"""Built-in bench tasks; importing this package registers them."""
from .bank import BankMarketing
from .beebench_b01_prior import BeeBenchB01Prior
from .econ_games_b01 import EconGamesB01
from .ertan_wtp import ErtanWTP
from .gss import GSSWave
from .open_ecommerce import OpenEcommerce
from .opinionqa import OpinionQAGroups
from .simbench import SimBench
from .twin import Twin2K500
from .upworthy import Upworthy

__all__ = ["BankMarketing", "BeeBenchB01Prior", "EconGamesB01", "ErtanWTP", "GSSWave", "OpenEcommerce", "OpinionQAGroups",
           "SimBench", "Twin2K500", "Upworthy"]
