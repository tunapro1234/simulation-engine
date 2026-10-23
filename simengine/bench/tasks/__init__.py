"""Built-in bench tasks; importing this package registers them.

    from simengine import bench
    bench.list()                                   # every registered task name
    from simengine.bench.tasks import movielens    # module-level helpers (metrics, readers, persona text) per task
    task = bench.get("movielens", target="next_item")

Dataset adapters: bank, gss, open_ecommerce, opinionqa, simbench, twin, upworthy (the first seven); instacart, m5,
movielens, retailrocket (purchase + next item), risky_choice (choices13k + cpc18), socsci210, subpop (subpop_opinionqa
+ global_opinionqa), swissmetro, uplift (hillstrom / criteo / x5) and web_behaviour (webshop next action + buy).
Economic-game tasks: econ_games_b01, ertan_wtp, beebench_b01_prior. Per-task docs: docs/modules/bench/<name>.md.
"""
from .bank import BankMarketing
from .beebench_b01_prior import BeeBenchB01Prior
from .econ_games_b01 import EconGamesB01
from .ertan_wtp import ErtanWTP
from .gss import GSSWave
from .instacart import Instacart
from .m5 import M5
from .movielens import MovieLens1M
from .open_ecommerce import OpenEcommerce
from .opinionqa import OpinionQAGroups
from .retailrocket import RetailRocketNext, RetailRocketPurchase
from .risky_choice import CPC18, Choices13k
from .simbench import SimBench
from .socsci210 import SocSci210
from .subpop import GlobalOpinionQA, SubPOPOpinionQA
from .swissmetro import SwissmetroModeChoice
from .twin import Twin2K500
from .uplift import CriteoUplift, HillstromUplift, X5Uplift
from .upworthy import Upworthy
from .web_behaviour import WebShopBuy, WebShopNextAction

__all__ = ["BankMarketing", "BeeBenchB01Prior", "CPC18", "Choices13k", "CriteoUplift", "EconGamesB01", "ErtanWTP",
           "GSSWave", "GlobalOpinionQA", "HillstromUplift", "Instacart", "M5", "MovieLens1M", "OpenEcommerce",
           "OpinionQAGroups", "RetailRocketNext", "RetailRocketPurchase", "SimBench", "SocSci210", "SubPOPOpinionQA",
           "SwissmetroModeChoice", "Twin2K500", "Upworthy", "WebShopBuy", "WebShopNextAction", "X5Uplift"]
