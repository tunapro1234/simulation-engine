"""Game library: 15 one-shot economic games as plain GAME dicts, a contract check and the prompt renderer.

Provenance. 14 games are ported verbatim (text, conditions, bounds, references) from the edsl-test BeeBench b01
battery (`benchmark/beebench/b01/games/*.py`: 10 calibration games, `holdout/*.py`: 4 holdout games; same owner,
internal reuse). The 15th, `ertan_vignette`, is new wording built on the parameters of the replicant project's
one-shot Ertan, Page & Putterman (2009) vignette (`papers/ertan2009/config.py`: ENDOWMENT, GROUP_SIZE, MPCR,
PUNISHMENT_RATIO, PROFILES; replicant is MIT licensed). The oTree app text itself is not ported.

GAME contract (unchanged from b01):

    {
      "game": "dictator",                 unique snake_case name
      "role": "dictator",                 decision role label
      "holdout": False,                   holdout games are never used for persona tuning
      "measures": "altruism",             the parameter this game identifies
      "conditions": [{"id": "base", ...params}],   >= 1; params fill the text; a condition may carry its own
                                          "min"/"max" that override the game-level bounds
      "text": "...",                      one canonical wording, text.format(**params); the persona is NOT part of it
      "question_type": "numerical" | "mc",
      "options": [...],                   mc only: exact option strings
      "min": 0, "max": 10,                numerical only
      "belief": None | {"text": "...", "min": .., "max": ..},   numerical, asked BEFORE the decision
      "references": {"fact": "value + SOURCE (paper, table/page)"},
    }

Optional, not used by the shipped games: "wordings": {wording_id: {"text": ..., "belief_text": ...}} for
paraphrase variants (wording_id 0 is always the canonical "text").
"""
from __future__ import annotations

import copy
import string
from typing import Any, Mapping

# ---------------------------------------------------------------- calibration games (b01/games)

DICTATOR = {
    "game": "dictator",
    "role": "dictator",
    "holdout": False,
    "measures": "altruism",
    "conditions": [{"id": "base", "pot": 100}],
    "text": (
        "You have ${pot}. You may give any amount of it to a stranger and keep "
        "the rest. The stranger has no say in the matter and you will never "
        "meet them. How many dollars do you give to the stranger (0 to {pot})?"
    ),
    "question_type": "numerical",
    "min": 0,
    "max": 100,
    "belief": None,
    "references": {
        "mean_give_share": "0.2835 of the pie — Engel 2011 dictator meta-analysis, MPI preprint 2010/07 ('dictators on average give 28.35 % of the pie')",
        "share_give_zero": "0.3611 — Engel 2011, same source",
        "share_equal_split": "0.1674 — Engel 2011, same source",
    },
}

ELEVEN_TWENTY = {
    "game": "eleven_twenty",
    "role": "requester",
    "holdout": False,
    "measures": "level_k_depth",
    "conditions": [{"id": "base", "lo": 11, "hi": 20, "bonus": 20}],
    "text": (
        "You and another player are playing a one-shot game. The other player is "
        "anonymous and you will never meet them. Each player requests an amount "
        "of money; the amount must be an integer between {lo} and {hi} dollars. "
        "Each player receives the amount they request. A player receives an "
        "additional {bonus} dollars if they ask for exactly one dollar less than "
        "the other player. What amount of money do you request ({lo} to {hi})?"
    ),
    "question_type": "numerical",
    "min": 11,
    "max": 20,
    "belief": None,
    "references": {
        "choice_distribution_pct": "11:4, 12:0, 13:3, 14:6, 15:1, 16:6, 17:32, 18:30, 19:12, 20:6 — Arad & Rubinstein 2012 AER 102(7), Table 1 p.3565 (basic version, n=108)",
        "modal_choices": "17 (32%) and 18 (30%) — same Table 1",
        "share_17_to_19": "0.74 — 'The vast majority of subjects (74 percent) chose the actions 17-18-19' (p.3565); 17-20 = 0.80 (p.3566)",
        "level0_action": "20 — 'The choice of 20 is a natural anchor for an iterative reasoning process' (p.3562)",
        "level_k_type_estimates": "L0=0.05, L1=0.13, L2=0.39, L3=0.43, error rate 0.32 — footnote 6, p.3566",
        "wording_note": "text follows the paper's basic-version instructions (p.3562) with two spec-mandated clauses NOT in the original: 'one-shot game' and 'The other player is anonymous and you will never meet them' (in the experiment, subjects were classmates seated in the same classroom, randomly matched); original currency was shekels (Tel Aviv University students), b.01 uses dollars",
    },
}

HOLT_LAURY = {
    "game": "holt_laury",
    "role": "chooser",
    "holdout": False,
    "measures": "risk_aversion",
    # Table 1 (low payoffs): row k gives the high payoff a k-in-10 chance in BOTH options.
    "conditions": [
        {"id": f"row{k}", "p": k, "q": 10 - k,
         "a_hi": "2.00", "a_lo": "1.60", "b_hi": "3.85", "b_lo": "0.10"}
        for k in range(1, 11)
    ],
    "text": (
        "You must choose between two lotteries. Option A pays ${a_hi} with a "
        "{p} in 10 chance and ${a_lo} with a {q} in 10 chance. Option B pays "
        "${b_hi} with a {p} in 10 chance and ${b_lo} with a {q} in 10 chance. "
        "You will be paid the outcome of the lottery you choose. Which do you "
        "choose: Option A or Option B?"
    ),
    "question_type": "mc",
    "options": ["Option A", "Option B"],
    "belief": None,
    "references": {
        "option_a_payoffs": "$2.00 / $1.60 — Holt & Laury 2002, AER 92(5), Table 1 p.1645 (low-payoff treatment)",
        "option_b_payoffs": "$3.85 / $0.10 — Holt & Laury 2002, Table 1 p.1645",
        "risk_neutral_safe_choices": "4 (switches to B at row 5) — Holt & Laury 2002 pp.1645-46: 'a risk-neutral person would choose A four times before switching to B'",
        "share_risk_averse": "~2/3 make more than 4 safe choices at low real payoffs — Holt & Laury 2002 p.1648",
        "modal_safe_choices": "4 and 5 (0.26 each; 4-6 covers 0.75) — Holt & Laury 2002, Table 3 p.1649, 'low real' column",
    },
}

LOSS_GAMBLES = {
    "game": "loss_gambles",
    "role": "gambler",
    "holdout": False,
    "measures": "loss_aversion",
    "conditions": [
        {"id": "gain2", "gain": 2, "loss": 5},
        {"id": "gain4", "gain": 4, "loss": 5},
        {"id": "gain5", "gain": 5, "loss": 5},
        {"id": "gain6", "gain": 6, "loss": 5},
        {"id": "gain8", "gain": 8, "loss": 5},
        {"id": "gain10", "gain": 10, "loss": 5},
        # Ladder extension (b01 reviewer finding #2): ratios up to 4 so that a human-median lambda ~2.25
        # (indifference at $11.25) is not right-censored; follows Tom, Fox, Trepel & Poldrack 2007.
        {"id": "gain12", "gain": 12, "loss": 5},
        {"id": "gain16", "gain": 16, "loss": 5},
        {"id": "gain20", "gain": 20, "loss": 5},
    ],
    "text": (
        "You are offered the following one-time gamble. A fair coin is flipped "
        "once: if it lands heads, you win ${gain}; if it lands tails, you lose "
        "${loss} of your own money. If you reject the gamble, the coin is not "
        "flipped and you neither win nor lose anything. Do you accept or "
        "reject this gamble (Accept or Reject)?"
    ),
    "question_type": "mc",
    "options": ["Accept", "Reject"],
    "belief": None,
    "references": {
        "lambda_median": "2.25 — Tversky & Kahneman 1992, J. Risk and Uncertainty 5:297-323, pp. 311-312: 'The median λ was 2.25, indicating pronounced loss aversion' (quoted in Regenwetter et al. 2022, doi:10.1177/25152459221074653); linear value at these stakes puts indifference near gain = 2.25 x $5 = $11.25",
        "accept_threshold": "gain ≈ 2 x loss — TK 1992 loss-aversion test: median theta = gain/loss at indifference ≈ 2 in problems 1-6 (reproduced in Regenwetter et al. 2022, Table 3); predicts a typical subject rejects gain2..gain8 and accepts only around gain10 ($10 = 2 x $5)",
        "lambda_meta": "mean 1.955, 95% interval [1.820, 2.102] — Brown, Imai, Vieider & Camerer 2024 meta-analysis of 607 estimates from 150 articles, J. Economic Literature, doi:10.1257/jel.20221698",
    },
}

_PG_N, _PG_ENDOW, _PG_MULT = 4, 20, 2
_PG_MPCR = _PG_MULT / _PG_N          # fund doubled and split 4 ways: $0.50 back per $1 put in


def _pg_cond(cid: str, info: str) -> dict:
    return {"id": cid, "endowment": _PG_ENDOW, "n_players": _PG_N, "n_others": _PG_N - 1,
            "multiplier": _PG_MULT, "mpcr": _PG_MPCR, "info": info}


PUBLIC_GOODS_STRATEGY = {
    "game": "public_goods_strategy",
    "role": "contributor",
    "holdout": False,
    "measures": "cooperation (conditional-cooperator type)",
    "conditions": [
        _pg_cond("unconditional", "You decide without knowing what the other players contribute."),
    ] + [
        _pg_cond(f"cond{x}", f"Suppose the other {_PG_N - 1} players contribute ${x} each on average.")
        for x in (0, 5, 10, 15, 20)
    ],
    "text": (
        "You are in a group of {n_players} players: you and {n_others} others. "
        "Each player gets ${endowment} and decides how many dollars of it to "
        "put into a group fund, keeping the rest. The total in the fund is "
        "multiplied by {multiplier} and divided equally among all {n_players} "
        "players, no matter who contributed — so each dollar you put in pays "
        "back ${mpcr:.2f} to you and ${mpcr:.2f} to each other player. The "
        "game is played exactly once and the players are anonymous. {info} "
        "How many dollars do you put into the fund (0 to {endowment})?"
    ),
    "question_type": "numerical",
    "min": 0,
    "max": _PG_ENDOW,
    "belief": {
        "text": (
            "You are in a group of {n_players} players: you and {n_others} "
            "others. Each player gets ${endowment} and at the same time puts "
            "some of it into a group fund; the fund is multiplied by "
            "{multiplier} and divided equally among all {n_players} players. "
            "Before you decide, predict the average number of dollars the "
            "other {n_others} players put into the fund. If your prediction "
            "is within $1 of their true average, you earn an extra $2. What "
            "is your prediction (0 to {endowment})?"
        ),
        "min": 0,
        "max": _PG_ENDOW,
    },
    "references": {
        "share_conditional_cooperators": "0.50 (22/44) — Fischbacher, Gächter & Fehr 2001, Economics Letters 71:397-404, Section 3 p. 401 & Fig. 1",
        "share_free_riders": "0.295 (13/44; schedule of all zeros) — FGF 2001, Section 3 p. 401",
        "share_hump_shaped": "0.136 (6/44) — FGF 2001, Section 3 p. 401",
        "perfect_conditional_slope": "4/44 schedules exactly on the diagonal (slope 1); only 11.9% of conditional cooperators' entries strictly above it ('conditional cooperation with a self-serving bias') — FGF 2001, p. 401",
        "mean_unconditional_contribution": "6.7 of 20 tokens = 0.335 of endowment — FGF 2001, p. 401",
        "design_difference": "FGF payoff was 20 - g_i + 0.4*sum(g_j), i.e. MPCR 0.4 (eq. 1, p. 398); b.01 uses MPCR 0.5 (fund doubled) per the b.01 game-6 spec; FGF elicited all 21 entries with a random-payoff mechanism, b.01 asks five hypothetical rows; the belief is asked before every row and is meaningful on the 'unconditional' row only",
    },
}

TIME_CHOICES = {
    "game": "time_choices",
    "role": "chooser",
    "holdout": False,
    "measures": "patience + present bias",
    "conditions": [
        {"id": "now103", "sooner_amount": 100, "sooner_when": "today", "later_amount": 103, "later_when": "in 30 days"},
        {"id": "now110", "sooner_amount": 100, "sooner_when": "today", "later_amount": 110, "later_when": "in 30 days"},
        {"id": "now125", "sooner_amount": 100, "sooner_when": "today", "later_amount": 125, "later_when": "in 30 days"},
        {"id": "later103", "sooner_amount": 100, "sooner_when": "in 30 days", "later_amount": 103, "later_when": "in 60 days"},
        {"id": "later110", "sooner_amount": 100, "sooner_when": "in 30 days", "later_amount": 110, "later_when": "in 60 days"},
        {"id": "later125", "sooner_amount": 100, "sooner_when": "in 30 days", "later_amount": 125, "later_when": "in 60 days"},
    ],
    "text": (
        "You are choosing between two payments. If you choose the sooner "
        "payment, you receive ${sooner_amount} {sooner_when}. If you choose "
        "the later payment, you receive ${later_amount} {later_when}. Both "
        "payments are completely certain and will be paid exactly on time. "
        "Which do you choose: 'Sooner payment' (${sooner_amount} "
        "{sooner_when}) or 'Later payment' (${later_amount} {later_when})?"
    ),
    "question_type": "mc",
    "options": ["Sooner payment", "Later payment"],
    "belief": None,
    "references": {
        "present_bias_beta_money": "0.82, 95% CI [0.74, 0.90] (0.87 [0.82, 0.92] after correcting for selective reporting) — Cheung, Tymula & Wang 2021, 'Quasi-Hyperbolic Present Bias: A Meta-Analysis', IZA DP 14625, abstract",
        "canonical_beta_delta": "beta=0.7, delta=0.957 — canonical quasi-hyperbolic parameterization (Laibson 1997 tradition) in Cohen, Ericson, Laibson & White, 'Measuring Time Preferences', JEL 2020, Figure 2 caption; field structural estimate beta=0.50, delta=0.96 (Laibson, Maxted, Repetto & Tobacman 2017, as reported ibid., p. 22)",
        "discount_rate_spread": "implicit annual discount rates across studies range from -6 percent to infinity — Frederick, Loewenstein & O'Donoghue 2002, 'Time Discounting and Time Preference: A Critical Review', JEL 40(2), p. 377 and Table 1 pp. 378-79",
    },
}

TRUST_RETURNER = {
    "game": "trust_returner",
    "role": "returner",
    "holdout": False,
    "measures": "positive_reciprocity",
    # game-level max is the largest received amount; each condition carries its own bound (b01 reviewer #4)
    "conditions": [
        {"id": "low", "endow": 50, "sent": 10, "received": 30, "max": 30},
        {"id": "mid", "endow": 50, "sent": 30, "received": 90, "max": 90},
        {"id": "high", "endow": 50, "sent": 50, "received": 150, "max": 150},
    ],
    "text": (
        "A stranger was given ${endow} and chose to send you ${sent} of it. "
        "Every dollar sent was tripled, so you received ${received}. The "
        "stranger kept what they did not send. You now choose how many "
        "dollars of the ${received} to return to the stranger; you keep "
        "whatever you do not return, and the game ends. You will never meet "
        "the stranger. How many dollars do you return to the stranger "
        "(0 to {received})?"
    ),
    "question_type": "numerical",
    "min": 0,
    "max": 150,
    "belief": None,
    "references": {
        "mean_return_share_of_available": "0.37 of the tripled amount — Johnson & Mislin 2011, 'Trust games: A meta-analysis', J. Econ. Psychology 32(5):865-889 (162 replications); their working-paper version ('Cultures of Kindness', Mercatus/GMU) Table 1: retavail mean 0.3651, sd 0.0942, N=75",
        "mean_sent_share_of_endowment": "0.51 of endowment — same source, Table 1: sentfraction mean 0.5088, sd 0.1377, N=84",
        "trustworthiness_measure": "amount returned / amount available to return (= sent x multiplier), following Glaeser et al. 2000 — Johnson & Mislin 2011, Data section",
    },
}

TRUST_SENDER = {
    "game": "trust_sender",
    "role": "sender",
    "holdout": False,
    "measures": "trust",
    "conditions": [{"id": "base", "pot": 100, "mult": 3, "tripled_pot": 300,
                    "belief_bonus": 2, "belief_tol": 5}],
    "text": (
        "You have ${pot}. You may send any amount of it to a stranger and keep "
        "the rest. Every dollar you send is multiplied by {mult} before the "
        "stranger receives it. The stranger then decides how much of what they "
        "received, if anything, to send back to you; whatever they send back is "
        "yours to keep. You will never meet the stranger and the game is played "
        "only once. How many dollars do you send to the stranger (0 to {pot})?"
    ),
    "question_type": "numerical",
    "min": 0,
    "max": 100,
    "belief": {
        "text": (
            "You are about to play the following game: you have ${pot} and may "
            "send any amount of it to a stranger, keeping the rest. Every dollar "
            "you send is multiplied by {mult} before the stranger receives it. "
            "The stranger then decides how much of what they received, if "
            "anything, to send back to you. Before you decide, predict how many "
            "dollars the stranger will send back to you; a prediction within "
            "${belief_tol} of the true amount earns you a ${belief_bonus} bonus. "
            "How many dollars do you predict the stranger will send back "
            "(0 to {tripled_pot})?"
        ),
        "min": 0,
        "max": 300,
    },
    "references": {
        "mean_sent_share": "0.50 of endowment — Johnson & Mislin 2011, 'Trust games: A meta-analysis', J. Econ. Psychology 32(5):865-889 (162 replications, >23,000 subjects); authors' WP version ('Cultures of Kindness', Mercatus) Table 1: mean sentfraction = 0.509 (SD 0.138, n=84 replications)",
        "mean_return_share_of_available": "0.37 of the amount available to the receiver (belief target: ~0.37 x tripled amount) — Johnson & Mislin 2011, same source; WP Table 1: mean retavail = 0.365 (SD 0.094, n=75)",
    },
}

ULTIMATUM_PROPOSER = {
    "game": "ultimatum_proposer",
    "role": "proposer",
    "holdout": False,
    "measures": "strategic_fairness",
    "conditions": [{"id": "base", "pot": 100}],
    "text": (
        "You have ${pot} to split with a stranger. You propose how many "
        "dollars to offer them and keep the rest. The stranger then either "
        "accepts or rejects your proposal: if they accept, the money is split "
        "exactly as you proposed; if they reject, you BOTH get $0. This "
        "happens once and you will never meet them. How many dollars do you "
        "offer to the stranger (0 to {pot})?"
    ),
    "question_type": "numerical",
    "min": 0,
    "max": 100,
    "belief": None,
    "references": {
        "mean_offer_share": "0.40 of the pie — Oosterbeek, Sloof & van de Kuilen 2004 meta-analysis (37 papers, 75 results), Experimental Economics 7(2):171-188, abstract: 'on average the proposer offers 40% of the pie'",
        "share_offers_rejected": "0.16 — Oosterbeek, Sloof & van de Kuilen 2004, same source, abstract: 'On average 16% of the offers is rejected'",
    },
}

ULTIMATUM_RESPONDER = {
    "game": "ultimatum_responder",
    "role": "responder",
    "holdout": False,
    "measures": "negative_reciprocity",
    "conditions": [
        {"id": "low", "pot": 100, "offer": 10, "keep": 90},
        {"id": "mid", "pot": 100, "offer": 25, "keep": 75},
        {"id": "fair", "pot": 100, "offer": 40, "keep": 60},
    ],
    "text": (
        "A stranger was given ${pot} and had to propose how to split it with "
        "you. They offered you ${offer} and kept ${keep} for themselves. If "
        "you accept, you get ${offer} and they get ${keep}. If you reject, "
        "you both get $0. This is a one-time decision and you will never "
        "meet them. Do you accept the offer (Accept or Reject)?"
    ),
    "question_type": "mc",
    "options": ["Accept", "Reject"],
    "belief": None,
    "references": {
        "reject_rate_low_offers": "offers below 20% of the pie are rejected about half the time — Camerer 2003, Behavioral Game Theory, ch. 2 stylized facts ('Offers below 20 percent are very rare and they are rejected about half of the times', quoted in PMC4934685)",
        "reject_rate_fair_offers": "offers of 40-50% of the pie are rarely rejected — Camerer 2003, Behavioral Game Theory, ch. 2 stylized facts",
        "mean_rejection_rate": "0.16 of all offers rejected — Oosterbeek, Sloof & van de Kuilen 2004 meta-analysis, Experimental Economics 7(2):171-188, abstract ('On average 16% of the offers is rejected')",
        "mean_offer_share": "0.40 of the pie — Oosterbeek et al. 2004, abstract (context: the typical offer responders face)",
    },
}

# ---------------------------------------------------------------- holdout games (b01/holdout + ertan_vignette)

BEAUTY_CONTEST = {
    "game": "beauty_contest",
    "role": "guesser",
    "holdout": True,
    "measures": "strategic_depth",
    "conditions": [{"id": "base", "lo": 0, "hi": 100, "frac": "2/3", "prize": 20}],
    "text": (
        "You are playing a guessing game with many other players, all choosing "
        "at the same time. Each player picks a number between {lo} and {hi}. "
        "The player whose number is closest to {frac} of the average of all "
        "chosen numbers (including yours) wins ${prize}; everyone else gets "
        "nothing, and ties split the prize. What number do you pick "
        "({lo} to {hi})?"
    ),
    "question_type": "numerical",
    "min": 0,
    "max": 100,
    "belief": None,
    "references": {
        "first_period_mean": "36.73 — Nagel 1995 AER 85(5), Figure 1B (p=2/3 sessions 4-7), p. 1316",
        "first_period_median": "33 — Nagel 1995, Figure 1B, p. 1316",
        "spike_level1_33": "modal neighborhood interval 30-37 around 50*(2/3)=33 (iteration step 1) — Nagel 1995, Figure 2B, p. 1317",
        "spike_level2_22": "'about 25 percent of all observations, the second-highest frequency' in interval 20-25 around 50*(2/3)^2~22 (iteration step 2) — Nagel 1995, p. 1318",
        "share_choosing_zero": "0 — 'No subject chose 0 in the 2/3 and 1/2 sessions', Nagel 1995, p. 1316",
        "share_above_67": "0.10 chose numbers greater than 67 (weakly dominated); 0.06 chose 66 or 67 — Nagel 1995, p. 1316",
        "design_note": "original prize 20 DM (~$13) per round, winner-take-all with ties split, 15-18 subjects per session — Nagel 1995, pp. 1315-1316; b.01 uses $20",
    },
}

_FG = {"n": 4, "endow": 20, "mpcr": "0.40", "cost": 1, "fine": 3, "tol": 1, "bonus": 2}
_FG_PMAX = 10   # max deduction points per target (Fehr & Gächter 2002 Nature technology)
_FG_RULES = (
    "You are in a group of {n} people, playing exactly once and anonymously. "
    "Each player gets ${endow} and chooses how many dollars (0 to {endow}) "
    "to put into a group project, keeping the rest. Every dollar put into "
    "the project pays out ${mpcr} to EACH of the {n} players, no matter who "
    "put it in. After everyone has chosen, all contributions are revealed "
    "and each player may assign deduction points to any other player: each "
    "point costs the punisher ${cost} and reduces the punished player's "
    "earnings by ${fine}. "
)
_FG_ASK_CONTRIBUTE = (
    "It is now the contribution stage. How many dollars do you put into the "
    "group project (0 to {endow})?"
)
_FG_ASK_PUNISH = (
    "The contribution stage is over: you put in ${own}, and the other "
    "players put in ${c1}, ${c2}, and ${c3}. It is now the punishment "
    "stage. How many deduction points do you assign to the player who put "
    "in ${target} (0 to {pmax})?"
)

PG_PUNISHMENT = {
    "game": "pg_punishment",
    "role": "group_member",
    "holdout": True,
    "measures": "cooperation + altruistic punishment",
    "conditions": [
        {"id": "contribute", **_FG, "stage": _FG_ASK_CONTRIBUTE.format(**_FG)},
        {"id": "punish_freerider", "max": _FG_PMAX, **_FG,
         "stage": _FG_ASK_PUNISH.format(own=15, c1=2, c2=18, c3=15, target=2, pmax=_FG_PMAX)},
        {"id": "punish_cooperator", "max": _FG_PMAX, **_FG,
         "stage": _FG_ASK_PUNISH.format(own=15, c1=18, c2=15, c3=15, target=18, pmax=_FG_PMAX)},
    ],
    "text": _FG_RULES + "{stage}",
    "question_type": "numerical",
    "min": 0,
    "max": _FG["endow"],
    "belief": {
        "text": _FG_RULES + (
            "Before anything is decided, predict the average number of "
            "dollars the other players will put into the group project "
            "(0 to {endow}). If your prediction is within ${tol} of the "
            "true average you earn an extra ${bonus}. What is your "
            "prediction (0 to {endow})?"
        ),
        "min": 0,
        "max": _FG["endow"],
    },
    "references": {
        "design_params": "n=4, endowment y=20, marginal payoff a=0.4, up to ten punishment points per target — Fehr & Gächter 2000 AER 90(4):980-994, Section II.C pp. 982-983",
        "punish_technology": "each point costs the punisher 1 MU and the punished member 3 MUs, 0-10 points — Fehr & Gächter 2002 Nature 415:137-140, p. 137. NOTE: FG2000 AER instead used a convex point-cost schedule (Table 2, p. 983); b.01 adopts the simpler linear 2002 technology",
        "contrib_share_with_punishment": "0.58 of endowment (Stranger-treatment average) — FG2000 AER Result 1, p. 984; free-riders raise contributions by 10-12 tokens (50-60% of endowment) vs no punishment, p. 992",
        "punishment_rises_with_negative_deviation": "received points per token below others' average: +0.2428*** Stranger / +0.4168*** Partner, positive deviation insignificant — FG2000 AER Result 7 + Table 5, pp. 990-991; FG2002 Nature p. 138: Tobit coef on negative deviation 0.622 (z=18.1)",
        "punish_freerider_level": "a Partner-treatment subject contributing 14-20 tokens below others' average received on average 6.8 punishment points — FG2000 AER Figure 5, p. 991",
        "perverse_punishment_share": "~0.20 of punishments target above-average contributors across VCM-with-punishment studies — Ertan, Page & Putterman 2005 WP printed p. 1, citing FG2000 p. 990 (edsl-test benchmark/human_targets.json)",
        "belief_note": "the belief is asked before every condition; it is meaningful for 'contribute' only",
    },
}

_TPP = {"pot": 100, "cap": 50, "endow": 50, "mult": 3}

THIRD_PARTY_PUNISHMENT = {
    "game": "third_party_punishment",
    "role": "third_party",
    "holdout": True,
    "measures": "norm_enforcement",
    "conditions": [
        {"id": "gave0", "gave": 0, **_TPP},
        {"id": "gave10", "gave": 10, **_TPP},
        {"id": "gave25", "gave": 25, **_TPP},
        {"id": "gave50", "gave": 50, **_TPP},
    ],
    "text": (
        "Two strangers, Person A and Person B, just played the following game: "
        "Person A received ${pot} and could give any amount of it, up to "
        "${cap}, to Person B, "
        "who started with nothing and had no say in the matter. Person A gave "
        "${gave} to Person B and kept the rest. You observed this but were not "
        "part of it; you have ${endow} of your own money and the split does not "
        "affect you. You may now spend any amount of your money to reduce "
        "Person A's earnings: each $1 you spend takes ${mult} away from Person "
        "A. The money you spend is gone; neither you nor Person B receives it. "
        "How many dollars do you spend on reducing Person A's earnings "
        "(0 to {endow})?"
    ),
    "question_type": "numerical",
    "min": 0,
    "max": 50,
    "belief": None,
    "references": {
        "design": "A endowed 100 points, transfers in {0,10,...,50}; C endowed 50 points; each deduction point costs C 1 and cuts A by 3; strategy method over all transfer levels — Fehr & Fischbacher 2004, Evol. Human Behav. 25(2), Sec. 2.1, pp. 66-67",
        "share_punish_below_50": "roughly 0.60 of third parties punish at EACH transfer level below 50 (n=22) — F&F 2004, Sec. 2.2, p. 68 and Fig. 1 (p. 69)",
        "mean_spend_gave0": "14 deduction points when A gave nothing (cuts A's income by 42) — F&F 2004, p. 68",
        "punish_slope": "OLS: punishment = .28 x (50 - transfer), constant -0.45 n.s. (P=.230); each 10-point drop in transfer adds 2.8 deduction points (8.4 cut to A) — F&F 2004, p. 68",
        "punish_at_equal_split": "near zero: punishment at transfer 50 occurred but its average was not significantly different from zero — F&F 2004, p. 68 and Fig. 1 (~5% punish)",
        "deviation_note": "gave25 is not a canonical F&F transfer level (multiples of 10); points are rendered as dollars with the same structure",
    },
}

_WTP_BALLOT = {"low": "LESS than the group average", "avg": "exactly the group average",
               "high": "MORE than the group average"}
_WTP_COMMON = {"endowment": 10, "group_size": 4, "multiplier": 1.6,
               "return_each": "0.40", "punish_cost": "0.25", "punish_reduction": "1.00"}

WHO_TO_PUNISH = {
    "game": "who_to_punish",
    "role": "voter",
    "holdout": True,
    "measures": "punishment norms (which contributors deserve punishment)",
    "conditions": [{"id": f"ballot_{k}", "target_group": v, **_WTP_COMMON} for k, v in _WTP_BALLOT.items()],
    "text": (
        "You are in a fixed group of {group_size} people playing a repeated "
        "public goods game. Each period every player receives ${endowment} and "
        "privately decides how much to put into a group fund, keeping the rest. "
        "The fund total is multiplied by {multiplier} and split equally, so each "
        "dollar contributed pays ${return_each} to every one of the {group_size} "
        "players. After contributions are revealed, punishment may happen: "
        "paying ${punish_cost} reduces another player's earnings by "
        "${punish_reduction} — but only players of the kinds your group votes "
        "punishable may be punished. Your group now votes (majority rule) on "
        "the punishment rules. Ballot item: should it be ALLOWED to punish a "
        "player who contributed {target_group}? Answer Yes, No, or No preference."
    ),
    "question_type": "mc",
    "options": ["Yes", "No", "No preference"],
    "belief": None,
    "references": {
        "indiv_yes_punish_low": "0.641 (410 Yes of 640 individual votes) — Ertan, Page & Putterman 2005 WP (publ. EER 2009), Table 2, p. 10",
        "indiv_yes_punish_avg": "0.072 (46 of 640) — same Table 2, p. 10",
        "indiv_yes_punish_high": "0.173 (111 of 640) — same Table 2, p. 10",
        "group_votes_allow_punish_high": "0 of 160 group votes — Result 1, p. 10 ('No group ever voted to allow punishment of higher-than-average contributors')",
        "design_params": "endowment $10, each $1 contributed pays $0.40 to each of 4 (fund x1.6 split equally), punishing costs $0.25 per $1.00 reduction — Section 1.2, eqs. (1)-(2), printed p. 7 (PDF p. 8) of the 2005 working paper",
        "ballot_options": "Yes / No / No preference; majority rule, ties fail — Section 1.3, printed pp. 7-8 and footnote 9 (printed p. 8)",
    },
}

# replicant papers/ertan2009/config.py: ENDOWMENT=20, GROUP_SIZE=5, MPCR=0.4, PUNISHMENT_RATIO=3, PROFILES;
# own contribution 15 and the 0-5 punishment budget follow the replicant vignette's punishment page.
_EV = {"endowment": 20, "group_size": 5, "n_others": 4, "mpcr": "0.4", "ratio": 3, "own": 15, "pmax": 5}
ERTAN_PROFILES = {
    "all_cooperate": {"others": "18, 17, 19, 16", "avg": "17.5"},
    "mixed": {"others": "15, 5, 18, 2", "avg": "10.0"},
    "one_freerider": {"others": "17, 18, 16, 2", "avg": "13.25"},
}

ERTAN_VIGNETTE = {
    "game": "ertan_vignette",
    "role": "punisher",
    "holdout": True,
    "measures": "punishment targeting under a punish-low-only rule (one-shot vignette)",
    "conditions": [{"id": k, **_EV, **v} for k, v in ERTAN_PROFILES.items()],
    "text": (
        "You are one of {group_size} people in a group. Everyone receives {endowment} tokens and "
        "decides how many of them to put into a group project; each token put into the project "
        "gives {mpcr} tokens to every member of the group, including the person who put it in. "
        "The rule in force for this round: members may punish only those who put in less than "
        "the group average. Every token you spend on punishment removes {ratio} tokens from the "
        "person you punish. This round you put in {own} tokens. The other {n_others} members put "
        "in {others} tokens (their average is {avg}). How many tokens (0 to {pmax}) do you spend "
        "on punishment?"
    ),
    "question_type": "numerical",
    "min": 0,
    "max": 5,
    "belief": None,
    "references": {
        "design": "endowment 20, group of 5, MPCR 0.4, 1 token removes 3, own contribution 15, budget 0-5 tokens, three hypothetical profiles of the other four (18,17,19,16 / 15,5,18,2 / 17,18,16,2) — replicant papers/ertan2009/config.py (MIT); rescaled one-shot vignette, NOT the original repeated Ertan, Page & Putterman (2009) game",
        "human_reference": "no individual human punishment amounts exist for these hypothetical profiles; the closest benchmark is qualitative: under a punish-low-only rule, punishment received tracks the shortfall below the average about 1:1 ($0.97-1.22 per $1, Ertan, Page & Putterman 2009 WP Table 4, printed p. 17)",
        "unsourced_note": "the replicant config's PAPER_FINDINGS (0.85 punish-low votes) has no page cite and is not used here; see who_to_punish for the Table 2 vote shares",
    },
}

CALIBRATION = (DICTATOR, ELEVEN_TWENTY, HOLT_LAURY, LOSS_GAMBLES, PUBLIC_GOODS_STRATEGY, TIME_CHOICES,
               TRUST_RETURNER, TRUST_SENDER, ULTIMATUM_PROPOSER, ULTIMATUM_RESPONDER)
HOLDOUT = (BEAUTY_CONTEST, PG_PUNISHMENT, THIRD_PARTY_PUNISHMENT, WHO_TO_PUNISH, ERTAN_VIGNETTE)

GAMES: dict[str, dict] = {g["game"]: g for g in CALIBRATION + HOLDOUT}

REQUIRED_KEYS = ("game", "role", "holdout", "measures", "conditions", "text", "question_type", "belief",
                 "references")


# ---------------------------------------------------------------- registry helpers

def list_games(holdout: bool | None = None) -> list[str]:
    """Game names in library order (calibration first). holdout=True/False filters; None returns all."""
    return [n for n, g in GAMES.items() if holdout is None or bool(g["holdout"]) == bool(holdout)]


def get_game(name: str | Mapping) -> dict:
    """A deep copy of one GAME dict (editing it never changes the library). A dict is returned as a copy."""
    if isinstance(name, Mapping):
        return copy.deepcopy(dict(name))
    if name not in GAMES:
        raise KeyError(f"unknown game {name!r}; known: {list(GAMES)}")
    return copy.deepcopy(GAMES[name])


def _game(game: str | Mapping) -> Mapping:
    return GAMES[game] if isinstance(game, str) and game in GAMES else (
        game if isinstance(game, Mapping) else get_game(game))


def _fields(template: str) -> set[str]:
    out = set()
    for _, name, _, _ in string.Formatter().parse(template):
        if name is not None and name != "":
            out.add(name.split(".")[0].split("[")[0])
    return out


def params(condition: Mapping) -> dict:
    """The format parameters of a condition (everything except its id)."""
    return {k: v for k, v in condition.items() if k != "id"}


def condition(game: str | Mapping, cond: str | Mapping) -> dict:
    """Look up a condition by id (a dict is returned unchanged)."""
    if isinstance(cond, Mapping):
        return dict(cond)
    g = _game(game)
    for c in g["conditions"]:
        if c["id"] == cond:
            return dict(c)
    raise KeyError(f"game {g['game']!r} has no condition {cond!r}; known: {[c['id'] for c in g['conditions']]}")


def bounds(game: str | Mapping, cond: str | Mapping) -> tuple[float | None, float | None]:
    """(min, max) for a numerical decision, with the per-condition override (b01 reviewer #4)."""
    g = _game(game)
    c = condition(g, cond)
    return c.get("min", g.get("min")), c.get("max", g.get("max"))


def _wording(g: Mapping, wording_id: int) -> dict:
    if int(wording_id) == 0:
        return {"text": g["text"], "belief_text": (g.get("belief") or {}).get("text")}
    w = (g.get("wordings") or {}).get(wording_id, (g.get("wordings") or {}).get(str(wording_id)))
    if w is None:
        raise ValueError(f"game {g['game']!r} has no wording_id {wording_id!r} (only 0 ships in the library)")
    return {"text": w["text"], "belief_text": w.get("belief_text", (g.get("belief") or {}).get("text"))}


def render(game: str | Mapping, cond: str | Mapping, wording_id: int = 0) -> str:
    """The decision text of one condition: text.format(**params). Deterministic; no persona, no answer format."""
    g = _game(game)
    return _wording(g, wording_id)["text"].format(**params(condition(g, cond)))


def render_belief(game: str | Mapping, cond: str | Mapping, wording_id: int = 0) -> str | None:
    """The belief question of one condition (None when the game has no belief layer)."""
    g = _game(game)
    if not g.get("belief"):
        return None
    t = _wording(g, wording_id)["belief_text"]
    return t.format(**params(condition(g, cond))) if t else None


def check_game(game: Mapping) -> list[str]:
    """Contract check. Returns a list of problems (empty = the GAME dict is valid)."""
    probs: list[str] = []
    name = game.get("game", "?")
    for k in REQUIRED_KEYS:
        if k not in game:
            probs.append(f"{name}: missing key {k!r}")
    if probs:
        return probs
    qt = game["question_type"]
    if qt not in ("numerical", "mc"):
        probs.append(f"{name}: question_type must be 'numerical' or 'mc', got {qt!r}")
    if qt == "mc":
        opts = game.get("options")
        if not opts or len(set(opts)) != len(opts):
            probs.append(f"{name}: mc game needs unique, non-empty options")
    if qt == "numerical":
        for k in ("min", "max"):
            if not isinstance(game.get(k), (int, float)):
                probs.append(f"{name}: numerical game needs numeric {k!r}")
    if not isinstance(game["references"], Mapping) or not game["references"]:
        probs.append(f"{name}: references must be a non-empty dict")
    conds = game["conditions"]
    if not conds:
        probs.append(f"{name}: needs at least one condition")
    ids = [c.get("id") for c in conds]
    if len(set(ids)) != len(ids) or any(i in (None, "") for i in ids):
        probs.append(f"{name}: condition ids must be unique and non-empty")
    belief = game.get("belief")
    if belief is not None:
        for k in ("text", "min", "max"):
            if k not in belief:
                probs.append(f"{name}: belief needs {k!r}")
    templates = {"text": game["text"]}
    if belief and "text" in belief:
        templates["belief"] = belief["text"]
    for c in conds:
        p = params(c)
        for label, t in templates.items():
            missing = _fields(t) - set(p)
            if missing:
                probs.append(f"{name}/{c.get('id')}: {label} placeholders not filled: {sorted(missing)}")
                continue
            try:
                out = t.format(**p)
            except (KeyError, IndexError, ValueError) as e:
                probs.append(f"{name}/{c.get('id')}: {label} does not format: {e!r}")
                continue
            if "{" in out or "}" in out:
                probs.append(f"{name}/{c.get('id')}: {label} still has braces after formatting")
        if qt == "numerical":
            lo, hi = c.get("min", game.get("min")), c.get("max", game.get("max"))
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo > hi:
                probs.append(f"{name}/{c.get('id')}: min > max")
    return probs


def answer_spec(game: str | Mapping, cond: str | Mapping) -> dict:
    """{"kind": "numerical", "min", "max"} or {"kind": "mc", "options"} for the answer parser."""
    g = _game(game)
    if g["question_type"] == "mc":
        return {"kind": "mc", "options": list(g["options"])}
    lo, hi = bounds(g, cond)
    return {"kind": "numerical", "min": lo, "max": hi}


def belief_spec(game: str | Mapping) -> dict | None:
    g = _game(game)
    b = g.get("belief")
    return None if not b else {"kind": "numerical", "min": b["min"], "max": b["max"]}


__all__ = ["GAMES", "CALIBRATION", "HOLDOUT", "ERTAN_PROFILES", "REQUIRED_KEYS", "list_games", "get_game",
           "check_game", "render", "render_belief", "condition", "params", "bounds", "answer_spec", "belief_spec"]
