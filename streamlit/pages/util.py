from tkinter import ROUND
import streamlit as st
from numpy.random import default_rng
import random
import pandas as pd
import plotnine as p9
from mizani.formatters import percent_format
import copy
from enum import Enum


ID = 'id_'
REVIEW_DRAW = 'review_'
WRITEUP_DRAW = 'writeup_'
SKILL = 'skill_'
REPUTATION = 'reputation_'
SCORE = 'score_'
FUNDS = 'funds_'
Y_OFFSET = 'y_offset_'
ROUND_NUM = 'round_'
WON = 'won_'
REASON = 'reason_'
ALGORITHM = 'algo_'

RNG = default_rng()

def init(proj_start_values, names):
    template = [None] * len(proj_start_values)
    jitter = -1
    for i in reversed(range(len(proj_start_values))): #keeps chart legend order
        jitter += 1
        template[i] = {
                    ID: names[i], 
                    REVIEW_DRAW: None,
                    WRITEUP_DRAW: None,
                    SKILL: proj_start_values[i], 
                    REPUTATION: 0.0,                    
                    SCORE: None, 
                    FUNDS: 0.0,
                    Y_OFFSET: jitter,
                    ROUND_NUM: 0,
                    WON: False,
                    ALGORITHM: None,
                    REASON: None}
    return (template)

# def compute_benefit(winner):
#     if winner['cumulative benefit'] == 0:
#         winner['cumulative benefit'] = 1
#     else:
#         winner['cumulative benefit'] += winner['cumulative benefit'] * .75

def select_random_n(num_funding_slots, rand_n, funding_threshold):
    random_candidates = []
    threshold_drop = 0.0
    while len(random_candidates) < num_funding_slots:
        random_candidates = [s for s in rand_n if s[SCORE] >=  
                                funding_threshold - threshold_drop]
        if len(random_candidates) < num_funding_slots:
            threshold_drop += .5
    #if threshold_drop > 0:
        #st.info(f"Standards have been lowered! Threshold lowered by {threshold_drop:.2f} for iteration")
    return random.sample(random_candidates, num_funding_slots)

def add_score(algorithm_simulation_data, sd_writeup, sd_review,
                 round):
    for i in range(len(algorithm_simulation_data[0])):
        proposal_draw = RNG.normal(0, sd_writeup)
        review_draw = RNG.normal(0, sd_review)
        for algo_sim_data in algorithm_simulation_data:
            algo_sim_data[i][ROUND_NUM] = round
            algo_sim_data[i][WRITEUP_DRAW] = proposal_draw
            algo_sim_data[i][REVIEW_DRAW] = review_draw
            algo_sim_data[i][SCORE] = \
                (proposal_draw + 
                 review_draw +                        algo_sim_data[i][SKILL] + 
                 algo_sim_data[i][REPUTATION])

def select_top_n(candidates, n):
    candidates = random.sample(candidates,len(candidates)) #scramble to keep ties from being resolved by insert order
    sorted_candidates = sorted(candidates, 
                                key=lambda s:s[SCORE], 
                                reverse=True)
    return sorted_candidates[0:n]

def distribute_awards(winners, funds, reputation):
    for winner in winners:
        winner[REPUTATION] +=  reputation
        winner[FUNDS] += funds
        winner[WON] = True

# def run3(top_n, rand_n, hybrid, num_funding_rounds, budget,
#          reputation_increase_from_funding, funding_threshold):
#     rounds_funding = []
#     for funding_round in range(num_funding_rounds + 1):
#         for proj_round in top_n + rand_n + hybrid:
#             result = copy.deepcopy(proj_round)
#             result['round'] = funding_round
#             rounds_funding.append(result)
#         if funding_round > num_funding_rounds:
#             break
#         add_score
#     #top_n
#         top_n_candidates = random.sample(top_n_candidates,len(top_n_candidates))
#         sorted_candidates = sorted(top_n, 
#                                 key=lambda s:['score'], 
#                                 reverse=True)    
#         for winner in sorted_candidates[0:budget]:
#             winner['reputation'] +=  reputation_increase_from_funding
#             winner['total funds'] += 1
#             #winner['cumulative benefit'] = compute_benefit(winner)
#     #random_n
#         winners = select_random_n(budget, rand_n, funding_threshold)
#         for winner in winners:
#             winner['reputation'] += reputation_increase_from_funding
#             winner['total funds'] += 1
#     #hybrid
#         sorted_candidates = sorted(hybrid, 
#                                 key=lambda s:['score'], 
#                                 reverse=True)
#         top_n_slots = budget // 2
#         top_n_winners = sorted_candidates[0:top_n_slots]
#         random_candidates = [candidate for candidate in hybrid if 
#                                 candidate not in top_n_winners]
#         random_n_winners = select_random_n(budget - top_n_slots,
#                                             random_candidates,
#                                             funding_threshold)
#         for winner in top_n_winners + random_n_winners:
#             winner['reputation'] += reputation_increase_from_funding
#             winner['total funds'] += 1
#     # return rounds_funding

def render3_util(df, column_to_show, show_top_n, show_random_n, 
            show_hybrid):
    if not (show_top_n or show_random_n or show_hybrid):
        st.write("select a dataset to view")
        return
    offset_scale = \
        (max(df[column_to_show]) - min(df[column_to_show])) / 100
    offset_scale = max(offset_scale, .01)
    df['y'] = df[column_to_show] + df['y_offset'] * offset_scale

    plot = (p9.ggplot(mapping=p9.aes(x=ROUND, y='y', group=ID)))
    if show_top_n:
        plot = plot + p9.geom_line(data=df[df[ALGORITHM] == 'Top N'],
                                    mapping=p9.aes(color=ID), size=.7)
    if show_random_n:
        plot = plot + p9.geom_line(data=df[df[ALGORITHM] == 'Random N'],   
                                    mapping=p9.aes(color=ID), size=.7, 
                                    linetype='dotted')
    if show_hybrid:
        plot = plot + p9.geom_line(data=df[df[ALGORITHM] == 'Hybrid'], 
                                    mapping=p9.aes(color=ID), size=.7, 
                                    linetype='dashdot')
    if (column_to_show == SKILL 
        and max(df[column_to_show]) == min(df[column_to_show])):
        single_y = max(df[column_to_show])
        plot = plot + p9.scale_y_continuous(
            label="Y projects offset by {offset_scale}",
            limits=[single_y - 0.5, single_y + .5])
    plot = plot + p9.scale_y_continuous(label="Foo")
    #plot = plot + p9.theme_xkcd()
    st.pyplot(p9.ggplot.draw(plot))
