from collections import defaultdict
from gzip import WRITE
from numpy import generic
import streamlit as st
#st.set_page_config(layout="wide")
import pandas as pd
import os
import sys
import copy
import plotnine as p9
import re
from mizani.formatters import percent_format
from numpy.random import default_rng
import random
sys.path.append("pages/")
import util

RNG = default_rng()


ID = util.ID
REVIEW_DRAW = util.REVIEW_DRAW
WRITEUP_DRAW = util.WRITEUP_DRAW
SKILL = util.SKILL
REPUTATION = util.REPUTATION
SCORE = util.SCORE
FUNDS = util.FUNDS
Y_OFFSET = util.Y_OFFSET
ROUND_NUM = util.ROUND_NUM
WON = util.WON
REASON = util.REASON
ALGORITHM = util.ALGORITHM
SS = st.session_state

pd.set_option('display.max_colwidth', None)

def reset():
    SS.df = None
    SS.accum_df = None
    SS.sd_writeup_reflection_of_skill = .25
    SS.budget = 2
    SS.hybrid_top_n_budget = SS.budget//2
    SS.funding_amount_in_millions = 1.0
    SS.num_funding_rounds = 5
    SS.reputation_increase = .75
    SS.minimum_threshold = 1.0
    SS.num_sims = 1
    SS.algo_names = ['Top N', 'Random N', 'Hybrid']
    SS.num_projects = 10
    SS.skills = [0.0, 1.0, 2.0, 3.0, 4.0]
    SS.skills.reverse()
    SS.proj_skill_values = [2.0] * SS.num_projects
    SS.names = "abcdefghijklmnopqrstuvwxyz".upper()
    SS.names = ['Amit', 'Beth', 'Chris', 'Drew', 'Enid', 
                'Fred', 'Gina', 'Hank', 'Ivor', 'Jude']
    SS.show_explanation = True
    SS.Notes = 'reset()'
    SS.show_top_n = True,
    SS.show_rand_n = True,
    SS.show_hybrid = False
    SS.y_dimension = FUNDS
    SS.sel_run_1 = 'latest'
    SS.sel_run_2 = 'latest'
    SS.y_units_1 = 'percent'
    SS.y_units_2 = 'percent'
    SS.currently_being_scored = 0
    SS.sd_writeup_reflection_of_skill = .25
    SS.sd_reviewer_accuracy = .25
    SS.current_round = 1
    SS.algo_for_spend = 'Top N'

session_config_values = ['Notes', 'sd_writeup_reflection_of_skill', 
'sd_reviewer_accuracy', 'budget', 
'funding_amount_in_millions', 'num_funding_rounds', 
'reputation_increase', 'minimum_threshold', 'num_sims',
'proj_skill_values', 'accum_df']

def generic_handler(widget_name, variable):
    SS[variable] = SS[widget_name]

if 'df' not in SS: #initialization when opened
    reset()
    SS.sessions = [] #persist across reset()

def run_simulation(proj_data_2, 
                    num_funding_rounds, 
                    sd_writeup,
                    sd_review,
                    reput_increase, 
                    budget,
                    hybrid_top_n_budget, 
                    minimum_threshold):
    top_n = copy.deepcopy(proj_data_2)
    for proj in top_n:
        proj[ALGORITHM] = 'Top N'
    rand_n = copy.deepcopy(proj_data_2)
    for proj in rand_n:
        proj[ALGORITHM] = 'Random N'
    hybrid = copy.deepcopy(proj_data_2)
    for proj in hybrid:
        proj[ALGORITHM] = 'Hybrid'
    results = []
    results.extend(top_n)
    results.extend(rand_n)
    results.extend(hybrid)
    for round_num in range(1, num_funding_rounds + 1):
        top_n = copy.deepcopy(top_n)
        rand_n = copy.deepcopy(rand_n)
        hybrid = copy.deepcopy(hybrid)
        for proj in top_n + rand_n + hybrid:
            proj[WON] = False 
        util.add_score([top_n, rand_n, hybrid], 
                        sd_writeup,
                        sd_review,
                        round_num)

        top_n_winners = util.select_top_n(top_n, budget)
        util.distribute_awards(top_n_winners, 1,
                            reput_increase)
        results.extend(top_n)

        random_n_winners = \
            util.select_random_n(budget, rand_n, 
                                minimum_threshold)
        util.distribute_awards(random_n_winners, 1,
                                reput_increase)
        results.extend(rand_n)

        hybrid_top_n_winners = \
            util.select_top_n(hybrid, hybrid_top_n_budget)
        hybrid_random_n_candidates = \
            [candidate for candidate in hybrid if 
                                    candidate not in 
                                    hybrid_top_n_winners]
        
        hybrid_random_n_winners = \
            util.select_random_n(budget - hybrid_top_n_budget, 
                                hybrid_random_n_candidates, 
                                minimum_threshold)
        util.distribute_awards(hybrid_top_n_winners +\
                                 hybrid_random_n_winners,
                                1,
                                reput_increase)
        results.extend(hybrid)
    return pd.DataFrame(results)

def run_n_simulations(num_sims, proj_skill_values, names, 
                      num_funding_rounds,
                      sd_writeup,
                      sd_review, reputation_increase, 
                      budget, minimum_threshold, 
                      hybrid_top_n_budget, algo_names, num_projects):
    proj_data = util.init(proj_skill_values, names)
    SS.df = None
    bins = list(range(0, num_funding_rounds + 1))
    SS.accum_df = pd.DataFrame({
        'algo': sorted(algo_names * len(bins), reverse=True), 
        'funding bin':  bins * len(algo_names), 
        'count': [0] * len(bins) * len(algo_names),
        '>= count': [0] * len(bins) * len(algo_names)})
    for i in range(num_sims):
        SS.df = run_simulation(proj_data, 
                                num_funding_rounds, 
                                sd_writeup,
                                sd_review,
                                reputation_increase, 
                                budget,
                                hybrid_top_n_budget,
                                minimum_threshold)
        accum_final_counts(SS.accum_df, SS.df, algo_names, 
                            num_funding_rounds)
#    accum_gt_counts(num_funding_rounds, algo_names, SS.accum_df)
    SS.accum_df['percent'] = SS.accum_df['count']/( num_projects * num_sims)
    #SS.accum_df['>= percent'] = SS.accum_df['>= count']/total_funds
    #st.dataframe(SS.accum_df)
    config = {k:SS[k] for k in session_config_values}
    for algo in SS.algo_names:
        bins = []
        bins_perc = []
        total = sum(SS.accum_df.loc[(SS.accum_df.algo == algo), 
                                    'count'])
        for n in range(SS.num_funding_rounds + 1):
            #st.info(n)
            count = SS.accum_df.loc[(SS.accum_df.algo == algo) &\
                                    (SS.accum_df['funding bin'] == n), 
                                    'count'].values[0]
            amount = '$0' if n == 0 else f'${n} million'
            bins.append(f'{amount}: {count}')
            bins_perc.append(f'{count/total:.0%} @ {amount}')
            #config[f"{algo}: ${n}M count"] = count
            if n == 0:
                config[f'{algo}: ${n}M'] = f'{count/total:.0%}'
        #config[f'{algo} count of projects awarded amount in millions'] = \
        #        '\n'.join(bins)
        #config[f'{algo} % of projects awarded amount in $ millions'] = \
        #        ', '.join(bins_perc)
    SS.sessions.append(config)

def run_sim_as_configured():
    run_n_simulations(SS.num_sims, 
                    SS.proj_skill_values, 
                    SS.names,
                    SS.num_funding_rounds,
                    SS.sd_writeup_reflection_of_skill,
                    SS.sd_reviewer_accuracy,
                    SS.reputation_increase, 
                    SS.budget, 
                    SS.minimum_threshold,
                    SS.hybrid_top_n_budget,
                    SS.algo_names, 
                    SS.num_projects)

def run_n_wrapper():
    SS.num_sims = SS.simulation_radio_btn
    SS.Notes = ''
    run_sim_as_configured()

def render3(df, column_to_show, show_top_n, show_random_n,
            show_hybrid):
    if not (show_top_n or show_random_n or show_hybrid):
        st.info("select a dataset to view")
        return None
    offset_scale =  (max(df[column_to_show]) - min(df[column_to_show])) / 100
    offset_scale = max(offset_scale, .01)
    df['y'] = df[column_to_show] + df[Y_OFFSET] * offset_scale

    plot = (p9.ggplot(mapping=p9.aes(x=ROUND_NUM, y='y', group = ID)))
    if show_top_n:
        plot = (plot +
                p9.geom_line(data=df[df[ALGORITHM] == 'Top N'],
                mapping=p9.aes(color=ID), size=.7)
        )
    if show_random_n:
        plot = plot + p9.geom_line(data=df[df[ALGORITHM] \
                                    == 'Random N'],   
                                    mapping=p9.aes(color=ID), 
                                    size=.7, 
                                    linetype='dotted')
    if show_hybrid:
        plot = plot + p9.geom_line(data=df[df[ALGORITHM] \
                                    == 'Hybrid'], 
                                    mapping=p9.aes(color=ID), 
                                    size=.7, 
                                    linetype='dashdot')
    if (column_to_show == 'skill' 
        and max(df[column_to_show]) == min(df[column_to_show])):
        single_y = max(df[column_to_show])
        plot = plot + p9.scale_y_continuous(
            label=f"Y projects jittered by {offset_scale:.2f}",
            limits=[single_y - 0.5, single_y + .5])

    plot = (plot + p9.xlim([0,SS.num_funding_rounds]) 
            + p9.ylim([0,SS.num_funding_rounds + .5]))
    
    plot = (plot + 
    p9.labels.ylab(f"Y projects jittered by {offset_scale:.2f}")
    )
    #plot = plot + p9.theme_xkcd()
    return plot

def accum_final_counts(sim_df, df, algo_names, num_funding_rounds):
    for algo in algo_names:
        last_round_algo_df = df[(df[ALGORITHM] == algo) & 
                            (df[ROUND_NUM] == num_funding_rounds)]
        for total_funds in last_round_algo_df[FUNDS]:
            total_funds_int = int(total_funds)
            sim_df.loc[(sim_df['algo'] == algo) & 
                        (sim_df['funding bin'] == total_funds_int), 
                        'count'] += 1

def accum_gt_counts(num_funding_rounds, algo_names, sim_df):
    for algo in algo_names:
        for value in range(1, num_funding_rounds + 1):
                gt_count = \
                    sum(sim_df[(sim_df[ALGORITHM] == algo) &
                            (sim_df['funding bin'] >= value)]['count'])
                sim_df.loc[(sim_df[ALGORITHM] == algo) &
                            (sim_df['funding bin'] == value), 
                            '>= count'] = gt_count

def reset_current_simulation(out):
    out.info("Resetting current run, previous runs are preserved")
    SS.df = None
    SS.currently_being_scored = 0
    SS.proj_data = util.init(SS.proj_skill_values, SS.names)

def apply_skills_rb():
    skills = None
    if SS.skills_rb == "Bell Curve":
        skills = [0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 
                    3.0, 3.0, 4.0]
    if SS.skills_rb == "God's Gift is Amongst Us":
        skills = [4.0] + [1.0] * 9
    if SS.skills_rb == "A Mother's Love":
        skills = [2.0] * 10
    for i, skill in enumerate(skills):
        SS.proj_skill_values[i] = skill

def plot_draws(author_draws, reviewer_draws, proj_names):
    x_dim = "Writeup's Reflection of Author Skill"
    y_dim = "Reviewer's Accuracy of Writeup Evaluation"
    plot = p9.ggplot(data=pd.DataFrame(
                    {x_dim:  author_draws,
                     y_dim: reviewer_draws,
                    'id': proj_names}),
                    mapping=p9.aes(x=x_dim, y=y_dim,
                    label='id'))
    plot = plot + p9.geom_point()
    plot = plot + \
        p9.geom_text(nudge_x=.1, nudge_y=.1)
    plot = plot + \
        p9.geom_point(data=pd.DataFrame({'x':[0.0],
                                           'y':[0.0]}),
                        mapping=p9.aes(x='x', y='y'),
                        fill='red')
    plot = plot + p9.stat_ellipse(geom='polygon', level= 0.68,
                                    type='t',
                                    alpha=.2,
                                    fill='red')
    plot = (plot 
             + p9.scales.ylim([min([-1.0] + author_draws) - .1,
                               max([1.0] + author_draws) + .1]) 
             + p9.scales.xlim([min([-1.0] + reviewer_draws) - .1,
                               max([1.0] + reviewer_draws) + .1])
    )
    return(plot)

def std_dev_handler(widget_name, variable, out):
    generic_handler(widget_name, variable)
    reset_current_simulation(out)

def add_score():
    reasons_positive_score = ['Strong morning coffee',
                            'Pun in title',
                            'Nice graph colors',
                            'Beautiful day']

    reasons_negative_score = ['Weak morning coffee',
                            'Bad pun in title',
                            'Gloomy day',
                            'Denied promotion']

    draw_project_presentation = RNG.normal(0, SS.sd_writeup_reflection_of_skill)
    draw_reviewer_accuracy = RNG.normal(0, SS.sd_reviewer_accuracy)
    proj_datum = SS.proj_data[SS.currently_being_scored]
    proj_datum[SKILL] =\
         SS.proj_skill_values[SS.currently_being_scored]
    proj_datum[WRITEUP_DRAW] = draw_project_presentation
    proj_datum[REVIEW_DRAW] = draw_reviewer_accuracy
    proj_datum[SCORE] = (proj_datum[SKILL] 
                            + draw_project_presentation 
                            + draw_reviewer_accuracy)
    reason = ''
    if draw_reviewer_accuracy > 0.0:
        reason = random.sample(reasons_positive_score, 1)
    else:
        reason = random.sample(reasons_negative_score, 1)
    proj_datum[REASON] = reason
    SS.currently_being_scored = \
            (SS.currently_being_scored + 1) % SS.num_projects

def render_scoring_df(df, out):
    if WRITEUP_DRAW not in df:
        out.write("Click on button")
        return
    disp_df = pd.DataFrame()
    disp_df = df[[SKILL, WRITEUP_DRAW, REVIEW_DRAW, REPUTATION, 
                  SCORE, REASON, ID]]
    display_cols = [ID, SKILL, WRITEUP_DRAW, REVIEW_DRAW,
                    SCORE, REPUTATION]
    disp_df = disp_df.loc[:, 
              disp_df.columns.isin(display_cols)]
    
    out.dataframe(disp_df.style.\
            format(subset=[WRITEUP_DRAW, 
                           REVIEW_DRAW,
                           SKILL,
                            REPUTATION,
                            SCORE],
                    formatter='{:.2f}'))

def plot_details(out):
    options = [FUNDS, WRITEUP_DRAW, REVIEW_DRAW,  
                REPUTATION, SCORE, SKILL]
    plot = render3(SS.df, SS.y_dimension, SS.show_top_n, 
                    SS.show_rand_n, SS.show_hybrid)
    if plot is not None:
        out.pyplot(p9.ggplot.draw(plot))
    out.checkbox("Top N", value=SS.show_top_n, 
                on_change=generic_handler,
                args= ('show_top_n_cb', 'show_top_n'),
                key='show_top_n_cb')
    out.checkbox("Random N", value=SS.show_rand_n,
                on_change=generic_handler,
                args= ('show_rand_n_cb', 'show_rand_n'),
                key='show_rand_n_cb')
    out.checkbox("Hybrid", value=SS.show_hybrid, 
                on_change=generic_handler,
                args= ('show_hybrid_cb', 'show_hybrid'),
                key='show_hybrid_cb')
    out.selectbox("Y dimension",
                options=options,
                index=options.index(SS.y_dimension),
                on_change=generic_handler,
                args=('select_y_dim', 'y_dimension'),
                key='select_y_dim')

def generate_result_plot(accum_df, y_units, fill):
    return (p9.ggplot(accum_df, p9.aes(x='funding bin',   
                                y=y_units,       
                                fill=fill))
            + p9.geom_col(position='dodge')
    )


def plot_results(col, i):
    y_units_key_i = f'y_units_{i}'
    y_run_id_select_box_i = f'run_id_selectbox_{i}'
    sel_run_i = f'sel_run_{i}'
    y_radio_i = f'y_radio_{i}'
    units = ['count', 'percent']
    runs = list(range(len(SS.sessions))) + ['latest']
    run_id = SS[sel_run_i]
    if run_id == 'latest':
        run_id = -1
    accum_df = SS.sessions[run_id]['accum_df']
    #st.dataframe(SS.accum_df)
    plot = generate_result_plot(accum_df, SS[y_units_key_i],
                                'algo')
    #plot = (p9.ggplot(accum_df, p9.aes(x='funding bin',   
    #                            y=SS[y_units_key_i],       
    #                            fill=ALGORITHM)) 
    #        + p9.geom_col(position='dodge')
    #)
    if SS[y_units_key_i] == 'percent':
        plot = plot + p9.scale_y_continuous(labels=percent_format())
    col.pyplot(p9.ggplot.draw(plot))
    col.radio("Y units", units,
               index=units.index(SS[y_units_key_i]),
               on_change=generic_handler,
               args=(y_radio_i, y_units_key_i),
               key=y_radio_i)
    col.selectbox("Run to show",
                    runs,
                    index=runs.index(SS[sel_run_i]),
                    on_change=generic_handler,
                    args=(y_run_id_select_box_i, sel_run_i),
                    key=y_run_id_select_box_i)

def show_next_round():
    if SS.current_round == SS.num_funding_rounds:
        SS.current_round = 1
    else:
        SS.current_round += 1

def reset_current_round_and_run_sim_once(out):
    SS.num_sims = 1
    run_sim_as_configured()
    SS.current_round = 1
    if out is not None:
        out.info("Ran Simulation")

def render_round_iterator(col1, col2):
    col1.button(f"Show next round",
            on_click=show_next_round)
    col2.radio(f"Round {SS.current_round}", 
                options=SS.algo_names,
                index=SS.algo_names.index(SS.algo_for_spend),
                horizontal=True,
                on_change=generic_handler,
                args=('algo_cb', 'algo_for_spend'),
                key='algo_cb')
    sub_df = SS.df[(SS.df[ALGORITHM] == SS.algo_cb) &
                    (SS.df[ROUND_NUM] == SS.current_round)]
        
    sub_df = sub_df.loc[:, 
                                sub_df.columns.isin([
                                    ID, REPUTATION, 
                                    SKILL,
                                    WRITEUP_DRAW, 
                                    REVIEW_DRAW,
                                    WON,
                                    SCORE, FUNDS])]
        
    col2.dataframe(sub_df\
        .style.format(subset=[WRITEUP_DRAW, 
                            REVIEW_DRAW,
                            REPUTATION,
                            SKILL, 
                            SCORE], formatter='{:.2f}')\
        .format(subset=[FUNDS], 
                        formatter='${:.0f} M'))

def apply_options():
    params = SS.interesting_params.split(', ')
    notes = ''
    for param_value in params:
        if param_value == 'reset()':
            reset()
            continue
        split = param_value.split(': ')
        if len(split) == 2:
            param = split[0]
            if param not in SS:
                st.info(f"No such parameter {param}")
                continue;
            value= split[1]
            st.info(value)
            if re.match(r'^\d+\.\d+$', value):
                value = float(value)
            elif re.match(r'^\d+$', value):
                value = int(value)
            if param == 'Notes':
                notes = value
            SS[param] = value
    SS.Notes = notes
    run_sim_as_configured()

def notes_handler():
    SS.Notes = SS.notes_input
    SS.sessions[-1]['Notes'] = SS.notes_input

def render_session():
    SS.accum_df = SS.sessions[SS.graph_session_select]['accum_df']

def set_skills_exp():
    exp = st.expander("Set skills for projects")
    exp.radio("Suggested Skill Templates--modifiable", 
    options=["Bell Curve", "God's Gift is Amongst Us", "A Mother's Love"],
    index=2,
    on_change=apply_skills_rb,
    key="skills_rb",
    horizontal=True)

    cols = exp.columns(SS.num_projects)
    for i in range(SS.num_projects):
        cols[i].radio(f"{SS.names[i]}",
                    SS.skills, 
                    index=SS.skills.index(SS.proj_skill_values[i]),
                    key=f"skill{i}",
                    horizontal=False)
    if exp.button("Apply changes, resets current run"):
        for i in range(10):
            SS.proj_skill_values[i] = SS[f"skill{i}"]
        reset_current_simulation(exp)

def writeup_sd_slider(col, label):
    col.slider(label,
        min_value=0.0, max_value=1.0, step=0.25,
        on_change=std_dev_handler,
        args=('sd_writeup_slider', 
                'sd_writeup_reflection_of_skill',
                col1), 
        value=SS.sd_writeup_reflection_of_skill,
        key='sd_writeup_slider')

def reviewer_sd_slider(col, label):
    col.slider(label, min_value=0.0, max_value=1.0, step=0.25,
            on_change=std_dev_handler,
            args=('sd_reviewer_slider',
                   'sd_reviewer_accuracy',
                   col1),
            value=SS.sd_reviewer_accuracy,
            key='sd_reviewer_slider')

def render_iterative_draws(col1, col2):
    if 'proj_data' not in SS:
        SS.proj_data = util.init(SS.proj_skill_values, SS.names)
    df = pd.DataFrame(SS.proj_data)
    df = df[df[WRITEUP_DRAW].notnull()]
    if len(df) < SS.num_projects:
        col1.button(f"Draw writeup and review variation for {SS.names[SS.currently_being_scored]}'s project",
            on_click=add_score,
            key=f"draw_button")
    if len(df) > 0:
        try:
            plot = plot_draws(list(df[WRITEUP_DRAW]), 
                            list(df[REVIEW_DRAW]),
                            list(df[ID]))
            col2_wide.pyplot(p9.ggplot.draw(plot))
        except p9.exceptions.PlotnineWarning as e:
            st.info(e)
        col1.write(f"Reason for reviewer draw: {df[REASON].iloc[-1][0]}")
        render_scoring_df(df, col2)
    if len(df) < SS.num_projects:
        exp.info(f"{SS.num_projects - len(df)} projects left to draw")

def budget_slider(col, label): #col2 for non-narr
    col.slider(label, 
            min_value=2, max_value=10, step=2, value=SS.budget, 
            on_change=generic_handler,
            args=('budget slider', 'budget'),
            key='budget slider')

def reputation_slider(col, label):
    col.slider(label,
                min_value=0.0, 
                max_value=2.0, 
                step=.25,
                value=SS.reputation_increase,
                on_change=generic_handler,
                args=('reputation slider', 
                      'reputation_increase'),
                key='reputation slider')
#--------------Direct Rendering Below Here---------

st.title("How Algorithims Can Influence Research Diversity, Education and Publishing")
st.markdown("A simulation fueled exploration")
st.markdown(("**Breck Baldwin**, breckbaldwin@gmail.com" +
                 "\nSeptember, 2022"))

st.checkbox("Show Narrative", value=SS.show_explanation, 
            on_change=generic_handler,
            args=('show_explanation_cb', 'show_explanation'),
            key='show_explanation_cb')

if SS.show_explanation:
    exp = st.expander("Introduction", expanded=False)
    exp.markdown("""
## Welcome to the simulation

Society allocates scarce resources in all sorts of ways--a common one relies on convincing others that you deserve membership in an elete cohort, for example: 

- The 2023 incoming class at Harvard
- Presentation at an academic conference
- Being among the awardees of a research grant. 

This simulation focuses on cumulative effects of the last example, research grant funding, with three award algorithms across 10 individuals over 5 funding cycles. Academic admissions and research publications can be thought of in the context of the simulations parameters as well--although perhaps with different consequences. See the discussion section.

foot note: This work more fully explores a screed I wrote for the Department of Energy's conference on funding scientific software, reference to the paper and accompanying simulation is at [Funding Strategies for Scientific Software](https://breckbaldwin.github.io/S3rd/presentations/DOE2021/FundingStrategiesForSciSoftware.html).

The algorithms are: 

1. A _Top N_ approach which selects the highest scoring candidates with N funding slots, e.g., an American meritocracy.
2. A _Random N_ approach that awards N slots randomly to candidates that pass a minumum score threshold.
3. A _Hybrid_ approach that blends the two. 

## Scoring candidates

The simulation setup goes as follows:

- We have 10 researchers applying for funding for 5 funding cycles--many government programs have 5 year cycles with funds awarded each year. In education, we could consider linked admissions events, high school, college, graduate school and one's first job. For publication, we could consider conference or journal publications. You can control the number of awards/admission slots/publication slots. 
- The researchers have a 'skill' value between 0.0, an F, to 4.0, an A on US style grading scale that are their actual ability/smarts/training. The distribution of skills across the 10 particpants can be controlled. 
- It is assumed that there are cumulative effects from success that increase the chance of subsequent success. This effect can be adjusted from zero upwards.

## Playing God

You'll be simulating wrecked careers as well as meteoric ascensions to greatness in no time, but skills have to be assigned first. There are three options plus just setting scores as your omnipotence decrees:

1. Bell Curve: Mostly C's (2.0), some B's (3.0) and D's (1.0) and an outlier A (4.0) and F (0.0).
2. God's Gift: One genius, A, in a collection of mediocrity, D's.
3. A Mother's Love: All researchers have the same mother who raised them equally skilled but mother is a realist and knows they are pretty average (C's).
""")
    set_skills_exp()

    exp = st.expander("Fickle Human Simulation", expanded=False)
    exp.markdown("""
While you, God-like, know the skills assigned above, mere mortals write proposals above or below their ability and reviewers even less reliably assess the skill reflected in the written proposal. We simulate this by drawing from a Gaussian/normal random distribution--think throwing darts with the bull's eye at 0,0 on a [Cartesian plot](https://en.wikipedia.org/wiki/Cartesian_coordinate_system):

- Horizontal distance from the bull's eye is how accurately the proposal refelects the actual skill of the researcher.

- Vertical distance from the bull's eye is the reviewer's accuracy reviewing said proposal. 

You get to control where 68 percent of the darts will land, e.g., where one (1) [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation) of random draws, aka dart throws, will land. Why the controls? 

- If you think the writeup/proposal is always a perfect expression of the author's skill then set the standard deviation to 0.0. If you think writeups vary a lot in reflecting underlying skill then pick how much--there are only three remaining options: 0.25, 0.5, 0.75 and 1.0.  
- Likewise, if you think reviewers have perfect recognition of assessing the skill in a writeup, then go with 0.0. If not, then how accurate are they? 

With just 10 throws the results will not be perfectly scattered but an approximate 68\% area of the actual throws will be drawn when there are enough data points.

Below are controls and a button to throw darts one at a time. You will see the draw and the math to compute the score which is:
`skill + writeup draw + revew draw + reputation (0.00 for now) = score` 
Review draws have a 'reason' for why the review was higher or lower than a perfectly accurate assessment at 0. As often seen with humans, the reason is a complete fabrication.""")

    (col1, col2_wide) = exp.columns([1,2])
    writeup_sd_slider(col1, 
    "Standard deviation writeup: 68% of project writeup fall within specified +/- range in conveying skill")
    
    reviewer_sd_slider(col1,
    "Standard deviation reviewer: '68% of reviewer evaluations within specified +/- range in grade points")
    render_iterative_draws(col1, col2_wide)
  
    exp_rep = st.expander("Reputation")
    exp_rep.markdown(f"""

## Success begets success

Reputation currently increments by {SS.reputation_increase} for all the algorithms per award. This value can be changed below. Reputation is the mechanism that reflects the impact of having resources from earlier funding. For school admissions it would be the impact that getting into a good high school has on getting into a good college, or for publication, having a previously published conference paper on getting another conference paper accepted.

All awards are \$1 million and the budget is exhausted each round. The budget can be raised to up to \$10 million but in even incremnts to allow for even split of the _Hybrid_ algorithm.
""")

    cols = st.columns(3)
    exp = cols[0].expander("Top N Algorithm")
    exp.markdown(f"""
## Algorithmic meritocracy:

The _Top N_ algorithm will take the available budget of \${SS.budget} million, and parcel out $1 million at a time starting at the top scoring project until the budget is spent. 

The incentive is to score in the top N which generally requires an excellent writeup (lots of work) and a good revew (some luck).

Hard work and reviewers have to 'like the cut of your jib'. 

If there are tied projects then pick randomly from them.
""")

    exp = cols[1].expander("Random N algorithm")
    exp.markdown("""
## Work hard AND get lucky:

The _Random N_ algorithm sets a minimum score for consideration in a lottery for the funds. There is no advantage to writing a proposal better than the threshold or having a greater reputation. 

The incentive is to write an adequate proposal (work but less work than an 'excellent' proposal) to get above threshold and leave the rest to chance (reviewer luck and selection luck)

There may not be sufficient candidate proposals to qualify however, in that case the threshold is dropped by .1 and selection starts over.

Medium hard work, reviewer impact much diminished since there is a random selection process that follows. 
""")

    exp = cols[2].expander("Hybrid Algorithm")
    exp.markdown("""
## The comprimise:
 
Proposals tend to either be OMG this should be funded with a long tail of less extraordinary efforts. Program managers, admission committees and other selection processes do feel that judgement has an important and predictively useful role which is the driving force behind the Top N algorithm. 

So the hybrid algorithm acknoledges the desire for discretion but changes that to _Top N/2_ where half the funding is done that way, the remainder is _Random N_. The ratio could be adjusted but I am trying to keep it simple.

Applicants can focus on winning _Top N_ or _Random N_. Presumably the true stars of the field will reliably rise while the rest continue along with random funding.  

""")

    exp_spend = st.expander("Let's spend some money!")
    exp_spend.markdown("## Watching the algorithms at work")
    (col1, col2) = exp_spend.columns([1,1])
    budget_slider(col1, 
    "Budget: In millions per funding cycle--each award is $1 million")

    reputation_slider(col2,
    "Reputation increase: How much increase in reputation per funding award which is added to project score.")

    if col1.button("(Re)run Algorithms"):
        reset_current_round_and_run_sim_once(None)
    if SS.df is None:
        st.stop()

    df = SS.df[(SS.df[ALGORITHM] == SS.algo_for_spend) &
                (SS.df[ROUND_NUM] <= SS.current_round)]

    plot = render3(df, FUNDS, 
                        SS.algo_for_spend == 'Top N', 
                        SS.algo_for_spend == 'Random N', 
                        SS.algo_for_spend == 'Hybrid')

    col1.pyplot(p9.ggplot.draw(plot))
    result_plot = generate_result_plot(df, 'funding bin', 'algo')
    col1.pyplot(p9.ggplot.draw(result_plot))
    if col1.button("Set to defaults"):
        reset()
        reset_current_round_and_run_sim_once(col1)
    render_round_iterator(col1, col2)

    exp_spend.markdown("""
## Cumulative effects of the algorithms

The above graph/table and controls allow exploration of the qualities of the algorithms individually, one round at a time. 
I suggest you 'Set to defaults' and iterate through one round at time for each algorithm. 

- With default settings, _Top N_ almost always allocates funds to a few projects. Since reputation increases with funding, there tends to be the same set of winners on each round. I suggest you run it a few times with defaults to get the feel for the configuration.  

- _Random N_ tends to distribute funds much more evenly in comparision to the _Top N_. You can see this on Round 5 of funding while switching between the algorithms. All algorithms share the same skill, writeup and review values across organizations per round so the differences between algorithms shown is deterministic.

- _Hybrid_ tends to have some strong winners from _Top N_ rounded out with a more even allocation from _Random N_ as expected. r 
    
""")

options = ['Custom', 
'Notes: A) Reset to defaults, reset()',
'Notes: B) No reputation increase--algos roughly same, reset(), reputation_increase: 0.0, num_sims: 100',
'Notes: C) Sufficient funding for all programs--all algos the same, reset(), budget: 10',
'Notes: D) Really poor reviewers--algos roughly same!, reset(), standard_deviation: 1.0, num_sims: 100',
'Notes: E) High miniumum score--algos same, reset(), minimum_threshold: 2.0, num_sims: 100']

if SS.show_explanation:
    exp_impact = st.expander("Impact of the algorithms")
    exp_impact.markdown(f"""
## Exploring the differences with algorithms and parameters

The algorithms determine how the same amount of money is distributed across the same candidates. The larger consequences that follow get picked up in the discussion but here we want to introduce the tools to better understand the properties of the algorithm and corresponding paramterizations.

Below is a histogram that counts the amount of funding a program got. If program {SS.names[1]} got $5 million then we record that fact.
""")
    (col1, col2,  col3) = exp_impact.columns(3)

    if SS.df is None:
        run_sim_as_configured()
    plot_results(exp_impact, 1)



col3.selectbox("Interesting Parameterizations", options=options,
        on_change=apply_options,
        key='interesting_params')

show_detail_graph = col1.checkbox("Show Last Simulation Detail Graph", 
                                value=True)

options = [1, 2, 10, 100]
col3.radio("Run N simulations", options=options, 
        index=options.index(SS.num_sims), 
        on_change=run_n_wrapper, 
        key='simulation_radio_btn',
        horizontal=True)
col1.button("(Re)run as configured", on_click=run_n_wrapper)
#if SS.num_sims > 1:
#    col1.info("Only last simulation results graphed since the number of simulations is > 1")

num_plots = col2.selectbox("Number of plots", options=[1,2])
if show_detail_graph:
    num_plots += 1
    cols = st.columns(num_plots)
    for i in range(num_plots):
        if i == 0:
            plot_details(cols[i])
        else:
            plot_results(cols[i], i)
else:
    cols = st.columns(num_plots)
    for i in range(num_plots):
        plot_results(cols[i], i + 1)



st.text_input("Notes:", value=SS.Notes, 
                on_change=notes_handler,
                key='notes_input')



st.selectbox("Graph run", options=range(len(SS.sessions)),
            on_change=render_session,
            key='graph_session_select')
                
st.info(SS.Notes)
sessions_df = pd.DataFrame(SS.sessions)
cols = list(sessions_df.columns)
cols.remove('accum_df')
leftmost_cols = ['Notes', 'Top N: $0M', 'Random N: $0M', 'Hybrid: $0M']
rightmost_cols = \
    [col for col in cols if col not in leftmost_cols]
reordered_cols = leftmost_cols + rightmost_cols
st.dataframe(sessions_df[reordered_cols])

discussion = """
## Discussion

THe DOE paper that prompted this work was focused on open source scientific software. I chose to discuss the funding selection model and oddly enough the Stan project got its start from a DOE grant that needed a project in the back channels of Columbia University--quite a random process actuall. I had be executive director of the Stan project (Bayesian model fitting) and had been through a few rounds of proposal writing to NASA, NSF, Chan-Zuckerberg Initiative and others. 

In the context of funding open source projects, the _Top N_ approach is actually destrucitive from my perspective. 

- Proposal writing takes a lot of work and from my particpation in NumFocus proposal writing jams (lots of projects, NumPy, SciPy, MathPlotLib and the like) apply for a call and talk about it on Slack). These proposals take at least a person month of effort and a lot of work goes into making them good with an around 20% chance of success. This is a huge distraction from key people on these projects for funding at around $100,000 to $500,000 over three years. This pays for things like continuous integration, professional programmers for some of the booring but necessary work and confernecs/get togethers. 

- It would be a much better allocation of resources to reduce the proposal writing load to a thresholded evaluation model. Have the proposal be demonstrating things like:
    + Userbase: Have N users active on forums
    + Significance: X dependencies from other libraries
    + Livelyness: Pull request aging over time
    + Developer Base: Q developers submititng a pull request in the last X days
    + Documentation/Tutorials: Demonstraated trainings/docs etc..
    + Research citations over time.
Then the rest of the proposal woiuld be how the funds would be used and who is in charge. Maybe a 2-3 pager? 

- Since there often are insufficient funds to fund all proposals, delect randomly for qualifying proposals, _Random N_. 

The runaway funding model really makes no sense for open source scientific sofware in my opinion. All the projects would benefit from funding and the first million goes a lot farther than the 5th million. 

I ackowledge the reality of largely volunteer selection committees who feel like some proposals are much stronger than others, so the hybrid model makes sense. 

## Academic papers/conferenc presentations

Academic conferences and journals are intensly competitive domains where the algorithms are less straight-forwardly evaluated. The reputation score is what accumulates and publication runs off of reputation. Journals have page counts, conferences have speaking slots which are the "budget". The number of cycles reflects the accumulation of reputation over time. The value of that reputation translates into promotion, ability to attract funding and students all of which increase reputation.

The impact of _Top N_ allocation is that rockstars get baked into the system and franky I think it kills diversity of thought. True rockstars are going to be recognized in any case if they really stand out but the rest of the field needs to exist too with less extraordinary talents. Turning non-rockstar talents into rockstars because of the the publication selection model seems nuts if it kills off the other non-rockstar talents. Fields need a diversity of ability. 

## Education

Getting into a good high-school helps get into a good college and so on, but are the cumulative effects a problem? Harvard is only going to graduate so may students a year so seems like the ones that get past the 5% acceptance rate are fine--what percentage of those that apply could function perfecty well in that environment? 50%?

While I think it applies to all the scenerious, addmissions hits hard on the insane levels of effort required to craft the life profile of an 18 year old to be accepted that starts years prior to the application. I believe the hybrid model is in effect somewhat since legacy applicants (alumni children) are presumably thresholded at some level and the rest are picked based on a total order in a space that includes whether the orchstra needs an oboe player. I am guessing that Harvard's product, bright-eyed-and-bushy-tailed members of society, would be improved if the entire incoming class were held to the legacy threshold and _Random N_ selection. 

This brings us to level 1 bias, which is that given a task to pick the top 20% from a list of people based on a criteria, those 20% will probably be biased by some quality that the evaluator is sensitive to. The easiest fix for that is to instead threshold at a significanly higher level and pick randomly from there--reduces the oppotunity for bias. Level 2 bias is to consider the impact of the threshold which has its own issues but I'd conjecture it is easier to spot. 













The simulation has been focused on research funding but I also mentioned academic admissions and research publications. 

### Academic admissions



With

"""

g = """


Based on paper at: https://breckbaldwin.github.io/S3rd/presentations/DOE2021/FundingStrategiesForSciSoftware.html


**TL;DR** Ranking research proposals by quality and choosign Top N leads to concentration of resources to N organizations. Random N selection of proposals that are above a quality threshold distributes funding opportunties more broadly. 

- All projects started with the same merit and reputation values. 
- Score = The score for a round of funding is a random draw from a normal distribution centered on merit with standard deviation .1 then added to the reputation value.
- Top N: Award top N scored proposals for the round.
- Random N: Award random N selected from Score > funding threshold.
    + The default threshold for funding is .2, so candidates have to get a bit lucky to draw a high value for the score initially to clear the threshold and then get lucky by being drawn from the set of candidates above threshold.
- If a candidate is funded then their reputation increases .1 and the resource count increases by 1. Resources/reputation can only go up, merit stays the same. """


fii = """Now we have some place to start before we pitch off into the dreaded algorithims. Just a bit more to do before the **judging** begins.

As alluded to above, only god or the simulation runner knows the true skill behind a project represented in a value between 0.0 and 4.0. But mere humans will be attempting to assign a score to each project's funding application.

The score is the sum of:

- Draw: The proposal writer attempts to convey their skill and the reviewer attempts to assess the skill behind the proposal. Both of these processes are imperfectly accurate. The writer may be inspired and write a better proposal than their actual skill. The other end of the continium is that they have a bad day and write a worse proposal than their actual skill or something in between. The reader has the same issues, in a good mood with a strong cup of coffee, they may assess the proposal at a higher level of skill. Weak coffee and a gloomy day may carry over to a lower assessment of skill than actual.

## Smart humans who figured out they are stupid

This score business is gonna be noisy. For example the smartest people at the world's smartest artificial intelligence conference (Neurips) can't agree on the quality of submitted research papers. Respect to the smart people who worked hard to figure out they were stupid--twice. To be fair the entire task is roughly impossible, but there are places of agreement--more on that later. 

Above you had the option to use the bell-curve to assign skill as god which in this simulation is the truth and a statement about how true skill varies among applicants. But now we are asking a mere human to **play** god by assigning a score to a research proposal that will be used to make a decision. If we had 100 reviewers we could assume the god assigned skill to be recovered most often of any score. The next most common scores would be plus/minus one grade (1.0), even less common would be those scores plus/minus 2.0 and so on. 

But just like whether Y,GR chose a bell curve for true skill, the bell curve, aka Normal or Gaussian distribution, is an assumption. But as an assumption it covers a broad range of sins and problems that I'll decriment my bullet-point budget to articulate:

- The key idea behind the bell curve as an error model for measurement is that errors tend to cancel each other out but only on average. So Dr. Faultenroy's utter ignorance of Bayesian statistics is canceled out by the fact that the title had a good pun. 
- Because averages don't always end up at 50\% for two choices you get skewed heads/tails lots of times.
- But most of the time you get 50/50, next most 49/51, 51/49, next most 48/52 and so on.
- In violation of the 3 bullet point rule: There are other error models, bell curve is just very common and robust. 

<img bell curve for coins>

That's enough for now, go take a statistics course if you want to know more and for the love of god make sure it is a Bayesian one.

So how do we manage our stupid human reviewer simulation? We assume they are going to be wrong a bunch but on average they will hit their numbers or be pretty close. We "virtually" throw a dart at the dark area of the above bell curve and if we hit the dark area we return the number on the x axis. If you think about it if we throw a million darts, the most common value will be the god decided skill. Pretty neat. But remember Y,GR, if you chose a bell curve for skills it has nothing to do with the measurements above, you could have been a unitarian. 

The approach to scoring is very simple. We draw, 'throw a dart', at the bell curve that is centered at 0, take the value, positive or negative, and add it to the skill. You can see the result in the below table.
"""
