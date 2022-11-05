from collections import defaultdict
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

RNG = default_rng()
sys.path.append("pages/")

import util

SS = st.session_state

pd.set_option('display.max_colwidth', None)

def reset():
    SS.df = None
    SS.accum_df = None
    SS.sd_writeup_reflection_of_skill = .5
    SS.budget = 2
    SS.hybrid_top_n_budget = SS.budget//2
    SS.funding_amount_in_millions = 1.0
    SS.num_funding_rounds = 5
    SS.reputation_increase_per_funding_round = .5
    SS.minimum_threshold = 1.0
    SS.num_sims = 1
    SS.algo_names = ['Top N', 'Random N', 'Hybrid']
    SS.num_projects = 10
    SS.skills = [0.0, 1.0, 2.0, 3.0, 4.0]
    SS.skills.reverse()
    SS.names = "abcdefghijklmnopqrstuvwxyz".upper()
    SS.names = ['Amit', 'Beth', 'Chris', 'Drew', 'Enid', 'Fred', 'Gina', 'Hank',
                'Ivor', 'Jude']
    SS.proj_skill_values = [1.0] * SS.num_projects
    SS.show_explanation = True
    SS.Notes = 'reset()'
    SS.show_top_n = True,
    SS.show_rand_n = True,
    SS.show_hybrid = False
    SS.y_dimension = 'total funds'
    SS.sel_run_1 = 'latest'
    SS.sel_run_2 = 'latest'
    SS.y_units_1 = 'percent'
    SS.y_units_2 = 'percent'
    SS.currently_being_scored = 0
    SS.score_individually = True
    SS.sd_writeup_reflection_of_skill = .5
    SS.sd_reviewer_accuracy = .5
    SS.current_round = 1

session_config_values = ['Notes', 'sd_writeup_reflection_of_skill', 
'sd_reviewer_accuracy', 'budget', 
'funding_amount_in_millions', 'num_funding_rounds', 
'reputation_increase_per_funding_round', 'minimum_threshold', 'num_sims',
'proj_skill_values', 'accum_df']

def generic_handler(widget_name, variable):
    SS[variable] = SS[widget_name]

if 'df' not in SS:
    reset()
    SS.sessions = []

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
        proj['algo'] = 'Top N'
    rand_n = copy.deepcopy(proj_data_2)
    for proj in rand_n:
        proj['algo'] = 'Random N'
    hybrid = copy.deepcopy(proj_data_2)
    for proj in hybrid:
        proj['algo'] = 'Hybrid'
    results = []
    results.extend(top_n)
    results.extend(rand_n)
    results.extend(hybrid)
    for round_num in range(1, num_funding_rounds + 1):
        top_n = copy.deepcopy(top_n)
        rand_n = copy.deepcopy(rand_n)
        hybrid = copy.deepcopy(hybrid)
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
                      sd_review, reputation_increase_per_funding_round, 
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
                                reputation_increase_per_funding_round, 
                                budget,
                                hybrid_top_n_budget,
                                minimum_threshold)
        accum_final_counts(SS.accum_df, SS.df, algo_names, num_funding_rounds)
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
                    SS.reputation_increase_per_funding_round, 
                    SS.budget, 
                    SS.minimum_threshold,
                    SS.hybrid_top_n_budget,
                    SS.algo_names, 
                    SS.num_projects)

def run_n_wrapper():
    SS.num_sims = SS.simulation_radio_btn
    SS.Notes = ''
    run_sim_as_configured()


def render3(df, column_to_show, show_top_n, show_random_n, show_hybrid):
    if not (show_top_n or show_random_n or show_hybrid):
        st.info("select a dataset to view")
        return None
    offset_scale =  (max(df[column_to_show]) - min(df[column_to_show])) / 100
    offset_scale = max(offset_scale, .01)
    df['y'] = df[column_to_show] + df['y_offset'] * offset_scale

    plot = (p9.ggplot(mapping=p9.aes(x='round', y='y', group = 'id')))
    if show_top_n:
        plot = plot + p9.geom_line(data=df[df['algo'] == 'Top N'],
                                    mapping=p9.aes(color='id'), size=.7)
    if show_random_n:
        plot = plot + p9.geom_line(data=df[df['algo'] == 'Random N'],   
                                    mapping=p9.aes(color='id'), size=.7, 
                                    linetype='dotted')
    if show_hybrid:
        plot = plot + p9.geom_line(data=df[df['algo'] == 'Hybrid'], 
                                    mapping=p9.aes(color='id'), size=.7, 
                                    linetype='dashdot')
    if (column_to_show == 'skill' 
        and max(df[column_to_show]) == min(df[column_to_show])):
        single_y = max(df[column_to_show])
        plot = plot + p9.scale_y_continuous(
            label=f"Y projects jittered by {offset_scale:.2f}",
            limits=[single_y - 0.5, single_y + .5])

    plot = (plot + p9.xlim([0,SS.num_funding_rounds]) 
            + p9.ylim([0,SS.num_funding_rounds]))
    
    plot = plot + p9.labels.ylab(f"Y projects jittered by {offset_scale:.2f}")
    #plot = plot + p9.theme_xkcd()
    return plot

def accum_final_counts(sim_df, df, algo_names, num_funding_rounds):
    for algo in algo_names:
        last_round_algo_df = df[(df['algo'] == algo) & 
                            (df['round'] == num_funding_rounds)]
        for total_funds in last_round_algo_df['total funds']:
            total_funds_int = int(total_funds)
            sim_df.loc[(sim_df['algo'] == algo) & 
                        (sim_df['funding bin'] == total_funds_int), 
                        'count'] += 1

def accum_gt_counts(num_funding_rounds, algo_names, sim_df):
    for algo in algo_names:
        for value in range(1, num_funding_rounds + 1):
                gt_count = \
                    sum(sim_df[(sim_df['algo'] == algo) &
                            (sim_df['funding bin'] >= value)]['count'])
                sim_df.loc[(sim_df['algo'] == algo) &
                            (sim_df['funding bin'] == value), 
                            '>= count'] = gt_count

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

Society allocates scarce resources in all sorts of ways--a common one relies on convincing others that you deserve membership in an elete cohort: The 2023 incoming class at Harvard, presentation at an academic conference and my focus--being among the awardees of a research grant. This work more fully explores a screed I wrote for the Department of Energy's conference on funding scientific software, reference to the paper and accompanying simulation is https://breckbaldwin.github.io/S3rd/presentations/DOE2021/FundingStrategiesForSciSoftware.html.

TL;DR The simulation covers 

1. A _Top N_ approach which selects the highest scoring candidates with N funding slots, e.g., an American meritocracy.
2. A _Random N_ approach that awards N slots randomly to candidates that pass a minumum score threshold.
3. A _Hybrid_ approach that blends the two. 

## Scoring candidates

I'll use research funding as the use-case for the simulation with the following properties:

- We have 10 researchers applying for funding for 5 funding cycles--many government programs have 5 year cycles with funds awarded each year. Similar rationals can be made for the other use-cases of admissions or conference papers. 
- The researchers have a 'skill' value between 0.0, an F, to 4.0, an A on US style grading scale that are their actual ability/smarts/training. 

## Playing God

You'll be simulating wrecked careers as well as meteoric ascensions to greatness in no time, but skills have to be assigned first. There are three options plus just setting scores as your omnipotence decrees:

1. Bell Curve: Mostly C's, some B's and D's and an outlier A and F.
2. God's Gift: One genius, A, in a collection of mediocrity, D's.
3. A Mother's Love: All researchers have the same mother who raised them equally skilled but mother is a realist and knows they are pretty average (C's).

""")

exp = st.expander("Set skills for projects")
exp.radio("Suggested Skill Collections", 
    options=["Bell Curve", "God's Gift is Amongst Us", "A Mother's Love"],
    index=2,
    horizontal=True)

SS.proj_skill_values = [0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0]

SS.proj_skill_values = [2.0] * SS.num_projects

cols = exp.columns(SS.num_projects)
for i in range(SS.num_projects):
    SS.proj_skill_values[i] = \
        cols[i].radio(f"{SS.names[i]}", 
                        SS.skills, 
                        index=SS.skills.index(SS.proj_skill_values[i]),
                        key=i, 
                        horizontal=False)


#top_n_df = pd.DataFrame(top_n)
if SS.show_explanation:
    exp = st.expander("Fickle Human Simulation", expanded=False)
    exp.markdown("""
While you, God-like, know the skills assigned above, mere mortals write proposals above or below their ability and reviewers even less reliably assess the skill reflected in the written proposal. We simulate this by drawing randomly from a Gaussian/normal random distribution--think throwing darts with the bull's eye at 0,0:

- Horizontal distance from the bull's eye is how accurately the proposal refelects the actual skill of the researcher.

- Vertical distance from the bull's eye is the reviewer's accuracy reviewing said proposal. 

You get to control where 68 percent of the darts will land with the two sliders below, e.g., where one (1) standard deviation of random draws, aka dart throws, will land. With just 10 throws the results will not be perfectly scattered but an approximate 68\% area of the actual throws will be drawn when there are enough data points.

Below are controls and a button to throw darts one at a time. You will see the draw and the math to compute the score which is just skill + writeup draw + revew draw + reputation (0.00 for now) = score. Review draws have a 'reason' for why the review was higher or lower than a perfectly accurate assessment at 0. As often seen with humans, the reason is a complete fabrication.""")


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


def plot_draws(author_draws, reviewer_draws, proj_names):
    plot = p9.ggplot(data=pd.DataFrame({'author_draw': author_draws,
                                        'reviewer_draw': reviewer_draws,
                                        'id': proj_names}),
                    mapping=p9.aes(x='author_draws', y='reviewer_draws',
                                    label='id'))
    plot = plot + p9.geom_point()
    plot = plot + \
        p9.geom_text(nudge_x=.1, nudge_y=.1)
    
    plot = plot + \
        p9.geom_point(data=pd.DataFrame({'x':[0.0],
                                           'y':[0.0]}),
                        mapping=p9.aes(x='x', y='y'),
                        fill='red')
#    plot = plot + p9.stat_ellipse(geom='polygon', level= 0.95,
#                                    type='norm',
#                                    alpha=.2,
#                                    fill='blue')
    plot = plot + p9.stat_ellipse(geom='polygon', level= 0.68,
                                    type='t',
                                    alpha=.2,
                                    fill='red')
    #plot = plot + p9.stat_ellipse(geom='polygon', level= 0.1,
    #                                type='t',
    #                                alpha=.2,
    #                                fill='green')
    plot = (plot 
             + p9.scales.ylim([min([-1.0] + author_draws) - .1,
                               max([1.0] + author_draws) + .1]) 
             + p9.scales.xlim([min([-1.0] + reviewer_draws) - .1,
                               max([1.0] + reviewer_draws) + .1])
    )
    return(plot)

(col1, col2, col3) = st.columns(3)

if SS.show_explanation:
    (col1, col2_wide) = st.columns([1,2])

def std_dev_handler(widget_name, variable):
    generic_handler(widget_name, variable)
    SS.currently_being_scored = 0
    SS.proj_data = util.init(SS.proj_skill_values, SS.names)
    col1.info("Resetting previous scores")

col1.slider(f"Standard deviation writeup: {'68% of project writeup fall within specified +/- range in conveying skill' if SS.show_explanation else ''} ",   
            min_value=0.0, max_value=1.0, step=0.25,
            on_change=std_dev_handler,
            args=('sd_writeup_slider', 
                  'sd_writeup_reflection_of_skill'), 
            value=SS.sd_writeup_reflection_of_skill,
            key='sd_writeup_slider')

col1.slider(f"Standard deviation reviewer: {'68% of reviewer evaluations within specified +/- range in grade points' if SS.show_explanation else ''}",
            min_value=0.0, max_value=1.0, step=0.25,
            on_change=std_dev_handler,
            args=('sd_reviewer_slider',
                   'sd_reviewer_accuracy'),
            value=SS.sd_reviewer_accuracy,
            key='sd_reviewer_slider')

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
    proj_datum['draw writeup'] = draw_project_presentation
    proj_datum['draw review'] = draw_reviewer_accuracy
    proj_datum['score'] = (proj_datum['skill'] 
                            + draw_project_presentation 
                            + draw_reviewer_accuracy)
    reason = ''
    if draw_reviewer_accuracy > 0.0:
        reason = random.sample(reasons_positive_score, 1)
    else:
        reason = random.sample(reasons_negative_score, 1)
    proj_datum['reason why review is skewed'] = reason
    SS.currently_being_scored = \
            (SS.currently_being_scored + 1) % SS.num_projects

def render_scoring_df(df, out):
    if 'draw writeup' not in df:
        out.write("Click on button")
        return
    disp_df = pd.DataFrame()
    disp_df['skill'] = df['skill']
    disp_df['writeup'] = df['draw writeup']
    disp_df['review'] = df['draw review']
    disp_df['rep'] = df['reputation']
    disp_df['score'] = df['score']
    disp_df['reason for skew'] = df['reason why review is skewed']
    disp_df['Proj Id'] = df['id']
    display_cols = ['Proj Id', 'skill', 
                                'writeup', 
                                'review',
                                'score', 
                                'rep',
                                #'reason for skew'
                                ]

    disp_df = disp_df.loc[:, 
              disp_df.columns.isin(display_cols)]
    
#    disp_df = pd.DataFrame(disp_df.iloc[-1])
    out.dataframe(disp_df.style.format(subset=['writeup', 'review',
                                                'skill', 'rep', 
                                                'score'], formatter='{:.2f}'))

if SS.show_explanation:
#    col1.checkbox("Score Individually", 
#                value=SS.score_individually,
##                on_change=generic_handler,
#                args=('score_indiv_cb', 'score_individually'),
#                key='score_indiv_cb')
    if SS.score_individually:
        if 'proj_data' not in SS:
            SS.proj_data = util.init(SS.proj_skill_values, SS.names)
        df = pd.DataFrame(SS.proj_data)
        df = df[df['draw writeup'].notnull()]
        if len(df) < SS.num_projects:
            col1.button(f"Draw writeup and review variation for {SS.names[SS.currently_being_scored]}'s project",
                on_click=add_score,
                key=f"draw_button")
        if len(df) > 0:
            try:
                plot = plot_draws(list(df['draw writeup']), 
                              list(df['draw review']),
                              list(df['id']))
                col2_wide.pyplot(p9.ggplot.draw(plot))
            except p9.exceptions.PlotnineWarning as e:
                st.info(e)
            col1.write(f"Reason for reviewer draw: {df['reason why review is skewed'].iloc[-1][0]}")
            render_scoring_df(df, col2_wide)
            st.markdown(f"""
Each of the algorithms is briefly explained below, the algorithms share the same draws across each round of funding but the cumulative effects will differ as reflected in the reputation value and accumulated funding. 

Reputation increments at {SS.reputation_increase_per_funding_round} for all the algorithms per award. This value can be changed below. Reputation is the mechanism that reflects the impact of having resources from previous funding. For school admissions it would be the impact that getting into a good high school has on getting into a good college, or having a previously published conference paper on getting another conference paper accepted.

All awards are \$1 million and the budget is exhausted each round. The budget can be raised to up to \$10 million but in even incremnts to allow for even split of the _Hybrid_ algorithm.
""")
        
        if len(df) < SS.num_projects:
            st.info(f"{SS.num_projects - len(df)} projects left to draw")
            st.stop()

        #df id factor=['id', 'skill', 'draw', 'score', 'reputation] value
#        disp_long_df = pd.melt(disp_df, id_vars=['id'],
#                value_vars=['id', 'skill', 'draw', 'score', 'reputation'])
        #st.dataframe(disp_long_df)

#        plot = (p9.ggplot(data=disp_long_df)
#                + p9.geom_col(p9.aes(x='id', y='value', fill='variable')))
#        st.pyplot(p9.ggplot.draw(plot))
        
#     exp.markdown("""
# Each round of funding will draw a score and add it to the skill + reputation scores for the project. The reputation is 0 now, but with successful funding it will grow which reflects the benefit of a project being funded for subsequent rounds of funding. Reputation is how the rich get richer in this simulation which may or may not be a good idea--and it is central to the algorithms that we are experimenting with below.
# """)

if 'df' not in SS:
    SS.df = None
    SS.accum_df = None
#one_run_button_description = "Draw scores for all projects"
#if SS.show_explanation:
#    if st.button(one_run_button_description):
#        run_sim_as_configured()
#        render_scoring_df(SS.df[(SS.df['algo'] == 'Top N') & 
#r                                (SS.df['round'] == 1)], col2_wide)

# if SS.df is None:
#     if SS.show_explanation:
#         st.info(f"Push {one_run_button_description} to evaluate/draw evaluations")
#         st.stop()
#     else:
#         run_n_simulations(1, 
#                           SS.proj_skill_values, 
#                           SS.names,
#                           SS.num_funding_rounds,
#                           SS.standard_deviation, 
#                           SS.reputation_increase_per_funding_round, 
#                           SS.budget, 
#                           SS.minimum_threshold,
#                           SS.hybrid_top_n_budget,
#                           SS.algo_names, 
#                           SS.num_projects)

def plot_details(out):
    options = ['total funds', 'draw', 'reputation', 'score', 'skill']
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
    plot = (p9.ggplot(accum_df, p9.aes(x='funding bin',   
                                          y=SS[y_units_key_i],       
                                          fill='algo')) 
            + p9.geom_col(position='dodge')
    )
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


if SS.show_explanation:
    cols = st.columns(3)
    exp = cols[0].expander("Top N Algorithm")
    exp.markdown(f"""
## Algorithmic meritocracy:

The _Top N_ algorithm will take the available budget of \${SS.budget} million, and parcel out $1 million at a time starting at the top scoring project until the budget is spent. 
 
If there are tied projects then pick randomly from them.
""")

    exp = cols[1].expander("Random N algorithm")
    exp.markdown("""
## Work hard AND get lucky:

The _Random N_ algorithm sets a minimum score for consideration in a lottery for the funds. Work hard, get a PhD, write a difficult proposal to score above a threshold. If more proposals are above threshold than the budget allows then select randomly. Brutal no? 

There may not be sufficient candidate proposals to qualify however, in that case the threshold is dropped by .1 until funds are exhausted iteratively with a fresh start on selection. 
""")

    exp = cols[2].expander("Hybrid Algorithm")
    exp.markdown("""
## The comprimise:
 
Proposals tend to either be OMG this should be funded with a long tail of less extraordnary efforts. Program managers, admission committees and other selection processes do feel that judgement has an important and predictively useful role which is the driving force behind the Top N algorithm. 

So the hybrid algorithm acknoledges the desire for discretion but changes that to _Top N/2_ where half the funding is done that way, the remainder is _Random N_. The ratio could be adjusted but trying to keep it simple.
""")

if SS.show_explanation:
    st.markdown("## Lets spend some money!")
    (col1, col2) = st.columns([1,1])
    slider_col = col1
    reputation_col = col2
else:
    slider_col = col2
    reputation_col = col3
slider_col.slider((f"Budget: {'In millions per funding cycle--each award is $1 million' if SS.show_explanation else ''}"), 
            min_value=2, max_value=10, step=2, value=SS.budget, 
            on_change=generic_handler,
            args=('budget slider', 'budget'),
            key='budget slider')

reputation_col.slider(f"Reputation increase: {'How much increase in reputation per funding award which is added to project score.' if SS.show_explanation else ''}", 
                min_value=0.0, 
                max_value=2.0, 
                step=.25,
                value=SS.reputation_increase_per_funding_round,
                on_change=generic_handler,
                args=('reputation slider', 
                      'reputation_increase_per_funding_round'),
                key='reputation slider')


def show_next_round():
    if SS.current_round == SS.num_funding_rounds:
        SS.current_round = 1
    else:
        SS.current_round += 1

if SS.show_explanation:
    if SS.df is None:
        if col1.button("Run Algorithms"):
            SS.num_sims = 1
            run_sim_as_configured()
            SS.current_round = 1
    #       st.dataframe(SS.df)
        else:
            st.stop()
    col1.button(f"Show next round",
                on_click=show_next_round)
        
    col2.radio(f"Algorithm to show: Round {SS.current_round}", 
                options=['Top N', 'Random N', 'Hybrid'], 
                horizontal=True,
                key="algo_cb")
    top_n_df = SS.df[(SS.df['algo'] == SS.algo_cb) &
                     (SS.df['round'] == SS.current_round)]
        
    top_n_df = top_n_df.loc[:, 
                                top_n_df.columns.isin([
                                    'id', 'reputation', 
                                    'skill',
                                    'draw writeup', 
                                    'draw review',
                                    'score', 'total funds'])]
        
    col2.dataframe(top_n_df\
        .style.format(subset=['draw writeup', 
                              'draw review',
                              'reputation',
                              'skill', 
                              'score'], formatter='{:.2f}'))
    plot = render3(SS.df[(SS.df['algo'] == SS.algo_cb) &
                        (SS.df['round'] <= SS.current_round)],
                        'total funds', 
                        SS.algo_cb == 'Top N', 
                        SS.algo_cb == 'Random N', 
                        SS.algo_cb == 'Hybrid')
    col1.pyplot(p9.ggplot.draw(plot))
    col1.write("Winners")
    SS.df[SS.df['total funds']]
    #col1.dataframe(display_df[display_df['total funds'] == 1.0])
    col1.dataframe()


#exp = st.expander("Applying Top N algorithm for multiple rounds of funding")


#if SS.show_explanation:
#    (col1_empty, col2_, col3_empty) = st.columns(3)

# col2.slider("Minimum threshold for funding",
#             min_value=0.0,
#             max_value=3.0,
#             step=.25,
#             value=SS.minimum_threshold,
#             on_change=generic_handler,
#             args=('threshold slider', 'minimum_threshold'),
#             key='threshold slider')





if SS.show_explanation:
    exp = st.expander("Show one round of funding")
    (col1, col2) = exp.columns(2)
    col1.write("All Projects")
    rand_n_df = SS.df[SS.df['algo'] == 'Random N']
    display_df = rand_n_df.loc[:, SS.df.columns.isin(['algo', 'id', 'skill', 'draw',
                                             'score', 'total funds'])]
    col1.dataframe(display_df)
    col2.write("Winning Projects")
    col2.dataframe(display_df[display_df['total funds'] == 1.0])



hybrid_random_n_budget = SS.budget - SS.hybrid_top_n_budget

if SS.show_explanation:
    exp1 = st.expander("Details")
    exp1.markdown(top_n_doc)

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

options = ['Custom', 
'Notes: A) Reset to defaults, reset()',
'Notes: B) No reputation increase--algos roughly same, reset(), reputation_increase_per_funding_round: 0.0, num_sims: 100',
'Notes: C) Sufficient funding for all programs--all algos the same, reset(), budget: 10',
'Notes: D) Really poor reviewers--algos roughly same!, reset(), standard_deviation: 1.0, num_sims: 100',
'Notes: E) High miniumum score--algos same, reset(), minimum_threshold: 2.0, num_sims: 100']

if SS.show_explanation:
    st.stop()

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
col1.button("Rerun as configured", on_click=run_n_wrapper)
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

def notes_handler():
    SS.Notes = SS.notes_input
    SS.sessions[-1]['Notes'] = SS.notes_input

st.text_input("Notes:", value=SS.Notes, 
                on_change=notes_handler,
                key='notes_input')

def render_session():
    SS.accum_df = SS.sessions[SS.graph_session_select]['accum_df']

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

g = """


Based on paper at: https://breckbaldwin.github.io/S3rd/presentations/DOE2021/FundingStrategiesForSciSoftware.html


**TL;DR** Ranking research proposals by quality and choosign Top N leads to concentration of resources to N organizations. Random N selection of proposals that are above a quality threshold distributes funding opportunties more broadly. 

- All projects started with the same merit and reputation values. 
- Score = The score for a round of funding is a random draw from a normal distribution centered on merit with standard deviation .1 then added to the reputation value.
- Top N: Award top N scored proposals for the round.
- Random N: Award random N selected from Score > funding threshold.
    + The default threshold for funding is .2, so candidates have to get a bit lucky to draw a high value for the score initially to clear the threshold and then get lucky by being drawn from the set of candidates above threshold.
- If a candidate is funded then their reputation increases .1 and the resource count increases by 1. Resources/reputation can only go up, merit stays the same. """


