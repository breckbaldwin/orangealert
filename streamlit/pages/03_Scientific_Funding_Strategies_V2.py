from collections import defaultdict
from turtle import onclick
import streamlit as st
#st.set_page_config(layout="wide")
import pandas as pd
import os
import sys
import copy
import plotnine as p9
import re


sys.path.append("pages/")

import util

SS = st.session_state
if 'df' not in SS:
    SS.df = None
    SS.accum_df = None
    SS.standard_deviation = .25
    SS.budget = 2
    SS.hybrid_top_n_budget = SS.budget//2
    SS.funding_amount_in_millions = 1.0
    SS.num_funding_rounds = 4
    SS.reputation_increase_per_funding_round = .5
    SS.minimum_threshold = 1.0
    SS.num_sims = 1
    SS.algo_names = ['Top N', 'Random N', 'Hybrid']
    SS.num_projects = 10
    SS.skills = [0.0, 1.0, 2.0, 3.0, 4.0]
    SS.skills.reverse()
    SS.names = "abcdefghijklmnopqrstuvwxyz".upper()
    SS.proj_skill_values = [1.0] * SS.num_projects

def run_simulation(proj_data_2, 
                    num_funding_rounds, 
                    standard_deviation,
                    reputation_increase_per_funding_round, 
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
        util.add_score([top_n, rand_n, hybrid], standard_deviation, round_num)

        top_n_winners = util.select_top_n(top_n, budget)
        util.distribute_awards(top_n_winners, 1,
                            reputation_increase_per_funding_round)
        results.extend(top_n)

        random_n_winners = util.select_random_n(budget, rand_n, 
                                                minimum_threshold)
        util.distribute_awards(random_n_winners, 1,
                                reputation_increase_per_funding_round)
        results.extend(rand_n)

        hybrid_top_n_winners = util.select_top_n(hybrid, hybrid_top_n_budget)
        hybrid_random_n_candidates = [candidate for candidate in hybrid if 
                                    candidate not in hybrid_top_n_winners]
        
        hybrid_random_n_winners = \
            util.select_random_n(budget - hybrid_top_n_budget, 
                                hybrid_random_n_candidates, minimum_threshold)
        util.distribute_awards(hybrid_top_n_winners + hybrid_random_n_winners,
                                1,
                                reputation_increase_per_funding_round)
        results.extend(hybrid)
    return pd.DataFrame(results)

def run_n_simulations(num_sims, proj_skill_values, names, num_funding_rounds,
                      standard_deviation, reputation_increase_per_funding_round, 
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
                                standard_deviation,
                                reputation_increase_per_funding_round, 
                                budget,
                                hybrid_top_n_budget,
                                minimum_threshold)
        accum_final_counts(SS.accum_df, SS.df, algo_names, num_funding_rounds)
#    accum_gt_counts(num_funding_rounds, algo_names, SS.accum_df)
    SS.accum_df['percent'] = SS.accum_df['count']/( num_projects * num_sims)
    #SS.accum_df['>= percent'] = SS.accum_df['>= count']/total_funds
    #st.dataframe(SS.accum_df)
    SS.draw_done = True

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

show_explanation = True
st.title("How Algorithims Influence Research Diversity")
if show_explanation:
    st.markdown("A simulation fueled exploration")
    st.markdown(("**Breck Baldwin**, breckbaldwin@gmail.com" +
                 "\nSeptember, 2022"))

show_explanation = st.checkbox("Show Explanation", value=show_explanation)

if show_explanation:
    exp = st.expander("Introduction", expanded=False)
    exp.markdown("""
## Welcome to the simulation

This is going to be a _little_ different from a standard blog post and an experiment in exposing you, gentle reader, (Y,GR) to a more dynamic way to explore ideas you may not be familliar with. 

## Where is the juice? 

To put it bluntly, this article talks about approaches to research grant funding which I assume you don't really have opinions about or particularly care about. The juice or what you get out of reading/interacting with this is:
- Lots of important decisions get made with the same algorithms I will be covering. Admissions, job applications and who gets picked for each side in a school yard kickball game.
- Y,GR will learn properties of the algorithms by running simulations with brutally short explanations. Experiential learners rejoyce! I'm your boy.
- No rule of three third point--got simulations to cover....

## Scoring candidates

Selection of 'winners' of limited resources often score candidates as a first step. This work focuses on what is done **after** the candidate scoring step has happened. But we still need a scoring step. The structure is as follows:

- We have 10 projects, A-J
- The projects have a 'skill' value between 0.0, an F, to 4.0, an A on US style grading scales. 
- The projects can have the same skill or different skills, you get to play around with it.
""")

    exp.markdown("""### Simulate score as a function of skill

You'll be simulating wrecked lives as well as meteoric ascensions to greatness in no time, be patient. Who lets Y,GR play god? We do.

Steps are:

1. Assign skill values to the projects. The default is everyone is a '1.0'. If you choose to remain a communist you can leave them, otherwise you can mix them up a bit. 
2. There is a 'Bell Curve' button to recreate Y,GR's harsher grading environments. 
3. The skill is unavailable to mere human evaluators, but Y,GR are functioning as god here-- so you get access to the source code.
""")

if show_explanation:
    st.write("Initial Skills Assignment")

exp = st.expander("Set project abilities")
if exp.checkbox("Apply Bell curve grading", value=False):
    SS.proj_skill_values = [0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0]
cols = exp.columns(SS.num_projects)
for i in range(SS.num_projects):
    SS.proj_skill_values[i] = \
        cols[i].radio(f"Proj {SS.names[i]}", 
                        SS.skills, 
                        index=SS.skills.index(SS.proj_skill_values[i]),
                        key=i, 
                        horizontal=False)

#top_n_df = pd.DataFrame(top_n)
if show_explanation:
    exp = st.expander("Stupid Human Simulation: Drawing a score from the bell curve", expanded=False)
    exp.markdown("""
Now we have some place to start before we pitch off into the dreaded algorithims. Just a bit more to do before the **judging** begins.

As alluded to above, only god, knows the true skill behind a project. This could be people, resources, research area etc... But mere humans will be attempting to assign a score to each project's funding application. 

Starting to drift off? Swap out 'score' with 'college application', 'Mr. America profile', 'audition' or any of how much one's 'jib' the-cut-of is apprecatiated by another human. 

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
""")

standard_deviation =\
    st.slider("67% of scores fall within specified +/- range in grade points",   
              min_value=0.0, max_value=1.0, step=0.25, value=SS.standard_deviation)

if 'df' not in SS:
    SS.df = None
    SS.accum_df = None



one_run_button_description = "Run simulation once"
if show_explanation:
    if st.button(one_run_button_description):
        run_n_simulations(SS.num_sims, 
                          SS.proj_skill_values, 
                          SS.names,
                          SS.num_funding_rounds,
                          SS.standard_deviation, 
                          SS.reputation_increase_per_funding_round, 
                          SS.budget, 
                          SS.minimum_threshold,
                          SS.hybrid_top_n_budget,
                          SS.algo_names, 
                          SS.num_projects)

if SS.df is None:
    if show_explanation:
        st.info(f"Push {one_run_button_description} to evaluate/draw evaluations")
        st.stop()
    else:
        run_n_simulations(1, 
                          SS.proj_skill_values, 
                          SS.names,
                          SS.num_funding_rounds,
                          SS.standard_deviation, 
                          SS.reputation_increase_per_funding_round, 
                          SS.budget, 
                          SS.minimum_threshold,
                          SS.hybrid_top_n_budget,
                          SS.algo_names, 
                          SS.num_projects)
        
if show_explanation:
    exp = st.expander("Show drawn Scores", expanded=False)
    res_df = SS.df[(SS.df['algo'] == 'Top N') & (SS.df['round'] == 1)]
    exp.dataframe(res_df.loc[:, res_df.columns.isin(['id', 'draw'])])
    exp.markdown("""
Each round of funding will draw a score and add it to the skill + reputation scores for the project. The reputation is 0 now, but with successful funding it will grow which reflects the benefit of a project being funded for subsequent rounds of funding. Reputation is how the rich get richer in this simulation which may or may not be a good idea--and it is central to the algorithms that we are experimenting with below.
""")

    exp = st.expander("The Top N Algorithm")
    exp.markdown("""
## Algorithmic Meritocracy: Top N

The Top N algorithm will take the available budget, constrained to \$1 million awards, and parcel out budget starting at the top scoring project. If we have \$3 million in the budget, then the top 3 scoring projects get funding. If there are ties for the score then pick from the order that happens to be in the list. 

Since the default order in this implementation will always choose alphabetically 'Proj A' before 'Proj B' if they have the same score we see programmer lazyness introducing bias.

Repeating the algorithm:

1. Select Top N proposals by score where N = millions of dollars of budget.
2. For each awarded proposal, increment the total funding by $1 million 
3. For each awarded proposal Increment the reputation by .5 (half a grade point)

## Lets spend some money!

Below we have the controls for a Top N algorithm simulation. 


""")

def generic_handler(widget_name, variable):
    SS[variable] = SS[widget_name]

st.slider(("Budget in millions per cycle--each" + 
            "award is $1 million?"), 
            min_value=2, max_value=10, step=2, value=SS.budget, 
            on_change=generic_handler,
            args=('budget slider', 'budget'),
            key='budget slider')

hybrid_random_n_budget = SS.budget - SS.hybrid_top_n_budget
st.write(f"Hybrid Top N={SS.hybrid_top_n_budget} Random N={hybrid_random_n_budget}")


if show_explanation:
    exp = st.expander("Show one round of funding")
    (col1, col2) = exp.columns(2)
    col1.write("All Projects")
    top_n_df = res_df.loc[:, SS.df.columns.isin(['id', 'skill', 'draw',
                                             'score', 'total funds'])]
    col1.dataframe(top_n_df)
    col2.write("Winning Projects")
    col2.dataframe(top_n_df[top_n_df['total funds'] == 1])

#exp = st.expander("Applying Top N algorithm for multiple rounds of funding")

top_n_doc = """
## Cumulative Effects of Many Rounds of Funding: Empire building

The Top N algorithm really shows its properties with repeated application. The key insight is the role of accumulated reputation which will give an advantage to projects that previously recieved funding. The analog in other domains would be generational wealth, personal wealth, and legacy applicants to colleges. 

The below slider controls the number of funding rounds which in turn creates a graph showing the accumulated funding for projects over time. 
"""

if show_explanation:
    exp= st.expander("Random N algorithm description")

    exp.markdown("""
## Random funding above a threshold: Work hard **and** get lucky

This here is **very** threatening to metocratic ideals by making clear that all that hard work may not pay off because of luck. Try it on for size: 

- Work hard, get a PhD, write a difficult proposal to qualifiy to be considered for funding with a score above a threshold. If more proposals are above threshold than the budget allows then select randomly. Brutal no? 
""")

    exp = st.expander("Hybrid: The reality program managers would accept")

    exp.markdown("""
Proposals tend to either be OMG this should be funded with a long tail of less extraordnary efforts. Program managers, admission committees and other selection processes do feel that judgement has an important and predictively useful role which is the driving force behind the Top N algorithm. So the hybrid algorithm acknoledges that but changes that to Top N/2 where half the funding is done that way, the remainder is Random N. The ratio could be adjusted but trying to keep it simple.
""")

(col1, col2, col3) = st.columns(3)

col1.slider("How many funding cycles?", min_value=1, 
            max_value=10, step=1, value=SS.num_funding_rounds,
            on_change=generic_handler,
            args=('funding slider', 'num_funding_rounds'),
            key='funding slider')

col2.slider("How much increase in reputation per funding award", 
                min_value=0.0, 
                max_value=2.0, 
                step=.25,
                value=SS.reputation_increase_per_funding_round,
                on_change=generic_handler,
                args=('reputation slider', 
                      'reputation_increase_per_funding_round'),
                key='reputation slider')

col3.slider("Minimum threshold for funding",
            min_value=0.0,
            max_value=3.0,
            step=.25,
            value=SS.minimum_threshold,
            on_change=generic_handler,
            args=('threshold slider', 'minimum_threshold'),
            key='threshold slider')

if show_explanation:
    exp1 = col1.expander("Details")
    exp1.markdown(top_n_doc)

def apply_options():
    params = SS.interesting_params.split(', ')
    for param in params:
        split = param.split(': ')
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
            SS[param] = value

options = ['Custom', 
        'reputation_increase_per_funding_round: 0.0, All algorithms perform same',
        'reputation: 1.5, Reputation increase: .25, Funding Threshold: 1.0, Score variation:.25, Budget: 2 million']
st.selectbox("Interesting Parameterizations", options=options,
            on_change=apply_options,
            key='interesting_params')

exp = st.expander("Show Last Simulation Detail Graph")
if SS.num_sims > 1:
    exp.info("Only last simulation results graphed since the number of simulations is > 1")

(col1, col2, col3) = exp.columns(3)
show_top_n = col1.checkbox("Top N", value=True)
show_rand_n = col2.checkbox("Random N", value=True)
show_hybrid = col3.checkbox("Hybrid", value=False)

y_dimension = 'total funds'
y_dimension = exp.radio("Y dimension",
                options=['total funds', 'draw', 'reputation', 'score', 'skill'],
                horizontal=True)
plot = render3(SS.df, y_dimension, show_top_n, show_rand_n, show_hybrid)
if plot is not None:
    exp.pyplot(p9.ggplot.draw(plot))


y_units= 'count'
y_units = st.radio("Y units", ['count', 'percent'], index=0)

from mizani.formatters import percent_format
#st.dataframe(SS.accum_df)
plot = (p9.ggplot(SS.accum_df, p9.aes(x='funding bin', y=y_units, fill='algo')) 
      + p9.geom_col(position='dodge')
)

if y_units == 'percent':
    plot = plot + p9.scale_y_continuous(labels=percent_format())

st.pyplot(p9.ggplot.draw(plot))

exp = st.expander("Show Parameter Selections")


def run_n_wrapper():
    SS.num_sims = SS.simulation_radio_btn
    run_n_simulations(SS.num_sims, 
                    SS.proj_skill_values, 
                    SS.names,
                    SS.num_funding_rounds,
                    SS.standard_deviation, 
                    SS.reputation_increase_per_funding_round, 
                    SS.budget, 
                    SS.minimum_threshold,
                    SS.hybrid_top_n_budget,
                    SS.algo_names, 
                    SS.num_projects)

options = [1, 2, 10, 100]
st.radio("Run N simulations", options=options, 
          index=options.index(SS.num_sims), 
          on_change=run_n_wrapper, 
          key='simulation_radio_btn')
st.button("Rerun as configured", on_click=run_n_wrapper)


g = """


Based on paper at: https://breckbaldwin.github.io/S3rd/presentations/DOE2021/FundingStrategiesForSciSoftware.html


**TL;DR** Ranking research proposals by quality and choosign Top N leads to concentration of resources to N organizations. Random N selection of proposals that are above a quality threshold distributes funding opportunties more broadly. 

- All projects started with the same merit and reputation values. 
- Score = The score for a round of funding is a random draw from a normal distribution centered on merit with standard deviation .1 then added to the reputation value.
- Top N: Award top N scored proposals for the round.
- Random N: Award random N selected from Score > funding threshold.
    + The default threshold for funding is .2, so candidates have to get a bit lucky to draw a high value for the score initially to clear the threshold and then get lucky by being drawn from the set of candidates above threshold.
- If a candidate is funded then their reputation increases .1 and the resource count increases by 1. Resources/reputation can only go up, merit stays the same. """


