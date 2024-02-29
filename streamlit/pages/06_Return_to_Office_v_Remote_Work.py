import streamlit as st


UI = st
SS = st.session_state

UI.title("Return to Office (RTO) vs Work from Home (WFH): Simulating an Thorny Issue")

UI.markdown("""
The pandemic shutdown broke a foundational pillar of most work: you have to go into the office. Gone in a snap by executive fiat and yet work still got done, maybe more, innovation happened, maybe less and work life did not melt down into anarchy. All in all a remarkable event that would change work forever except some employers want it the way it was. 
            
Asserting RTO comes with very vauge justifications and little in the way of hard evidence that this is needed for the bottom line. Generally the lame excuse is that "water cooler conversations" are important for X. Counter arguments are that execs are trying to recover sunk costs in offices, managers feel useless unless their minions are before them for physical inspection and so on. 

Lets pretend there is a rational case for RTO and simulate how things are different in the WFH environment. 
            """)

# productivity
# innovation
# coordination
# onboarding
# slacking
# control
# civic duty to downtowns
# 

# innovation = execution * cross pollination * focus * resouces * effort

UI.markdown("""
I'll start with a really mathematically boring model, just a bunch of multiplicative factors but I'm hoping it helps structure my thinking. 
            
innovation = execution * cross pollination * focus * resouces * effort
            
Innovation was the reason management offered at my day job so lets start with that.
            
I'll stick with a three way distinction picking from "Office", "About the same", and "Home". 
            
We will take the office
            """)

office_same_home = ['Office', 'About the same', 'Home']
office_same_home_values = [.5, 1, 1.5]

def num_fr_ordered_cat_selectbox(text, ordered_options, numeric_values):
    choice = UI.selectbox(text,
             options=ordered_options)
    return numeric_values[ordered_options.index(choice)]
    
work_harder_at = num_fr_ordered_cat_selectbox("People work harder at:", office_same_home, office_same_home_values)

problem_solving_better_at = num_fr_ordered_cat_selectbox("Problem solving with others works best at:", office_same_home, office_same_home_values)


y = work_harder_at + problem_solving_better_at

UI.markdown(f"{y} = {work_harder_at} + {problem_solving_better_at}")


