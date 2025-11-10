import streamlit as st


# dummy_state = {
#     "researcher": {
#         "bull": [
#             "Bull Round 1: Market shows strong upward momentum.",
#             "Bull Round 2: Strong earnings forecast supports growth."
#         ],
#         "bear": [
#             "Bear Round 1: Rising interest rates may slow the market.",
#             "Bear Round 2: High volatility indicates potential downturn."
#         ],
#         "facilitator": [
#             "Facilitator Round 1: Bull is optimistic, Bear highlights macro risks.",
#             "Facilitator Round 2: Bull leans on earnings, Bear emphasizes volatility."
#         ]
#     }
# }



def display_researcher_rounds(state: dict):
    """
    Display Bull, Bear, Facilitator arguments inside
    the container placeholders stored inside the same state dictionary.
    """

    researcher = state.get("researcher", {})

    bull_list = researcher.get("bull", [])
    bear_list = researcher.get("bear", [])
    facilitator_list = researcher.get("facilitator", [])

    total_rounds = len(bull_list)

    # ✅ Loop through rounds based on number of bull responses
    for i in range(total_rounds):
        # container name: researcher_round1_container, researcher_round2_container, ...
        container_key = f"researcher_round{i+1}_container"
        container = state.get(container_key)

        if not container:
            continue  # skip if container not found

        with container:
            st.subheader(f"📍 Round {i+1}")

            st.markdown("### 🟢 Bull Agent")
            st.write(bull_list[i] if i < len(bull_list) else "No bull data")

            st.markdown("### 🔴 Bear Agent")
            st.write(bear_list[i] if i < len(bear_list) else "No bear data")

            st.markdown("### 🟡 Facilitator Summary")
            st.write(facilitator_list[i] if i < len(facilitator_list) else "No facilitator data")

    return state
# if __name__ == "__main__":
#     display_debate_ui(dummy_state)

#import streamlit as st


# dummy_state = {
#     "researcher": {
#         "bull": [
#             "Bull Round 1: Market shows strong upward momentum.",
#             "Bull Round 2: Strong earnings forecast supports growth."
#         ],
#         "bear": [
#             "Bear Round 1: Rising interest rates may slow the market.",
#             "Bear Round 2: High volatility indicates potential downturn."
#         ],
#         "facilitator": [
#             "Facilitator Round 1: Bull is optimistic, Bear highlights macro risks.",
#             "Facilitator Round 2: Bull leans on earnings, Bear emphasizes volatility."
#         ]
#     }
# }



def display_researcher_rounds_withexpander(state: dict):
    """
    Display Bull, Bear, Facilitator arguments inside
    the container placeholders stored inside the same state dictionary.
    """

    researcher = state.get("researcher", {})

    bull_list = researcher.get("bull", [])
    bear_list = researcher.get("bear", [])
    facilitator_list = researcher.get("facilitator", [])

    total_rounds = len(bull_list)

    # ✅ Loop through rounds based on number of bull responses
    for i in range(total_rounds):
        # container name: researcher_round1_container, researcher_round2_container, ...
        container_key = f"researcher_round{i+1}_container"
        container = state.get(container_key)

        if not container:
            continue  # skip if container not found
        
        with container:
            with st.expander(f"📍 Round {i+1}"):

                st.markdown("### 🟢 Bull Agent")
                st.write(bull_list[i] if i < len(bull_list) else "No bull data")

                st.markdown("### 🔴 Bear Agent")
                st.write(bear_list[i] if i < len(bear_list) else "No bear data")

                st.markdown("### 🟡 Facilitator Summary")
                st.write(facilitator_list[i] if i < len(facilitator_list) else "No facilitator data")

    return state
# if __name__ == "__main__":
#     display_debate_ui(dummy_state)