# ChatInject multi-turn data

The multi-turn ChatInject attacks (`chat_inject_*_with_utility_*_multiturn_*`) load pre-generated
conversation JSON from this directory, keyed by injection-task GOAL. These files are NOT vendored — copy
them from ChatInject (https://github.com/hwanchang00/ChatInject/tree/main/src/agentdojo/attacks/multi_turn_data):

    curl -L -o with_utility_multi_turn_7_generation_output.json \
      https://raw.githubusercontent.com/hwanchang00/ChatInject/main/src/agentdojo/attacks/multi_turn_data/with_utility_multi_turn_7_generation_output.json
    curl -L -o with_utility_multi_turn_authority_endorsement_7_generation_output.json \
      https://raw.githubusercontent.com/hwanchang00/ChatInject/main/src/agentdojo/attacks/multi_turn_data/with_utility_multi_turn_authority_endorsement_7_generation_output.json

Coverage: banking / slack / travel injection GOALs only — NOT shopping. Match is exact string equality on
`injection_task.GOAL`; verify your suite's GOALs are keys in the JSON, or the attack raises ValueError.
