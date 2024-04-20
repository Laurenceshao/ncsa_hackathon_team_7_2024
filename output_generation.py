import json

path_to_json_file = "step_summary.json" # we will just update this file, no need to create a new file


def generate_output(executor_output: (str, str)) -> str:
    f = open(path_to_json_file)
    step_summary_list = json.load(f)
    new_id = 10
    new_step_number = 10
    new_step_summary = {
        "id": new_id, 
        "step number": new_step_number,
        "step summary": executor_output[0],
        "output of step": executor_output[1]
    }
    step_summary_list["step summary list"].append(new_step_summary)
    with open('step_summary.json', 'w') as f:
        json.dump(step_summary_list, f)
    
    summary_of_steps = ""
    for step_summary in step_summary_list["step summary list"]:
        summary_of_steps += "Then, " + step_summary["step summary"] + "\n"

    return summary_of_steps



example_json_object_as_dict = {
    "step summary list": [
        {
            "id": 10001010100, 
            "step number": 1,
            "step summary": "In this step I did...",
            "output of step": "53.3"
        },
        {
            "id": 10001010101, 
            "step number": 2,
            "step summary": "In this step I did...",
            "output of step": "12.1"
        }
    ]
}

with open('step_summary.json', 'w') as f:
    json.dump(example_json_object_as_dict, f)

executor_output = ("executor result", "executor code that generated result")
summary_of_steps = generate_output(executor_output)
# send_to_planner(summary_of_steps, path_to_json_file)
print(summary_of_steps)
