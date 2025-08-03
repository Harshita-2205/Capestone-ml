import os
from dotenv import load_dotenv
import yaml
from crewai import Agent, Task, Crew, LLM
from src.output_pydantic import NERValidationOutput, PreliminaryDiagnosisListOutput


load_dotenv()

# Initialize the LLM
llm = LLM(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini/gemini-1.5-flash",
)

files = {
    "agents" : "crew/config/agents.yaml",
    "tasks" :  "crew/config/tasks.yaml"
    }

## Load configurations from YAML files

configs = {}

for config_type, file_path in files.items():
    with open(file_path, "r") as file:
        configs[config_type] = yaml.safe_load(file)


## Agents-Configs
ner_validaton_agent_config = configs["agents"]["ner_validaton_agent"]
preliminary_diagnosis_agent_config = configs["agents"]["preliminary_diagnose_agent"]
report_writer_agent_config = configs["agents"]["report_writer_agent"]

## Tasks-Configs
ner_validaton_task_config = configs["tasks"]["ner_validaton_task"]
preliminary_diagnose_task_config = configs["tasks"]["preliminary_diagnose_task"]
report_writing_task_config = configs["tasks"]["report_writing_task"]


## Agents
ner_validation_agent = Agent(
    config = ner_validaton_agent_config,
    llm=llm
)

preliminary_diagnosis_agent = Agent(
    config = preliminary_diagnosis_agent_config,
    llm=llm
)

report_writer_agent = Agent(
    config=report_writer_agent_config,
    llm=llm,
)



## Tasks
ner_validation_task = Task(
    config = ner_validaton_task_config,
    agent = ner_validation_agent,
    output_pydantic= NERValidationOutput
)

preliminary_diagnosis_task = Task(
    config = preliminary_diagnose_task_config,
    agent = preliminary_diagnosis_agent,
    output_pydantic= PreliminaryDiagnosisListOutput,
)

report_writing_task = Task(
    config=report_writing_task_config,
    agent=report_writer_agent,
    output_file="report.md"
)

## Crew

ner_validation_crew = Crew(
    agents=[ner_validation_agent, ],
    tasks=[ner_validation_task, ],
    verbose=True
)

prelim_diag_crew = Crew(
    agents=[preliminary_diagnosis_agent],
    tasks=[preliminary_diagnosis_task],
    verbose=True
)

report_writing_crew = Crew(
    agents=[report_writer_agent,],
    tasks=[report_writing_task,],
    verbose=True
)
