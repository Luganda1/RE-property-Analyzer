# src/latest_ai_development/crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
import yaml




@CrewBase
class LatestAiDevelopmentCrew():
    """ LatestAiDevelopment crew """
    
    def load_yaml_file(file_path):
        with open(file_path, 'r') as f:
        return yaml.safe_load(f)
    
    agents_config = load_yaml_file('./agent.yaml')
    tasks_config = load_yaml_file('./task.yaml')

    @agent
    def researcher(self) -> Agent:
        return Agent(
        config=self.agents_config['researcher'],
        verbose=True,
        tools=[SerperDevTool()]
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
        config=self.agents_config['reporting_analyst'],
        verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
        config=self.tasks_config['research_task'],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
        config=self.tasks_config['reporting_task'],
        output_file='output/report.md' # This is the file that will be contain the final report.
        )

    @crew
    def crew(self) -> Crew:
        """Creates the LatestAiDevelopment crew"""
        return Crew(
        agents=self.agents, # Automatically created by the @agent decorator
        tasks=self.tasks, # Automatically created by the @task decorator
        process=Process.sequential,
        verbose=True,
        )
