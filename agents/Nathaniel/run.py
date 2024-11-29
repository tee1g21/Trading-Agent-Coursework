from mable.examples import environment, fleets

from agents import AGENTS

if __name__ == '__main__':
	specifications_builder = environment.get_specification_builder(environment_files_path="resources/")
	fleet = fleets.example_fleet_1()

	for Agent in AGENTS:
		specifications_builder.add_company(Agent.Data(Agent, fleet, Agent.__name__))

	sim = environment.generate_simulation(specifications_builder)
	sim.run()
