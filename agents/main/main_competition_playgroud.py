from mable.examples import environment, fleets, companies

from agents.main.final import CompanyWhatever
from agents.Tom.SuperCoolCompany import SuperCoolCompany


def build_specification():
    specifications_builder = environment.get_specification_builder(
        trade_occurrence_frequency=30,
        trades_per_occurrence=5,
        num_auctions=12,
        environment_files_path='resources/')
    # my_fleet = fleets.mixed_fleet(num_suezmax=1, num_aframax=1, num_vlcc=1)
    # specifications_builder.add_company(SuperCoolCompany.Data(SuperCoolCompany, my_fleet, SuperCoolCompany.__name__))
    my_fleet = fleets.mixed_fleet(num_suezmax=1, num_aframax=1, num_vlcc=1)
    specifications_builder.add_company(CompanyWhatever.Data(CompanyWhatever, my_fleet, CompanyWhatever.__name__))
    # my_fleet = fleets.mixed_fleet(num_suezmax=1, num_aframax=1, num_vlcc=1)
    # specifications_builder.add_company(Dumbass.Data(Dumbass, my_fleet, Dumbass.__name__))
    arch_enemy_fleet = fleets.mixed_fleet(num_suezmax=1, num_aframax=1, num_vlcc=1)
    specifications_builder.add_company(
        companies.MyArchEnemy.Data(
            companies.MyArchEnemy, arch_enemy_fleet, "Arch Enemy Ltd.",
            profit_factor=1.65))
    the_scheduler_fleet = fleets.mixed_fleet(num_suezmax=1, num_aframax=1, num_vlcc=1)
    specifications_builder.add_company(
        companies.TheScheduler.Data(
            companies.TheScheduler, the_scheduler_fleet, "The Scheduler LP",
            profit_factor=1.65))
    sim = environment.generate_simulation(
        specifications_builder,
        show_detailed_auction_outcome=True,
        global_agent_timeout=60)
    sim.run()


if __name__ == '__main__':
    build_specification()
