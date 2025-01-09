from mable.examples import environment, fleets, companies

import SuperCoolCompany
import Dumbass3

def build_specification():
    number_of_month = 12
    trades_per_auction = 5
    specifications_builder = environment.get_specification_builder(
        trades_per_occurrence=trades_per_auction,
        num_auctions=number_of_month, environment_files_path="../../resources")
    my_fleet = fleets.mixed_fleet(num_suezmax=1, num_aframax=1, num_vlcc=1)
    specifications_builder.add_company(Dumbass3.Dumbass3.Data(Dumbass3.Dumbass3, my_fleet, Dumbass3.Dumbass3.__name__))
    arch_enemy_fleet = fleets.mixed_fleet(num_suezmax=1, num_aframax=1, num_vlcc=1)
    specifications_builder.add_company(
        companies.MyArchEnemy.Data(
            companies.MyArchEnemy, arch_enemy_fleet, "Arch Enemy Ltd.",
            profit_factor=1.5))
    the_scheduler_fleet = fleets.mixed_fleet(num_suezmax=1, num_aframax=1, num_vlcc=1)
    specifications_builder.add_company(
        companies.TheScheduler.Data(
            companies.TheScheduler, the_scheduler_fleet, "The Scheduler LP",
            profit_factor=1.4))
    sim = environment.generate_simulation(
        specifications_builder,
        show_detailed_auction_outcome=True,
        global_agent_timeout=60)
    sim.run()


if __name__ == '__main__':
    build_specification()
