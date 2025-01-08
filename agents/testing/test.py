# performance testing

# Usage
from mable.examples import environment, fleets
import os
import inspect

# typing
import typing
from mable.cargo_bidding import TradingCompany
from mable.competition.generation import AuctionSimulationEngine
from mable.observers import EventObserver
from mable.extensions.fuel_emissions import FuelSpecsBuilder


from loguru import logger

class PerformanceTest():

    _resources = "./resources"
    _output_directory = "./out"

    def setup(self) -> None:
        if not os.path.exists(self._resources):
            raise ValueError(f"Cannot find Resources Directory relative to {os.path.abspath(os.path.curdir)}")
        self._specifications_builder = environment.get_specification_builder(
            environment_files_path=self._resources,
            trade_occurrence_frequency=30,
            trades_per_occurrence=1,
            num_auctions=2,
            fixed_trades=None,
            use_only_precomputed_routes=True
        )
    
    def add_company_random_fleet(self, company: type[TradingCompany]) -> None:
        if not inspect.isclass(company):
            raise TypeError("company must be a class")
        if not issubclass(company, TradingCompany):
            raise TypeError("company must be TradingCompany")
        if self._specifications_builder is None:
            raise AssertionError("Must run Performancvetest.setup first")
        
        logger.debug(f"Adding Company {company.__name__}")
        my_fleet = fleets.mixed_fleet(num_suezmax=1, num_aframax=1, num_vlcc=1)
        self._specifications_builder.add_company(company.Data(company, my_fleet, company.__name__))

    def __init__(self):
        self.sim: AuctionSimulationEngine | None = None
        self._specifications_builder: FuelSpecsBuilder | None = None

    def test(self) -> None:
        if not os.path.exists(self._output_directory):
            os.makedirs(self._output_directory)

        self.sim = typing.cast(AuctionSimulationEngine, environment.generate_simulation(
            self._specifications_builder,
            show_detailed_auction_outcome=False,
            output_directory=self._output_directory,
            global_agent_timeout=60
        ))

        if self.sim:
            self.sim.run()
        else:
            logger.error("No sim to run!")
    
    def get_sim_results(self):
        if self.sim is None:
            return
        observers = self.sim.get_event_observers()
        for observer in observers:
            observer = typing.cast(EventObserver, observer)
            logger.debug(observer)
