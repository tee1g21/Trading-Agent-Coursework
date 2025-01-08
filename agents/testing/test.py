# performance testing

import inspect
import os
import typing

from loguru import logger
from mable.cargo_bidding import TradingCompany
from mable.competition.generation import AuctionSimulationEngine
from mable.examples import environment, fleets
from mable.extensions.fuel_emissions import FuelSpecsBuilder
from mable.observers import EventObserver


class PerformanceTest:

    _resources = "./resources"
    _output_directory = "./out"

    def setup(self) -> None:
        if not os.path.exists(self._resources):
            raise ValueError(
                f"Cannot find Resources Directory relative to {os.path.abspath(os.path.curdir)}"
            )
        self._specifications_builder = environment.get_specification_builder(
            environment_files_path=self._resources,
            trade_occurrence_frequency=30,
            trades_per_occurrence=1,
            num_auctions=2,
            fixed_trades=None,
            use_only_precomputed_routes=True,
        )

    def add_company_random_fleet(
        self,
        company: type[TradingCompany],
        fleet_nums: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        my_fleet = fleets.mixed_fleet(
            num_suezmax=fleet_nums[0], num_aframax=fleet_nums[1], num_vlcc=fleet_nums[2]
        )
        self.add_company_with_fleet(company, my_fleet)

    def add_company_with_fleet(
        self, company: type[TradingCompany], fleet: list
    ) -> None:
        if not inspect.isclass(company):
            raise TypeError("company must be a class")
        if not issubclass(company, TradingCompany):
            raise TypeError("company must be TradingCompany")
        if self._specifications_builder is None:
            raise AssertionError("Must run Performancetest.setup first")

        logger.debug(f"Adding Company {{{company.__name__}, {fleet}}}")
        self._specifications_builder.add_company(
            company.Data(company, fleet, company.__name__)
        )

    def __init__(self):
        self.sim: AuctionSimulationEngine | None = None
        self._specifications_builder: FuelSpecsBuilder | None = None

    def set_output_directory(self, dir: str) -> None:
        self._output_directory = dir

    def get_output_directory(self) -> str:
        return self._output_directory

    def test(self) -> None:
        if not os.path.exists(self._output_directory):
            os.makedirs(self._output_directory)

        self.sim = typing.cast(
            AuctionSimulationEngine,
            environment.generate_simulation(
                self._specifications_builder,
                show_detailed_auction_outcome=False,
                output_directory=self._output_directory,
                global_agent_timeout=60,
            ),
        )

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


class TestingEnvironment:

    def __init__(self):
        self.all_test_results = []
        self._test_companies: list[type[TradingCompany]] = []
        self._test_fleet_combos: list[tuple[int, int, int]] = []

    def setup_companies(self, companies: list[type[TradingCompany]]):
        self._test_companies = companies

    def setup_random_fleets(self, max_suezmax=1, max_aframax=1, max_vlcc=1):
        self._test_fleet_combos = []
        for suez_num in range(max_suezmax):
            for afra_num in range(max_aframax):
                for vlcc_num in range(max_vlcc):
                    self._test_fleet_combos.append((suez_num, afra_num, vlcc_num))

    def run_test(self):
        logger.info("Creating Performance Test")
        performance_test = PerformanceTest()
        logger.info("Setup Performance Test")
        performance_test.setup()

        logger.info("Adding Companies")
        performance_test.add_company_random_fleet(Dumbass2)
        performance_test.add_company_random_fleet(SuperCoolCompany)

        logger.info("Running Test")
        performance_test.test()
        logger.info("End Performance Test")

    def print_results(self):
        pass
