import dataclasses
import timeit
from typing import List, Dict

from mable.cargo_bidding import TradingCompany
from mable.event_management import TravelEvent, CargoTransferEvent
from mable.extensions.fuel_emissions import VesselWithEngine
from mable.shipping_market import Trade
from mable.simulation_space import Location, OnJourney
from mable.transport_operation import ScheduleProposal
from mable.transportation_scheduling import Schedule


@dataclasses.dataclass
class SubScheduleProposal:
    schedule: Schedule
    start_price: float
    extra_trades: List[Trade]

    def copy(self):
        return SubScheduleProposal(
            schedule=self.schedule.copy(),
            start_price=self.start_price,
            extra_trades=self.extra_trades.copy()
        )


@dataclasses.dataclass
class ValuedSubScheduleProposal:
    proposal: SubScheduleProposal
    costs: Dict[Trade, float]
    heuristic: float

    def copy(self):
        return ValuedSubScheduleProposal(
            proposal=self.proposal.copy(),
            costs=self.costs.copy(),
            heuristic=self.heuristic
        )


@dataclasses.dataclass
class ValuedScheduleProposal:
    proposals: Dict[VesselWithEngine, ValuedSubScheduleProposal]
    heuristic_sum: float

    def __init__(
            self,
            proposals: Dict[VesselWithEngine, ValuedSubScheduleProposal],
            heuristic_sum: float
    ):
        self.proposals = proposals
        self.heuristic_sum = heuristic_sum

    @classmethod
    def calc_heuristic(
            cls,
            proposals: Dict[VesselWithEngine, ValuedSubScheduleProposal]
    ):
        return cls(
            proposals=proposals,
            heuristic_sum=sum(map(lambda x: x.heuristic, proposals.values()))
        )

    def copy(self):
        return ValuedScheduleProposal(
            proposals={
                vessel: schedule.copy()
                for vessel, schedule in self.proposals.items()
            },
            heuristic_sum=self.heuristic_sum
        )

    def aggregate(self) -> ScheduleProposal:
        agg_schedules = {}
        agg_trades = []
        agg_costs = {}

        for vessel, valued_proposal in self.proposals.items():
            agg_schedules[vessel] = valued_proposal.proposal.schedule
            agg_trades += valued_proposal.proposal.extra_trades
            agg_costs.update(valued_proposal.costs)

        return ScheduleProposal(agg_schedules, agg_trades, agg_costs)


class Dumbass2(TradingCompany):
    def __init__(self, fleet, name):
        super().__init__(fleet, name)
        self.profit_factor = 1.4

    def receive(self, contracts, auction_ledger=None, *args, **kwargs):
        if len(contracts) == 0:
            print("no contracts won")
        else:
            print(f'{self.name} won trades:')
            for contract in contracts:
                print(f'\t[{contract.trade.origin_port} => {contract.trade.destination_port}]: {contract.payment}')
        trades = [one_contract.trade for one_contract in contracts]
        scheduling_proposal = self.propose_schedules(trades, True)
        _ = self.apply_schedules(scheduling_proposal.schedules)

    def propose_schedules(self, trades, strict=False) -> ScheduleProposal:
        start_schedules = {
            vessel: SubScheduleProposal(
                vessel.schedule,
                self.calculate_schedule_cost(vessel, vessel.schedule),
                [])
            for vessel in self.fleet
        }

        start = timeit.default_timer()
        evaluated_schedules = self.__propose_schedules(trades, start_schedules, strict)
        stop = timeit.default_timer()
        aggregate = evaluated_schedules.aggregate()

        if not strict:
            if len(aggregate.costs) == 0:
                print("no bids")
            else:
                print("----- bids -----")
                for trade, cost in aggregate.costs.items():
                    print(f"[{trade.origin_port} => {trade.destination_port}]: {cost}")
                print("----------------")
            print(f'time: {stop - start}')

        return aggregate

    def __propose_schedules(
            self,
            trades: List[Trade],
            schedules: Dict[VesselWithEngine, SubScheduleProposal],
            strict: bool
    ) -> ValuedScheduleProposal:
        if len(trades) == 0:
            return ValuedScheduleProposal.calc_heuristic({
                vessel: self.evaluate_proposal(vessel, proposal, strict)
                for vessel, proposal in schedules.items()
            })

        current_trade = trades[0]
        rest_trades = trades[1:]

        best_option = self.__propose_schedules(rest_trades, schedules, strict)
        for vessel, proposal in schedules.items():
            old_schedule = proposal.schedule.copy()
            proposal.extra_trades.append(current_trade)

            for option in Dumbass2.generate_options(current_trade, proposal.schedule):
                proposal.schedule = option
                evaluation = self.__propose_schedules(rest_trades, schedules, strict)
                if evaluation.heuristic_sum > best_option.heuristic_sum:
                    best_option = evaluation.copy()

            proposal.schedule = old_schedule
            proposal.extra_trades.pop()

        return best_option

    @staticmethod
    def generate_options(
            trade: Trade,
            schedule: Schedule
    ) -> List[Schedule]:
        options = []
        insertion_points = schedule.get_insertion_points()[-8:]
        for k in range(len(insertion_points)):
            idx_pick_up = insertion_points[k]
            insertion_point_after_idx_k = insertion_points[k:]
            for m in range(len(insertion_point_after_idx_k)):
                idx_drop_off = insertion_point_after_idx_k[m]

                schedule_copy = schedule.copy()
                schedule_copy.add_transportation(trade, idx_pick_up, idx_drop_off)
                if schedule_copy.verify_schedule():
                    options.append(schedule_copy)
        return options

    def evaluate_proposal(
            self,
            vessel: VesselWithEngine,
            proposal: SubScheduleProposal,
            strict: bool
    ) -> ValuedSubScheduleProposal:
        if strict:
            return self.evaluate_proposal_strict(vessel, proposal)
        else:
            return self.evaluate_proposal_heuristic(vessel, proposal)

    def evaluate_proposal_strict(
            self,
            vessel: VesselWithEngine,
            proposal: SubScheduleProposal
    ) -> ValuedSubScheduleProposal:
        return ValuedSubScheduleProposal(
            proposal=proposal,
            costs={},
            heuristic=-self.calculate_schedule_cost(vessel, proposal.schedule)
        )

    def evaluate_proposal_heuristic(
            self,
            vessel: VesselWithEngine,
            proposal: SubScheduleProposal
    ) -> ValuedSubScheduleProposal:
        if len(proposal.extra_trades) == 0:
            return ValuedSubScheduleProposal(
                proposal=proposal,
                costs={},
                heuristic=0
            )

        schedule_cost = self.calculate_schedule_cost(vessel, proposal.schedule)
        schedule_cost_delta = schedule_cost - proposal.start_price

        trade_costs = {}
        trade_cost_total = 0
        for trade in proposal.extra_trades:
            trade_cost = self.calculate_trade_cost(vessel, trade)
            trade_costs[trade] = trade_cost
            trade_cost_total += trade_cost

        distributed_cost = (schedule_cost_delta - trade_cost_total) / len(proposal.extra_trades)
        trade_costs = {
            trade: cost + distributed_cost
            for trade, cost in trade_costs.items()
        }

        efficiency = trade_cost_total / schedule_cost_delta

        return ValuedSubScheduleProposal(
            proposal=proposal,
            costs=trade_costs,
            heuristic=efficiency
        )

    def calculate_trade_cost(
            self,
            vessel: VesselWithEngine,
            trade: Trade
    ) -> float:
        load_unload_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
        return (
            # loading cost
            vessel.get_loading_consumption(load_unload_time) +
            # moving cost
            vessel.get_laden_consumption(
                vessel.get_travel_time(
                    self.headquarters.get_network_distance(trade.origin_port, trade.destination_port)),
                vessel.speed) +
            # unloading cost
            vessel.get_unloading_consumption(load_unload_time)
        )

    def calculate_schedule_cost(
            self,
            vessel: VesselWithEngine,
            schedule: Schedule
    ) -> float:
        total_cost = 0
        current_location = vessel.location

        if isinstance(current_location, OnJourney):
            current_location = current_location.destination

        for type, trade in schedule.get_simple_schedule():
            if type == 'PICK_UP':
                dst = trade.origin_port
            elif type == 'DROP_OFF':
                dst = trade.destination_port
            else:
                raise ValueError(f'Unknown schedule type {type}')

            total_cost += vessel.get_ballast_consumption(
                vessel.get_travel_time(
                    self.headquarters.get_network_distance(
                        current_location,
                        dst)),
                vessel.speed)

            total_cost += vessel.get_loading_consumption(
                vessel.get_loading_time(trade.cargo_type, trade.amount))
            current_location = dst
        return total_cost