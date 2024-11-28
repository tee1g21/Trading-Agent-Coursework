from typing import List, Dict, Tuple

from mable.cargo_bidding import TradingCompany
from mable.event_management import TravelEvent, CargoTransferEvent
from mable.extensions.fuel_emissions import VesselWithEngine
from mable.shipping_market import TimeWindowTrade
from mable.transport_operation import ScheduleProposal, Bid
from mable.transportation_scheduling import Schedule


# TODO: Known bugs.
# the cost of a vessels already accepted contracts is considered when calculating bids for new contracts,
#     this causes the current accepted contracts costs to be counted as distributed costs towards new contracts also
# I think cost calculation is slightly wrong, in theory this bot trading alone should always have a profit of 0 but somehow it manages to have a slight profit so yay?
# contracts are considered atomically, we should also consider Pickup, Pickup, Drop-off, Drop-off style schedules to reduce travel
# future contracts are not currently considered, we should account for them and add a bias (or reduced cost) to contracts that end near the start of future contracts we like

# tries to find the best ordering of contracts and ships, ignoring the splitting of contracts and other bots
class GreedyBestOrder(TradingCompany):
    def __init__(self, fleet, name):
        super().__init__(fleet, name)

        self.future_trades = set()

    def pre_inform(self, trades, time):
        for trade in trades:
            self.future_trades.add(trade)

    def inform(self, trades, *args, **kwargs):
        for trade in trades:
            self.future_trades.add(trade)

        print('---auction start----')
        proposed_scheduling = self.propose_schedules(trades)

        if proposed_scheduling is None:
            print('---auction end, no bids----')
            return []

        bids = [Bid(trade=trade,
                    amount=proposed_scheduling.costs.get(trade, 0))
                for trade
                in proposed_scheduling.scheduled_trades]
        print(f'bids: {bids}')
        print('---auction end----')
        return bids

    def receive(self, contracts, auction_ledger=None, *args, **kwargs):
        print(f'{self.name} won trades: {contracts}')

        trades = [one_contract.trade for one_contract in contracts]
        scheduling_proposal = self.propose_schedules(trades)

        if scheduling_proposal is None:
            return

        rejected_trades = self.apply_schedules(scheduling_proposal.schedules)

        if len(rejected_trades) > 0:
            print(f"{len(rejected_trades)} rejected trades.")

    def propose_schedules(self, trades: List[TimeWindowTrade]) -> ScheduleProposal | None:
        # sort the trades by latest pickup from earliest to latest,
        trades = sorted(trades, key=lambda t: t.latest_pickup)
        # initialise the datastructures that represents a vessels info, include the current schedule as we must
        # continue from that
        schedules = {
            vessel: (vessel.schedule.copy(), [])
            for vessel
            in self._fleet
        }

        _, vessel_data = self.__propose_schedules(trades, schedules)

        # restructure data as required for SP
        schedules = {}
        trades = []
        costs = {}

        for vessel, (schedule, vessel_trades) in vessel_data.items():
            schedules[vessel] = schedule
            for trade, cost in vessel_trades:
                trades.append(trade)
                costs[trade] = cost

        return ScheduleProposal(schedules, trades, costs)

    def __propose_schedules(self,
                            trades: List[TimeWindowTrade],
                            # similar to the return type
                            # a dict of vessels to pairs of that vessels schedule and all new trades
                            schedules: Dict[VesselWithEngine, Tuple[Schedule, List[TimeWindowTrade]]]
                            # this datastructures is fucked but
                            # ( total profit,
                            #     dict[ key: vessel
                            #           value: (
                            #                      total vesel profit,
                            #                      vessel schedule,
                            #                      new trades for this vessel
                            #                  )
                            #         ]
                            # )
                            ) -> Tuple[float,
                                       Dict[VesselWithEngine, Tuple[Schedule,
                                                                    List[Tuple[TimeWindowTrade,
                                                                               float]]]]]:
        if len(trades) == 0:
            # base case, there are no more trades to consider

            vessels_trades = {}
            total_cost = 0
            for vessel, (schedule, vessel_trades) in schedules.items():
                if len(vessel_trades) == 0:
                    vessels_trades[vessel] = (schedule, [])
                    continue

                contract_costs = []
                total_contract_cost = 0
                for trade in vessel_trades:
                    contract_cost = self.calculate_contract_cost(trade, vessel)
                    total_contract_cost += contract_cost
                    contract_costs.append((trade, contract_cost))

                total_schedule_cost = self.__calculate_schedule_cost(schedule, vessel)
                distributed_cost = (total_schedule_cost - total_contract_cost) / len(vessel_trades)

                # guard in case of bugs
                if distributed_cost <= 0:
                    distributed_cost = 0

                # add distributed costs to all contracts
                contract_costs = list(map(lambda trade_data: (trade_data[0], trade_data[1] + distributed_cost),
                                          contract_costs))

                vessels_trades[vessel] = (schedule, contract_costs)
                total_cost += total_schedule_cost
            return total_cost, vessels_trades

        # consider all options for the current trade

        # option of not accepting this trade
        options = [self.__propose_schedules(trades[1:], schedules)]
        for vessel in self._fleet:
            # try to assign the contract to each vesel
            # copy the datastructures to prevent weird shared ref errors
            schedules_copy = GreedyBestOrder.__copy_schedule_data(schedules)

            schedules_copy[vessel][0].add_transportation(trades[0])
            schedules_copy[vessel][1].append(trades[0])

            # ensure the vessel can conduct the trade, skip if it can't
            if not schedules_copy[vessel][0].verify_schedule():
                continue

            options.append(self.__propose_schedules(trades[1:],
                                                    schedules_copy))

        # pick the best allocation of trades (highest profit)
        best = max(options,
                   key=lambda o: o[0])

        print(f'best trades are: {best}')

        return best

    @staticmethod
    def __copy_schedule_data(data: Dict[VesselWithEngine, Tuple[Schedule, List[TimeWindowTrade]]]
                             ) -> Dict[VesselWithEngine, Tuple[Schedule, List[TimeWindowTrade]]]:
        return {
            vessel: (schedule.copy(), trades.copy())
            for vessel, (schedule, trades)
            in data.items()
        }

    def calculate_contract_cost(self,
                                contract: TimeWindowTrade,
                                vessel: VesselWithEngine
                                ) -> float:
        return (
            # loading cost
            vessel.get_loading_consumption(vessel.get_loading_time(contract.cargo_type, contract.amount)) +
            # moving cost
            vessel.get_ballast_consumption(
                vessel.get_travel_time(
                    self.headquarters.get_network_distance(contract.origin_port,
                                                           contract.destination_port)),
                vessel.speed) +
            # unloading cost
            vessel.get_unloading_consumption(vessel.get_loading_time(contract.cargo_type, contract.amount))
        )

    def __calculate_schedule_cost(self,
                                  schedule: Schedule,
                                  vessel: VesselWithEngine
                                  ) -> int:
        # calculate the total cost of the schedule for a vessel
        cost = 0
        for x in schedule:
            if isinstance(x, TravelEvent):
                cost += vessel.get_laden_consumption(self.headquarters.get_network_distance(x.location.origin,
                                                                                            x.location.destination) / vessel.speed,
                                                     vessel.speed)
            elif isinstance(x, CargoTransferEvent):
                cost += vessel.get_loading_consumption(vessel.get_loading_time(x.trade.cargo_type, x.trade.amount))

        return cost