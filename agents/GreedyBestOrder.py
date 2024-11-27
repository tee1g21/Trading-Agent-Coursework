from typing import List, Dict, Tuple

from mable.cargo_bidding import TradingCompany
from mable.event_management import TravelEvent, CargoTransferEvent
from mable.extensions.fuel_emissions import VesselWithEngine
from mable.shipping_market import TimeWindowTrade
from mable.transport_operation import ScheduleProposal, Bid
from mable.transportation_scheduling import Schedule


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
        trades = sorted(trades, key=lambda t: t.latest_pickup)
        schedules = {
            vessel: (vessel.schedule.copy(), [])
            for vessel
            in self._fleet
        }

        total_profit, vessel_data = self.__propose_schedules(trades, schedules)

        if total_profit <= 0 or not any(map(lambda x: x[2], vessel_data.values())):
            return None

        schedule_proposal = {vessel: schedule for vessel, (_, schedule, _) in vessel_data.items()}

        trade_values = {}
        for _, (vessel_profit, _, trades) in vessel_data.items():
            trade_values.update({
                trade: vessel_profit/len(trades)
                for trade in trades
            })

        return ScheduleProposal(schedule_proposal, list(trade_values.keys()), trade_values)

    def __propose_schedules(self,
                            trades: List[TimeWindowTrade],
                            schedules: Dict[VesselWithEngine, Tuple[Schedule, List[TimeWindowTrade]]],
                            ) -> Tuple[int,
                                       Dict[VesselWithEngine, Tuple[int,
                                                                    Schedule,
                                                                    List[TimeWindowTrade]]]]:
        if len(trades) == 0:
            vessel_profits = {
                vessel: (self.calculate_schedule_cost(schedule, vessel), schedule, trades)
                for vessel, (schedule, trades)
                in schedules.items()
            }
            return sum(map(lambda x: x[0], vessel_profits.values())), vessel_profits

        options = [self.__propose_schedules(trades[1:], schedules)]
        for vessel in self._fleet:
            schedules_copy = GreedyBestOrder.__copy_schedule_data(schedules)

            schedules_copy[vessel][0].add_transportation(trades[0])
            schedules_copy[vessel][1].append(trades[0])

            if not schedules_copy[vessel][0].verify_schedule():
                continue

            options.append(self.__propose_schedules(trades[1:],
                                                    schedules_copy))

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

    def calculate_schedule_cost(self,
                                schedule: Schedule,
                                vessel: VesselWithEngine
                                ) -> int:
        cost = 0
        for x in schedule:
            if isinstance(x, TravelEvent):
                cost += vessel.get_laden_consumption(self.headquarters.get_network_distance(x.location.origin,
                                                                                            x.location.destination) / vessel.speed,
                                                     vessel.speed)
            elif isinstance(x, CargoTransferEvent):
                cost += vessel.get_loading_consumption(vessel.get_loading_time(x.trade.cargo_type, x.trade.amount))

        return cost