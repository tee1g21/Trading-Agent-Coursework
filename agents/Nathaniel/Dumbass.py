import math

from mable.cargo_bidding import TradingCompany
from mable.extensions.fuel_emissions import VesselWithEngine
from mable.shipping_market import Trade
from mable.transport_operation import ScheduleProposal


class Dumbass(TradingCompany):
    def __init__(self, fleet, name):
        super().__init__(fleet, name)
        self.profit_factor = 1.4

    def receive(self, contracts, auction_ledger=None, *args, **kwargs):
        print(f'{self.name} won trades: {contracts}')

    def propose_schedules(self, trades):
        # safety in case this blows up
        trades = trades.copy()

        schedules = {}
        scheduled_trades = []
        costs = {}

        for vessel in self.fleet:
            current_schedule = vessel.schedule
            simple_schedule = current_schedule.get_simple_schedule()

            print(simple_schedule)
            if len(simple_schedule) == 0:
                end_location = vessel.location
            else:
                assert simple_schedule[-1][0] == "DROP_OFF"
                end_location = simple_schedule[-1][1].destination_port

            closest_trade = None
            closest_dist = math.inf
            for trade in trades:
                dist = self.headquarters.get_network_distance(end_location, trade.origin_port)

                if dist < closest_dist:
                    closest_trade = trade
                    closest_dist = dist

            if closest_trade is not None:
                new_schedule = current_schedule.copy()
                new_schedule.add_transportation(closest_trade)

                if new_schedule.verify_schedule():
                    schedules[vessel] = new_schedule
                    scheduled_trades.append(closest_trade)
                    trades.remove(closest_trade)
                    cost = self.calculate_trade_cost(closest_dist, vessel, closest_trade) * self.profit_factor
                    print(f'bidding: [{closest_trade.origin_port.name} => {closest_trade.origin_port.name}]: {cost}')
                    costs[closest_trade] = cost

        return ScheduleProposal(schedules, scheduled_trades, costs)

    def calculate_trade_cost(self,
                             relocate_dist: float,
                             vessel: VesselWithEngine,
                             trade: Trade
                             ) -> float:
        load_unload_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
        return (
            # relocate cost
            vessel.get_ballast_consumption(vessel.get_travel_time(relocate_dist), vessel.speed) +
            # loading cost
            vessel.get_loading_consumption(load_unload_time) +
            # moving cost
            vessel.get_laden_consumption(
                vessel.get_travel_time(
                    self.headquarters.get_network_distance(trade.origin_port,
                                                           trade.destination_port)),
                vessel.speed) +
            # unloading cost
            vessel.get_unloading_consumption(load_unload_time)
        )