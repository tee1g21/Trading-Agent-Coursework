from mable.cargo_bidding import TradingCompany
from mable.transport_operation import ScheduleProposal

import attrs
from marshmallow import fields


class SuperCoolCompany(TradingCompany):
    
    def __init__(self, fleet, name, profit_factor=1.65):
        """
        :param fleet: The companies fleet.
        :param name: The companies name.
        :param profit_factor: The companies profit factor, i.e. factor applied to cost to determine bids.
        """
        super().__init__(fleet, name)
        self._profit_factor = profit_factor
        self.auction_ledger = {}
        
    @attrs.define
    class Data(TradingCompany.Data):
        profit_factor: float = 1.65

        class Schema(TradingCompany.Data.Schema):
            profit_factor = fields.Float(default=1.65)
        
    def pre_inform(self, trades, time):
        return super().pre_inform(trades, time)
    
    def inform(self, trades, *args, **kwargs):
        return super().inform(trades, *args, **kwargs)
    
    def receive(self, contracts, auction_ledger=None, *args, **kwargs):
        self.auction_ledger = auction_ledger
        return super().receive(contracts, auction_ledger, *args, **kwargs)
    
    def propose_schedules(self, trades):
        return super().propose_schedules(trades)
    
    def predict_cost(self, vessel, trade):
        
        # calculate loading and unloading costs
        loading_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
        loading_costs = vessel.get_loading_consumption(loading_time)
        unloading_costs = vessel.get_unloading_consumption(loading_time)
        
        # calculate travel costs
        travel_distance = self.headquarters.get_network_distance(
            trade.origin_port, trade.destination_port)
        travel_time = vessel.get_travel_time(travel_distance)
        travel_cost = vessel.get_laden_consumption(travel_time, vessel.speed)
        
        # calculate total cost
        total_cost = loading_costs + unloading_costs + travel_cost  
        
        return total_cost
    
    def find_competing_vessels(self, trade):
        competing_vessels = {}                
        companies = self.headquarters.get_companies()

        # Find the closest vessel for each company
        if companies:
            for company in companies:
                closest_vessel = None
                min_distance = float('inf')

                for vessel in company.fleet:                    
                    print(f"{company.name}'s {vessel.name}")                    
                    # distance from vessel to trade origin port
                    distance = self.headquarters.get_network_distance(vessel.location, trade.origin_port)
                    # update min distance and closest vessel
                    if distance < min_distance:
                        min_distance = distance
                        closest_vessel = vessel

                # add the closest vessel to competing vessels
                if closest_vessel:
                    competing_vessels[company] = closest_vessel
                    
        return competing_vessels
    
    def calculate_competitor_profit_factors(self, auction_ledger):
        profit_factors = {}

        for company_name, won_contracts in auction_ledger.items():
            company_fleet = [c for c in self.headquarters.get_companies() if c.name == company_name].pop().fleet
            company_bid_factors = []

            for contract in won_contracts:
                trade = contract.trade
                competitor_bid = contract.payment

                best_cost = float('inf')
                for vessel in company_fleet:
                    predicted_cost = self.predict_cost(vessel, trade)
                    if predicted_cost < best_cost:
                        best_cost = predicted_cost

                bid_factor = competitor_bid / best_cost if best_cost > 0 else float('inf')
                company_bid_factors.append(bid_factor)

                print(f"Company: {company_name}, Trade: {trade.origin_port.name} -> {trade.destination_port.name}")
                print(f"  Bid: {competitor_bid}, Cost: {best_cost}, Factor: {bid_factor}")

            profit_factors[company_name] = company_bid_factors

        return profit_factors
    
    def order_trades_by_future_distance(self, trades):
        trade_distances = {}

        for current_trade in trades:
            closest_future_trade = None
            min_distance = float('inf')

            if self._future_trades:  # Ensure there are future trades available
                for future_trade in self._future_trades:
                    distance = self.headquarters.get_network_distance(
                        current_trade.destination_port, future_trade.origin_port
                    )
                    if distance < min_distance:
                        min_distance = distance
                        closest_future_trade = future_trade

            # Store the minimum distance for the current trade
            trade_distances[current_trade] = min_distance

            # Debug output
            print(f"Trade: {current_trade.origin_port.name} -> {current_trade.destination_port.name}")
            if closest_future_trade:
                print(f"\tClosest future trade: {closest_future_trade.origin_port.name} -> {closest_future_trade.destination_port.name}")
                print(f"\tDistance: {min_distance}")
            else:
                print("\tNo future trades available")

        # Sort trades by distance to the closest future trade
        sorted_trades = sorted(trade_distances.keys(), key=lambda trade: trade_distances[trade])

        return sorted_trades
    
    def order_trades_by_profit(self, trades):
        trade_profits = {}

        for trade in trades:
            # Predict cost for executing the trade
            best_cost = float('inf')
            for vessel in self.fleet:  # Consider each vessel for the trade
                predicted_cost = self.predict_cost(vessel, trade)
                if predicted_cost < best_cost:
                    best_cost = predicted_cost

            # Calculate revenue (assume trade.amount * some constant per unit)
            revenue = trade.payment * 1  # Replace 1 with actual revenue multiplier

            # Calculate profit
            profit = revenue - best_cost
            trade_profits[trade] = profit

        # Sort trades by profit
        sorted_trades = sorted(trade_profits.keys(), key=lambda trade: trade_profits[trade], reverse=True)
        return sorted_trades

                        
                        
                        