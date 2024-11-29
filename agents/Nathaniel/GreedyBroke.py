from mable.cargo_bidding import TradingCompany


# default bot, bids on everything it can but bids 0
class GreedyBroke(TradingCompany):
	def receive(self, contracts, auction_ledger=None, *args, **kwargs):
		print(f'{self.name} won trades: {contracts}')