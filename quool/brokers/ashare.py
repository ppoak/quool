from quool import Broker


class AShareBroker(Broker):

    def create(
        self,
        type: str,
        code: str,
        quantity: int,
        exectype: str,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ):
        quantity = int(quantity // 100 * 100)
        if quantity > 0:
            return super().create(
                type, code, quantity, exectype, limit, trigger, id, valid
            )
