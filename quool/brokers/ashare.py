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
                type=type,
                code=code,
                quantity=quantity,
                exectype=exectype,
                limit=limit,
                trigger=trigger,
                id=id,
                valid=valid,
            )
